"""Routes: POST /connect/api, POST /connect/database"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    IMPORTED_FLAT_PATH,
    ALLOWED_DB_TYPES, DB_QUERY_TIMEOUT, DB_MAX_ROWS,
    API_FETCH_TIMEOUT, API_MAX_ROWS,
)

router = APIRouter(tags=["connectors"])


@router.post("/connect/api")
async def connect_api(body: dict) -> dict:
    """
    Fetch data from a REST API / OData endpoint.

    Body:
      url         : str (required)
      method      : str (GET or POST, default GET)
      headers     : dict (optional)
      auth_type   : str (none, bearer, api_key, basic)
      auth_value  : str (token or key value)
      json_path   : str (dot-separated path to records array, e.g. "data.records")
      preview_only: bool (default false)
    """
    import httpx

    url = body.get("url")
    if not url:
        raise HTTPException(400, "URL is required.")

    method = body.get("method", "GET").upper()
    headers = body.get("headers") or {}
    auth_type = body.get("auth_type", "none")
    auth_value = body.get("auth_value", "")
    json_path = body.get("json_path", "")
    preview_only = body.get("preview_only", False)

    if auth_type == "bearer" and auth_value:
        headers["Authorization"] = f"Bearer {auth_value}"
    elif auth_type == "api_key" and auth_value:
        headers["X-API-Key"] = auth_value

    try:
        async with httpx.AsyncClient(timeout=API_FETCH_TIMEOUT) as client:
            if method == "POST":
                resp = await client.post(url, headers=headers)
            else:
                resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(400, f"API returned HTTP {exc.response.status_code}: {exc.response.text[:500]}")
    except Exception as exc:
        raise HTTPException(400, f"Failed to fetch from API: {exc}")

    if json_path:
        for key in json_path.split("."):
            if isinstance(data, dict) and key in data:
                data = data[key]
            else:
                raise HTTPException(400, f"JSON path '{json_path}' not found in response.")

    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        raise HTTPException(400, "API response could not be converted to tabular data.")

    if len(df) > API_MAX_ROWS:
        raise HTTPException(400, f"API returned {len(df)} rows, exceeding limit of {API_MAX_ROWS}.")

    from data_ingestion.loader import validate_flat_schema
    validation = validate_flat_schema(df)

    if preview_only:
        return {
            "status": "preview",
            "valid": validation["valid"],
            "row_count": validation["row_count"],
            "columns": validation["columns"],
            "missing": validation["missing"],
            "sample_rows": validation["sample_rows"],
        }

    if not validation["valid"]:
        raise HTTPException(
            400,
            f"Schema validation failed. Missing columns: {validation['missing']}. "
            f"Found: {validation['columns']}",
        )

    IMPORTED_FLAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(IMPORTED_FLAT_PATH, index=False)

    return {
        "status": "imported",
        "source_type": "api",
        "url": url,
        "row_count": validation["row_count"],
        "columns": validation["columns"],
        "sample_rows": validation["sample_rows"],
        "message": f"API data imported ({validation['row_count']} rows). Call POST /pipeline/run to process it.",
    }


@router.post("/connect/database")
async def connect_database(body: dict) -> dict:
    """
    Connect to a database and import data.

    Body:
      db_type  : str (postgresql, mysql, sqlite, mssql)
      host, port, database, username, password : connection details
      query    : str (SQL SELECT to execute)
      test_only: bool (default false)
    """
    db_type = body.get("db_type", "").lower()
    if db_type not in ALLOWED_DB_TYPES:
        raise HTTPException(400, f"Unsupported database type '{db_type}'. Allowed: {ALLOWED_DB_TYPES}")

    host = body.get("host", "localhost")
    port = body.get("port")
    database = body.get("database", "")
    username = body.get("username", "")
    password = body.get("password", "")
    query = body.get("query", "")
    test_only = body.get("test_only", False)

    if not query:
        raise HTTPException(400, "SQL query is required.")

    try:
        from sqlalchemy import create_engine, text

        if db_type == "sqlite":
            url = f"sqlite:///{database}"
        elif db_type == "postgresql":
            port = port or 5432
            url = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
        elif db_type == "mysql":
            port = port or 3306
            url = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        elif db_type == "mssql":
            port = port or 1433
            try:
                import pyodbc  # noqa: F401
                url = f"mssql+pyodbc://{username}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
            except ImportError:
                raise HTTPException(
                    400,
                    "SQL Server connector requires pyodbc and ODBC Driver 17. "
                    "Install with: pip install pyodbc",
                )
        else:
            raise HTTPException(400, f"Unsupported db_type: {db_type}")

        engine = create_engine(
            url,
            connect_args={"connect_timeout": DB_QUERY_TIMEOUT} if db_type != "sqlite" else {},
        )

        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchmany(DB_MAX_ROWS)
            columns = list(result.keys())
            total_count = len(rows)

        df = pd.DataFrame(rows, columns=columns)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Database connection failed: {exc}")

    from data_ingestion.loader import validate_flat_schema
    validation = validate_flat_schema(df)

    if test_only:
        return {
            "status": "connected",
            "valid": validation["valid"],
            "db_type": db_type,
            "row_count": total_count,
            "columns": validation["columns"],
            "missing": validation["missing"],
            "sample_rows": validation["sample_rows"],
        }

    if not validation["valid"]:
        raise HTTPException(
            400,
            f"Schema validation failed. Missing columns: {validation['missing']}. "
            f"Found: {validation['columns']}",
        )

    IMPORTED_FLAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(IMPORTED_FLAT_PATH, index=False)

    return {
        "status": "imported",
        "source_type": "database",
        "db_type": db_type,
        "row_count": validation["row_count"],
        "columns": validation["columns"],
        "sample_rows": validation["sample_rows"],
        "message": f"Database data imported ({validation['row_count']} rows). Call POST /pipeline/run to process it.",
    }
