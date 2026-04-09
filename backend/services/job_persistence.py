"""
Job persistence layer using SQLite for saving/loading prepared data between sessions.
Allows users to resume previous forecasting jobs.
"""

import sqlite3
import json
import pickle
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Database path: services/ -> backend/ -> artifacts/
DB_DIR = Path(__file__).resolve().parent.parent / "artifacts"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "jobs.db"


def init_db():
    """Initialize the SQLite database schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            stage TEXT NOT NULL,
            rows INTEGER,
            cols INTEGER,
            created_at TEXT,
            updated_at TEXT,
            data_pickle BLOB,
            metadata TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_job(job_id: str, filename: str, stage: str, df: Any, rows: int = 0, cols: int = 0, metadata: Dict = None) -> bool:
    """
    Save a job with its prepared data to the database.

    Args:
        job_id: Unique job identifier
        filename: Original filename
        stage: Current stage (imported, prepped, forecasted)
        df: Pandas DataFrame to save
        rows: Number of rows in data
        cols: Number of columns in data
        metadata: Additional metadata as dict

    Returns:
        bool: True if saved successfully
    """
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Pickle the DataFrame
        data_pickle = pickle.dumps(df)

        # Metadata as JSON
        meta_json = json.dumps(metadata or {})

        now = datetime.utcnow().isoformat()

        cursor.execute("""
            INSERT OR REPLACE INTO jobs
            (job_id, filename, stage, rows, cols, created_at, updated_at, data_pickle, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (job_id, filename, stage, rows, cols, now, now, data_pickle, meta_json))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving job: {e}")
        return False


def load_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a saved job from the database.

    Args:
        job_id: Job identifier

    Returns:
        dict with keys: job_id, filename, stage, rows, cols, df, metadata
        or None if not found
    """
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT job_id, filename, stage, rows, cols, data_pickle, metadata, updated_at
            FROM jobs WHERE job_id = ?
        """, (job_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        job_id, filename, stage, rows, cols, data_pickle, meta_json, updated_at = row

        # Unpickle the DataFrame
        df = pickle.loads(data_pickle)
        metadata = json.loads(meta_json) if meta_json else {}

        return {
            "job_id": job_id,
            "filename": filename,
            "stage": stage,
            "rows": rows,
            "cols": cols,
            "df": df,
            "metadata": metadata,
            "updated_at": updated_at,
        }
    except Exception as e:
        print(f"Error loading job: {e}")
        return None


def list_jobs() -> list[Dict[str, Any]]:
    """List all saved jobs (without loading data)."""
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT job_id, filename, stage, rows, cols, updated_at
            FROM jobs ORDER BY updated_at DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "job_id": row[0],
                "filename": row[1],
                "stage": row[2],
                "rows": row[3],
                "cols": row[4],
                "updated_at": row[5],
            }
            for row in rows
        ]
    except Exception as e:
        print(f"Error listing jobs: {e}")
        return []


def delete_job(job_id: str) -> bool:
    """Delete a saved job."""
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error deleting job: {e}")
        return False


def cleanup_old_jobs(keep_recent: int = 10):
    """Keep only the N most recent jobs."""
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM jobs WHERE job_id NOT IN (
                SELECT job_id FROM jobs ORDER BY updated_at DESC LIMIT ?
            )
        """, (keep_recent,))

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error cleaning up jobs: {e}")
