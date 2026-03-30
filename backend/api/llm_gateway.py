"""
LLM Gateway — Multi-provider adapter with tool-calling for Predictify.

Supports:
  - Google Gemini (native SDK)
  - OpenRouter (OpenAI-compatible API)

Tools execute read-only queries against the existing FastAPI endpoints
internally (no HTTP roundtrip — direct function calls).
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "llm_config.yaml"

# ─── Config loader ──────────────────────────────────────────────────────────

_config_cache: dict | None = None


def load_config(force: bool = False) -> dict:
    global _config_cache
    if _config_cache and not force:
        return _config_cache
    with open(CONFIG_PATH, "r") as f:
        _config_cache = yaml.safe_load(f)
    return _config_cache


def save_config(cfg: dict) -> None:
    global _config_cache
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    _config_cache = cfg


# ─── Tool Definitions ───────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "get_data_summary",
        "description": "Get a summary of the imported dataset: row count, columns, date range, stores, categories. Call this first to understand the data.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_forecast",
        "description": "Get forecast data for a specific grain (total/store/category), horizon (7/14/30 days), and model. Returns dates, forecasted values, confidence intervals, and actuals.",
        "parameters": {
            "type": "object",
            "properties": {
                "grain": {"type": "string", "enum": ["total", "store", "category"], "description": "Aggregation level"},
                "horizon": {"type": "integer", "enum": [7, 14, 30], "description": "Forecast horizon in days"},
                "model": {"type": "string", "description": "Model name e.g. RidgeForecaster, LassoForecaster, NaiveMean"},
                "group": {"type": "string", "description": "Store ID or category name (optional, for store/category grain)"},
            },
            "required": ["grain"],
        },
    },
    {
        "name": "get_metrics",
        "description": "Get model evaluation metrics (MAE, RMSE) across all models and grains. Use this to compare model performance.",
        "parameters": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Filter by model name (optional)"},
            },
            "required": [],
        },
    },
    {
        "name": "get_pipeline_status",
        "description": "Check if the forecasting pipeline has been run, its status (idle/running/complete/error), and timing.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "query_data",
        "description": "Run an analytical query on the imported dataset. Specify a pandas-style operation described in natural language. Examples: 'top 5 stores by revenue', 'daily sales trend', 'category breakdown'.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Natural language analytical question about the data"},
            },
            "required": ["question"],
        },
    },
    {
        "name": "get_screen_context",
        "description": "Get information about what the user is currently viewing on screen: current view, selected grain, model, horizon, and visible chart data.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ─── Tool Execution (direct, no HTTP) ───────────────────────────────────────

def _safe_json(obj: Any) -> Any:
    """Make objects JSON-serializable."""
    if isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
        return str(obj)
    if isinstance(obj, float) and (pd.isna(obj)):
        return None
    return obj


def execute_tool(tool_name: str, args: dict, app_state: dict, prep_state: dict, screen_context: dict | None = None) -> str:
    """
    Execute a tool by name. Returns a JSON string result.
    app_state and prep_state are the in-memory dicts from app.py.
    """
    cfg = load_config()
    max_rows = cfg.get("safety", {}).get("max_query_rows", 1000)

    try:
        if tool_name == "get_data_summary":
            return _tool_data_summary(prep_state)

        elif tool_name == "get_forecast":
            return _tool_get_forecast(args)

        elif tool_name == "get_metrics":
            return _tool_get_metrics(args)

        elif tool_name == "get_pipeline_status":
            return json.dumps({
                "status": app_state.get("status", "unknown"),
                "job_id": app_state.get("job_id"),
                "started_at": app_state.get("started_at"),
                "finished_at": app_state.get("finished_at"),
                "error": app_state.get("error"),
            }, default=_safe_json)

        elif tool_name == "query_data":
            return _tool_query_data(args, prep_state, max_rows)

        elif tool_name == "get_screen_context":
            return json.dumps(screen_context or {"info": "No screen context available"}, default=_safe_json)

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        logger.exception("Tool execution error: %s", tool_name)
        return json.dumps({"error": str(e)})


def _tool_data_summary(prep_state: dict) -> str:
    """Summarize the loaded dataset."""
    from config import DATA_DIR, REPORT_DIR, IMPORTED_FLAT_PATH, WORKBOOK_PATH, WORKBOOK_PATH_RAW

    result: dict[str, Any] = {"has_data": False}

    # Check summary.json first
    summary_path = REPORT_DIR / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            result = json.load(f)
        result["has_data"] = True

    # Add prep data info
    df = prep_state.get("df")
    if df is not None:
        result["has_data"] = True
        result["row_count"] = len(df)
        result["columns"] = list(df.columns)
        result["dtypes"] = {col: str(df[col].dtype) for col in df.columns}
        # Date range
        for dc in ["date", "trans_date", "Trans. Date"]:
            if dc in df.columns:
                try:
                    dates = pd.to_datetime(df[dc], errors="coerce").dropna()
                    if len(dates):
                        result["date_range"] = {"min": str(dates.min().date()), "max": str(dates.max().date()), "days": (dates.max() - dates.min()).days}
                except Exception:
                    pass
                break
        # Basic stats
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            stats = df[num_cols].describe().round(2)
            result["numeric_stats"] = {col: stats[col].to_dict() for col in num_cols[:5]}  # limit to 5
    elif IMPORTED_FLAT_PATH.exists():
        try:
            df_peek = pd.read_csv(IMPORTED_FLAT_PATH, nrows=5)
            row_count = sum(1 for _ in open(IMPORTED_FLAT_PATH)) - 1
            result["has_data"] = True
            result["row_count"] = row_count
            result["columns"] = list(df_peek.columns)
        except Exception:
            pass

    return json.dumps(result, default=_safe_json)


def _tool_get_forecast(args: dict) -> str:
    from config import FORECAST_DIR, DATA_DIR

    grain = args.get("grain", "total")
    horizon = args.get("horizon", 7)
    model = args.get("model", "RidgeForecaster")
    group = args.get("group")

    fc_path = FORECAST_DIR / f"forecast_{grain}_{model}.csv"
    if not fc_path.exists():
        available = list(FORECAST_DIR.glob(f"forecast_{grain}_*.csv"))
        if not available:
            return json.dumps({"error": f"No forecasts for grain='{grain}'. Run pipeline first.", "available_grains": [p.stem.split("_")[1] for p in FORECAST_DIR.glob("forecast_*.csv")]})
        fc_path = available[0]
        model = fc_path.stem.replace(f"forecast_{grain}_", "")

    df = pd.read_csv(fc_path, parse_dates=["date"])
    df = df[df["horizon"] == horizon]

    if group and "group_key" in df.columns:
        df = df[df["group_key"] == group]
    elif "group_key" in df.columns and len(df) > 0:
        group = df["group_key"].iloc[0]
        df = df[df["group_key"] == group]

    if df.empty:
        return json.dumps({"error": "No forecast data for these filters."})

    df = df.sort_values("date")

    result = {
        "grain": grain, "model": model, "horizon": horizon, "group": group or "total",
        "forecast_count": len(df),
        "date_range": {"start": str(df["date"].iloc[0].date()), "end": str(df["date"].iloc[-1].date())},
        "forecast_values": df["forecast"].round(2).tolist(),
        "mean_forecast": round(df["forecast"].mean(), 2),
        "total_forecast": round(df["forecast"].sum(), 2),
    }

    if "ci_lower" in df.columns:
        result["ci_lower_mean"] = round(df["ci_lower"].mean(), 2)
        result["ci_upper_mean"] = round(df["ci_upper"].mean(), 2)

    # Load actuals for comparison
    act_map = {"total": "fact_total_daily.csv", "store": "fact_store_daily.csv", "category": "fact_category_daily.csv"}
    act_path = DATA_DIR / act_map.get(grain, "")
    if act_path.exists():
        act = pd.read_csv(act_path, parse_dates=["date"])
        if group and grain != "total":
            col = "store_id" if grain == "store" else "category"
            if col in act.columns:
                act = act[act[col] == group]
        if "sales_value" in act.columns and len(act):
            result["actual_mean_daily"] = round(act["sales_value"].mean(), 2)
            result["actual_total"] = round(act["sales_value"].sum(), 2)
            result["actual_last_7d"] = round(act.sort_values("date").tail(7)["sales_value"].mean(), 2)

    return json.dumps(result, default=_safe_json)


def _tool_get_metrics(args: dict) -> str:
    from config import REPORT_DIR

    metrics_path = REPORT_DIR / "metrics_all.json"
    if not metrics_path.exists():
        return json.dumps({"error": "No metrics found. Run the pipeline first."})

    with open(metrics_path) as f:
        records = json.load(f)

    model_filter = args.get("model")
    if model_filter:
        records = [r for r in records if r.get("model_name") == model_filter]

    records = sorted(records, key=lambda r: r.get("mean_mae", float("inf")))

    # Summarize
    summary = []
    for r in records[:15]:
        summary.append({
            "model": r.get("model_name"),
            "grain": r.get("grain"),
            "horizon": r.get("horizon"),
            "MAE": round(r.get("mean_mae", 0), 2),
            "RMSE": round(r.get("mean_rmse", 0), 2),
        })

    return json.dumps({"metrics": summary, "total_records": len(records)}, default=_safe_json)


def _tool_query_data(args: dict, prep_state: dict, max_rows: int) -> str:
    """Analytical query on the dataset — translates natural language to pandas ops."""
    question = args.get("question", "")
    df = prep_state.get("df")

    if df is None:
        from config import IMPORTED_FLAT_PATH
        if IMPORTED_FLAT_PATH.exists():
            df = pd.read_csv(IMPORTED_FLAT_PATH)
        else:
            return json.dumps({"error": "No data loaded. Import data first."})

    result: dict[str, Any] = {"question": question, "row_count": len(df), "columns": list(df.columns)}

    # Parse common analytical patterns
    q = question.lower().strip()

    try:
        if any(w in q for w in ["top", "best", "highest"]):
            n = 5
            nums = re.findall(r'\d+', q)
            if nums:
                n = min(int(nums[0]), max_rows)
            if "store" in q and "store_id" in df.columns:
                agg_col = "net_amount" if "net_amount" in df.columns else "quantity" if "quantity" in df.columns else df.select_dtypes("number").columns[0]
                top = df.groupby("store_id")[agg_col].sum().nlargest(n)
                result["data"] = [{"store_id": k, agg_col: round(v, 2)} for k, v in top.items()]
            elif "categor" in q and "category" in df.columns:
                agg_col = "net_amount" if "net_amount" in df.columns else "quantity"
                top = df.groupby("category")[agg_col].sum().nlargest(n)
                result["data"] = [{"category": k, agg_col: round(v, 2)} for k, v in top.items()]
            elif "item" in q and "item_id" in df.columns:
                agg_col = "net_amount" if "net_amount" in df.columns else "quantity"
                top = df.groupby("item_id")[agg_col].sum().nlargest(n)
                result["data"] = [{"item_id": k, agg_col: round(v, 2)} for k, v in top.items()]
            else:
                result["data"] = df.head(n).to_dict(orient="records")

        elif any(w in q for w in ["trend", "daily", "time series", "over time"]):
            date_col = next((c for c in ["date", "trans_date"] if c in df.columns), None)
            if date_col:
                agg_col = "net_amount" if "net_amount" in df.columns else "quantity"
                daily = df.groupby(date_col)[agg_col].sum().reset_index()
                daily = daily.sort_values(date_col).tail(max_rows)
                result["data"] = daily.to_dict(orient="records")

        elif any(w in q for w in ["breakdown", "distribution", "split", "by"]):
            group_col = None
            if "store" in q and "store_id" in df.columns:
                group_col = "store_id"
            elif "categor" in q and "category" in df.columns:
                group_col = "category"
            if group_col:
                agg_col = "net_amount" if "net_amount" in df.columns else "quantity"
                breakdown = df.groupby(group_col)[agg_col].agg(["sum", "mean", "count"]).round(2).reset_index()
                result["data"] = breakdown.head(max_rows).to_dict(orient="records")

        elif any(w in q for w in ["total", "sum", "overall"]):
            num_cols = df.select_dtypes("number").columns.tolist()
            result["data"] = {col: round(df[col].sum(), 2) for col in num_cols[:5]}

        elif any(w in q for w in ["average", "mean"]):
            num_cols = df.select_dtypes("number").columns.tolist()
            result["data"] = {col: round(df[col].mean(), 2) for col in num_cols[:5]}

        else:
            # Default: basic stats + sample
            num_cols = df.select_dtypes("number").columns.tolist()
            result["data"] = {
                "shape": list(df.shape),
                "numeric_summary": {col: {"mean": round(df[col].mean(), 2), "sum": round(df[col].sum(), 2)} for col in num_cols[:5]},
                "sample": df.head(3).to_dict(orient="records"),
            }

    except Exception as e:
        result["error"] = str(e)

    return json.dumps(result, default=_safe_json)


# ─── Audit Log ───────────────────────────────────────────────────────────────

def _audit_log(entry: dict) -> None:
    cfg = load_config()
    if not cfg.get("audit", {}).get("enabled", False):
        return
    log_path = BASE_DIR / cfg["audit"].get("log_file", "artifacts/audit_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry["timestamp"] = datetime.datetime.utcnow().isoformat()
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, default=_safe_json) + "\n")


# ─── Provider Adapters ───────────────────────────────────────────────────────

def _build_tool_declarations_gemini() -> list[dict]:
    """Convert our tool defs to Gemini function declaration format."""
    decls = []
    for t in TOOL_DEFINITIONS:
        # Build clean properties without 'required' at property level
        props = {}
        for k, v in t["parameters"].get("properties", {}).items():
            prop = {"type": v.get("type", "string").upper()}
            if "description" in v:
                prop["description"] = v["description"]
            if "enum" in v:
                prop["enum"] = [str(e) for e in v["enum"]]
            props[k] = prop

        decls.append({
            "name": t["name"],
            "description": t["description"],
            "parameters": {
                "type": "OBJECT",
                "properties": props,
                "required": t["parameters"].get("required", []),
            } if props else {"type": "OBJECT", "properties": {}},
        })
    return decls


def _build_tools_openai() -> list[dict]:
    """Convert our tool defs to OpenAI function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in TOOL_DEFINITIONS
    ]


# ─── Gemini streaming with tool calling ──────────────────────────────────────

async def stream_gemini(
    messages: list[dict],
    system_prompt: str,
    app_state: dict,
    prep_state: dict,
    screen_context: dict | None = None,
    model_override: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream a Gemini response with automatic tool execution.
    Yields SSE-formatted chunks: data: {json}\n\n
    """
    cfg = load_config()
    provider_cfg = cfg["providers"]["gemini"]
    api_key = provider_cfg.get("api_key") or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        yield _sse({"type": "error", "content": "Gemini API key not configured. Edit backend/llm_config.yaml or set GEMINI_API_KEY env var."})
        return

    model_id = model_override or cfg.get("active_model", "gemini-2.0-flash")

    # Build Gemini request
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"

    # Convert messages to Gemini format
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    body: dict[str, Any] = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "tools": [{"functionDeclarations": _build_tool_declarations_gemini()}],
        "generationConfig": {
            "maxOutputTokens": cfg.get("safety", {}).get("max_tokens_per_response", 4096),
            "temperature": 0.7,
        },
    }

    max_tool_rounds = 5
    for round_num in range(max_tool_rounds):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=body)
                if resp.status_code != 200:
                    yield _sse({"type": "error", "content": f"Gemini API error {resp.status_code}: {resp.text[:500]}"})
                    return
                result = resp.json()
        except Exception as e:
            yield _sse({"type": "error", "content": f"Gemini request failed: {e}"})
            return

        candidates = result.get("candidates", [])
        if not candidates:
            yield _sse({"type": "error", "content": "No response from Gemini."})
            return

        parts = candidates[0].get("content", {}).get("parts", [])

        # Check for function calls
        fn_calls = [p for p in parts if "functionCall" in p]
        text_parts = [p.get("text", "") for p in parts if "text" in p]

        if text_parts and not fn_calls:
            # Pure text response — stream it out
            full_text = "".join(text_parts)
            # Simulate streaming by sending in chunks
            chunk_size = 40
            for i in range(0, len(full_text), chunk_size):
                yield _sse({"type": "chunk", "content": full_text[i:i+chunk_size]})
            yield _sse({"type": "done"})

            _audit_log({"event": "llm_response", "model": model_id, "provider": "gemini", "tool_rounds": round_num})
            return

        if fn_calls:
            # Execute tool calls
            tool_responses = []
            for fc in fn_calls:
                fn = fc["functionCall"]
                tool_name = fn["name"]
                tool_args = fn.get("args", {})

                yield _sse({"type": "tool_call", "tool": tool_name, "args": tool_args})

                _audit_log({"event": "tool_call", "tool": tool_name, "args": tool_args, "model": model_id})

                tool_result = execute_tool(tool_name, tool_args, app_state, prep_state, screen_context)

                tool_responses.append({
                    "functionResponse": {
                        "name": tool_name,
                        "response": {"result": tool_result},
                    }
                })

            # Add model response + tool results to contents and loop
            contents.append({"role": "model", "parts": parts})
            contents.append({"role": "user", "parts": tool_responses})
            body["contents"] = contents
            continue

    yield _sse({"type": "error", "content": "Max tool rounds exceeded."})


# ─── OpenRouter streaming with tool calling ──────────────────────────────────

async def stream_openrouter(
    messages: list[dict],
    system_prompt: str,
    app_state: dict,
    prep_state: dict,
    screen_context: dict | None = None,
    model_override: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream an OpenRouter response (OpenAI-compatible) with tool calling.
    """
    cfg = load_config()
    provider_cfg = cfg["providers"]["openrouter"]
    api_key = provider_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        yield _sse({"type": "error", "content": "OpenRouter API key not configured. Edit backend/llm_config.yaml or set OPENROUTER_API_KEY env var."})
        return

    model_id = model_override or cfg.get("active_model", "deepseek/deepseek-r1")
    base_url = provider_cfg.get("base_url", "https://openrouter.ai/api/v1")

    # Build OpenAI-format messages
    oai_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        oai_messages.append({"role": msg["role"], "content": msg["content"]})

    body: dict[str, Any] = {
        "model": model_id,
        "messages": oai_messages,
        "tools": _build_tools_openai(),
        "max_tokens": cfg.get("safety", {}).get("max_tokens_per_response", 4096),
        "temperature": 0.7,
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://predictify.local",
        "X-Title": "Predictify",
    }

    max_tool_rounds = 5
    for round_num in range(max_tool_rounds):
        try:
            full_content = ""
            tool_calls_acc: dict[int, dict] = {}

            async with httpx.AsyncClient(timeout=90.0) as client:
                async with client.stream("POST", f"{base_url}/chat/completions", json=body, headers=headers) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        yield _sse({"type": "error", "content": f"OpenRouter error {resp.status_code}: {error_body.decode()[:500]}"})
                        return

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        delta = chunk.get("choices", [{}])[0].get("delta", {})

                        # Text content
                        if delta.get("content"):
                            full_content += delta["content"]
                            yield _sse({"type": "chunk", "content": delta["content"]})

                        # Tool calls accumulation
                        if delta.get("tool_calls"):
                            for tc in delta["tool_calls"]:
                                idx = tc.get("index", 0)
                                if idx not in tool_calls_acc:
                                    tool_calls_acc[idx] = {"id": tc.get("id", ""), "name": "", "arguments": ""}
                                if tc.get("id"):
                                    tool_calls_acc[idx]["id"] = tc["id"]
                                fn = tc.get("function", {})
                                if fn.get("name"):
                                    tool_calls_acc[idx]["name"] = fn["name"]
                                if fn.get("arguments"):
                                    tool_calls_acc[idx]["arguments"] += fn["arguments"]

            # No tool calls — done
            if not tool_calls_acc:
                if full_content:
                    yield _sse({"type": "done"})
                else:
                    yield _sse({"type": "chunk", "content": "I couldn't generate a response. Please try again."})
                    yield _sse({"type": "done"})

                _audit_log({"event": "llm_response", "model": model_id, "provider": "openrouter", "tool_rounds": round_num})
                return

            # Execute tool calls
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": full_content or None, "tool_calls": []}
            for idx in sorted(tool_calls_acc):
                tc = tool_calls_acc[idx]
                assistant_msg["tool_calls"].append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                })

            oai_messages.append(assistant_msg)

            for idx in sorted(tool_calls_acc):
                tc = tool_calls_acc[idx]
                tool_name = tc["name"]
                try:
                    tool_args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    tool_args = {}

                yield _sse({"type": "tool_call", "tool": tool_name, "args": tool_args})
                _audit_log({"event": "tool_call", "tool": tool_name, "args": tool_args, "model": model_id})

                tool_result = execute_tool(tool_name, tool_args, app_state, prep_state, screen_context)
                oai_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_result})

            body["messages"] = oai_messages
            body["stream"] = True
            continue

        except Exception as e:
            yield _sse({"type": "error", "content": f"OpenRouter request failed: {e}"})
            return

    yield _sse({"type": "error", "content": "Max tool rounds exceeded."})


# ─── Unified entry point ─────────────────────────────────────────────────────

async def stream_chat(
    messages: list[dict],
    app_state: dict,
    prep_state: dict,
    screen_context: dict | None = None,
    persona: str = "default",
    provider_override: str | None = None,
    model_override: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Main entry: route to the active provider's streaming function.
    """
    cfg = load_config()
    provider = provider_override or cfg.get("active_provider", "gemini")

    # Get persona system prompt
    personas = cfg.get("personas", {})
    persona_cfg = personas.get(persona, personas.get("default", {}))
    system_prompt = persona_cfg.get("system_prompt", "You are a helpful assistant.")

    # Add screen context to system prompt
    if screen_context:
        system_prompt += f"\n\nCurrent user screen context: {json.dumps(screen_context, default=_safe_json)}"

    if provider == "gemini":
        async for chunk in stream_gemini(messages, system_prompt, app_state, prep_state, screen_context, model_override):
            yield chunk
    elif provider == "openrouter":
        async for chunk in stream_openrouter(messages, system_prompt, app_state, prep_state, screen_context, model_override):
            yield chunk
    else:
        yield _sse({"type": "error", "content": f"Unknown provider: {provider}"})


def _sse(data: dict) -> str:
    """Format a dict as an SSE line."""
    return f"data: {json.dumps(data, default=_safe_json)}\n\n"
