"""Routes: POST /chat, GET /llm/config, POST /llm/config"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.state import _state, _prep

router = APIRouter(tags=["llm"])


@router.post("/chat")
async def chat_endpoint(body: dict):
    """
    Stream an LLM response via Server-Sent Events.

    Body:
      messages      : list[{role, content}]
      screen_context: dict (optional) — {currentView, grain, horizon, model, groupValue}
      persona       : str (optional)  — default | ceo | supply_chain | reasoning
      provider      : str (optional)  — override active_provider
      model         : str (optional)  — override active_model
    """
    from api.llm_gateway import stream_chat

    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(400, "messages list is required.")

    screen_context = body.get("screen_context")
    persona = body.get("persona", "default")
    provider = body.get("provider")
    model = body.get("model")

    async def event_stream():
        async for chunk in stream_chat(
            messages=messages,
            app_state=_state,
            prep_state=_prep,
            screen_context=screen_context,
            persona=persona,
            provider_override=provider,
            model_override=model,
        ):
            yield chunk

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/llm/config")
async def llm_config_get() -> dict:
    """Return current LLM configuration (API keys redacted)."""
    from api.llm_gateway import load_config

    cfg = load_config(force=True)
    safe = json.loads(json.dumps(cfg, default=str))
    for prov in safe.get("providers", {}).values():
        key = prov.get("api_key", "")
        if key:
            prov["api_key"] = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
    return safe


@router.post("/llm/config")
async def llm_config_update(body: dict) -> dict:
    """
    Update LLM configuration.

    Body (all optional):
      active_provider, active_model, api_key, provider
    """
    from api.llm_gateway import load_config, save_config

    cfg = load_config(force=True)

    if "active_provider" in body:
        cfg["active_provider"] = body["active_provider"]
    if "active_model" in body:
        cfg["active_model"] = body["active_model"]
    if "api_key" in body:
        prov = body.get("provider", cfg.get("active_provider", "gemini"))
        if prov in cfg.get("providers", {}):
            cfg["providers"][prov]["api_key"] = body["api_key"]

    save_config(cfg)
    return {
        "status": "updated",
        "active_provider": cfg["active_provider"],
        "active_model": cfg.get("active_model"),
    }
