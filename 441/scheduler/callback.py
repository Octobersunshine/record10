from __future__ import annotations

import logging

import httpx

from scheduler.models import Task

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient()
    return _client


async def execute_callback(task: Task):
    cb = task.callback
    client = await _get_client()
    try:
        kwargs: dict = {
            "method": cb.method.value,
            "url": cb.url,
            "headers": cb.headers,
            "timeout": cb.timeout,
        }
        if cb.body and cb.method.value in ("POST", "PUT"):
            kwargs["content"] = cb.body
        resp = await client.request(**kwargs)
        logger.info(
            "Callback for task %s returned %s: %s",
            task.id,
            resp.status_code,
            resp.text[:200],
        )
    except Exception:
        logger.exception("HTTP callback error for task %s", task.id)


async def close_client():
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
    _client = None
