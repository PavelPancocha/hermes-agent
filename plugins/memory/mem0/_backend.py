"""REST backend adapted from the pinned Mem0 vendor provider."""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Any


def _add_kwargs(user_id: str, agent_id: str, infer: bool, metadata: dict | None) -> dict[str, Any]:
    return {"user_id": user_id, "agent_id": agent_id, "infer": infer, **({"metadata": metadata} if metadata else {})}


def _unwrap_results(response: Any) -> list:
    """Normalize API response — extract results list from dict or pass through."""
    return response.get("results", []) if isinstance(response, dict) else response if isinstance(response, list) else []


class Mem0Backend(ABC):
    """Interface for the self-hosted HTTP backend.
    update()/delete() are template methods: subclasses implement raw ``_update``/``_delete``."""

    @abstractmethod
    def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = False) -> list[dict]: ...
    @abstractmethod
    def get_all(self, *, filters: dict, page: int = 1, page_size: int = 100) -> dict: ...
    @abstractmethod
    def add(self, messages: list, *, user_id: str, agent_id: str, infer: bool = False, metadata: dict | None = None) -> dict: ...
    @abstractmethod
    def get(self, memory_id: str) -> dict | None: ...
    @abstractmethod
    def _update(self, memory_id: str, text: str) -> None: ...
    @abstractmethod
    def _delete(self, memory_id: str) -> None: ...

    def update(self, memory_id: str, text: str) -> dict:
        self._update(memory_id, text)
        return {"result": "Memory updated.", "memory_id": memory_id}

    def delete(self, memory_id: str) -> dict:
        self._delete(memory_id)
        return {"result": "Memory deleted.", "memory_id": memory_id}

    def close(self) -> None:
        pass


class SelfHostedBackend(Mem0Backend):
    """Direct HTTP backend for a self-hosted Mem0 server (the FastAPI ``server/``).
    mem0.MemoryClient is hardwired to the cloud API (``Authorization: Token``, ``GET /v1/ping/`` in ``__init__``),
    so this speaks the server's real contract: ``X-API-Key`` auth and the ``/memories`` / ``/search`` routes."""

    def __init__(self, api_key: str, host: str, transport=None):
        import httpx
        headers = {"Content-Type": "application/json", **({"X-API-Key": api_key} if api_key else {})}  # key omitted only for AUTH_DISABLED servers
        # Connect-level retries keep one dropped SYN from counting toward the breaker. ``transport`` is injectable for tests.
        self._client = httpx.Client(base_url=host.rstrip("/"), headers=headers, timeout=30.0, transport=transport or httpx.HTTPTransport(retries=2))
        self._capture_timeout = httpx.Timeout(120.0, connect=30.0)

    def _json(self, method: str, path: str, **kwargs) -> Any:
        resp = self._client.request(method, path, **kwargs)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = False) -> list[dict]:
        # rerank is platform-only; the self-hosted /search ignores it. user_id belongs in filters (top-level is deprecated).
        return _unwrap_results(self._json("POST", "/search", json={"query": query, "top_k": top_k, **({"filters": filters} if filters else {})}))

    def get_all(self, *, filters: dict, page: int = 1, page_size: int = 100) -> dict:
        page = max(1, page)
        page_size = min(max(1, page_size), 200)
        # shortcut: only the first 1000 records, use server pagination for larger audits.
        response = self._json("GET", "/memories", params={
            "user_id": filters["user_id"], "top_k": min(page * page_size, 1000),
        })
        results = _unwrap_results(response)
        count = response.get("count", len(results)) if isinstance(response, dict) else len(results)
        start = (page - 1) * page_size
        return {"results": results[start:start + page_size], "count": count}

    def add(self, messages: list, *, user_id: str, agent_id: str, infer: bool = False, metadata: dict | None = None) -> dict:
        # Server-side extraction takes longer than a search or verbatim write.
        return self._json("POST", "/memories", json={"messages": messages, **_add_kwargs(user_id, agent_id, infer, metadata)},
                          timeout=self._capture_timeout if infer else self._client.timeout)

    def get(self, memory_id: str) -> dict | None:
        return self._json("GET", f"/memories/{memory_id}")

    def _update(self, memory_id: str, text: str) -> None:
        self._json("PUT", f"/memories/{memory_id}", json={"text": text})

    def _delete(self, memory_id: str) -> None:
        self._json("DELETE", f"/memories/{memory_id}")

    def close(self) -> None:
        with suppress(Exception):
            self._client.close()
