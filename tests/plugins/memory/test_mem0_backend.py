"""Tests for Mem0Backend abstraction — PlatformBackend and OSSBackend."""

import pytest

from plugins.memory.mem0._backend import Mem0Backend, PlatformBackend, SelfHostedHTTPBackend, OSSBackend


class FakePlatformClient:
    """Fake MemoryClient for PlatformBackend tests."""

    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append(("search", query, kwargs))
        return {"results": [{"id": "m1", "memory": "fact1", "score": 0.9}]}

    def get_all(self, **kwargs):
        self.calls.append(("get_all", kwargs))
        return {"count": 1, "next": None, "results": [{"id": "m1", "memory": "fact1"}]}

    def add(self, messages, **kwargs):
        self.calls.append(("add", messages, kwargs))
        return {"status": "PENDING", "event_id": "evt-1"}

    def update(self, **kwargs):
        self.calls.append(("update", kwargs))
        return {"id": kwargs["memory_id"], "text": kwargs["text"]}

    def delete(self, **kwargs):
        self.calls.append(("delete", kwargs))


class TestPlatformBackend:

    def _make(self):
        client = FakePlatformClient()
        backend = PlatformBackend.__new__(PlatformBackend)
        backend._client = client
        return backend, client

    def test_search_forwards_params(self):
        backend, client = self._make()
        result = backend.search("test query", filters={"user_id": "u1"}, top_k=5)
        assert client.calls[0][0] == "search"
        assert client.calls[0][1] == "test query"
        assert client.calls[0][2]["filters"] == {"user_id": "u1"}
        assert client.calls[0][2]["top_k"] == 5

    def test_search_forwards_rerank(self):
        backend, client = self._make()
        backend.search("q", filters={}, rerank=False)
        assert client.calls[0][2]["rerank"] is False

    def test_search_rerank_default_true(self):
        backend, client = self._make()
        backend.search("q", filters={})
        assert client.calls[0][2]["rerank"] is True

    def test_search_returns_list(self):
        backend, _ = self._make()
        result = backend.search("q", filters={})
        assert isinstance(result, list)
        assert result[0]["id"] == "m1"

    def test_get_all_forwards_pagination(self):
        backend, client = self._make()
        result = backend.get_all(filters={"user_id": "u1"}, page=2, page_size=50)
        assert client.calls[0][1]["page"] == 2
        assert client.calls[0][1]["page_size"] == 50
        assert "count" in result

    def test_add_forwards_kwargs(self):
        backend, client = self._make()
        msgs = [{"role": "user", "content": "hi"}]
        result = backend.add(msgs, user_id="u1", agent_id="hermes", infer=False)
        call = client.calls[0]
        assert call[2]["user_id"] == "u1"
        assert call[2]["infer"] is False
        # metadata kwarg should be omitted entirely when not provided so we
        # don't surprise older mem0 client versions with an unknown kwarg.
        assert "metadata" not in call[2]

    def test_add_forwards_metadata_when_present(self):
        backend, client = self._make()
        msgs = [{"role": "user", "content": "hi"}]
        backend.add(
            msgs,
            user_id="u1",
            agent_id="hermes",
            infer=False,
            metadata={"channel": "telegram"},
        )
        assert client.calls[0][2]["metadata"] == {"channel": "telegram"}

    def test_add_omits_empty_metadata(self):
        backend, client = self._make()
        msgs = [{"role": "user", "content": "hi"}]
        backend.add(msgs, user_id="u1", agent_id="hermes", infer=False, metadata={})
        assert "metadata" not in client.calls[0][2]

    def test_update_forwards(self):
        backend, client = self._make()
        backend.update("m1", "new text")
        assert client.calls[0][1] == {"memory_id": "m1", "text": "new text"}

    def test_delete_forwards(self):
        backend, client = self._make()
        backend.delete("m1")
        assert client.calls[0][1] == {"memory_id": "m1"}


class FakeHTTPResponse:
    def __init__(self, data=None):
        self._data = data if data is not None else {}
        self.content = b"{}"

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class FakeHTTPClient:
    def __init__(self):
        self.calls = []

    def post(self, path, **kwargs):
        self.calls.append(("post", path, kwargs))
        if path == "/search":
            return FakeHTTPResponse({"results": [{"id": "m1", "memory": "fact1"}]})
        return FakeHTTPResponse({"event_id": "evt-1"})

    def get(self, path, **kwargs):
        self.calls.append(("get", path, kwargs))
        return FakeHTTPResponse({"count": 1, "results": [{"id": "m1", "memory": "fact1"}]})

    def put(self, path, **kwargs):
        self.calls.append(("put", path, kwargs))
        return FakeHTTPResponse({})

    def delete(self, path, **kwargs):
        self.calls.append(("delete", path, kwargs))
        return FakeHTTPResponse({})

    def close(self):
        self.calls.append(("close",))


class TestSelfHostedHTTPBackend:

    def _make(self):
        client = FakeHTTPClient()
        backend = SelfHostedHTTPBackend.__new__(SelfHostedHTTPBackend)
        backend._client = client
        return backend, client

    def test_search_posts_to_self_hosted_search(self):
        backend, client = self._make()
        result = backend.search("hello", filters={"user_id": "beka"}, top_k=3)
        assert client.calls[0][0] == "post"
        assert client.calls[0][1] == "/search"
        assert client.calls[0][2]["json"] == {
            "query": "hello",
            "filters": {"user_id": "beka"},
            "top_k": 3,
        }
        assert result[0]["id"] == "m1"

    def test_get_all_scopes_by_user_id_query_param(self):
        backend, client = self._make()
        result = backend.get_all(filters={"user_id": "beka", "tenant": "ignored"}, page=2, page_size=50)
        assert client.calls[0][0] == "get"
        assert client.calls[0][1] == "/memories"
        assert client.calls[0][2]["params"] == {"user_id": "beka", "page": 2, "page_size": 50}
        assert result["count"] == 1

    def test_add_posts_to_self_hosted_memories(self):
        backend, client = self._make()
        messages = [{"role": "user", "content": "fact"}]
        result = backend.add(messages, user_id="beka", agent_id="hermes:beka", metadata={"channel": "cli"})
        assert client.calls[0][0] == "post"
        assert client.calls[0][1] == "/memories"
        assert client.calls[0][2]["json"] == {
            "messages": messages,
            "user_id": "beka",
            "agent_id": "hermes:beka",
            "infer": False,
            "metadata": {"channel": "cli"},
        }
        assert result["event_id"] == "evt-1"

    def test_update_and_delete_use_self_hosted_memory_id_paths(self):
        backend, client = self._make()
        assert backend.update("mem-1", "new") == {"result": "Memory updated.", "memory_id": "mem-1"}
        assert backend.delete("mem-1") == {"result": "Memory deleted.", "memory_id": "mem-1"}
        assert client.calls[0] == ("put", "/memories/mem-1", {"json": {"text": "new"}})
        assert client.calls[1] == ("delete", "/memories/mem-1", {})


class FakeOSSMemory:
    """Fake mem0.Memory for OSSBackend tests."""

    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append(("search", query, kwargs))
        return {"results": [{"id": "m1", "memory": "fact1", "score": 0.8}]}

    def get_all(self, **kwargs):
        self.calls.append(("get_all", kwargs))
        return {"results": [{"id": "m1", "memory": "fact1"}]}

    def add(self, messages, **kwargs):
        self.calls.append(("add", messages, kwargs))
        return {"results": [{"id": "m1", "memory": "fact1", "event": "ADD"}]}

    def update(self, memory_id, **kwargs):
        self.calls.append(("update", memory_id, kwargs))
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        self.calls.append(("delete", memory_id))
        return {"message": "Memory deleted successfully!"}


class TestOSSBackend:

    def _make(self):
        memory = FakeOSSMemory()
        backend = OSSBackend.__new__(OSSBackend)
        backend._memory = memory
        return backend, memory

    def test_search_returns_list(self):
        backend, _ = self._make()
        result = backend.search("test", filters={"user_id": "u1"})
        assert isinstance(result, list)
        assert result[0]["id"] == "m1"

    def test_search_passes_filters(self):
        backend, memory = self._make()
        backend.search("q", filters={"user_id": "u1"}, top_k=3)
        assert memory.calls[0][2]["filters"] == {"user_id": "u1"}
        assert memory.calls[0][2]["top_k"] == 3

    def test_search_ignores_rerank(self):
        """OSS backend accepts rerank param but does not forward it to Memory."""
        backend, memory = self._make()
        backend.search("q", filters={}, rerank=True)
        assert "rerank" not in memory.calls[0][2]

    def test_get_all_ignores_pagination(self):
        """OSSBackend accepts page/page_size but does NOT forward to Memory.get_all()."""
        backend, memory = self._make()
        result = backend.get_all(filters={"user_id": "u1"}, page=2, page_size=50)
        call_kwargs = memory.calls[0][1]
        assert "page" not in call_kwargs
        assert "page_size" not in call_kwargs
        assert result["count"] == 1

    def test_get_all_returns_envelope(self):
        backend, _ = self._make()
        result = backend.get_all(filters={"user_id": "u1"})
        assert "results" in result
        assert "count" in result

    def test_add_forwards_kwargs(self):
        backend, memory = self._make()
        msgs = [{"role": "user", "content": "hi"}]
        backend.add(msgs, user_id="u1", agent_id="hermes", infer=False)
        assert memory.calls[0][2]["user_id"] == "u1"
        assert memory.calls[0][2]["infer"] is False

    def test_update_maps_text_to_data(self):
        """OSS Memory.update uses `data=` param, not `text=`."""
        backend, memory = self._make()
        backend.update("m1", "new text")
        assert memory.calls[0][0] == "update"
        assert memory.calls[0][1] == "m1"
        assert memory.calls[0][2] == {"data": "new text"}

    def test_delete_positional_arg(self):
        backend, memory = self._make()
        backend.delete("m1")
        assert memory.calls[0] == ("delete", "m1")

    def test_update_normalizes_response(self):
        backend, _ = self._make()
        result = backend.update("m1", "text")
        assert result == {"result": "Memory updated.", "memory_id": "m1"}

    def test_delete_normalizes_response(self):
        backend, _ = self._make()
        result = backend.delete("m1")
        assert result == {"result": "Memory deleted.", "memory_id": "m1"}
