import json
import asyncio

import multi_agent_app.history as history
import multi_agent_app.routes as routes


def test_append_to_chat_history_writes(tmp_path, monkeypatch):
    chat_path = tmp_path / "chat_history.json"
    monkeypatch.setattr(history, "_PRIMARY_CHAT_HISTORY_PATH", chat_path)
    monkeypatch.setattr(history, "_FALLBACK_CHAT_HISTORY_PATH", chat_path)
    monkeypatch.setattr(history, "_refresh_memory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(history, "_consolidate_short_into_long", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(history, "load_memory_settings", lambda: {"enabled": False})

    history._append_to_chat_history("user", "hello", broadcast=False)

    data = json.loads(chat_path.read_text(encoding="utf-8"))
    assert data[-1]["content"] == "hello"


class _FakeRequest:
    def __init__(self, method="GET", payload=None):
        self.method = method
        self._payload = payload

    async def json(self):
        return self._payload


def test_api_memory_does_not_expose_history_sync():
    res = asyncio.run(routes.api_memory(_FakeRequest(method="GET")))
    assert "history_sync_enabled" not in res
