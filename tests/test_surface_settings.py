"""Dashboard-editable breath/dream caps and the 'last surface' readout."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import server
from surface_settings import SurfaceSettings


def _bucket(index: int, content: str = "", **meta):
    return {
        "id": f"bucket-{index:02d}",
        "content": content or f"memory-{index:02d}",
        "metadata": {
            "name": f"memory-{index:02d}",
            "type": "dynamic",
            "created": f"2026-07-{index:02d}T00:00:00",
            "last_active": f"2026-07-{index:02d}T00:00:00",
            "importance": index,
            "activation_count": 1,
            "resolved": False,
            "pinned": False,
            **meta,
        },
    }


@pytest.fixture
def settings(tmp_path, monkeypatch):
    store = SurfaceSettings(str(tmp_path))
    monkeypatch.setattr(server, "surface_settings", store)
    return store


def test_defaults_match_previous_hardcoded_caps(settings):
    assert settings.all() == {
        "breath_max_results": 20,
        "breath_max_tokens": 10000,
        "dream_max_results": 10,
        "dream_max_tokens": 10000,
    }


def test_update_validates_and_persists(settings, tmp_path):
    settings.update({"breath_max_results": "30", "dream_max_results": 5})
    assert settings.get("breath_max_results") == 30
    assert settings.get("dream_max_results") == 5
    assert json.loads((tmp_path / ".surface_settings.json").read_text())["breath_max_results"] == 30
    # A fresh instance reads the same file.
    assert SurfaceSettings(str(tmp_path)).get("breath_max_results") == 30

    with pytest.raises(ValueError):
        settings.update({"breath_max_results": 0})
    with pytest.raises(ValueError):
        settings.update({"breath_max_tokens": 99999})
    with pytest.raises(ValueError):
        settings.update({"nope": 3})
    with pytest.raises(ValueError):
        settings.update({"dream_max_tokens": "abc"})
    # Failed updates leave stored values untouched.
    assert settings.get("breath_max_results") == 30


def test_corrupt_stored_values_fall_back_or_clamp(settings, tmp_path):
    (tmp_path / ".surface_settings.json").write_text(
        json.dumps({"breath_max_results": 500, "dream_max_results": "x", "breath_max_tokens": True})
    )
    values = settings.all()
    assert values["breath_max_results"] == 50
    assert values["dream_max_results"] == 10
    assert values["breath_max_tokens"] == 10000


def test_dream_reservation_honours_count_and_token_caps(settings):
    settings.update({"dream_max_results": 4, "dream_max_tokens": 600})
    long_text = "这是一段很长的记忆正文。" * 60  # ~500 chars after truncation
    buckets = [_bucket(i, long_text) for i in range(1, 9)]

    recent = server._select_dream_recent(buckets)

    assert len(recent) < 4, "token cap should bite before the count cap"
    assert [b["id"] for b in recent] == [f"bucket-{i:02d}" for i in range(8, 8 - len(recent), -1)]
    total = sum(server.count_tokens_approx(server._render_dream_part(b)) for b in recent)
    assert total <= 600


@pytest.mark.asyncio
async def test_breath_and_dream_use_dashboard_caps(settings, monkeypatch):
    settings.update({"breath_max_results": 3, "dream_max_results": 2})
    buckets = [_bucket(i) for i in range(1, 13)]
    monkeypatch.setattr(server.bucket_mgr, "list_all", AsyncMock(return_value=buckets))
    monkeypatch.setattr(server.bucket_mgr, "soft_touch", AsyncMock())
    monkeypatch.setattr(server.decay_engine, "ensure_started", AsyncMock())
    monkeypatch.setattr(server.decay_engine, "calculate_score", lambda m: float(m["importance"]))
    monkeypatch.setattr(server.dehydrator, "dehydrate", AsyncMock(side_effect=lambda c, metadata=None: c))
    monkeypatch.setattr(server.embedding_engine, "enabled", False)
    monkeypatch.setattr(server, "_fire_webhook", AsyncMock())
    monkeypatch.setattr(server, "_select_importance_floor", lambda scored, **kw: [])
    audit = MagicMock()
    monkeypatch.setattr(server, "surface_audit", audit)
    monkeypatch.setattr(server.decay_engine, "relationship_clock", MagicMock())

    breath_text = await server.breath()
    dream_text = await server.dream()

    calls = {call.args[0]: call for call in audit.record.call_args_list}
    assert calls["breath"].kwargs["max_results"] == 3
    assert calls["breath"].kwargs["dynamic_returned_count"] == 3
    assert calls["dream"].kwargs["max_results"] == 2
    assert calls["dream"].kwargs["returned_count"] == 2
    assert calls["dream"].kwargs["max_tokens"] == 10000
    assert calls["dream"].kwargs["remaining_tokens"] < 10000
    # dream took the newest 2; breath must not repeat them
    assert "memory-12" in dream_text and "memory-11" in dream_text
    assert "memory-12" not in breath_text and "memory-11" not in breath_text
    # explicit caller values still win over the dashboard default
    audit.reset_mock()
    await server.breath(max_results=5)
    assert audit.record.call_args.kwargs["max_results"] == 5


def test_last_surface_stats_splits_pinned_from_dynamic():
    events = [
        {"flow": "feel", "timestamp": "t3"},
        {
            "flow": "breath", "timestamp": "t2", "returned_count": 3,
            "pinned_returned_count": 1, "dynamic_returned_count": 2,
            "max_results": 20, "max_tokens": 10000, "remaining_tokens": 9400,
            "entries": [
                {"channel": "pin", "outcome": "surfaced", "summary_tokens": 200},
                {"channel": "dynamic", "outcome": "surfaced", "summary_tokens": 250},
                {"channel": "dynamic", "outcome": "surfaced", "summary_tokens": 150},
                {"channel": "dynamic", "outcome": "token_exhausted"},
            ],
        },
        {
            "flow": "dream", "timestamp": "t1", "returned_count": 2, "max_results": 10,
            "entries": [
                {"channel": "dream", "outcome": "surfaced", "summary_tokens": 300},
                {"channel": "dream", "outcome": "surfaced", "summary_tokens": 100},
            ],
        },
        {"flow": "breath", "timestamp": "t0", "returned_count": 99, "entries": []},
    ]

    last = server._last_surface_stats(events)

    assert last["breath"]["timestamp"] == "t2"
    assert last["breath"]["pinned_returned_count"] == 1
    assert last["breath"]["dynamic_returned_count"] == 2
    assert last["breath"]["pinned_tokens"] == 200
    assert last["breath"]["dynamic_tokens"] == 400
    assert last["breath"]["used_tokens"] == 600
    # old dream events have no budget fields: fall back to summing entries
    assert last["dream"]["used_tokens"] == 400
    assert last["dream"]["max_tokens"] is None


@pytest.mark.asyncio
async def test_surface_settings_api_roundtrip(settings, monkeypatch):
    monkeypatch.setattr(server, "_require_auth", lambda _r: None)
    monkeypatch.setattr(server, "surface_audit", MagicMock(recent=MagicMock(return_value=[]), max_events=50))

    post = MagicMock(); post.json = AsyncMock(return_value={"breath_max_results": 25})
    resp = await server.api_surface_settings_update(post)
    assert resp.status_code == 200
    assert json.loads(resp.body)["settings"]["breath_max_results"] == 25

    bad = MagicMock(); bad.json = AsyncMock(return_value={"breath_max_results": 0})
    assert (await server.api_surface_settings_update(bad)).status_code == 400

    get = MagicMock()
    data = json.loads((await server.api_surface_settings_get(get)).body)
    assert data["settings"]["breath_max_results"] == 25
    assert data["limits"]["breath_max_results"] == {"min": 1, "max": 50}
    assert data["last"] == {"breath": None, "dream": None}
