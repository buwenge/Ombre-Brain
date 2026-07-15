"""Regression tests for non-overlapping startup memory channels."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import server


def _bucket(index: int, *, bucket_type="dynamic", pinned=False):
    return {
        "id": f"bucket-{index:02d}",
        "content": f"memory-{index:02d}",
        "metadata": {
            "name": f"memory-{index:02d}",
            "type": bucket_type,
            "created": f"2026-07-{index:02d}T00:00:00",
            "last_active": f"2026-07-{index:02d}T00:00:00",
            "importance": index,
            "activation_count": 1,
            "resolved": False,
            "pinned": pinned,
        },
    }


def test_dream_reservation_is_newest_ten_and_excludes_private_types():
    buckets = [_bucket(index) for index in range(1, 13)]
    buckets.extend([
        _bucket(20, bucket_type="letter"),
        _bucket(21, bucket_type="feel"),
        _bucket(22, bucket_type="permanent"),
        _bucket(23, pinned=True),
    ])

    recent = server._select_dream_recent(buckets)

    assert [item["id"] for item in recent] == [
        f"bucket-{index:02d}" for index in range(12, 2, -1)
    ]


@pytest.mark.asyncio
async def test_breath_and_dream_outputs_are_disjoint(monkeypatch):
    buckets = [_bucket(index) for index in range(1, 13)]
    monkeypatch.setattr(
        server.bucket_mgr,
        "list_all",
        AsyncMock(return_value=buckets),
    )
    monkeypatch.setattr(server.bucket_mgr, "soft_touch", AsyncMock())
    monkeypatch.setattr(server.decay_engine, "ensure_started", AsyncMock())
    monkeypatch.setattr(
        server.decay_engine,
        "calculate_score",
        lambda metadata: float(metadata["importance"]),
    )
    monkeypatch.setattr(
        server.dehydrator,
        "dehydrate",
        AsyncMock(side_effect=lambda content, metadata=None: content),
    )
    monkeypatch.setattr(server.embedding_engine, "enabled", False)
    monkeypatch.setattr(server, "_fire_webhook", AsyncMock())
    audit = MagicMock()
    monkeypatch.setattr(server, "surface_audit", audit)
    relationship_clock = MagicMock()
    monkeypatch.setattr(server.decay_engine, "relationship_clock", relationship_clock)

    breath_text = await server.breath()
    dream_text = await server.dream()

    relationship_clock.resume.assert_called_once_with("breath")

    assert "memory-01" in breath_text
    assert "memory-02" in breath_text
    for index in range(3, 13):
        assert f"memory-{index:02d}" not in breath_text
        assert f"memory-{index:02d}" in dream_text
    assert "memory-01" not in dream_text
    assert "memory-02" not in dream_text

    calls = {call.args[0]: call for call in audit.record.call_args_list}
    breath_entries = calls["breath"].args[1]
    dream_entries = calls["dream"].args[1]

    surfaced = [entry for entry in breath_entries if entry.get("outcome") == "surfaced"]
    reserved = [entry for entry in breath_entries if entry.get("outcome") == "reserved_for_dream"]
    assert {entry["id"] for entry in surfaced} == {"bucket-01", "bucket-02"}
    assert {entry["weight_rank"] for entry in surfaced} == {11, 12}
    assert {entry["id"] for entry in reserved} == {
        f"bucket-{index:02d}" for index in range(3, 13)
    }
    assert [entry["weight_rank"] for entry in dream_entries] == list(range(1, 11))
    assert [entry["newest_position"] for entry in dream_entries] == list(range(1, 11))
