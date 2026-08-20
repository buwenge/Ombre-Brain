"""Regression tests for breath(bucket_id=...) supporting comma-separated batch fetch."""

from unittest.mock import AsyncMock

import pytest

import server


def _bucket(bucket_id: str, content: str):
    return {
        "id": bucket_id,
        "content": content,
        "metadata": {"name": bucket_id, "type": "dynamic"},
    }


def _make_get(store):
    async def _get(bucket_id):
        return store.get(bucket_id)
    return _get


@pytest.mark.asyncio
async def test_bucket_id_batch_fetches_all_and_touches_each(monkeypatch):
    store = {
        "aaa": _bucket("aaa", "content-aaa"),
        "bbb": _bucket("bbb", "content-bbb"),
    }
    monkeypatch.setattr(server.bucket_mgr, "get", _make_get(store))
    touch = AsyncMock()
    monkeypatch.setattr(server.bucket_mgr, "touch", touch)
    monkeypatch.setattr(
        server.dehydrator,
        "dehydrate",
        AsyncMock(side_effect=lambda content, metadata=None: content),
    )

    result = await server.breath(bucket_id="aaa,bbb")

    assert "[bucket_id:aaa] content-aaa" in result
    assert "[bucket_id:bbb] content-bbb" in result
    assert touch.await_count == 2


@pytest.mark.asyncio
async def test_bucket_id_batch_reports_missing_ids(monkeypatch):
    store = {"aaa": _bucket("aaa", "content-aaa")}
    monkeypatch.setattr(server.bucket_mgr, "get", _make_get(store))
    monkeypatch.setattr(server.bucket_mgr, "touch", AsyncMock())
    monkeypatch.setattr(
        server.dehydrator,
        "dehydrate",
        AsyncMock(side_effect=lambda content, metadata=None: content),
    )

    result = await server.breath(bucket_id="aaa,missing")

    assert "[bucket_id:aaa] content-aaa" in result
    assert "missing" in result


@pytest.mark.asyncio
async def test_bucket_id_batch_stops_at_token_budget(monkeypatch):
    store = {
        "aaa": _bucket("aaa", "x" * 400),
        "bbb": _bucket("bbb", "y" * 400),
    }
    monkeypatch.setattr(server.bucket_mgr, "get", _make_get(store))
    monkeypatch.setattr(server.bucket_mgr, "touch", AsyncMock())
    monkeypatch.setattr(
        server.dehydrator,
        "dehydrate",
        AsyncMock(side_effect=lambda content, metadata=None: content),
    )

    # Each 400-char summary is ~21 tokens (count_tokens_approx: len*0.05 + one
    # english-word run). max_tokens=25 fits exactly one, not both.
    result = await server.breath(bucket_id="aaa,bbb", max_tokens=25)

    assert "aaa" in result
    assert "bbb" not in result
