"""Regression tests for decay-engine startup after a service redeploy."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import server


@pytest.fixture
def stopped_decay_engine(monkeypatch):
    engine = MagicMock()
    engine.is_running = False

    async def start_engine():
        engine.is_running = True

    engine.ensure_started = AsyncMock(side_effect=start_engine)
    engine.calculate_score.return_value = 1.0
    monkeypatch.setattr(server, "decay_engine", engine)
    return engine


@pytest.mark.asyncio
async def test_health_bootstraps_decay_engine(monkeypatch, stopped_decay_engine):
    stats = {
        "permanent_count": 1,
        "dynamic_count": 2,
    }
    monkeypatch.setattr(
        server.bucket_mgr,
        "get_stats",
        AsyncMock(return_value=stats),
    )

    response = await server.health_check(None)

    stopped_decay_engine.ensure_started.assert_awaited_once_with()
    assert json.loads(response.body) == {
        "status": "ok",
        "buckets": 3,
        "decay_engine": "running",
    }


@pytest.mark.asyncio
async def test_pulse_bootstraps_decay_engine_before_reporting(
    monkeypatch,
    stopped_decay_engine,
):
    stats = {
        "permanent_count": 1,
        "dynamic_count": 2,
        "archive_count": 3,
        "total_size_kb": 4.0,
    }
    monkeypatch.setattr(
        server.bucket_mgr,
        "get_stats",
        AsyncMock(return_value=stats),
    )
    monkeypatch.setattr(
        server.bucket_mgr,
        "list_all",
        AsyncMock(return_value=[]),
    )

    result = await server.pulse()

    stopped_decay_engine.ensure_started.assert_awaited_once_with()
    assert "衰减引擎: 运行中" in result
