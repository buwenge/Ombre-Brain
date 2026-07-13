"""Regression tests for decay-engine startup after a service redeploy."""

import json
from datetime import datetime
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
    assert "总桶数: 0" in result
    assert "异常摘要（最多3条）:" in result


@pytest.mark.asyncio
async def test_pulse_is_fixed_compact_and_never_leaks_bucket_content(
    monkeypatch,
    stopped_decay_engine,
):
    today = datetime.now().date().isoformat()
    buckets = [
        {
            "id": "private-memory-id",
            "content": "PRIVATE-CONTENT-MUST-NOT-LEAK",
            "metadata": {
                "type": "dynamic",
                "created": today + "T01:00:00",
                "domain": ["未分类"],
                "digested": False,
            },
        },
        {
            "id": "digested-id",
            "content": "digested",
            "metadata": {
                "type": "dynamic",
                "created": today + "T02:00:00",
                "domain": ["生活"],
                "digested": True,
            },
        },
        {
            "id": "feel-id",
            "content": "feel",
            "metadata": {
                "type": "feel",
                "created": "2026-01-01T00:00:00",
                "domain": [],
            },
        },
        {
            "id": "bad-date-id",
            "content": "bad date",
            "metadata": {
                "type": "archived",
                "created": "not-a-date",
                "domain": ["生活"],
            },
        },
    ]
    monkeypatch.setattr(
        server.bucket_mgr,
        "list_all",
        AsyncMock(return_value=buckets),
    )
    monkeypatch.setattr(
        server.dehydrator,
        "recent_tagging_failures",
        [{"timestamp": today}],
    )
    monkeypatch.setattr(server.embedding_engine, "enabled", True)
    monkeypatch.setattr(
        server.embedding_engine,
        "list_all_ids",
        MagicMock(return_value={"digested-id", "orphan-id"}),
    )

    result = await server.pulse(include_archive=True)

    assert "总桶数: 4" in result
    assert "今日新增: 2" in result
    assert "未消化/feel数: 1/1" in result
    assert "打标失败计数: 1" in result
    assert result.count("\n- ") == 3
    assert "PRIVATE-CONTENT-MUST-NOT-LEAK" not in result
    assert "private-memory-id" not in result
    assert server.count_tokens_approx(result) <= server.PULSE_MAX_TOKENS


def test_pulse_cap_truncates_oversized_output():
    result = server._cap_pulse_output("异常" * 2000)

    assert result.endswith("[输出已截断]")
    assert server.count_tokens_approx(result) <= server.PULSE_MAX_TOKENS


@pytest.mark.asyncio
async def test_admin_diagnostics_is_paginated_and_excludes_content(
    monkeypatch,
    stopped_decay_engine,
):
    buckets = [
        {
            "id": f"bucket-{index}",
            "content": f"SECRET-{index}",
            "metadata": {
                "name": f"name-{index}",
                "type": "dynamic",
                "created": f"2026-07-{index + 1:02d}T00:00:00",
                "domain": ["生活"],
                "tags": ["tag"],
            },
        }
        for index in range(3)
    ]
    monkeypatch.setattr(server, "_require_auth", lambda request: None)
    monkeypatch.setattr(
        server.bucket_mgr,
        "list_all",
        AsyncMock(return_value=buckets),
    )
    monkeypatch.setattr(server.embedding_engine, "enabled", False)

    request = MagicMock()
    request.query_params = {
        "offset": "1",
        "limit": "1",
        "include_archive": "false",
    }
    response = await server.api_admin_diagnostics(request)
    payload = json.loads(response.body)

    assert payload["pagination"]["returned"] == 1
    assert payload["pagination"]["total"] == 3
    assert len(payload["buckets"]) == 1
    assert "content" not in payload["buckets"][0]
    assert "content_preview" not in payload["buckets"][0]
    assert "SECRET" not in response.body.decode("utf-8")


@pytest.mark.asyncio
async def test_admin_diagnostics_requires_authentication():
    request = MagicMock()
    request.cookies = {}

    response = await server.api_admin_diagnostics(request)

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_admin_diagnostics_is_not_registered_as_mcp_tool():
    tool_names = {tool.name for tool in await server.mcp.list_tools()}

    assert "diagnostics" not in tool_names
    assert "api_admin_diagnostics" not in tool_names
