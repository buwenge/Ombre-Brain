"""Regression tests for maintenance edits versus real activation."""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import frontmatter
import pytest

import server


def _set_activation_state(bucket_mgr, bucket_id, *, last_active, activation_count):
    path = bucket_mgr._find_bucket_file(bucket_id)
    post = frontmatter.load(path)
    post["last_active"] = last_active
    post["activation_count"] = activation_count
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(frontmatter.dumps(post))


@pytest.mark.asyncio
async def test_maintenance_update_preserves_activation_state(bucket_mgr):
    bucket_id = await bucket_mgr.create(content="需要纠错的记忆", name="原名")
    old_time = (datetime.now() - timedelta(days=20)).isoformat()
    _set_activation_state(
        bucket_mgr,
        bucket_id,
        last_active=old_time,
        activation_count=4.2,
    )

    assert await bucket_mgr.update(bucket_id, touch=False, name="纠正后的名字") is True

    bucket = await bucket_mgr.get(bucket_id)
    assert bucket["metadata"]["name"] == "纠正后的名字"
    assert str(bucket["metadata"]["last_active"]) == old_time
    assert float(bucket["metadata"]["activation_count"]) == 4.2


@pytest.mark.asyncio
async def test_maintenance_update_does_not_emit_activation(bucket_mgr):
    events = []
    bucket_mgr.set_activation_callback(
        lambda bucket_id, mode: events.append((bucket_id, mode))
    )
    bucket_id = await bucket_mgr.create(content="只做维护的记忆")

    assert await bucket_mgr.update(bucket_id, touch=False, name="修正名字") is True

    assert events == []


@pytest.mark.asyncio
async def test_explicit_touch_refreshes_time_without_changing_count(bucket_mgr):
    bucket_id = await bucket_mgr.create(content="后来发生了相关的新事情", name="旧桶")
    old_time = (datetime.now() - timedelta(days=20)).isoformat()
    _set_activation_state(
        bucket_mgr,
        bucket_id,
        last_active=old_time,
        activation_count=4.2,
    )

    before = datetime.now()
    assert await bucket_mgr.update(bucket_id, touch=True, content="合并后的内容") is True

    bucket = await bucket_mgr.get(bucket_id)
    refreshed = datetime.fromisoformat(str(bucket["metadata"]["last_active"]))
    assert refreshed >= before.replace(microsecond=0)
    assert float(bucket["metadata"]["activation_count"]) == 4.2


@pytest.mark.asyncio
async def test_successful_activation_paths_emit_callback(bucket_mgr):
    events = []
    bucket_mgr.set_activation_callback(
        lambda bucket_id, mode: events.append((bucket_id, mode))
    )
    bucket_id = await bucket_mgr.create(content="会被真正唤醒的记忆")

    assert await bucket_mgr.update(bucket_id, touch=True, name="合并后的记忆") is True
    await bucket_mgr.soft_touch(bucket_id)
    await bucket_mgr.touch(bucket_id)

    assert events == [
        (bucket_id, "update"),
        (bucket_id, "soft_touch"),
        (bucket_id, "touch"),
    ]


@pytest.mark.asyncio
async def test_dashboard_edit_explicitly_disables_touch(monkeypatch):
    request = MagicMock()
    request.path_params = {"bucket_id": "dashboard-bucket"}
    request.json = AsyncMock(return_value={"name": "纠正后的名字"})

    monkeypatch.setattr(server, "_require_auth", lambda _request: None)
    monkeypatch.setattr(
        server.bucket_mgr,
        "get",
        AsyncMock(return_value={"id": "dashboard-bucket", "metadata": {}}),
    )
    update = AsyncMock(return_value=True)
    monkeypatch.setattr(server.bucket_mgr, "update", update)

    response = await server.api_bucket_update(request)

    assert json.loads(response.body) == {
        "ok": True,
        "updated": ["name"],
        "activation_preserved": True,
    }
    update.assert_awaited_once_with(
        "dashboard-bucket",
        touch=False,
        name="纠正后的名字",
    )
