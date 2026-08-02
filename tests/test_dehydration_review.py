import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import server
from dehydrator import Dehydrator


def _response(content, finish_reason="stop"):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def test_review_split_preserves_complete_source():
    source = ("第一段。" * 240) + "\n\n" + ("第二段很长！" * 240)

    chunks = Dehydrator.split_review_content(source, max_chars=180)

    assert len(chunks) > 2
    assert "".join(chunks) == source
    assert all(0 < len(chunk) <= 180 for chunk in chunks)


def test_deepseek_structured_calls_disable_thinking(test_config):
    test_config["dehydration"].update({
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-v4-flash",
    })
    dehydrator = Dehydrator(test_config)

    assert dehydrator._deepseek_thinking_body() == {
        "thinking": {"type": "disabled"}
    }


def test_other_providers_do_not_receive_deepseek_option(test_config):
    dehydrator = Dehydrator(test_config)

    assert dehydrator._deepseek_thinking_body() is None


def test_dehydration_parser_rejects_invented_role(test_config):
    dehydrator = Dehydrator(test_config)
    raw = json.dumps({"core_facts": ["用户很生气"], "summary": "用户生气"})

    with pytest.raises(ValueError, match="擅自引入称呼"):
        dehydrator._parse_dehydration(raw, source="我很生气")


@pytest.mark.asyncio
async def test_api_dehydrate_retries_truncated_json_and_returns_canonical(test_config):
    dehydrator = Dehydrator(test_config)
    good = json.dumps({"core_facts": ["我记住了"], "summary": "我记住了"})
    dehydrator._chat_completion = AsyncMock(side_effect=[
        _response("{", finish_reason="length"),
        _response(good),
    ])

    result = await dehydrator._api_dehydrate("我记住了")

    assert json.loads(result) == {"core_facts": ["我记住了"], "summary": "我记住了"}
    assert dehydrator._chat_completion.await_count == 2


@pytest.mark.asyncio
async def test_review_draft_keeps_failed_chunk_visible(test_config):
    dehydrator = Dehydrator(test_config)
    source = "甲" * 1800 + "乙" * 10
    valid = json.dumps({"core_facts": ["甲"], "summary": "甲"})
    dehydrator._api_dehydrate = AsyncMock(side_effect=[valid, RuntimeError("第二段失败")])

    result = await dehydrator.generate_review_draft(source)

    assert result["failed_chunks"] == 1
    assert len(result["chunks"]) == 2
    assert "".join(chunk["source"] for chunk in result["chunks"]) == source
    assert result["chunks"][1]["error"] == "第二段失败"
    assert dehydrator._get_cached_summary(source) is None


def _bucket(bucket_id, content, created, **metadata):
    meta = {
        "id": bucket_id,
        "name": bucket_id,
        "type": "dynamic",
        "created": created,
        **metadata,
    }
    return {"id": bucket_id, "metadata": meta, "content": content}


def test_queue_only_contains_unreviewed_long_normal_buckets():
    long_content = "今天发生了一件需要记住的事。" * 40
    reviewed_hash = __import__("hashlib").sha256(long_content.encode()).hexdigest()
    buckets = [
        _bucket("wanted", long_content, "2026-08-02T08:00:00"),
        _bucket("reviewed", long_content, "2026-08-02T07:00:00", dehydration_edited_hash=reviewed_hash),
        _bucket("raw", long_content, "2026-08-02T06:00:00", verbatim=True),
        _bucket("short", "很短", "2026-08-02T05:00:00"),
        _bucket("feel", long_content, "2026-08-02T04:00:00", type="feel"),
        _bucket("old", long_content, "2026-07-20T04:00:00"),
    ]

    today_rows = server._dehydration_queue_rows(buckets, "today", today=date(2026, 8, 2))
    all_rows = server._dehydration_queue_rows(buckets, "all", today=date(2026, 8, 2))

    assert [row["id"] for row in today_rows] == ["wanted"]
    assert [row["id"] for row in all_rows] == ["wanted", "old"]


def test_queue_requeues_manual_draft_after_source_changes():
    old_content = "旧正文。" * 100
    new_content = "新正文。" * 100
    old_hash = __import__("hashlib").sha256(old_content.encode()).hexdigest()
    buckets = [_bucket(
        "stale",
        new_content,
        "2026-08-02T08:00:00",
        dehydration_edited_hash=old_hash,
    )]

    rows = server._dehydration_queue_rows(buckets, "today", today=date(2026, 8, 2))

    assert rows[0]["id"] == "stale"
    assert rows[0]["stale_manual"] is True


def test_queue_uses_reviewer_local_day():
    content = "刚过午夜保存的中国时区记忆。" * 40
    buckets = [_bucket("after-midnight", content, "2026-08-01T16:30:00")]

    rows = server._dehydration_queue_rows(
        buckets,
        "today",
        today=date(2026, 8, 2),
        timezone_offset_minutes=-480,
    )

    assert [row["id"] for row in rows] == ["after-midnight"]


@pytest.mark.asyncio
async def test_manual_save_overwrites_cache_without_activating(monkeypatch):
    request = MagicMock()
    request.path_params = {"bucket_id": "reviewed"}
    request.json = AsyncMock(return_value={
        "core_facts": ["我记住了"],
        "summary": "我记住了这件事",
    })
    monkeypatch.setattr(server, "_require_auth", lambda _request: None)
    monkeypatch.setattr(server.bucket_mgr, "get", AsyncMock(return_value={
        "id": "reviewed",
        "metadata": {},
        "content": "我记住了这件事",
    }))
    monkeypatch.setattr(server.dehydrator, "set_manual_summary", MagicMock(return_value="content-hash"))
    update = AsyncMock(return_value=True)
    monkeypatch.setattr(server.bucket_mgr, "update", update)

    response = await server.api_dehydrate_preview_save(request)

    assert json.loads(response.body) == {"ok": True}
    update.assert_awaited_once_with(
        "reviewed",
        touch=False,
        dehydration_edited_hash="content-hash",
        verbatim=False,
    )


@pytest.mark.asyncio
async def test_keep_original_does_not_activate(monkeypatch):
    request = MagicMock()
    request.path_params = {"bucket_id": "raw"}
    monkeypatch.setattr(server, "_require_auth", lambda _request: None)
    monkeypatch.setattr(server.bucket_mgr, "get", AsyncMock(return_value={"id": "raw"}))
    update = AsyncMock(return_value=True)
    monkeypatch.setattr(server.bucket_mgr, "update", update)

    response = await server.api_dehydration_review_verbatim(request)

    assert json.loads(response.body) == {"ok": True}
    update.assert_awaited_once_with("raw", touch=False, verbatim=True)
