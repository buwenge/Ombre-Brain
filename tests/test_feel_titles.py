from unittest.mock import AsyncMock

import pytest

import server


@pytest.mark.asyncio
async def test_feel_without_self_written_title_is_not_saved(monkeypatch):
    monkeypatch.setattr(server.decay_engine, "ensure_started", AsyncMock())
    create = AsyncMock(return_value="should-not-exist")
    monkeypatch.setattr(server.bucket_mgr, "create", create)

    result = await server.hold(content="我留下了一点感受。", feel=True, valence=0.7)

    assert "title" in result
    assert "未保存" in result
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_feel_uses_title_written_by_xiaoyu(monkeypatch):
    monkeypatch.setattr(server.decay_engine, "ensure_started", AsyncMock())
    create = AsyncMock(return_value="feel-title-id")
    monkeypatch.setattr(server.bucket_mgr, "create", create)
    monkeypatch.setattr(server.embedding_engine, "generate_and_store", AsyncMock())

    result = await server.hold(
        content="我留下了一点感受。",
        feel=True,
        title="我终于明白的事",
        valence=0.7,
        arousal=0.4,
    )

    assert result == "🫧feel→feel-title-id"
    create.assert_awaited_once_with(
        content="我留下了一点感受。",
        tags=[],
        importance=5,
        domain=[],
        valence=0.7,
        arousal=0.4,
        name="我终于明白的事",
        bucket_type="feel",
    )
