from unittest.mock import AsyncMock

import pytest

import server


def _bucket(bucket_id, name, content="正文", bucket_type="dynamic"):
    return {
        "id": bucket_id,
        "metadata": {
            "id": bucket_id,
            "name": name,
            "type": bucket_type,
            "domain": ["回忆"],
            "tags": [],
        },
        "content": content,
    }


@pytest.mark.asyncio
async def test_dashboard_search_guarantees_id_and_vector_results_without_activation(monkeypatch):
    by_id = _bucket("e977f156d719", "远程情趣对话")
    semantic = _bucket("semantic00001", "没有字面重合的记忆")
    isolated = _bucket("feel00000001", "只在 feel 里的内容", bucket_type="feel")
    monkeypatch.setattr(
        server.bucket_mgr,
        "list_all",
        AsyncMock(return_value=[by_id, semantic, isolated]),
    )
    keyword_search = AsyncMock(return_value=[])
    monkeypatch.setattr(server.bucket_mgr, "search", keyword_search)
    monkeypatch.setattr(server.embedding_engine, "enabled", True)
    monkeypatch.setattr(
        server.embedding_engine,
        "search_similar",
        AsyncMock(return_value=[("semantic00001", 0.82), ("feel00000001", 0.95)]),
    )
    touch = AsyncMock()
    soft_touch = AsyncMock()
    monkeypatch.setattr(server.bucket_mgr, "touch", touch)
    monkeypatch.setattr(server.bucket_mgr, "soft_touch", soft_touch)

    results = await server._dashboard_readonly_search("e977f1", limit=30)

    assert [item["id"] for item in results] == ["e977f156d719", "semantic00001"]
    assert results[0]["match_reasons"] == ["桶ID"]
    assert results[1]["match_reasons"] == ["语义向量"]
    keyword_search.assert_awaited_once_with(
        "e977f1",
        limit=30,
        use_embedding_prefilter=False,
    )
    touch.assert_not_awaited()
    soft_touch.assert_not_awaited()


@pytest.mark.asyncio
async def test_dashboard_search_guarantees_partial_title_match(monkeypatch):
    title_match = _bucket("title0000001", "钓鱼游戏记录")
    other = _bucket("other0000001", "普通日记")
    monkeypatch.setattr(
        server.bucket_mgr,
        "list_all",
        AsyncMock(return_value=[other, title_match]),
    )
    monkeypatch.setattr(server.bucket_mgr, "search", AsyncMock(return_value=[]))
    monkeypatch.setattr(server.embedding_engine, "enabled", False)

    results = await server._dashboard_readonly_search("钓鱼游戏", limit=30)

    assert [item["id"] for item in results] == ["title0000001"]
    assert results[0]["match_reasons"] == ["标题"]


@pytest.mark.asyncio
async def test_dashboard_search_merges_keyword_reason_with_direct_match(monkeypatch):
    title_match = _bucket("title0000001", "冲突教训")
    keyword_copy = dict(title_match)
    keyword_copy["score"] = 73.5
    monkeypatch.setattr(
        server.bucket_mgr,
        "list_all",
        AsyncMock(return_value=[title_match]),
    )
    monkeypatch.setattr(
        server.bucket_mgr,
        "search",
        AsyncMock(return_value=[keyword_copy]),
    )
    monkeypatch.setattr(server.embedding_engine, "enabled", False)

    results = await server._dashboard_readonly_search("冲突", limit=30)

    assert len(results) == 1
    assert results[0]["match_reasons"] == ["标题", "关键词"]
    assert results[0]["score"] == 97.0
