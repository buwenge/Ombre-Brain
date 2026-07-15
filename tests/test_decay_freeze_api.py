"""Authenticated Dashboard API tests for decay freezing."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import server


@pytest.mark.asyncio
async def test_freeze_api_starts_persistent_freeze(monkeypatch):
    request = MagicMock()
    request.json = AsyncMock(return_value={"frozen": True})
    monkeypatch.setattr(server, "_require_auth", lambda _request: None)
    clock = MagicMock()
    clock.freeze.return_value = {"frozen": True, "frozen_since": "2026-07-15T00:00:00"}
    monkeypatch.setattr(server.decay_engine, "relationship_clock", clock)

    response = await server.api_decay_freeze_update(request)

    assert response.status_code == 200
    assert json.loads(response.body)["frozen"] is True
    clock.freeze.assert_called_once_with()
    clock.resume.assert_not_called()


@pytest.mark.asyncio
async def test_freeze_api_allows_manual_resume(monkeypatch):
    request = MagicMock()
    request.json = AsyncMock(return_value={"frozen": False})
    monkeypatch.setattr(server, "_require_auth", lambda _request: None)
    clock = MagicMock()
    clock.resume.return_value = {"frozen": False, "completed_intervals": 1}
    monkeypatch.setattr(server.decay_engine, "relationship_clock", clock)

    response = await server.api_decay_freeze_update(request)

    assert response.status_code == 200
    assert json.loads(response.body)["frozen"] is False
    clock.resume.assert_called_once_with("dashboard_manual")


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [[], {}, {"frozen": 1}, {"frozen": "yes"}])
async def test_freeze_api_rejects_non_boolean_input(monkeypatch, body):
    request = MagicMock()
    request.json = AsyncMock(return_value=body)
    monkeypatch.setattr(server, "_require_auth", lambda _request: None)

    response = await server.api_decay_freeze_update(request)

    assert response.status_code == 400
