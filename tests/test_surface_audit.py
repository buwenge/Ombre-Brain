import json
import stat
from unittest.mock import MagicMock

import pytest

import server
from surface_audit import SurfaceAuditLog


def test_surface_audit_persists_allowlisted_metadata_only(tmp_path):
    audit = SurfaceAuditLog(str(tmp_path), max_events=3)

    audit.record(
        "breath",
        [{
            "id": "bucket-1",
            "name": "test memory",
            "weight_rank": 2,
            "outcome": "surfaced",
            "content": "SECRET-CONTENT",
            "summary": "SECRET-SUMMARY",
        }],
        total_buckets=10,
        arbitrary_secret="SECRET-CONTEXT",
    )

    raw = (tmp_path / ".surface_audit.json").read_text(encoding="utf-8")
    assert "SECRET" not in raw
    assert stat.S_IMODE((tmp_path / ".surface_audit.json").stat().st_mode) == 0o600
    event = audit.recent(1)[0]
    assert event["flow"] == "breath"
    assert event["total_buckets"] == 10
    assert event["entries"] == [{
        "id": "bucket-1",
        "name": "test memory",
        "weight_rank": 2,
        "outcome": "surfaced",
    }]


def test_surface_audit_retains_fixed_number_and_returns_newest_first(tmp_path):
    audit = SurfaceAuditLog(str(tmp_path), max_events=2)
    audit.record("dream", [], returned_count=1)
    audit.record("feel", [], returned_count=2)
    audit.record("breath", [], returned_count=3)

    events = audit.recent(20)

    assert [event["flow"] for event in events] == ["breath", "feel"]
    payload = json.loads((tmp_path / ".surface_audit.json").read_text(encoding="utf-8"))
    assert len(payload["events"]) == 2


def test_surface_audit_recovers_from_corrupt_state(tmp_path):
    path = tmp_path / ".surface_audit.json"
    path.write_text("not json", encoding="utf-8")
    audit = SurfaceAuditLog(str(tmp_path), max_events=2)

    audit.record("dream", [], returned_count=0)

    assert audit.recent(1)[0]["flow"] == "dream"


def test_surface_audit_failure_does_not_escape_recall_path(monkeypatch):
    broken_audit = MagicMock()
    broken_audit.record.side_effect = OSError("disk unavailable")
    monkeypatch.setattr(server, "surface_audit", broken_audit)

    server._record_surface_audit("breath", [{"id": "bucket-1"}], status="complete")

    broken_audit.record.assert_called_once()


@pytest.mark.asyncio
async def test_surface_audit_api_is_authenticated_and_bounded(monkeypatch):
    fake_audit = MagicMock()
    fake_audit.max_events = 50
    fake_audit.recent.return_value = [{"flow": "breath", "entries": []}]
    monkeypatch.setattr(server, "surface_audit", fake_audit)
    monkeypatch.setattr(server, "_require_auth", lambda request: None)
    request = MagicMock()
    request.query_params = {"limit": "999"}

    response = await server.api_admin_surface_audit(request)
    payload = json.loads(response.body)

    fake_audit.recent.assert_called_once_with(50)
    assert payload["retention"] == 50
    assert payload["events"][0]["flow"] == "breath"


@pytest.mark.asyncio
async def test_surface_audit_api_requires_authentication():
    request = MagicMock()
    request.cookies = {}

    response = await server.api_admin_surface_audit(request)

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_surface_audit_api_is_not_registered_as_mcp_tool():
    tool_names = {tool.name for tool in await server.mcp.list_tools()}

    assert "surface_audit" not in tool_names
    assert "api_admin_surface_audit" not in tool_names
