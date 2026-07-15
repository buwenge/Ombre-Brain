"""Persistent, metadata-only audit trail for memory surfacing flows.

The audit file intentionally accepts only a small allowlist of fields.  Bucket
content, summaries, tags, and arbitrary exception messages can therefore never
be persisted even if a caller accidentally includes them.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger("ombre_brain.surface_audit")


class SurfaceAuditLog:
    """Small persistent ring buffer for Breath/Dream/Feel selection metadata."""

    ENTRY_FIELDS = {
        "id",
        "name",
        "channel",
        "type",
        "created",
        "score",
        "weight_rank",
        "breath_rank",
        "newest_position",
        "candidate_position",
        "output_position",
        "cold_start",
        "outcome",
        "reason",
        "summary_tokens",
        "budget_before",
    }
    CONTEXT_FIELDS = {
        "total_buckets",
        "pinned_count",
        "dynamic_pool_count",
        "dream_reserved_count",
        "candidate_count",
        "returned_count",
        "pinned_returned_count",
        "dynamic_returned_count",
        "max_results",
        "max_tokens",
        "remaining_tokens",
        "status",
        "error",
    }

    def __init__(self, buckets_dir: str, max_events: int = 50):
        self.path = Path(buckets_dir) / ".surface_audit.json"
        self.max_events = max(1, int(max_events))
        self._lock = threading.RLock()

    @staticmethod
    def _clean_value(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def _load_unlocked(self) -> list[dict]:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            events = payload.get("events", []) if isinstance(payload, dict) else []
            return events if isinstance(events, list) else []
        except FileNotFoundError:
            return []
        except Exception as exc:
            logger.warning("Surface audit state could not be read: %s", type(exc).__name__)
            return []

    def record(self, flow: str, entries: list[dict] | None = None, **context: Any) -> dict:
        clean_entries = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            clean_entries.append({
                key: self._clean_value(value)
                for key, value in entry.items()
                if key in self.ENTRY_FIELDS
            })

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "flow": str(flow),
            **{
                key: self._clean_value(value)
                for key, value in context.items()
                if key in self.CONTEXT_FIELDS
            },
            "entries": clean_entries,
        }

        with self._lock:
            events = self._load_unlocked()
            events.append(event)
            events = events[-self.max_events:]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.path.with_suffix(".tmp")
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump({"version": 1, "events": events}, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, self.path)
        return event

    def recent(self, limit: int = 20) -> list[dict]:
        bounded = max(1, min(self.max_events, int(limit)))
        with self._lock:
            return list(reversed(self._load_unlocked()[-bounded:]))
