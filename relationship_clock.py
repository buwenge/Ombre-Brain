"""Persistent relationship-time clock for pausing memory decay."""

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone

logger = logging.getLogger("ombre_brain.relationship_clock")


class RelationshipClock:
    """Exclude user-declared frozen intervals from decay elapsed time.

    The state lives beside the bucket directories on the persistent Volume.
    It stores only timestamps and resume reasons, never memory content.
    """

    STATE_FILENAME = ".decay_freeze.json"

    def __init__(self, buckets_dir: str, now_fn=None):
        self.buckets_dir = buckets_dir
        self.state_path = os.path.join(buckets_dir, self.STATE_FILENAME)
        self._now_fn = now_fn or datetime.now
        self._lock = threading.RLock()
        self._state = self._load_state()

    @staticmethod
    def _normalize(value: datetime) -> datetime:
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    @classmethod
    def _parse(cls, value) -> datetime | None:
        try:
            return cls._normalize(datetime.fromisoformat(str(value)))
        except (TypeError, ValueError):
            return None

    def _now(self) -> datetime:
        return self._normalize(self._now_fn())

    @staticmethod
    def _empty_state() -> dict:
        return {"version": 1, "frozen_since": None, "intervals": []}

    def _load_state(self) -> dict:
        if not os.path.exists(self.state_path):
            return self._empty_state()
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if not isinstance(raw, dict):
                raise ValueError("state is not an object")
            intervals = raw.get("intervals", [])
            if not isinstance(intervals, list):
                raise ValueError("intervals is not a list")
            return {
                "version": 1,
                "frozen_since": raw.get("frozen_since"),
                "intervals": intervals,
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load decay freeze state: %s", type(exc).__name__)
            return self._empty_state()

    def _save_state(self) -> None:
        os.makedirs(self.buckets_dir, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.buckets_dir,
                prefix=".decay_freeze.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = handle.name
                json.dump(self._state, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, self.state_path)
        except OSError:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            raise

    def freeze(self, at: datetime | None = None) -> dict:
        """Start a frozen interval; repeated calls are idempotent."""
        with self._lock:
            if not self._state.get("frozen_since"):
                moment = self._normalize(at) if at is not None else self._now()
                self._state["frozen_since"] = moment.isoformat(timespec="seconds")
                self._save_state()
            return self.status(at=at)

    def resume(self, reason: str, at: datetime | None = None) -> dict:
        """Close the active interval without re-counting its elapsed time."""
        with self._lock:
            frozen_since = self._parse(self._state.get("frozen_since"))
            if frozen_since is not None:
                moment = self._normalize(at) if at is not None else self._now()
                if moment < frozen_since:
                    moment = frozen_since
                self._state["intervals"].append({
                    "start": frozen_since.isoformat(timespec="seconds"),
                    "end": moment.isoformat(timespec="seconds"),
                    "resume_reason": str(reason)[:80],
                })
                self._state["frozen_since"] = None
                self._save_state()
            return self.status(at=at)

    def _interval_bounds(self, now: datetime) -> list[tuple[datetime, datetime]]:
        bounds = []
        for item in self._state.get("intervals", []):
            if not isinstance(item, dict):
                continue
            start = self._parse(item.get("start"))
            end = self._parse(item.get("end"))
            if start is not None and end is not None and end >= start:
                bounds.append((start, end))
        active_start = self._parse(self._state.get("frozen_since"))
        if active_start is not None:
            bounds.append((active_start, max(active_start, now)))
        bounds.sort(key=lambda pair: pair[0])

        merged = []
        for start, end in bounds:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return [(start, end) for start, end in merged]

    def effective_days_since(self, value, fallback_days: float = 30.0) -> float:
        """Return wall-clock days minus overlapping frozen intervals."""
        start = self._parse(value)
        if start is None:
            return fallback_days
        now = self._now()
        if start >= now:
            return 0.0
        with self._lock:
            excluded = 0.0
            for frozen_start, frozen_end in self._interval_bounds(now):
                overlap_start = max(start, frozen_start)
                overlap_end = min(now, frozen_end)
                if overlap_end > overlap_start:
                    excluded += (overlap_end - overlap_start).total_seconds()
        raw = (now - start).total_seconds()
        return max(0.0, raw - excluded) / 86400.0

    def status(self, at: datetime | None = None) -> dict:
        with self._lock:
            now = self._normalize(at) if at is not None else self._now()
            total_seconds = sum(
                (end - start).total_seconds()
                for start, end in self._interval_bounds(now)
            )
            frozen_since = self._state.get("frozen_since")
            return {
                "frozen": bool(self._parse(frozen_since)),
                "frozen_since": frozen_since,
                "excluded_total_seconds": round(max(0.0, total_seconds), 3),
                "completed_intervals": len(self._state.get("intervals", [])),
            }
