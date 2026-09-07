"""Dashboard-editable caps for Breath/Dream surfacing.

Stored as a tiny JSON file next to the buckets (survives restarts and git
pulls).  Readers must call ``get()`` at call time — never freeze a value at
import — so a change made in the Dashboard takes effect on the next breath.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path


logger = logging.getLogger("ombre_brain.surface_settings")


class SurfaceSettings:
    # key -> (default, minimum, maximum)
    SPEC = {
        "breath_max_results": (20, 1, 50),      # dynamic buckets only; pins never count
        "breath_max_tokens": (10000, 500, 20000),  # shared budget: pins are charged first
        "dream_max_results": (10, 1, 50),       # normal buckets only (not feel/crave)
        "dream_max_tokens": (10000, 500, 50000),
    }

    def __init__(self, buckets_dir: str):
        self.path = Path(buckets_dir) / ".surface_settings.json"
        self._lock = threading.RLock()

    @classmethod
    def defaults(cls) -> dict:
        return {key: spec[0] for key, spec in cls.SPEC.items()}

    @classmethod
    def limits(cls) -> dict:
        return {key: {"min": spec[1], "max": spec[2]} for key, spec in cls.SPEC.items()}

    def _load_unlocked(self) -> dict:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return {}
        except Exception as exc:
            logger.warning("Surface settings could not be read: %s", type(exc).__name__)
            return {}
        return payload if isinstance(payload, dict) else {}

    def all(self) -> dict:
        """Current effective values: stored overrides clamped onto defaults."""
        values = self.defaults()
        with self._lock:
            stored = self._load_unlocked()
        for key, (_default, lo, hi) in self.SPEC.items():
            raw = stored.get(key)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                continue
            values[key] = max(lo, min(hi, int(raw)))
        return values

    def get(self, key: str) -> int:
        return self.all()[key]

    def update(self, changes: dict) -> dict:
        """Validate and persist; raises ValueError naming the offending key."""
        cleaned = {}
        for key, raw in changes.items():
            if key not in self.SPEC:
                raise ValueError(f"unknown setting: {key}")
            _default, lo, hi = self.SPEC[key]
            if isinstance(raw, bool):
                raise ValueError(f"{key} must be an integer")
            try:
                value = int(raw)
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be an integer")
            if not lo <= value <= hi:
                raise ValueError(f"{key} must be between {lo} and {hi}")
            cleaned[key] = value
        with self._lock:
            stored = self._load_unlocked()
            stored.update(cleaned)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.path.with_suffix(".tmp")
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(stored, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        return self.all()
