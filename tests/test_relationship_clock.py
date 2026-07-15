"""Tests for persistent user-controlled relationship-time freezing."""

from datetime import datetime

import pytest

from relationship_clock import RelationshipClock


def test_freeze_excludes_wall_time_and_resume_keeps_it_excluded(tmp_path):
    now = [datetime(2026, 7, 1, 0, 0, 0)]
    clock = RelationshipClock(str(tmp_path), now_fn=lambda: now[0])
    last_active = datetime(2026, 7, 1, 0, 0, 0)

    now[0] = datetime(2026, 7, 2, 0, 0, 0)
    frozen = clock.freeze()
    assert frozen["frozen"] is True

    now[0] = datetime(2026, 7, 5, 0, 0, 0)
    assert clock.effective_days_since(last_active) == pytest.approx(1.0)

    resumed = clock.resume("test_activation")
    assert resumed["frozen"] is False
    assert resumed["completed_intervals"] == 1

    now[0] = datetime(2026, 7, 7, 0, 0, 0)
    assert clock.effective_days_since(last_active) == pytest.approx(3.0)


def test_freeze_state_survives_reload(tmp_path):
    now = [datetime(2026, 7, 10, 12, 0, 0)]
    first = RelationshipClock(str(tmp_path), now_fn=lambda: now[0])
    first.freeze()

    now[0] = datetime(2026, 7, 13, 12, 0, 0)
    reloaded = RelationshipClock(str(tmp_path), now_fn=lambda: now[0])

    assert reloaded.status()["frozen"] is True
    assert reloaded.effective_days_since(datetime(2026, 7, 9, 12, 0, 0)) == pytest.approx(1.0)

    reloaded.resume("after_reload")
    final = RelationshipClock(str(tmp_path), now_fn=lambda: now[0])
    assert final.status()["frozen"] is False
    assert final.status()["completed_intervals"] == 1


def test_repeated_freeze_is_idempotent(tmp_path):
    now = [datetime(2026, 7, 1, 0, 0, 0)]
    clock = RelationshipClock(str(tmp_path), now_fn=lambda: now[0])
    first = clock.freeze()

    now[0] = datetime(2026, 7, 3, 0, 0, 0)
    second = clock.freeze()

    assert second["frozen_since"] == first["frozen_since"]
    clock.resume("one_resume")
    assert clock.status()["completed_intervals"] == 1


def test_decay_score_stays_stable_while_relationship_time_is_frozen(decay_eng):
    now = [datetime(2026, 7, 2, 0, 0, 0)]
    decay_eng.relationship_clock._now_fn = lambda: now[0]
    metadata = {
        "type": "dynamic",
        "importance": 8,
        "activation_count": 5,
        "arousal": 0.8,
        "last_active": "2026-07-01T00:00:00",
    }
    score_before = decay_eng.calculate_score(metadata)
    decay_eng.relationship_clock.freeze()

    now[0] = datetime(2026, 7, 8, 0, 0, 0)
    score_during_freeze = decay_eng.calculate_score(metadata)
    assert score_during_freeze == score_before

    decay_eng.relationship_clock.resume("test_activation")
    now[0] = datetime(2026, 7, 10, 0, 0, 0)
    assert decay_eng.calculate_score(metadata) < score_during_freeze


def test_only_overlap_after_last_active_is_excluded(tmp_path):
    now = [datetime(2026, 7, 1, 0, 0, 0)]
    clock = RelationshipClock(str(tmp_path), now_fn=lambda: now[0])
    clock.freeze()

    now[0] = datetime(2026, 7, 5, 0, 0, 0)
    clock.resume("activation")
    now[0] = datetime(2026, 7, 7, 0, 0, 0)

    # This memory became active during the frozen interval, so only time after
    # the interval ended counts toward decay.
    assert clock.effective_days_since(datetime(2026, 7, 3, 0, 0, 0)) == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_successful_bucket_activation_resumes_clock(bucket_mgr, tmp_path):
    now = [datetime(2026, 7, 1, 0, 0, 0)]
    clock = RelationshipClock(str(tmp_path / "clock"), now_fn=lambda: now[0])
    bucket_mgr.set_activation_callback(
        lambda _bucket_id, mode: clock.resume(f"bucket_{mode}")
    )
    bucket_id = await bucket_mgr.create(content="会在重逢时浮现的记忆")
    clock.freeze()

    now[0] = datetime(2026, 7, 4, 0, 0, 0)
    await bucket_mgr.soft_touch(bucket_id)

    status = clock.status()
    assert status["frozen"] is False
    assert status["completed_intervals"] == 1
