import json
import time

import monitor
import tracker


class TestDecideCommand:
    def test_centered_dog_no_command(self):
        assert tracker.decide_command(320, 640) is None
        assert tracker.decide_command(320 + 640 * 0.13, 640) is None

    def test_dog_right_of_center_pans_right(self):
        direction, steps = tracker.decide_command(560, 640)
        assert direction == "right"
        assert 1 <= steps <= tracker.MAX_STEPS_PER_CMD

    def test_dog_left_of_center_pans_left(self):
        direction, _ = tracker.decide_command(80, 640)
        assert direction == "left"

    def test_inverted_sign_flips_direction(self):
        direction, _ = tracker.decide_command(560, 640, pan_sign=-1)
        assert direction == "left"

    def test_far_offset_capped_at_max_steps(self):
        _, steps = tracker.decide_command(639, 640)
        assert steps == tracker.MAX_STEPS_PER_CMD

    def test_half_gain_undershoots(self):
        off = 6 * tracker.STEP_PX          # 108 px, outside the deadband
        _, steps = tracker.decide_command(320 + off, 640)
        assert steps <= 3                  # half-gain: never the full offset


class TestPanSign:
    def test_correction_that_helps_keeps_sign(self):
        ps = tracker.PanSign()
        ps.sent(120.0)
        ps.observed(40.0)
        assert ps.sign == 1

    def test_correction_that_hurts_flips_sign(self):
        ps = tracker.PanSign()
        ps.sent(120.0)
        ps.observed(160.0)
        assert ps.sign == -1

    def test_observation_without_send_is_ignored(self):
        ps = tracker.PanSign()
        ps.observed(500.0)
        assert ps.sign == 1


class TestCommandBudget:
    def test_min_gap_enforced(self):
        b = tracker.CommandBudget(min_gap=4.0, per_min=8)
        assert b.allow(100.0)
        b.record(100.0)
        assert not b.allow(102.0)
        assert b.allow(104.5)

    def test_per_minute_cap(self):
        b = tracker.CommandBudget(min_gap=0.0, per_min=3)
        for t in (100.0, 101.0, 102.0):
            assert b.allow(t)
            b.record(t)
        assert not b.allow(103.0)
        assert b.allow(161.5)          # window rolled past the first sends


class TestMovingFlagRoundTrip:
    def test_monitor_honors_fresh_flag_and_ignores_stale(self, tmp_path):
        flag_path = tmp_path / "mi360_moving.json"
        now = time.time()
        tracker.write_moving_flag(flag_path, now + 5.0)
        mf = monitor.MovingFlag(str(flag_path))
        assert mf.moving(now) is True
        assert mf.moving(now + 10.0) is False   # stale window ignored

    def test_missing_or_malformed_flag_never_blinds(self, tmp_path):
        assert monitor.MovingFlag(None).moving(0.0) is False
        assert monitor.MovingFlag(str(tmp_path / "nope.json")).moving(0.0) is False
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        assert monitor.MovingFlag(str(bad)).moving(0.0) is False

    def test_rewritten_flag_is_picked_up(self, tmp_path):
        flag_path = tmp_path / "f.json"
        tracker.write_moving_flag(flag_path, 100.0)
        mf = monitor.MovingFlag(str(flag_path))
        assert mf.moving(99.0) is True
        time.sleep(0.02)
        tracker.write_moving_flag(flag_path, 200.0)
        assert mf.moving(150.0) is True
