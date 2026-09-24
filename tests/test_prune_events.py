import json
import os
import time

import prune_events

DAY = 86400


def _event(root, name, age_days, positive=None):
    ev = root / name
    ev.mkdir(parents=True)
    (ev / "frame.jpg").write_bytes(b"x" * 1000)
    if positive is not None:
        (ev / "analysis.json").write_text(
            json.dumps({"final_abnormal_event": positive}), encoding="utf-8")
    old = time.time() - age_days * DAY
    os.utime(ev, (old, old))
    return ev


class TestPrune:
    def test_old_negative_removed_recent_kept(self, tmp_path):
        _event(tmp_path, "event_old_neg", 30, positive=False)
        _event(tmp_path, "event_new_neg", 2, positive=False)
        removed, freed = prune_events.prune(tmp_path, keep_days=14)
        assert removed == ["event_old_neg"]
        assert freed >= 1000              # frame bytes + the analysis.json
        assert not (tmp_path / "event_old_neg").exists()
        assert (tmp_path / "event_new_neg").exists()

    def test_positive_events_kept_forever(self, tmp_path):
        _event(tmp_path, "event_old_pos", 400, positive=True)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == []

    def test_unverified_old_event_removed(self, tmp_path):
        _event(tmp_path, "event_old_unverified", 30, positive=None)
        _event(tmp_path, "event_recent", 1, positive=False)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == ["event_old_unverified"]

    def test_unreadable_verdict_is_kept(self, tmp_path):
        ev = _event(tmp_path, "event_old_broken", 30, positive=False)
        (ev / "analysis.json").write_text("{not json", encoding="utf-8")
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == []

    def test_dry_run_deletes_nothing(self, tmp_path):
        _event(tmp_path, "event_old_neg", 30, positive=False)
        _event(tmp_path, "event_recent", 1, positive=False)
        removed, _ = prune_events.prune(tmp_path, keep_days=14, dry_run=True)
        assert removed == ["event_old_neg"]
        assert (tmp_path / "event_old_neg").exists()

    def test_blind_spell_does_not_empty_the_archive(self, tmp_path):
        """Regression (2026-09): the monitor recorded nothing for 20 days and
        a now-based window deleted every negative from before the gap. The
        window now hangs off the newest event, so a gap removes nothing."""
        _event(tmp_path, "event_before_gap_a", 20, positive=False)
        _event(tmp_path, "event_before_gap_b", 25, positive=False)
        _event(tmp_path, "event_long_before", 40, positive=False)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == ["event_long_before"]

    def test_future_dated_event_does_not_slide_the_window(self, tmp_path):
        _event(tmp_path, "event_future", -30, positive=False)
        _event(tmp_path, "event_recent", 10, positive=False)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == []

    def test_empty_root(self, tmp_path):
        assert prune_events.prune(tmp_path, keep_days=14) == ([], 0)

    def test_non_event_dirs_untouched(self, tmp_path):
        other = tmp_path / "eval_work"
        other.mkdir()
        old = time.time() - 100 * DAY
        os.utime(other, (old, old))
        prune_events.prune(tmp_path, keep_days=14)
        assert other.exists()
