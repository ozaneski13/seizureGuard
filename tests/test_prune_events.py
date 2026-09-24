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


def _neg(root, name, age_days, present=0, failed=0):
    """A verified negative whose batches marked `present` signs as present."""
    ev = root / name
    ev.mkdir(parents=True)
    (ev / "frame.jpg").write_bytes(b"x" * 1000)
    signs = [{"sign": "paddling", "present": True}] * present
    (ev / "analysis.json").write_text(json.dumps({
        "final_abnormal_event": False, "failed_batches": failed,
        "batches": [{"abnormal_event": False, "observed_signs": signs}],
    }), encoding="utf-8")
    old = time.time() - age_days * DAY
    os.utime(ev, (old, old))
    return ev


class TestHardNegativeSamples:
    def _fresh(self, root):
        _event(root, "event_fresh", 0, positive=False)   # keeps the window at now

    def test_one_negative_per_camera_per_day_is_kept(self, tmp_path):
        self._fresh(tmp_path)
        _neg(tmp_path, "event_20260801_100000_mi360-pi", 30)
        _neg(tmp_path, "event_20260801_120000_mi360-pi", 30)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == ["event_20260801_120000_mi360-pi"]   # tie -> earliest kept

    def test_most_sign_like_negative_is_the_sample(self, tmp_path):
        self._fresh(tmp_path)
        _neg(tmp_path, "event_20260801_100000_mi360-pi", 30, present=0)
        _neg(tmp_path, "event_20260801_120000_mi360-pi", 30, present=2)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == ["event_20260801_100000_mi360-pi"]

    def test_each_camera_and_day_gets_its_own_sample(self, tmp_path):
        self._fresh(tmp_path)
        for name in ("event_20260801_100000_mi360-pi", "event_20260801_110000_mi360-pi",
                     "event_20260801_100000_c700-pi", "event_20260801_110000_c700-pi",
                     "event_20260802_100000_mi360-pi", "event_20260802_110000_mi360-pi"):
            _neg(tmp_path, name, 30)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert sorted(removed) == ["event_20260801_110000_c700-pi",
                                   "event_20260801_110000_mi360-pi",
                                   "event_20260802_110000_mi360-pi"]

    def test_ambiguous_names_never_merge_two_slots(self, tmp_path):
        # "_1_" may be monitor.py's collision suffix or part of a camera named
        # "1_..."; merging would delete one camera's only hard negative.
        self._fresh(tmp_path)
        _neg(tmp_path, "event_20260801_100000_mi360-pi", 30)
        _neg(tmp_path, "event_20260801_100000_1_mi360-pi", 30, present=1)
        _neg(tmp_path, "event_20260801_110000_2_cam", 30)
        _neg(tmp_path, "event_20260801_120000_cam", 30)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == []

    def test_malformed_verdicts_never_stop_the_run(self, tmp_path):
        # The claude-cli verifier stores observed_signs exactly as the model
        # returned them; one odd shape inside the window must not abort every
        # nightly prune (review finding, reproduced before the fix).
        self._fresh(tmp_path)
        shapes = [["none"], {"paddling": {"present": True}}, [None], "paddling", 7]
        for i, signs in enumerate(shapes):
            ev = tmp_path / f"event_202609{10 + i}_100000_odd-pi"
            ev.mkdir()
            (ev / "analysis.json").write_text(json.dumps({
                "final_abnormal_event": False, "failed_batches": 0,
                "batches": [{"abnormal_event": False, "observed_signs": signs}, "junk"],
            }), encoding="utf-8")
        weird = tmp_path / "event_20260920_100000_odd-pi"
        weird.mkdir()
        (weird / "analysis.json").write_text("null", encoding="utf-8")
        for hour in ("10", "11", "12"):
            _neg(tmp_path, f"event_20260801_{hour}0000_mi360-pi", 30)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == ["event_20260801_110000_mi360-pi",
                           "event_20260801_120000_mi360-pi"]

    def test_positive_does_not_take_the_negative_slot(self, tmp_path):
        self._fresh(tmp_path)
        _event(tmp_path, "event_20260801_090000_mi360-pi", 30, positive=True)
        _neg(tmp_path, "event_20260801_100000_mi360-pi", 30)
        _neg(tmp_path, "event_20260801_120000_mi360-pi", 30)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == ["event_20260801_120000_mi360-pi"]

    def test_failed_or_unverified_events_are_never_samples(self, tmp_path):
        self._fresh(tmp_path)
        _neg(tmp_path, "event_20260801_100000_mi360-pi", 30, present=3, failed=2)
        _neg(tmp_path, "event_20260801_120000_mi360-pi", 30)
        _event(tmp_path, "event_20260802_100000_mi360-pi", 30, positive=None)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert sorted(removed) == ["event_20260801_100000_mi360-pi",
                                   "event_20260802_100000_mi360-pi"]

    def test_samples_are_stable_across_runs(self, tmp_path):
        self._fresh(tmp_path)
        for hour, present in (("10", 1), ("11", 0), ("12", 1), ("13", 0)):
            _neg(tmp_path, f"event_20260801_{hour}0000_mi360-pi", 30, present=present)
        first, _ = prune_events.prune(tmp_path, keep_days=14)
        second, _ = prune_events.prune(tmp_path, keep_days=14)
        assert len(first) == 3 and second == []
        assert (tmp_path / "event_20260801_100000_mi360-pi").exists()
