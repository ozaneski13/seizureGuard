import json
import os
import time
from datetime import date, timedelta

import prune_events

DAY = 86400


def _event(root, name, age_days, positive=None, analysis=None, meta=False, handled=None):
    """analysis / handled: the analysis.json / handled.json text, or a value
    to dump; meta writes event_meta.json. All written before the mtime is
    set, since writing into the dir would reset it to now."""
    ev = root / name
    ev.mkdir(parents=True)
    (ev / "frame.jpg").write_bytes(b"x" * 1000)
    if positive is not None:
        analysis = {"final_abnormal_event": positive}
    for fname, content in (("analysis.json", analysis), ("handled.json", handled)):
        if content is not None:
            (ev / fname).write_text(
                content if isinstance(content, str) else json.dumps(content),
                encoding="utf-8")
    if meta:
        (ev / "event_meta.json").write_text(
            json.dumps({"t0": 0.0, "start": 1.0, "end": 9.0, "peaks": []}), encoding="utf-8")
    old = time.time() - age_days * DAY
    os.utime(ev, (old, old))
    return ev


def _cam(root, camera, ages, **kw):
    """One event per age in days for one camera, named the way monitor.py
    names them, with the mtime matching the day in the name."""
    names = []
    for age in ages:
        day = date.today() - timedelta(days=age)
        names.append(_event(root, f"event_{day:%Y%m%d}_120000_{camera}", age, **kw).name)
    return names


class TestPrune:
    def test_old_negative_removed_recent_kept(self, tmp_path):
        _event(tmp_path, "event_old_neg", 30, positive=False)
        _event(tmp_path, "event_new_neg", 2, positive=False)
        removed, freed = prune_events.prune(tmp_path, keep_days=14)
        assert removed == ["event_old_neg"]
        assert freed >= 1000              # frame bytes + the analysis.json
        assert not (tmp_path / "event_old_neg").exists()
        assert (tmp_path / "event_new_neg").exists()

    # Each keep-forever test puts a newer event beside the old one, so the
    # old event is outside the window and only the keep rule can save it.
    def test_positive_events_kept_forever(self, tmp_path):
        _event(tmp_path, "event_old_pos", 400, positive=True)
        _event(tmp_path, "event_recent", 1, positive=False)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == []

    def test_unverified_old_event_removed(self, tmp_path):
        # No analysis.json at all: with verify on, this event already got its
        # own "unverified" alert with the clip, so it ages out like a negative.
        _event(tmp_path, "event_old_unverified", 30, positive=None)
        _event(tmp_path, "event_recent", 1, positive=False)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == ["event_old_unverified"]

    def test_unreadable_verdict_is_kept(self, tmp_path):
        _event(tmp_path, "event_old_broken", 30, analysis="{not json")
        _event(tmp_path, "event_recent", 1, positive=False)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == []

    def test_unchecked_events_kept_forever(self, tmp_path):
        """A false verdict with failed batches or a backend outage means the
        event was never checked, not that it was negative. An outage sends no
        per-event alert or clip, so the event dir is the only copy (review
        finding: day 15 of an outage deleted day 1 unseen)."""
        for name, failed, outage in (
                ("event_old_outage", 2, "claude CLI error: 401 token expired"),
                ("event_old_partial", 1, None),
                ("event_old_outage_only", 0, "rate limited"),
                ("event_old_junk_fields", "two", ["not", "a", "string"])):
            _event(tmp_path, name, 30, analysis={
                "final_abnormal_event": False, "failed_batches": failed,
                "backend_outage": outage})
        _event(tmp_path, "event_old_neg", 30, positive=False)
        _event(tmp_path, "event_recent", 1, positive=False)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == ["event_old_neg"]

    def test_interrupted_verification_is_kept_forever(self, tmp_path):
        """verify_event writes analysis.json after every batch; one it never
        finished (timeout, restart, crash) says complete false with only the
        batches it reached. The unasked ones may hold the seizure, so it is
        unchecked even though its alert (UNVERIFIED) was delivered."""
        delivered = {"handled_at": 1.0, "alerted": True, "delivered": True,
                     "text": "UNVERIFIED"}
        _event(tmp_path, "event_old_interrupted", 30, meta=True, handled=delivered,
               analysis={"final_abnormal_event": False, "failed_batches": 0,
                         "backend_outage": None, "complete": False})
        _event(tmp_path, "event_old_finished", 30, meta=True, handled=delivered,
               analysis={"final_abnormal_event": False, "failed_batches": 0,
                         "backend_outage": None, "complete": True})
        _event(tmp_path, "event_recent", 1, positive=False)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == ["event_old_finished"]

    def test_captured_but_never_processed_event_is_kept(self, tmp_path):
        # event_meta.json without handled.json: the monitor died before it
        # verified or alerted, so nobody has seen this event.
        _event(tmp_path, "event_old_unprocessed", 30, meta=True)
        _event(tmp_path, "event_old_processed", 30, meta=True,
               handled={"handled_at": 1.0, "alerted": False, "delivered": None})
        _event(tmp_path, "event_recent", 1, positive=False)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == ["event_old_processed"]

    def test_undelivered_alert_is_kept(self, tmp_path):
        # The alert never reached the owner; the monitor keeps retrying it,
        # and this dir is the only copy of what it would show.
        _event(tmp_path, "event_old_undelivered", 30, meta=True,
               handled={"handled_at": 1.0, "alerted": True, "delivered": False})
        _event(tmp_path, "event_old_delivered", 30, meta=True,
               handled={"handled_at": 1.0, "alerted": True, "delivered": True})
        _event(tmp_path, "event_recent", 1, positive=False)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == ["event_old_delivered"]

    def test_odd_handled_markers_never_stop_the_run(self, tmp_path):
        # Unreadable or odd markers are kept, never guessed as done.
        kept = {"event_old_truncated": "{not json", "event_old_null": "null",
                "event_old_list": [True, False], "event_old_text": '"done"',
                "event_old_no_delivered": {"alerted": True},
                "event_old_odd_values": {"alerted": "yes", "delivered": "no"}}
        for name, handled in kept.items():
            _event(tmp_path, name, 30, handled=handled)
        _event(tmp_path, "event_old_not_alerted", 30,
               handled={"alerted": False, "delivered": False})
        _event(tmp_path, "event_recent", 1, positive=False)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == ["event_old_not_alerted"]

    def test_dry_run_deletes_nothing(self, tmp_path):
        _event(tmp_path, "event_old_neg", 30, positive=False)
        _event(tmp_path, "event_recent", 1, positive=False)
        removed, _ = prune_events.prune(tmp_path, keep_days=14, dry_run=True)
        assert removed == ["event_old_neg"]
        assert (tmp_path / "event_old_neg").exists()

    def test_blind_spell_does_not_empty_the_archive(self, tmp_path):
        """Regression (2026-09): the monitor recorded nothing for 20 days and
        a now-based window deleted every negative from before the gap. Names
        without a camera and day still hang the window off the newest of
        them, so a gap removes nothing."""
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


class TestActiveDayWindow:
    """Retention counts the days a camera actually recorded, per camera: a
    blind camera ages nothing, and its pre-gap archive stays until it has
    recorded keep_days new days. The events here have no verdict, so no
    sample or keep-forever rule can save them: only the window decides."""

    def test_active_camera_keeps_its_last_n_days(self, tmp_path):
        names = _cam(tmp_path, "mi360-pi", range(20))
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert sorted(removed) == sorted(names[14:])

    def test_every_event_of_a_kept_day_is_kept(self, tmp_path):
        _cam(tmp_path, "mi360-pi", range(1, 14))
        # Two events today still count as one day: the 14th, not a 15th.
        _cam(tmp_path, "mi360-pi", [0])
        _event(tmp_path, f"event_{date.today():%Y%m%d}_000100_mi360-pi", 0)
        dropped = _cam(tmp_path, "mi360-pi", [14])
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == dropped

    def test_blind_camera_loses_nothing_while_the_other_records(self, tmp_path):
        # c700 went blind 20 days ago; mi360 kept recording every day.
        c700 = _cam(tmp_path, "c700-pi", range(20, 25))
        mi360 = _cam(tmp_path, "mi360-pi", range(25))
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert sorted(removed) == sorted(mi360[14:])
        assert all((tmp_path / n).exists() for n in c700)

    def test_pre_gap_events_outlive_the_first_event_after_the_gap(self, tmp_path):
        # 10 recorded days, a blind spell, then recording resumes: the pre-gap
        # days go one by one as new days are recorded, not all at once.
        before = _cam(tmp_path, "mi360-pi", range(30, 40))
        _cam(tmp_path, "mi360-pi", [4])
        assert prune_events.prune(tmp_path, keep_days=14)[0] == []
        _cam(tmp_path, "mi360-pi", [1, 2, 3])
        assert prune_events.prune(tmp_path, keep_days=14)[0] == []
        _cam(tmp_path, "mi360-pi", [0])
        assert prune_events.prune(tmp_path, keep_days=14)[0] == [before[-1]]

    def test_day_comes_from_the_name_not_the_dir_mtime(self, tmp_path):
        # verify and make_clip write into the event dir after capture, and so
        # would a later re-check: a dir mtime is not the day it was recorded.
        # Counting it would turn old events into new active days and push
        # untouched pre-gap days out of the window.
        _cam(tmp_path, "mi360-pi", range(14))
        old = f"event_{date.today() - timedelta(days=30):%Y%m%d}_120000_mi360-pi"
        _event(tmp_path, old, 0)              # dir touched today
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == [old]

    def test_ambiguous_camera_names_keep_separate_windows(self, tmp_path):
        # "2_cam" may be a collision name of "cam" or a camera of its own; as
        # in daily_samples it is never merged, so a blind "2_cam" keeps its
        # archive while "cam" records.
        blind = _cam(tmp_path, "2_cam", range(20, 25))
        _cam(tmp_path, "cam", range(25))
        prune_events.prune(tmp_path, keep_days=14)
        assert all((tmp_path / n).exists() for n in blind)

    def test_unnamed_events_ignore_the_cameras_window(self, tmp_path):
        # A hand-made copy has no camera or day in its name. Its window hangs
        # off the newest unnamed event only: a recording camera must not age
        # it out on the calendar.
        _event(tmp_path, "event_vid20260111_seizure_window_fable", 60)
        _cam(tmp_path, "mi360-pi", range(3))
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == []


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
    def _fresh(self, root, *cameras):
        # 14 recent recorded days per camera put the 2026-08 events below
        # outside the window, so only the sample rule can keep them.
        for camera in cameras:
            _cam(root, camera, range(14), positive=False)

    def test_one_negative_per_camera_per_day_is_kept(self, tmp_path):
        self._fresh(tmp_path, "mi360-pi")
        _neg(tmp_path, "event_20260801_100000_mi360-pi", 30)
        _neg(tmp_path, "event_20260801_120000_mi360-pi", 30)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == ["event_20260801_120000_mi360-pi"]   # tie -> earliest kept

    def test_most_sign_like_negative_is_the_sample(self, tmp_path):
        self._fresh(tmp_path, "mi360-pi")
        _neg(tmp_path, "event_20260801_100000_mi360-pi", 30, present=0)
        _neg(tmp_path, "event_20260801_120000_mi360-pi", 30, present=2)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == ["event_20260801_100000_mi360-pi"]

    def test_each_camera_and_day_gets_its_own_sample(self, tmp_path):
        self._fresh(tmp_path, "mi360-pi", "c700-pi")
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
        self._fresh(tmp_path, "mi360-pi", "1_mi360-pi", "2_cam", "cam")
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
        self._fresh(tmp_path, "mi360-pi")
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
        self._fresh(tmp_path, "mi360-pi")
        _event(tmp_path, "event_20260801_090000_mi360-pi", 30, positive=True)
        _neg(tmp_path, "event_20260801_100000_mi360-pi", 30)
        _neg(tmp_path, "event_20260801_120000_mi360-pi", 30)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        assert removed == ["event_20260801_120000_mi360-pi"]

    def test_failed_or_unverified_events_are_never_samples(self, tmp_path):
        self._fresh(tmp_path, "mi360-pi")
        _neg(tmp_path, "event_20260801_100000_mi360-pi", 30, present=3, failed=2)
        _neg(tmp_path, "event_20260801_120000_mi360-pi", 30)
        _event(tmp_path, "event_20260802_100000_mi360-pi", 30, positive=None)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        # The failed event is kept as unchecked, but it did not take the
        # sample slot: the clean 12:00 negative is kept as well.
        assert removed == ["event_20260802_100000_mi360-pi"]
        assert (tmp_path / "event_20260801_120000_mi360-pi").exists()

    def test_interrupted_verification_is_never_a_sample(self, tmp_path):
        self._fresh(tmp_path, "mi360-pi")
        ev = _neg(tmp_path, "event_20260801_100000_mi360-pi", 30, present=3)
        a = json.loads((ev / "analysis.json").read_text(encoding="utf-8"))
        (ev / "analysis.json").write_text(json.dumps({**a, "complete": False}),
                                          encoding="utf-8")
        _neg(tmp_path, "event_20260801_120000_mi360-pi", 30)
        removed, _ = prune_events.prune(tmp_path, keep_days=14)
        # The interrupted event is kept as unchecked, but it did not take the
        # sample slot from the finished negative (no "complete" key: a file
        # from before the key existed is still a verified negative).
        assert removed == []

    def test_samples_are_stable_across_runs(self, tmp_path):
        self._fresh(tmp_path, "mi360-pi")
        for hour, present in (("10", 1), ("11", 0), ("12", 1), ("13", 0)):
            _neg(tmp_path, f"event_20260801_{hour}0000_mi360-pi", 30, present=present)
        first, _ = prune_events.prune(tmp_path, keep_days=14)
        second, _ = prune_events.prune(tmp_path, keep_days=14)
        assert len(first) == 3 and second == []
        assert (tmp_path / "event_20260801_100000_mi360-pi").exists()
