import json
import os
import re
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

import monitor
from monitor import MotionTrigger, RingBuffer, StreamWatchdog


class TestStreamWatchdog:
    def test_transient_failure_takes_no_action(self):
        wd = StreamWatchdog(retry_sec=3.0, alert_sec=60.0)
        assert wd.failed(100.0) == []
        assert wd.failed(101.0) == []

    def test_reconnect_after_retry_window(self):
        wd = StreamWatchdog(retry_sec=3.0, alert_sec=60.0)
        wd.failed(100.0)
        assert wd.failed(103.5) == ["reconnect"]

    def test_reconnect_repeats_per_interval_not_per_read(self):
        wd = StreamWatchdog(retry_sec=3.0, alert_sec=60.0)
        wd.failed(100.0)
        assert wd.failed(103.5) == ["reconnect"]
        assert wd.failed(103.9) == []          # just retried; wait
        assert wd.failed(107.0) == ["reconnect"]

    def test_blind_alert_fires_once_per_reminder_interval(self):
        wd = StreamWatchdog(retry_sec=3.0, alert_sec=60.0, remind_sec=3600.0)
        wd.failed(100.0)
        assert "blind_alert" in wd.failed(161.0)
        assert "blind_alert" not in wd.failed(300.0)

    def test_blind_alert_repeats_while_the_stream_stays_dead(self):
        """Regression (2026-09): one alert on day one of a 20-day blind
        spell was all the owner got."""
        wd = StreamWatchdog(retry_sec=3.0, alert_sec=60.0, remind_sec=3600.0)
        wd.failed(100.0)
        wd.failed(161.0)
        assert "blind_alert" not in wd.failed(3700.0)
        assert "blind_alert" in wd.failed(3761.0)
        assert "blind_alert" in wd.failed(7361.0)

    def test_recovery_after_alert_reports_how_long_it_was_down(self):
        wd = StreamWatchdog(retry_sec=3.0, alert_sec=60.0)
        wd.failed(100.0)
        wd.failed(161.0)
        assert wd.ok(400.0) == 300.0

    def test_recovery_before_alert_is_silent(self):
        wd = StreamWatchdog(retry_sec=3.0, alert_sec=60.0)
        wd.failed(100.0)
        assert wd.ok(130.0) is None
        assert wd.ok(131.0) is None            # healthy reads stay silent

    def test_recovery_resets_everything(self):
        wd = StreamWatchdog(retry_sec=3.0, alert_sec=60.0)
        wd.failed(100.0)
        wd.failed(161.0)
        wd.ok(170.0)
        assert wd.failed(200.0) == []          # fresh stall, fresh timers
        assert wd.failed(204.0) == ["reconnect"]

    def test_second_stall_after_recovery_alerts_again(self):
        """A recovery must clear the reminder slot: otherwise a stream that
        dies again within 6 h stays silently blind until the first alert's
        reminder is due."""
        wd = StreamWatchdog(retry_sec=3.0, alert_sec=60.0)
        wd.failed(100.0)
        assert "blind_alert" in wd.failed(161.0)
        wd.ok(170.0)
        wd.failed(200.0)
        assert "blind_alert" in wd.failed(261.0)

    def test_undelivered_alert_is_retried_soon_not_in_six_hours(self):
        """Regression: a blind alert lost to a WAN hiccup used up the 6 h
        reminder slot, so the owner heard nothing for 6 h."""
        wd = StreamWatchdog(retry_sec=3.0, alert_sec=60.0, remind_sec=3600.0,
                            undelivered_retry_sec=60.0)
        wd.failed(100.0)
        assert "blind_alert" in wd.failed(161.0)
        wd.alert_undelivered(161.0)
        assert "blind_alert" not in wd.failed(200.0)
        assert "blind_alert" in wd.failed(221.0)     # delivered this time
        assert "blind_alert" not in wd.failed(3800.0)
        assert "blind_alert" in wd.failed(3821.0)    # back on the reminder cadence

    def test_recovery_is_announced_even_if_no_blind_alert_got_through(self):
        wd = StreamWatchdog(retry_sec=3.0, alert_sec=60.0)
        wd.failed(100.0)
        wd.failed(161.0)
        wd.alert_undelivered(161.0)
        assert wd.ok(400.0) == 300.0


class TestFormatDuration:
    def test_units(self):
        assert monitor.format_duration(30) == "1 min"
        assert monitor.format_duration(61) == "1 min"
        assert monitor.format_duration(45 * 60) == "45 min"
        assert monitor.format_duration(6 * 3600) == "6.0 h"
        assert monitor.format_duration(20 * 86400) == "20.0 days"


class TestOpenLiveAlerts:
    def test_cannot_open_repeats_then_announces_recovery(self, monkeypatch):
        clock = [1000.0]
        sent = []
        attempts = []

        def fake_open(source):
            attempts.append(clock[0])
            if clock[0] < 1000.0 + 7 * 3600:
                raise RuntimeError("Could not open source")
            return ("cap", False, None)

        monkeypatch.setattr(monitor, "open_capture", fake_open)
        monkeypatch.setattr(monitor.time, "time", lambda: clock[0])
        monkeypatch.setattr(monitor.time, "sleep",
                            lambda s: clock.__setitem__(0, clock[0] + 600))
        monkeypatch.setattr(monitor.alerts, "send_alert", sent.append)

        assert monitor._open_live("rtsp://x/mi360") == ("cap", False, None)
        blind = [m for m in sent if m.startswith("Monitor cannot open")]
        assert len(blind) == 2                 # at ~10 min and ~6 h 10 min
        assert sent[-1].startswith("Monitor recovered: rtsp://x/mi360 opened after 7.0 h")

    def test_undelivered_cannot_open_alert_is_retried(self, monkeypatch):
        clock = [1000.0]
        sent = []

        def fake_open(source):
            if clock[0] < 1000.0 + 2 * 3600:
                raise RuntimeError("Could not open source")
            return ("cap", False, None)

        def fake_send(text):
            delivered = clock[0] >= 1300.0      # WAN back after 5 min
            sent.append((clock[0], delivered))
            return delivered

        monkeypatch.setattr(monitor, "open_capture", fake_open)
        monkeypatch.setattr(monitor, "time", types.SimpleNamespace(
            time=lambda: clock[0], monotonic=lambda: clock[0],
            sleep=lambda s: clock.__setitem__(0, clock[0] + 30)))
        monkeypatch.setattr(monitor.alerts, "send_alert", fake_send)
        monkeypatch.setattr(monitor.alerts, "telegram_configured", lambda: True)

        monitor._open_live("rtsp://x/mi360")
        cannot_open = sent[:-1]                 # the last one is the recovery
        assert [d for _, d in cannot_open] == [False] * (len(cannot_open) - 1) + [True]
        assert cannot_open[-1][0] <= 1300.0 + monitor.STREAM_BLIND_RETRY_SEC


class _FakeCap:
    def __init__(self, *args):
        self.args = args
        self.props = {}

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.props[prop] = value


class TestOpenCaptureSources:
    def test_url_source_forces_ffmpeg_with_timeouts(self, monkeypatch):
        """Regression (2026-09-04): with no backend pinned, a 404 from the
        restreamer made OpenCV fall back to GStreamer, which deadlocked in
        native code and left the monitor blind for 20 days."""
        created = []
        monkeypatch.setattr(monitor.cv2, "VideoCapture",
                            lambda *a: created.append(_FakeCap(*a)) or created[-1])
        cap, file_mode, fps = monitor.open_capture("rtsp://localhost:8554/mi360")
        url, backend, params = created[0].args
        assert url == "rtsp://localhost:8554/mi360"
        assert backend == monitor.cv2.CAP_FFMPEG
        settings = dict(zip(params[::2], params[1::2]))
        assert settings[monitor.cv2.CAP_PROP_OPEN_TIMEOUT_MSEC] > 0
        assert settings[monitor.cv2.CAP_PROP_READ_TIMEOUT_MSEC] > 0
        assert file_mode is False
        assert fps is None

    def test_camera_index_uses_directshow(self, monkeypatch):
        created = []
        monkeypatch.setattr(monitor.cv2, "VideoCapture",
                            lambda *a: created.append(_FakeCap(*a)) or created[-1])
        cap, file_mode, fps = monitor.open_capture("1")
        assert created[0].args == (1, monitor.cv2.CAP_DSHOW)
        assert file_mode is False


def _frame_time(path):
    return float(re.search(r"_t_([0-9.]+)s", path.name).group(1))


def _feed_span(trigger, t0, t1, score, is_global=False, step=1 / 30):
    """Feed constant-score frames over [t0, t1); return first completed event."""
    t = t0
    while t < t1:
        ev = trigger.feed(t, score, is_global)
        if ev is not None:
            return ev
        t += step
    return None


QUIET = monitor.MOTION_OFF - 1.0
MOVING = monitor.MOTION_ON + 4.0


class TestMotionTrigger:
    def test_sustained_motion_then_quiet_produces_event(self):
        tr = MotionTrigger()
        assert _feed_span(tr, 0.0, 5.0, QUIET) is None
        assert _feed_span(tr, 5.0, 12.0, MOVING) is None
        assert tr.state == "active"
        ev = _feed_span(tr, 12.0, 30.0, QUIET)
        assert ev is not None
        start, end = ev
        assert start == pytest.approx(5.0, abs=0.1)
        assert end == pytest.approx(12.0, abs=0.2)

    def test_brief_spike_does_not_trigger(self):
        tr = MotionTrigger()
        assert _feed_span(tr, 0.0, 1.0, MOVING) is None
        assert _feed_span(tr, 1.0, 30.0, QUIET) is None
        assert tr.state == "idle"

    def test_event_max_caps_duration(self):
        tr = MotionTrigger()
        ev = _feed_span(tr, 0.0, 100.0, MOVING)
        assert ev is not None
        start, end = ev
        assert end - start == pytest.approx(monitor.EVENT_MAX_SEC, abs=0.1)

    def test_global_change_never_triggers(self):
        tr = MotionTrigger()
        assert _feed_span(tr, 0.0, 30.0, 250.0, is_global=True) is None
        assert tr.state == "idle"

    def test_flush_finalizes_inflight_event(self):
        tr = MotionTrigger()
        _feed_span(tr, 0.0, 10.0, MOVING)
        assert tr.state == "active"
        ev = tr.flush()
        assert ev is not None
        assert ev[0] == pytest.approx(0.0, abs=0.1)
        assert tr.state == "idle"

    def test_flush_when_idle_is_none(self):
        assert MotionTrigger().flush() is None


class TestRingBuffer:
    def test_trims_old_entries(self):
        ring = RingBuffer(seconds=10.0)
        for t in range(0, 30):
            ring.append(float(t), b"x")
        times = [t for t, _ in ring.items]
        assert min(times) >= 29.0 - 10.0
        assert max(times) == 29.0

    def test_snapshot_bounds(self):
        ring = RingBuffer(seconds=100.0)
        for t in range(0, 50):
            ring.append(float(t), b"x")
        fb = ring.snapshot(10.0, 20.0)
        assert fb.times[0] == 10.0
        assert fb.times[-1] == 20.0


class TestIntegration:
    @pytest.fixture
    def alert_calls(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            monitor.alerts, "send_alert",
            lambda text, photo_path=None, video_path=None:
                calls.append((text, photo_path, video_path)))
        return calls

    @pytest.fixture
    def events(self, synthetic_video, tmp_path, monkeypatch, alert_calls):
        monkeypatch.setenv("SEIZUREGUARD_VERIFY", "0")
        monkeypatch.delenv("SEIZUREGUARD_POSE_PYTHON", raising=False)
        real_process = monitor.process_event

        def slow_process(*args, **kwargs):
            time.sleep(0.3)             # the worker lags the read loop, as in production
            return real_process(*args, **kwargs)

        monkeypatch.setattr(monitor, "process_event", slow_process)
        return monitor.run(str(synthetic_video), out_root=tmp_path / "events")

    def test_file_run_returns_only_after_every_event_is_handled(self, events):
        """An offline run must not return while the worker still holds
        events: their alerts would be lost with the process."""
        assert all((e / monitor.HANDLED_MARKER).exists() for e in events)

    def test_violent_segment_produces_event(self, events):
        assert len(events) >= 1
        first = events[0]
        base_times = [_frame_time(p) for p in (first / "base").glob("frame_*.jpg")]
        assert base_times, "event has no base frames"
        # violent motion starts at 20s; with pre-roll the window starts ~15s
        span = max(base_times)
        assert span >= 5.0, "event window suspiciously short"

    def test_flash_alone_never_creates_event(self, events, synthetic_video):
        # no event should *start* around the 40-41s flash; starts come from
        # the violent (20s) and moderate (50s) segments only
        # event dirs are timestamped by wall clock, so measure via alert count
        assert 1 <= len(events) <= 2

    def test_unverified_alert_sent_per_event(self, events, alert_calls):
        assert len(alert_calls) == len(events)
        text, photo, video = alert_calls[0]
        assert "unverified" in text
        assert photo is not None
        assert video is not None          # clip built around the first peak

    def test_event_dir_structure(self, events):
        first = events[0]
        assert (first / "base").is_dir()
        assert (first / "burst").is_dir()
        assert list((first / "burst").glob("frame_*.jpg"))
        # the full-rate video alert clips are cut from, written by the worker
        assert (first / "event.mp4").exists()


class TestAlertTextFor:
    """Regression: a verify run whose batches all failed was treated as
    'negative' and silenced the alert. 383 production events went through
    that path unnoticed."""

    def _verdict(self, positive=False, failed=0, batches=8, conf=0.0):
        return {
            "final_abnormal_event": positive,
            "final_confidence": conf,
            "failed_batches": failed,
            "batches": [{}] * batches,
        }

    def test_analyzed_negative_stays_silent(self):
        assert monitor.alert_text_for(self._verdict(), "ev") is None

    def test_positive_alerts_with_confidence(self):
        text = monitor.alert_text_for(self._verdict(positive=True, conf=0.65), "ev")
        assert "0.65" in text and "ev" in text

    def test_all_batches_failed_alerts_unverified(self):
        text = monitor.alert_text_for(self._verdict(failed=8, batches=8), "ev")
        assert text is not None
        assert "UNVERIFIED" in text and "8/8" in text

    def test_partial_failure_also_alerts(self):
        text = monitor.alert_text_for(self._verdict(failed=1, batches=8), "ev")
        assert text is not None and "1/8" in text

    def test_positive_wins_over_failures(self):
        text = monitor.alert_text_for(
            self._verdict(positive=True, failed=3, conf=0.4), "ev")
        assert "Abnormal" in text and "UNVERIFIED" not in text

    def test_no_verdict_is_unverified(self):
        assert "unverified" in monitor.alert_text_for(None, "ev")


class TestVerifyProbe:
    def _proc(self, stdout="", returncode=0, stderr=""):
        import types
        return types.SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)

    def test_working_backend_probes_ok(self, monkeypatch):
        monkeypatch.setattr(monitor.shutil, "which", lambda n: "claude")
        monkeypatch.setattr(monitor.subprocess, "run",
                            lambda *a, **k: self._proc("OK"))
        assert monitor.verify_probe() == (True, None)

    def test_expired_token_is_caught(self, monkeypatch):
        monkeypatch.setattr(monitor.shutil, "which", lambda n: "claude")
        monkeypatch.setattr(monitor.subprocess, "run", lambda *a, **k: self._proc(
            "", 1, "API Error: 401 OAuth access token has expired"))
        ok, reason = monitor.verify_probe()
        assert ok is False and "401" in reason

    def test_missing_cli_is_caught(self, monkeypatch):
        monkeypatch.setattr(monitor.shutil, "which", lambda n: None)
        ok, reason = monitor.verify_probe()
        assert ok is False and "not found" in reason

    def test_crash_never_raises(self, monkeypatch):
        monkeypatch.setattr(monitor.shutil, "which", lambda n: "claude")

        def boom(*a, **k):
            raise OSError("no process")

        monkeypatch.setattr(monitor.subprocess, "run", boom)
        assert monitor.verify_probe()[0] is False


class TestOutageNotifier:
    """A backend outage repeats identically on every event. Alerting per
    event trains the owner to ignore the one channel that must not be
    ignored, so the outage itself is announced instead."""

    def test_first_outage_notifies(self):
        n = monitor.OutageNotifier(interval=3600)
        notify, skipped = n.should_notify(1000.0)
        assert notify is True and skipped == 0

    def test_repeat_events_are_suppressed(self):
        n = monitor.OutageNotifier(interval=3600)
        n.should_notify(1000.0)
        for i in range(1, 6):
            notify, skipped = n.should_notify(1000.0 + i)
            assert notify is False
            assert skipped == i

    def test_reminder_after_interval_reports_backlog(self):
        n = monitor.OutageNotifier(interval=3600)
        n.should_notify(1000.0)
        for i in range(1, 4):
            n.should_notify(1000.0 + i)
        notify, skipped = n.should_notify(1000.0 + 3601)
        assert notify is True
        assert skipped == 3          # events silently unchecked meanwhile

    def test_recovery_is_reported_once(self):
        n = monitor.OutageNotifier()
        n.should_notify(1000.0)
        assert n.recovered() is True
        assert n.recovered() is False    # already announced

    def test_no_recovery_message_without_outage(self):
        assert monitor.OutageNotifier().recovered() is False

    def test_undelivered_notice_does_not_use_up_the_slot(self):
        n = monitor.OutageNotifier(interval=3600)
        assert n.should_notify(1000.0) == (True, 0)
        n.undelivered()
        # next event tries again; the event whose notice was lost is unchecked too
        assert n.should_notify(1001.0) == (True, 1)
        assert n.should_notify(1002.0) == (False, 1)

    def test_undelivered_reminder_keeps_the_backlog(self):
        n = monitor.OutageNotifier(interval=3600)
        n.should_notify(1000.0)
        n.should_notify(1001.0)
        n.should_notify(1002.0)
        assert n.should_notify(4700.0) == (True, 2)
        n.undelivered()
        assert n.should_notify(4701.0) == (True, 3)

    def test_undelivered_recovery_is_retried(self):
        n = monitor.OutageNotifier()
        n.should_notify(1000.0)
        assert n.recovered() is True
        n.undelivered()
        assert n.recovered() is True
        assert n.recovered() is False

    def test_recovery_after_a_lost_first_notice_is_still_announced(self):
        """The event behind a lost notice went unchecked; recovering without
        a word would leave the owner never hearing about it."""
        n = monitor.OutageNotifier()
        n.should_notify(1000.0)
        n.undelivered()
        assert n.recovered() is True


class TestOutageIsNotAPerEventAlert:
    def test_outage_verdict_produces_no_event_alert(self):
        verdict = {"final_abnormal_event": False, "failed_batches": 7,
                   "batches": [{}] * 7, "backend_outage": "You've hit your limit"}
        assert monitor.alert_text_for(verdict, "ev") is None

    def test_plain_failure_still_alerts_unverified(self):
        verdict = {"final_abnormal_event": False, "failed_batches": 7,
                   "batches": [{}] * 7, "backend_outage": None}
        assert "UNVERIFIED" in monitor.alert_text_for(verdict, "ev")

    def test_positive_event_alerts_even_during_an_outage(self):
        verdict = {"final_abnormal_event": True, "final_confidence": 0.6,
                   "positive_batches": 2, "failed_batches": 3,
                   "batches": [{}] * 7, "backend_outage": "limit"}
        assert "Abnormal" in monitor.alert_text_for(verdict, "ev")



class TestSystemdWatchdog:
    """The monitor once wedged in native code for 20 days while systemd said
    "active"; only an external watchdog can catch that, so pinging must be
    reliable when enabled and inert when not."""

    def _wd(self, monkeypatch, addr="/run/systemd/notify", interval=10.0):
        if addr is None:
            monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
        else:
            monkeypatch.setenv("NOTIFY_SOCKET", addr)
        wd = monitor.SystemdWatchdog(interval=interval)
        sent = []
        monkeypatch.setattr(wd, "_send", lambda msg: sent.append(msg))
        return wd, sent

    def test_noop_without_notify_socket(self, monkeypatch):
        wd, sent = self._wd(monkeypatch, addr=None)
        assert wd.ping(100.0) is False
        assert sent == []

    def test_first_ping_sends_watchdog_message(self, monkeypatch):
        wd, sent = self._wd(monkeypatch)
        assert wd.ping(100.0) is True
        assert sent == [b"WATCHDOG=1"]

    def test_pings_are_rate_limited(self, monkeypatch):
        wd, sent = self._wd(monkeypatch, interval=10.0)
        wd.ping(100.0)
        assert wd.ping(105.0) is False
        assert wd.ping(110.5) is True
        assert len(sent) == 2

    def test_abstract_socket_address(self, monkeypatch):
        wd, _ = self._wd(monkeypatch, addr="@/org/freedesktop/systemd1/notify")
        assert wd._address() == chr(0) + "/org/freedesktop/systemd1/notify"


# ------------------------------------------------------------ event pipeline

def _make_event(root, name="event_20260925_010000_mi360", n_frames=90, meta=True):
    """An event dir as capture_event leaves it: a frame every 0.5 s (their
    content does not matter here) and event_meta.json."""
    d = root / name
    (d / "base").mkdir(parents=True)
    (d / "burst").mkdir()
    for i in range(n_frames):
        (d / "base" / f"frame_{i:03d}_t_{i * 0.5:0.3f}s.jpg").write_bytes(b"")
    if meta:
        (d / monitor.EVENT_META).write_text(json.dumps(
            {"t0": 1000.0, "start": 1005.0, "end": 1044.5, "peaks": [1020.0]}))
    return d


class TestClipWindow:
    """Regression: an UNVERIFIED alert for a partly failed event came with a
    single still frame, although it is the one the owner must judge by eye."""

    frames = [Path(f"frame_{i:03d}_t_{i * 0.5:0.3f}s.jpg") for i in range(90)]

    def _verdict(self, *states):
        return {"batches": [{"abnormal_event": s, "confidence": 0.7} for s in states]}

    def test_positive_batch_wins(self):
        w = monitor.clip_window(self._verdict(False, None, True), self.frames,
                                [1010.0], 1000.0)
        assert w == monitor.alert_clip.clamp_window(30.0, 44.5)

    def test_failed_batch_when_nothing_was_positive(self):
        w = monitor.clip_window(self._verdict(False, None, False), self.frames,
                                [1010.0], 1000.0)
        assert w == monitor.alert_clip.clamp_window(15.0, 29.5)

    def test_peak_window_when_the_failed_batch_has_no_frames(self):
        w = monitor.clip_window(self._verdict(False, False, False, None), self.frames,
                                [1010.0], 1000.0)
        assert w == monitor.alert_clip.peak_window([1010.0], 1000.0)

    def test_peak_window_without_a_verdict(self):
        assert monitor.clip_window(None, self.frames, [1010.0], 1000.0) == \
            monitor.alert_clip.peak_window([1010.0], 1000.0)


class TestProcessEvent:
    @pytest.fixture
    def env(self, monkeypatch):
        monkeypatch.delenv("SEIZUREGUARD_POSE_PYTHON", raising=False)
        state = types.SimpleNamespace(sent=[], windows=[], verdict=None, delivered=True)

        def fake_send(text, photo_path=None, video_path=None):
            state.sent.append(text)
            return state.delivered

        monkeypatch.setattr(monitor.alerts, "send_alert", fake_send)
        monkeypatch.setattr(monitor.alerts, "telegram_configured", lambda: True)
        monkeypatch.setattr(monitor, "run_verify", lambda d: state.verdict)
        monkeypatch.setattr(monitor.alert_clip, "make_clip",
                            lambda d, w: state.windows.append(w))
        return state

    def _handled(self, event_dir):
        return json.loads((event_dir / monitor.HANDLED_MARKER).read_text())

    def test_failed_batch_alert_comes_with_a_clip(self, env, tmp_path):
        env.verdict = {"final_abnormal_event": False, "failed_batches": 1,
                       "batches": [{"abnormal_event": s} for s in (False, None, False)]}
        ev = _make_event(tmp_path)
        monitor.process_event(ev, True, monitor.OutageNotifier())
        assert env.sent[0].startswith("UNVERIFIED motion event")
        assert env.windows == [monitor.alert_clip.clamp_window(15.0, 29.5)]
        assert self._handled(ev)["alerted"] is True

    def test_negative_event_is_marked_handled_without_alert(self, env, tmp_path):
        env.verdict = {"final_abnormal_event": False, "failed_batches": 0,
                       "batches": [{"abnormal_event": False}]}
        ev = _make_event(tmp_path)
        monitor.process_event(ev, True, monitor.OutageNotifier())
        assert env.sent == []
        assert self._handled(ev)["alerted"] is False

    def test_undelivered_alert_is_recorded(self, env, tmp_path):
        env.delivered = False
        ev = _make_event(tmp_path)
        monitor.process_event(ev, False, monitor.OutageNotifier())
        handled = self._handled(ev)
        assert handled["alerted"] is True and handled["delivered"] is False

    def test_event_without_meta_file_still_alerts(self, env, tmp_path):
        ev = _make_event(tmp_path, meta=False)
        monitor.process_event(ev, False, monitor.OutageNotifier())
        assert env.sent == [f"Motion event captured (unverified) - {ev.name}"]
        assert self._handled(ev)["alerted"] is True

    def test_undelivered_outage_notice_is_retried_on_the_next_event(self, env, tmp_path):
        env.verdict = {"final_abnormal_event": False, "failed_batches": 1,
                       "batches": [{}], "backend_outage": "You've hit your limit"}
        notifier = monitor.OutageNotifier()
        env.delivered = False
        monitor.process_event(_make_event(tmp_path, "event_a"), True, notifier)
        env.delivered = True
        monitor.process_event(_make_event(tmp_path, "event_b"), True, notifier)
        monitor.process_event(_make_event(tmp_path, "event_c"), True, notifier)
        notices = [m for m in env.sent if "verification unavailable" in m]
        assert len(notices) == 2
        assert "(1 further events since the last notice)" in notices[1]

    def test_undelivered_recovery_message_is_retried(self, env, tmp_path):
        notifier = monitor.OutageNotifier()
        notifier.should_notify(1000.0)
        env.verdict = {"final_abnormal_event": False, "failed_batches": 0,
                       "batches": [{"abnormal_event": False}]}
        env.delivered = False
        monitor.process_event(_make_event(tmp_path, "event_a"), True, notifier)
        env.delivered = True
        monitor.process_event(_make_event(tmp_path, "event_b"), True, notifier)
        monitor.process_event(_make_event(tmp_path, "event_c"), True, notifier)
        assert env.sent.count("seizureGuard: AI verification is back online.") == 2

    def test_lost_outage_notice_then_healthy_event_still_says_something(self, env, tmp_path):
        env.verdict = {"final_abnormal_event": False, "failed_batches": 1,
                       "batches": [{}], "backend_outage": "529 overloaded"}
        notifier = monitor.OutageNotifier()
        env.delivered = False
        monitor.process_event(_make_event(tmp_path, "event_a"), True, notifier)
        env.verdict = {"final_abnormal_event": False, "failed_batches": 0,
                       "batches": [{"abnormal_event": False}]}
        env.delivered = True
        monitor.process_event(_make_event(tmp_path, "event_b"), True, notifier)
        assert env.sent[-1] == "seizureGuard: AI verification is back online."

    def test_event_stays_unhandled_until_processing_finishes(self, env, tmp_path, monkeypatch):
        """The marker is written last: a restart mid-verify must leave the
        event for the restart sweep to find."""
        ev = _make_event(tmp_path)

        class Restart(Exception):
            pass

        def verify(event_dir):
            assert not (event_dir / monitor.HANDLED_MARKER).exists()
            assert monitor.unhandled_events(tmp_path, "mi360", time.time()) == [ev]
            raise Restart

        monkeypatch.setattr(monitor, "run_verify", verify)
        with pytest.raises(Restart):
            monitor.process_event(ev, True, monitor.OutageNotifier())
        assert monitor.unhandled_events(tmp_path, "mi360", time.time()) == [ev]


class TestEventWorker:
    def test_stuck_only_past_the_deadline(self):
        w = monitor.EventWorker(lambda d, fb: None, deadline=100.0)
        try:
            assert w.stuck(5000.0) is False          # idle is never stuck
            w.busy_since = 1000.0
            assert w.stuck(1100.0) is False
            assert w.stuck(1100.5) is True
        finally:
            w.busy_since = None
            w.close()

    def test_deadline_is_above_every_bounded_wait(self):
        assert monitor.HANDLE_DEADLINE_SEC > 600 + 3600     # pose gate + verify

    def test_busy_only_while_processing(self):
        release, seen = threading.Event(), []

        def process(event_dir, fb):
            seen.append(w.busy_since)
            release.wait(5)

        w = monitor.EventWorker(process)
        w.submit("ev")
        for _ in range(500):
            if seen:
                break
            time.sleep(0.01)
        assert seen and seen[0] is not None
        release.set()
        w.close()
        assert w.busy_since is None

    def test_backlog_alert_once_per_episode(self):
        gate = [threading.Event()]
        w = monitor.EventWorker(lambda d, fb: gate[0].wait(5), backlog_alert=10)
        assert sum(w.submit(f"a{i}") for i in range(15)) == 1
        gate[0].set()
        w.queue.join()
        gate[0] = threading.Event()
        assert sum(w.submit(f"b{i}") for i in range(15)) == 1   # a new episode
        gate[0].set()
        w.close()

    def test_crashing_event_still_alerts_and_the_worker_goes_on(self, monkeypatch):
        sent, done = [], []
        monkeypatch.setattr(monitor.alerts, "send_alert", lambda text, **kw: sent.append(text))

        def process(event_dir, fb):
            if event_dir.name == "event_bad":
                raise OSError("disk full")
            done.append(event_dir)

        w = monitor.EventWorker(process)
        w.submit(Path("event_bad"))
        w.submit(Path("event_good"))
        w.close()
        assert done == [Path("event_good")]
        assert any(m.startswith("UNVERIFIED") and "event_bad" in m for m in sent)


class _Stop(Exception):
    """Ends a live run(); its loop has no other exit."""


class TestRestartSweep:
    def test_finds_only_this_monitors_recent_unhandled_events(self, tmp_path):
        now = time.time()
        own = _make_event(tmp_path, "event_20260925_010000_mi360", n_frames=1)
        again = _make_event(tmp_path, "event_20260925_010000_1_mi360", n_frames=1)
        handled = _make_event(tmp_path, "event_20260925_010100_mi360", n_frames=1)
        (handled / monitor.HANDLED_MARKER).write_text("{}")
        _make_event(tmp_path, "event_20260925_010200_c700", n_frames=1)
        _make_event(tmp_path, "event_20260925_010300_bigmi360", n_frames=1)
        old = _make_event(tmp_path, "event_20260924_010000_mi360", n_frames=1)
        os.utime(old, (now - 7 * 3600, now - 7 * 3600))
        (tmp_path / "event_20260925_010400_mi360").write_text("not a dir")
        assert monitor.unhandled_events(tmp_path, "mi360", now) == [own, again]

    def test_missing_root_is_empty(self, tmp_path):
        assert monitor.unhandled_events(tmp_path / "nope", "mi360", time.time()) == []

    def test_a_freshly_captured_event_is_found_by_the_sweep(self, tmp_path):
        ring = RingBuffer()
        jpg = monitor.encode_jpg(QUIET_FRAME)
        for i in range(300):
            ring.append(1000.0 + i / 15, jpg)
        history = [(1000.0 + i / 15, 3.0) for i in range(300)]
        event_dir, _ = monitor.capture_event(ring, history, 1008.0, 1015.0,
                                             tmp_path, "mi360")
        assert monitor.unhandled_events(tmp_path, "mi360", time.time()) == [event_dir]

    def _run_live(self, monkeypatch, root, verify):
        processed, done = [], threading.Event()

        def fake_process(event_dir, use_verify, outage_notifier, fb=None):
            processed.append((event_dir, fb))
            done.set()

        class Cap:
            def read(self):
                done.wait(5 if verify else 0.3)
                raise _Stop

            def release(self):
                pass

        monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
        monkeypatch.setattr(monitor, "verify_enabled", lambda: verify)
        monkeypatch.setattr(monitor, "verify_probe", lambda: (True, None))
        monkeypatch.setattr(monitor, "process_event", fake_process)
        monkeypatch.setattr(monitor, "_open_live", lambda source, *a: (Cap(), False, None))
        with pytest.raises(_Stop):
            monitor.run("rtsp://x/mi360", out_root=root, name="mi360")
        return processed

    def test_restart_requeues_an_event_left_mid_verify(self, tmp_path, monkeypatch):
        """A restart (deploy, token setup, watchdog kill) mid-verify used to
        drop the event: no verdict, no alert."""
        root = tmp_path / "events"
        own = _make_event(root, "event_20260925_010000_mi360", n_frames=1)
        _make_event(root, "event_20260925_010000_c700", n_frames=1)
        assert self._run_live(monkeypatch, root, verify=True) == [(own, None)]

    def test_no_sweep_with_verify_off(self, tmp_path, monkeypatch):
        root = tmp_path / "events"
        _make_event(root, "event_20260925_010000_mi360", n_frames=1)
        assert self._run_live(monkeypatch, root, verify=False) == []


QUIET_FRAME = np.full((360, 640, 3), 40, np.uint8)


def _motion_frame(i):
    """A 100 px square jumping between two spots: local motion well above
    MOTION_ON, far below the global-change cutoff."""
    frame = QUIET_FRAME.copy()
    x = 100 if i % 2 else 400
    frame[100:200, x:x + 100] = 255
    return frame


@pytest.fixture
def live(monkeypatch, tmp_path):
    """run() on a live source under a fake clock; each sleep() in the loop
    (one per failed read or open) advances it 60 s. The fake monotonic clock
    is offset from the fake wall clock, as in production, so mixing the two
    up breaks the tests."""
    clock = [1000.0]
    monkeypatch.setattr(monitor, "time", types.SimpleNamespace(
        time=lambda: clock[0], monotonic=lambda: clock[0] - 1e6,
        sleep=lambda s: clock.__setitem__(0, clock[0] + 60),
        strftime=time.strftime))
    for var in ("NOTIFY_SOCKET", "SEIZUREGUARD_MOVING_FLAG", "SEIZUREGUARD_POSE_PYTHON"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SEIZUREGUARD_VERIFY", "0")
    state = types.SimpleNamespace(clock=clock, sent=[], deliver=lambda: True,
                                  root=tmp_path / "events")

    def fake_send(text, photo_path=None, video_path=None):
        delivered = state.deliver()
        state.sent.append((clock[0], text, delivered))
        return delivered

    monkeypatch.setattr(monitor.alerts, "send_alert", fake_send)
    monkeypatch.setattr(monitor.alerts, "telegram_configured", lambda: True)

    def start(cap=None):
        """Runs until _Stop; with no cap, open_capture stays the test's own."""
        if cap is not None:
            monkeypatch.setattr(monitor, "_open_live", lambda source, *a: (cap, False, None))
            monkeypatch.setattr(monitor, "open_capture", lambda source: (cap, False, None))
        with pytest.raises(_Stop):
            monitor.run("rtsp://x/mi360", out_root=state.root, name="mi360")

    def verify_on(process):
        monkeypatch.setattr(monitor, "verify_enabled", lambda: True)
        monkeypatch.setattr(monitor, "verify_probe", lambda: (True, None))
        monkeypatch.setattr(monitor, "process_event", process)

    state.verify_on = verify_on

    state.start = start
    return state


class _DeadStreamCap:
    """One good frame, then a stream dead until `back_at`, then good frames;
    stops the run after three of them."""

    def __init__(self, clock, back_at):
        self.clock, self.back_at = clock, back_at
        self.reads = self.good_after = 0

    def read(self):
        self.reads += 1
        if self.reads > 1:
            if self.clock[0] < self.back_at:
                return False, None
            self.good_after += 1
            if self.good_after > 3:
                raise _Stop
        return True, QUIET_FRAME.copy()

    def release(self):
        pass


class _EventCap:
    """Live source scripted around the event worker: 5 s of motion, then
    quiet until event 1 is captured; once event 1 is being processed the
    stream dies until the blind alert goes out, then 5 s of motion and quiet
    again (event 2). Each good read is 1/15 s of fake time."""

    def __init__(self, live, processing, done):
        self.live, self.processing, self.done = live, processing, done
        self.i = 0
        self.motion_until = live.clock[0] + 5.0
        self.second = False

    def read(self):
        clock = self.live.clock
        if self.done() or clock[0] > 1000.0 + 3600:
            raise _Stop
        if self.processing.is_set() and not self.second:
            if not any(text.startswith("Monitor blind") for _, text, _ in self.live.sent):
                return False, None
            self.second = True
            self.motion_until = clock[0] + 5.0
        clock[0] += 1 / 15
        self.i += 1
        return True, (_motion_frame(self.i) if clock[0] < self.motion_until
                      else QUIET_FRAME.copy())

    def release(self):
        pass


class TestLiveRun:
    """run() on a live stream. File-source runs exit on the first failed
    read, so they never reach the stream-health alerts."""

    def _blind(self, live):
        return [(t, text) for t, text, _ in live.sent if text.startswith("Monitor blind")]

    def test_blind_alert_reminder_and_recovery(self, live):
        live.start(_DeadStreamCap(live.clock, back_at=1000.0 + 7 * 3600))
        blind = [text for _, text in self._blind(live)]
        assert len(blind) == 2
        assert "for 1 min" in blind[0] and "for 6.0 h" in blind[1]
        recovered = [text for _, text, _ in live.sent
                     if text.startswith("Monitor recovered: frames from")]
        assert len(recovered) == 1 and "after 7.0 h" in recovered[0]

    def test_undelivered_blind_alert_is_retried_every_minute(self, live):
        attempts = []

        def deliver():
            attempts.append(1)
            return len(attempts) > 2          # WAN down for the first two

        live.deliver = deliver
        live.start(_DeadStreamCap(live.clock, back_at=1000.0 + 7 * 3600))
        assert [t for t, _ in self._blind(live)] == [
            1060.0, 1120.0, 1180.0, 1180.0 + monitor.STREAM_BLIND_REMIND_SEC]

    def test_read_loop_keeps_watching_while_an_event_is_processed(self, live, monkeypatch):
        """Regression: verification ran inline in the read loop, so for the
        minutes it took no frame was read. A seizure starting meanwhile went
        unseen, and the stream's own watchdog never counted the gap."""
        processing, release = threading.Event(), threading.Event()
        captured, processed = [], []
        real_capture = monitor.capture_event

        def capture(*args, **kwargs):
            result = real_capture(*args, **kwargs)
            captured.append(result)
            if len(captured) == 2:
                release.set()
            return result

        def process(event_dir, use_verify, outage_notifier, fb=None):
            if processed:
                processed.append(None)
                return
            processing.set()
            released = release.wait(10)
            processed.append((released, len(captured), [text for _, text, _ in live.sent]))

        monkeypatch.setattr(monitor, "capture_event", capture)
        monkeypatch.setattr(monitor, "process_event", process)
        live.start(_EventCap(live, processing, lambda: len(processed) >= 2))
        released, n_captured, sent_meanwhile = processed[0]
        assert released and n_captured == 2         # event 2 seen while event 1 blocked
        assert any(t.startswith("Monitor blind") for t in sent_meanwhile)

    def test_wedged_worker_stops_the_systemd_pings(self, live, monkeypatch):
        """Regression: a keepalive thread pinged systemd for as long as event
        handling took, so a wedge inside it hid from the watchdog forever."""
        monkeypatch.setenv("NOTIFY_SOCKET", "/run/systemd/notify")
        pings = []
        monkeypatch.setattr(monitor.SystemdWatchdog, "_send",
                            lambda self, msg: pings.append(live.clock[0]))
        processing, unblock = threading.Event(), threading.Event()

        def process(event_dir, use_verify, outage_notifier, fb=None):
            processing.set()
            unblock.wait(10)

        class Cap:
            i, busy_from = 0, None

            def read(self):
                clock = live.clock
                if not processing.is_set():         # event, then quiet
                    clock[0] += 1 / 15
                    self.i += 1
                    return True, (_motion_frame(self.i) if clock[0] < 1005.0
                                  else QUIET_FRAME.copy())
                if self.busy_from is None:
                    self.busy_from = clock[0]
                clock[0] += 60
                if clock[0] > self.busy_from + monitor.HANDLE_DEADLINE_SEC + 600:
                    unblock.set()
                    raise _Stop
                return True, QUIET_FRAME.copy()

            def release(self):
                pass

        monkeypatch.setattr(monitor, "process_event", process)
        cap = Cap()
        live.start(cap)
        deadline = cap.busy_from + monitor.HANDLE_DEADLINE_SEC
        assert any(cap.busy_from < t <= deadline for t in pings)   # slow is fine
        assert max(pings) <= deadline + 1.0                          # wedged is not

    def test_restart_while_the_stream_is_down_still_checks_swept_events(self, live, monkeypatch):
        """Regression: the sweep ran only once the stream opened, so after a
        restart during a camera outage longer than SWEEP_MAX_AGE_SEC the
        interrupted event aged out and was never checked or alerted."""
        live.clock[0] = time.time()             # event dir mtimes are real
        start = live.clock[0]
        _make_event(live.root, "event_20260925_010000_mi360", n_frames=1)
        processed, done_before_open = threading.Event(), []
        live.verify_on(lambda event_dir, use_verify, notifier, fb=None: processed.set())

        class Cap:
            def read(self):
                raise _Stop

            def release(self):
                pass

        def fake_open(source):
            if not done_before_open:            # give the worker real time
                done_before_open.append(processed.wait(5))
            if live.clock[0] < start + 7 * 3600:
                raise RuntimeError("Could not open source")
            return Cap(), False, None

        monkeypatch.setattr(monitor, "open_capture", fake_open)
        live.start()
        assert done_before_open == [True]

    def test_wedged_swept_event_stops_the_pings_while_the_stream_is_down(self, live, monkeypatch):
        monkeypatch.setenv("NOTIFY_SOCKET", "/run/systemd/notify")
        pings = []
        monkeypatch.setattr(monitor.SystemdWatchdog, "_send",
                            lambda self, msg: pings.append(live.clock[0]))
        processing, unblock = threading.Event(), threading.Event()

        def process(event_dir, use_verify, outage_notifier, fb=None):
            processing.set()
            unblock.wait(10)

        live.verify_on(process)
        _make_event(live.root, "event_20260925_010000_mi360", n_frames=1)
        busy_from = []

        def fake_open(source):
            if not busy_from:
                assert processing.wait(5)
                busy_from.append(live.clock[0])
            if live.clock[0] > busy_from[0] + monitor.HANDLE_DEADLINE_SEC + 600:
                raise _Stop
            raise RuntimeError("Could not open source")

        monkeypatch.setattr(monitor, "open_capture", fake_open)
        try:
            live.start()
        finally:
            unblock.set()
        deadline = busy_from[0] + monitor.HANDLE_DEADLINE_SEC
        assert any(busy_from[0] < t <= deadline for t in pings)
        assert max(pings) <= deadline

    def _waiting_alerts(self, live, monkeypatch, swept, captures):
        """swept unhandled events on disk at startup, then `captures` live
        events, while the worker is stuck on the first one."""
        release = threading.Event()
        live.verify_on(lambda *a, **k: release.wait(10))
        for i in range(swept):
            _make_event(live.root, f"event_20260925_0100{i:02d}_mi360", n_frames=1)
        try:
            live.start(_MotionEventsCap(live.clock, captures))
        finally:
            release.set()
        return [text for _, text, _ in live.sent if "waiting for verification" in text]

    def test_backlog_from_the_restart_sweep_is_announced_once(self, live, monkeypatch):
        """Regression: sweep submits dropped the backlog signal, which also
        silenced it for the live captures queued behind them."""
        assert len(self._waiting_alerts(live, monkeypatch, swept=12, captures=2)) == 1

    def test_backlog_from_live_captures_is_announced_once(self, live, monkeypatch):
        assert len(self._waiting_alerts(live, monkeypatch, swept=9, captures=4)) == 1


class _MotionEventsCap:
    """`n` motion events, each 5 s of motion then 12 s of quiet (captured
    10 s into it); stops the run after the last one. Each read is 1/15 s."""

    PERIOD = 17.0

    def __init__(self, clock, n):
        self.clock, self.t0, self.n = clock, clock[0], n
        self.i = 0

    def read(self):
        elapsed = self.clock[0] - self.t0
        if elapsed >= self.n * self.PERIOD:
            raise _Stop
        self.clock[0] += 1 / 15
        self.i += 1
        return True, (_motion_frame(self.i) if elapsed % self.PERIOD < 5.0
                      else QUIET_FRAME.copy())

    def release(self):
        pass
