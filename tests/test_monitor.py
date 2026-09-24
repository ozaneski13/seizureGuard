import re

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

    def test_blind_alert_fires_exactly_once(self):
        wd = StreamWatchdog(retry_sec=3.0, alert_sec=60.0)
        wd.failed(100.0)
        actions = wd.failed(161.0)
        assert "blind_alert" in actions
        assert "blind_alert" not in wd.failed(300.0)

    def test_recovery_resets_everything(self):
        wd = StreamWatchdog(retry_sec=3.0, alert_sec=60.0)
        wd.failed(100.0)
        wd.failed(161.0)
        wd.ok()
        assert wd.failed(200.0) == []          # fresh stall, fresh timers
        assert wd.failed(204.0) == ["reconnect"]


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
        return monitor.run(str(synthetic_video), out_root=tmp_path / "events")

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
        with wd.keepalive():
            pass
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

    def test_keepalive_pings_while_main_loop_is_blocked(self, monkeypatch):
        import time as _time
        wd, sent = self._wd(monkeypatch, interval=0.02)
        with wd.keepalive():
            _time.sleep(0.15)
        assert len(sent) >= 2
        count = len(sent)
        _time.sleep(0.08)
        assert len(sent) == count          # thread stopped with the block
