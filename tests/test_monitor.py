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
    def test_url_source_passes_string_through(self, monkeypatch):
        created = []
        monkeypatch.setattr(monitor.cv2, "VideoCapture",
                            lambda *a: created.append(_FakeCap(*a)) or created[-1])
        cap, file_mode, fps = monitor.open_capture("rtsp://localhost:8554/mi360")
        assert created[0].args == ("rtsp://localhost:8554/mi360",)
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
