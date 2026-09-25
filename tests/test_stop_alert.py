import json

import pytest

import stop_alert

ABNORMAL = {"SERVICE_RESULT": "signal", "EXIT_CODE": "killed", "EXIT_STATUS": "KILL"}


@pytest.fixture
def sent(monkeypatch):
    """Telegram configured; records every message, delivery controlled per test."""
    box = {"messages": [], "deliver": True}

    def fake_send(text, photo_path=None, video_path=None):
        box["messages"].append(text)
        return box["deliver"]

    monkeypatch.setattr(stop_alert.alerts, "send_alert", fake_send)
    monkeypatch.setattr(stop_alert.alerts, "telegram_configured", lambda: True)
    return box


def stop(tmp_path, now, env=ABNORMAL):
    return stop_alert.main(["stop_alert.py", "seizureguard-c700"], env=env, now=now,
                           state_dir=tmp_path)


def state(tmp_path):
    return json.loads((tmp_path / "stop-alert-seizureguard-c700.json").read_text())


class TestStopAlert:
    def test_clean_stop_is_silent_and_leaves_no_state(self, tmp_path, sent):
        assert stop(tmp_path, 1000.0, env={"SERVICE_RESULT": "success"}) == 0
        assert sent["messages"] == []
        assert list(tmp_path.iterdir()) == []

    def test_first_abnormal_stop_is_announced_at_once(self, tmp_path, sent):
        stop(tmp_path, 1000.0)
        assert sent["messages"] == [
            "seizureGuard seizureguard-c700 stopped abnormally (signal, killed/KILL); "
            "systemd is restarting it - the camera is not watched until it is back"]
        assert state(tmp_path) == {"last_sent": 1000.0, "pending": 0}

    def test_crash_loop_is_rate_limited_and_counted(self, tmp_path, sent):
        """Regression: a monitor that crash-loops restarts every ~20 s, and
        each stop used to page."""
        stop(tmp_path, 1000.0)
        for i in range(1, 6):                      # 5 more stops, 20 s apart
            stop(tmp_path, 1000.0 + 20 * i)
        assert len(sent["messages"]) == 1
        assert state(tmp_path)["pending"] == 5
        stop(tmp_path, 1000.0 + stop_alert.STOP_ALERT_INTERVAL_SEC)
        assert len(sent["messages"]) == 2
        assert "(5 more abnormal stops since the last message)" in sent["messages"][1]
        assert state(tmp_path)["pending"] == 0

    def test_one_unreported_stop_is_singular(self, tmp_path, sent):
        stop(tmp_path, 1000.0)
        stop(tmp_path, 1020.0)
        stop(tmp_path, 1000.0 + stop_alert.STOP_ALERT_INTERVAL_SEC)
        assert "(1 more abnormal stop since the last message)" in sent["messages"][1]

    def test_undelivered_message_is_retried_on_the_next_stop(self, tmp_path, sent):
        sent["deliver"] = False
        stop(tmp_path, 1000.0)
        sent["deliver"] = True
        stop(tmp_path, 1020.0)                     # well inside the interval
        assert len(sent["messages"]) == 2
        assert "(1 more abnormal stop since the last message)" in sent["messages"][1]
        assert state(tmp_path) == {"last_sent": 1020.0, "pending": 0}

    def test_undelivered_keeps_the_old_interval_running(self, tmp_path, sent):
        stop(tmp_path, 1000.0)
        sent["deliver"] = False
        t = 1000.0 + stop_alert.STOP_ALERT_INTERVAL_SEC
        stop(tmp_path, t)
        assert state(tmp_path) == {"last_sent": 1000.0, "pending": 1}

    def test_console_only_mode_counts_as_delivered(self, tmp_path, sent, monkeypatch):
        monkeypatch.setattr(stop_alert.alerts, "telegram_configured", lambda: False)
        sent["deliver"] = False
        stop(tmp_path, 1000.0)
        stop(tmp_path, 1020.0)
        assert len(sent["messages"]) == 1

    def test_unreadable_state_announces(self, tmp_path, sent):
        (tmp_path / "stop-alert-seizureguard-c700.json").write_text("{truncated")
        stop(tmp_path, 1000.0)
        assert len(sent["messages"]) == 1

    def test_units_are_limited_separately(self, tmp_path, sent):
        stop(tmp_path, 1000.0)
        stop_alert.main(["stop_alert.py", "seizureguard-mi360"], env=ABNORMAL, now=1010.0,
                        state_dir=tmp_path)
        assert len(sent["messages"]) == 2

    def test_unwritable_state_dir_never_raises(self, tmp_path, sent):
        blocker = tmp_path / "data"
        blocker.write_text("a file where the dir should be")
        assert stop_alert.main(["stop_alert.py", "u"], env=ABNORMAL, now=1.0,
                               state_dir=blocker / "sub") == 0
        assert len(sent["messages"]) == 1
