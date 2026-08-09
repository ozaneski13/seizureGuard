import cv2
import numpy as np

import alert_clip


def _fake_event(tmp_path, times, size=(64, 48)):
    burst = tmp_path / "burst"
    burst.mkdir()
    (tmp_path / "base").mkdir()
    for i, t in enumerate(times):
        img = np.full((size[1], size[0], 3), (i * 7) % 255, np.uint8)
        cv2.imwrite(str(burst / f"frame_{i:03d}_t_{t:0.3f}s.jpg"), img)
    return tmp_path


class TestClampWindow:
    def test_short_window_padded_to_min(self):
        lo, hi = alert_clip.clamp_window(10.0, 12.0)
        assert hi - lo == alert_clip.MIN_CLIP_SEC
        assert (lo + hi) / 2 == 11.0

    def test_long_window_trimmed_to_max_around_center(self):
        lo, hi = alert_clip.clamp_window(0.0, 30.0)
        assert hi - lo == alert_clip.MAX_CLIP_SEC
        assert (lo + hi) / 2 == 15.0

    def test_in_range_window_kept(self):
        lo, hi = alert_clip.clamp_window(4.0, 11.0)
        assert (hi - lo) == 7.0


class TestBestBatchWindow:
    def _frames(self, tmp_path, n):
        _fake_event(tmp_path, [i / 10.0 for i in range(n)])
        return alert_clip.event_frames(tmp_path)

    def test_picks_highest_confidence_positive_batch(self, tmp_path):
        frames = self._frames(tmp_path, 90)          # 3 batches of 30
        analysis = {"batches": [
            {"abnormal_event": False, "confidence": 0.9},
            {"abnormal_event": True, "confidence": 0.4},
            {"abnormal_event": True, "confidence": 0.7},   # <- winner
        ]}
        lo, hi = alert_clip.best_batch_window(analysis, frames)
        # batch 2 covers t=6.0..8.9; clamped to 5s around its center
        assert lo <= 6.5 and hi >= 8.4
        assert 6.0 <= (lo + hi) / 2 <= 9.0

    def test_no_positive_batch_returns_none(self, tmp_path):
        frames = self._frames(tmp_path, 30)
        analysis = {"batches": [{"abnormal_event": False, "confidence": 0.9}]}
        assert alert_clip.best_batch_window(analysis, frames) is None

    def test_missing_analysis_returns_none(self):
        assert alert_clip.best_batch_window(None, []) is None


class TestMakeClip:
    def test_encodes_playable_clip(self, tmp_path):
        event = _fake_event(tmp_path, [i / 10.0 for i in range(80)])
        out = alert_clip.make_clip(event, (2.0, 6.0))
        assert out is not None and out.exists()
        cap = cv2.VideoCapture(str(out))
        n = 0
        while cap.read()[0]:
            n += 1
        cap.release()
        assert n >= 30                     # ~4s of 10fps frames made it in

    def test_too_few_frames_returns_none(self, tmp_path):
        event = _fake_event(tmp_path, [0.0, 0.1])
        assert alert_clip.make_clip(event, (0.0, 5.0)) is None

    def test_none_window_returns_none(self, tmp_path):
        event = _fake_event(tmp_path, [i / 10.0 for i in range(20)])
        assert alert_clip.make_clip(event, None) is None


class TestPeakWindow:
    def test_centered_on_first_peak(self):
        import pytest
        lo, hi = alert_clip.peak_window([33.7, 52.2], t_start=10.0)
        assert (lo + hi) / 2 == pytest.approx(23.7)
        assert hi - lo == pytest.approx(alert_clip.MAX_CLIP_SEC)

    def test_no_peaks_returns_none(self):
        assert alert_clip.peak_window([], 0.0) is None
