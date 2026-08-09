import re

import numpy as np

import sampling


def _frame_time(path):
    return float(re.search(r"_t_([0-9.]+)s", path.name).group(1))


def _gray(fill=0):
    return np.full((360, 640), fill, np.uint8)


class TestMotionScore:
    def test_no_motion_scores_zero(self):
        assert sampling.motion_score(_gray(40), _gray(40)) == 0.0

    def test_score_grows_with_changed_area(self):
        prev = _gray(40)
        small = _gray(40)
        small[100:130, 100:130] = 200
        large = _gray(40)
        large[50:250, 50:450] = 200
        s_small = sampling.motion_score(prev, small)
        s_large = sampling.motion_score(prev, large)
        assert 0.0 < s_small < s_large

    def test_full_frame_change_is_global(self):
        score = sampling.motion_score(_gray(40), _gray(220))
        assert sampling.is_global_change(score)

    def test_local_motion_is_not_global(self):
        prev = _gray(40)
        cur = _gray(40)
        cur[100:160, 100:160] = 200
        assert not sampling.is_global_change(sampling.motion_score(prev, cur))


class TestSelectPeaks:
    def test_picks_strongest_with_min_separation(self):
        series = [(10.0, 50.0), (10.5, 60.0), (30.0, 40.0), (31.0, 45.0), (55.0, 20.0)]
        peaks = sampling.select_peaks(series)
        assert peaks == [10.5, 31.0, 55.0]

    def test_sustained_motion_gets_full_span_coverage(self):
        # Regression: real seizure footage had a 40 s sustained-motion region
        # starved of bursts because 3 isolated spikes elsewhere took every slot.
        series = [(float(t), 10.0) for t in range(0, 100, 10)]
        peaks = sampling.select_peaks(series)
        assert len(peaks) >= sampling.NUM_BURSTS
        assert max(peaks) >= 60.0     # late buckets are covered too

    def test_spikes_cannot_starve_a_later_active_region(self):
        spikes = [(1.0, 125.0), (10.0, 90.0), (20.0, 80.0)]
        seizure = [(float(t), 5.0) for t in range(60, 100, 2)]
        peaks = sampling.select_peaks(spikes + seizure)
        assert any(60.0 <= p <= 100.0 for p in peaks)

    def test_drift_never_earns_a_coverage_burst(self):
        active = [(5.0, 50.0)]
        drift = [(float(t), 0.4) for t in range(30, 100, 5)]
        peaks = sampling.select_peaks(active + drift)
        assert all(p < 30.0 for p in peaks)

    def test_empty_series(self):
        assert sampling.select_peaks([]) == []


class TestFrameBuffer:
    def test_empty_returns_none(self):
        assert sampling.FrameBuffer().nearest(1.0) is None

    def test_nearest_picks_closest(self):
        fb = sampling.FrameBuffer()
        for t in (0.0, 1.0, 2.0):
            fb.append(t, b"jpg%f" % t)
        assert fb.nearest(0.4)[0] == 0.0
        assert fb.nearest(0.6)[0] == 1.0
        assert fb.nearest(-5.0)[0] == 0.0
        assert fb.nearest(99.0)[0] == 2.0


class TestExtractedEvent:
    def test_base_frame_count(self, synthetic_event_dir):
        base = list((synthetic_event_dir / "base").glob("frame_*.jpg"))
        assert 115 <= len(base) <= 125

    def test_burst_covers_violent_segment(self, synthetic_event_dir):
        times = [_frame_time(p) for p in (synthetic_event_dir / "burst").glob("frame_*.jpg")]
        assert times, "no burst frames produced"
        assert any(19.0 <= t <= 24.0 for t in times)

    def test_flash_never_selected_as_peak(self, synthetic_event_dir):
        times = [_frame_time(p) for p in (synthetic_event_dir / "burst").glob("frame_*.jpg")]
        in_flash = [t for t in times if 39.8 <= t <= 41.2]
        assert len(in_flash) == 0

    def test_no_duplicate_frames_across_base_and_burst(self, synthetic_event_dir):
        base = {_frame_time(p) for p in (synthetic_event_dir / "base").glob("frame_*.jpg")}
        burst = {_frame_time(p) for p in (synthetic_event_dir / "burst").glob("frame_*.jpg")}
        assert not (base & burst)

    def test_merged_order_is_chronological(self, synthetic_event_dir):
        frames = sorted(
            list((synthetic_event_dir / "base").glob("frame_*.jpg"))
            + list((synthetic_event_dir / "burst").glob("frame_*.jpg")),
            key=_frame_time,
        )
        times = [_frame_time(p) for p in frames]
        assert times == sorted(times)
