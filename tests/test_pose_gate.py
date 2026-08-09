import numpy as np
import pytest

import pose_gate

FPS = 10.0
T = 60          # 6 seconds of burst frames
K = 12          # keypoints


def _trajectory(jerk_hz=None, jerk_amp=15.0, conf=0.9, seed=7):
    """Dog walking slowly (0.3 Hz sway + drift); optionally add rhythmic
    limb jerking at jerk_hz on the last 4 keypoints (the 'legs')."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / FPS
    kpts = np.zeros((T, K, 3))
    for k in range(K):
        base_x = 100 + 5 * k + 8.0 * t                      # slow drift
        base_y = 100 + 3 * k + 10.0 * np.sin(2 * np.pi * 0.3 * t + k)
        kpts[:, k, 0] = base_x + rng.normal(0, 0.5, T)
        kpts[:, k, 1] = base_y + rng.normal(0, 0.5, T)
        kpts[:, k, 2] = conf
    if jerk_hz is not None:
        for k in range(K - 4, K):
            kpts[:, k, 0] += jerk_amp * np.sin(2 * np.pi * jerk_hz * t + k)
            kpts[:, k, 1] += jerk_amp * np.cos(2 * np.pi * jerk_hz * t + k)
    return kpts


class TestOscillationScore:
    def test_rhythmic_jerking_scores_high(self):
        assert pose_gate.oscillation_score(_trajectory(jerk_hz=4.0)) > 0.6

    def test_smooth_walk_scores_low(self):
        assert pose_gate.oscillation_score(_trajectory()) < pose_gate.GATE_THRESHOLD

    def test_separation_between_walk_and_seizure(self):
        walk = pose_gate.oscillation_score(_trajectory())
        seizure = pose_gate.oscillation_score(_trajectory(jerk_hz=3.0))
        assert seizure > walk + 0.3

    def test_band_edges(self):
        in_band = pose_gate.oscillation_score(_trajectory(jerk_hz=2.0))
        below_band = pose_gate.oscillation_score(_trajectory(jerk_hz=0.5, jerk_amp=25.0))
        assert in_band > below_band

    def test_too_few_frames_scores_zero(self):
        assert pose_gate.oscillation_score(_trajectory()[:5]) == 0.0

    def test_static_dog_scores_zero(self):
        kpts = np.zeros((T, K, 3))
        kpts[:, :, 2] = 0.9
        assert pose_gate.oscillation_score(kpts) == 0.0


class TestGateDecision:
    def test_high_score_escalates(self):
        assert pose_gate.gate_decision(0.8, mean_conf=0.9) == "escalate"

    def test_low_score_with_confident_dog_skips(self):
        assert pose_gate.gate_decision(0.1, mean_conf=0.9) == "skip"

    def test_low_confidence_fails_open(self):
        assert pose_gate.gate_decision(0.1, mean_conf=0.05) == "escalate"


class TestAnalyzeEvent:
    @pytest.fixture
    def event_dir(self, tmp_path):
        burst = tmp_path / "burst"
        burst.mkdir()
        for i in range(T):
            (burst / f"frame_{i:03d}_t_{i / FPS:0.3f}s.jpg").write_bytes(b"\xff\xd8")
        return tmp_path

    def test_seizure_trajectory_escalates(self, event_dir):
        out = pose_gate.analyze_event(
            event_dir,
            keypoint_fn=lambda frames: (_trajectory(jerk_hz=4.0), list(range(T))))
        assert out["decision"] == "escalate"
        assert out["score"] > 0.6

    def test_walking_dog_skips(self, event_dir):
        out = pose_gate.analyze_event(
            event_dir, keypoint_fn=lambda f: (_trajectory(), list(range(T))))
        assert out["decision"] == "skip"

    def test_backend_failure_fails_open(self, event_dir):
        def boom(frames):
            raise RuntimeError("no dog detected")

        out = pose_gate.analyze_event(event_dir, keypoint_fn=boom)
        assert out["decision"] == "escalate"
        assert "no dog" in out["reason"]

    def test_too_few_frames_fails_open(self, tmp_path):
        (tmp_path / "burst").mkdir()
        out = pose_gate.analyze_event(tmp_path, keypoint_fn=lambda f: None)
        assert out["decision"] == "escalate"

    def test_zero_confidence_fails_open(self, event_dir):
        def zero_conf(frames):
            k = _trajectory()
            k[:, :, 2] = 0.0
            return k, list(range(T))

        out = pose_gate.analyze_event(event_dir, keypoint_fn=zero_conf)
        assert out["decision"] == "escalate"


class TestSegments:
    def test_contiguous_is_one_segment(self):
        times = [i / 10.0 for i in range(30)]
        assert pose_gate._segments(times) == [list(range(30))]

    def test_gaps_split_segments(self):
        times = [i / 10.0 for i in range(20)] + \
            [10.0 + i / 10.0 for i in range(20)]
        segs = pose_gate._segments(times)
        assert len(segs) == 2
        assert segs[0] == list(range(20))
        assert segs[1] == list(range(20, 40))

    def test_jitter_in_one_window_detected_despite_gaps(self, tmp_path):
        burst = tmp_path / "burst"
        burst.mkdir()
        times = [2.0 + i / FPS for i in range(30)] + \
            [20.0 + i / FPS for i in range(30)]
        for i, t in enumerate(times):
            (burst / f"frame_{i:03d}_t_{t:0.3f}s.jpg").write_bytes(b"x")

        def kp(frames):
            calm = _trajectory()[:30]
            jerky = _trajectory(jerk_hz=3.0)[:30]
            import numpy as _np
            return _np.concatenate([calm, jerky]), list(range(60))

        out = pose_gate.analyze_event(tmp_path, keypoint_fn=kp)
        assert out["decision"] == "escalate"
        assert out["score"] > 0.6


class TestNonUniformSampling:
    def test_dedup_holes_do_not_hide_jitter(self):
        kpts = _trajectory(jerk_hz=3.0)
        keep = [i for i in range(T) if i % 5 != 4]   # drop every 5th sample
        times = np.array(keep) / FPS
        score = pose_gate.oscillation_score(kpts[keep], times=times)
        assert score > 0.6

    def test_dedup_holes_do_not_inflate_walk(self):
        kpts = _trajectory()
        keep = [i for i in range(T) if i % 5 != 4]
        times = np.array(keep) / FPS
        assert pose_gate.oscillation_score(kpts[keep], times=times) < 0.5


class TestPartialDetections:
    def _event_dir(self, tmp_path, n=60):
        burst = tmp_path / "burst"
        burst.mkdir()
        for i in range(n):
            (burst / f"frame_{i:03d}_t_{i / FPS:0.3f}s.jpg").write_bytes(b"x")
        return tmp_path

    def test_missing_detections_are_skipped_not_fatal(self, tmp_path):
        event_dir = self._event_dir(tmp_path)
        kept = [i for i in range(T) if i % 6 != 5]

        def kp(frames):
            return _trajectory(jerk_hz=3.0)[kept], kept

        out = pose_gate.analyze_event(event_dir, keypoint_fn=kp)
        assert out["decision"] == "escalate"
        assert out["score"] > 0.6
        assert 0.8 < out["detection_rate"] < 0.9

    def test_too_few_detections_fails_open(self, tmp_path):
        event_dir = self._event_dir(tmp_path)

        def kp(frames):
            return _trajectory()[:5], list(range(5))

        out = pose_gate.analyze_event(event_dir, keypoint_fn=kp)
        assert out["decision"] == "escalate"
        assert "too few frames with a detected dog" in out["reason"]

    def test_low_detection_rate_fails_open(self, tmp_path):
        # Regression: a convulsing dog on its side is detected in only ~1/3
        # of frames; the gate must escalate, not read the unscored 0.0 as
        # "confidently non-rhythmic" and silence a real seizure.
        event_dir = self._event_dir(tmp_path)
        kept = list(range(0, T, 3))       # 33% detection rate, spread out

        def kp(frames):
            return _trajectory()[kept], kept

        out = pose_gate.analyze_event(event_dir, keypoint_fn=kp)
        assert out["decision"] == "escalate"
        assert "detection rate" in out["reason"]

    def test_fragmented_segments_fail_open(self, tmp_path):
        # Detection rate above the floor but clustered in runs shorter than
        # MIN_FRAMES: nothing is scorable, so the gate must escalate.
        event_dir = self._event_dir(tmp_path, n=100)
        kept = [i for i in range(100) if (i % 12) < 8]   # runs of 8 < MIN_FRAMES

        def kp(frames):
            traj = _trajectory()
            import numpy as _np
            reps = _np.concatenate([traj, traj])[:len(kept)]
            return reps, kept

        out = pose_gate.analyze_event(event_dir, keypoint_fn=kp)
        assert out["decision"] == "escalate"
        assert "no segment long enough" in out["reason"]
