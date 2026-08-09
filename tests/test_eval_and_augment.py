import json
import sys
import textwrap

import cv2
import numpy as np
import pytest

import augment_seizure
import eval_clips


class TestSummarize:
    def test_perfect_split(self):
        rows = [
            {"expected": True, "predicted": True},
            {"expected": True, "predicted": True},
            {"expected": False, "predicted": False},
        ]
        m = eval_clips.summarize(rows)
        assert m["sensitivity"] == 1.0
        assert m["specificity"] == 1.0
        assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (2, 0, 1, 0)

    def test_miss_and_false_positive(self):
        rows = [
            {"expected": True, "predicted": False},
            {"expected": True, "predicted": True},
            {"expected": False, "predicted": True},
            {"expected": False, "predicted": False},
        ]
        m = eval_clips.summarize(rows)
        assert m["sensitivity"] == 0.5
        assert m["specificity"] == 0.5

    def test_single_class_gives_none_for_other_metric(self):
        m = eval_clips.summarize([{"expected": False, "predicted": False}])
        assert m["sensitivity"] is None
        assert m["specificity"] == 0.0 or m["specificity"] == 1.0


class TestEvaluateClip:
    def test_runs_extract_and_stub_verify(self, synthetic_video, tmp_path):
        stub = tmp_path / "stub_verify.py"
        stub.write_text(textwrap.dedent("""
            import json, sys
            from pathlib import Path
            event_dir = Path(sys.argv[1])
            out = {"final_abnormal_event": True, "final_confidence": 0.77,
                   "failed_batches": 0}
            (event_dir / "analysis.json").write_text(json.dumps(out))
        """), encoding="utf-8")
        row = eval_clips.evaluate_clip(
            synthetic_video, tmp_path / "work",
            verify_cmd=[sys.executable, str(stub)])
        assert row["predicted"] is True
        assert row["confidence"] == 0.77
        assert row["frames"] > 100
        assert row["window"] > 50


class TestShiftRegion:
    def _frame(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        frame[40:60, 40:60] = 255
        return frame

    def test_shift_moves_pixels(self):
        out = augment_seizure.shift_region(self._frame(), (40, 40, 60, 60), 10, 0)
        assert out[50, 65, 0] == 255
        assert not np.array_equal(out, self._frame())

    def test_clipping_at_edges_is_safe(self):
        frame = self._frame()
        out = augment_seizure.shift_region(frame, (40, 40, 60, 60), 90, 90)
        assert out.shape == frame.shape

    def test_degenerate_bbox_is_noop(self):
        frame = self._frame()
        out = augment_seizure.shift_region(frame, (50, 50, 50, 50), 5, 5)
        assert np.array_equal(out, frame)


class TestAugment:
    @pytest.fixture
    def plain_video(self, tmp_path):
        path = tmp_path / "plain.mp4"
        out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                              30, (160, 120))
        for _ in range(30 * 12):
            frame = np.full((120, 160, 3), 30, np.uint8)
            frame[40:80, 60:100] = (0, 180, 220)
            out.write(frame)
        out.release()
        return path

    def test_jitter_window_moves_only_inside_window(self, plain_video, tmp_path):
        out_path = tmp_path / "aug.mp4"
        jittered = augment_seizure.augment(
            plain_video, out_path, freq=3.0, amp=20, start=4.0, dur=4.0,
            bbox_fn=lambda frame: (60, 40, 100, 80))
        assert jittered > 100

        orig = cv2.VideoCapture(str(plain_video))
        aug = cv2.VideoCapture(str(out_path))
        diffs = []
        idx = 0
        while True:
            r1, f1 = orig.read()
            r2, f2 = aug.read()
            if not (r1 and r2):
                break
            diffs.append((idx / 30.0, float(np.abs(
                f1.astype(int) - f2.astype(int)).mean())))
            idx += 1
        orig.release()
        aug.release()

        inside = sum(d for t, d in diffs if 4.2 <= t <= 7.8) / \
            len([1 for t, d in diffs if 4.2 <= t <= 7.8])
        outside = sum(d for t, d in diffs if t <= 3.5 or t >= 8.5) / \
            len([1 for t, d in diffs if t <= 3.5 or t >= 8.5])
        # outside differs only by codec re-encode noise; the jitter window
        # must stand clearly above that floor
        assert inside > 2 * outside, (inside, outside)

    def test_no_dog_means_no_jitter(self, plain_video, tmp_path):
        out_path = tmp_path / "aug.mp4"
        jittered = augment_seizure.augment(
            plain_video, out_path, bbox_fn=lambda frame: None)
        assert jittered == 0
