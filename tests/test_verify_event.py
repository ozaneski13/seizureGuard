import json
import sys
import types

import pytest

import verify_event as ve


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(ve.time, "sleep", lambda s: None)
    monkeypatch.delenv("SEIZUREGUARD_BACKEND", raising=False)
    # CI runners have no claude binary; the which() preflight must not gate
    # tests whose subprocess layer is mocked anyway.
    monkeypatch.setattr(ve.shutil, "which", lambda name: "claude")


def _fake_run_claude(confirm_fn):
    def fake(prompt, model, images, timeout=600):
        return confirm_fn(prompt)
    return fake


def _run_main(monkeypatch, event_dir):
    monkeypatch.setattr(sys, "argv", ["verify_event.py", str(event_dir)])
    ve.main()
    return json.loads((event_dir / "analysis.json").read_text())


def _prompt_hits_window(prompt, lo, hi):
    import re
    m = re.search(r"attached images\): ([0-9., ]+)", prompt)
    times = [float(x) for x in m.group(1).split(",") if x.strip()] if m else []
    return any(lo <= t <= hi for t in times)


class TestPureCore:
    def test_frame_time(self, tmp_path):
        assert ve.frame_time(tmp_path / "frame_003_t_12.345s.jpg") == 12.345
        assert ve.frame_time(tmp_path / "whatever.jpg") == 0.0

    def test_iter_batches(self):
        items = list(range(70))
        batches = list(ve.iter_batches(items, 30))
        assert [len(b) for b in batches] == [30, 30, 10]
        assert [x for b in batches for x in b] == items

    def test_parse_json_verdict_plain(self):
        assert ve.parse_json_verdict('{"seen": "no", "confidence": 0.1}') == {
            "seen": "no", "confidence": 0.1}

    def test_parse_json_verdict_fenced(self):
        text = 'Here you go:\n```json\n{"seen": "yes", "confidence": 0.8}\n```'
        assert ve.parse_json_verdict(text)["seen"] == "yes"

    def test_parse_json_verdict_prose_wrapped(self):
        text = 'The answer is {"abnormal_event": false, "confidence": 0.2} as requested.'
        assert ve.parse_json_verdict(text)["abnormal_event"] is False

    def test_parse_json_verdict_garbage_raises(self):
        with pytest.raises(ValueError):
            ve.parse_json_verdict("I cannot read these images")


class TestSignRules:
    def _verdict(self, present_signs, abnormal=False):
        return {
            "abnormal_event": abnormal,
            "confidence": 0.5,
            "observed_signs": [
                {"sign": s, "present": True, "body_region": "x", "sustained": True}
                for s in present_signs
            ],
        }

    def test_one_hard_sign_is_positive(self):
        assert ve.decide_signs(self._verdict(["paddling"])) is True

    def test_two_soft_signs_are_positive(self):
        assert ve.decide_signs(self._verdict(["drooling", "head_tremor"])) is True

    def test_one_soft_sign_is_negative(self):
        assert ve.decide_signs(self._verdict(["drooling"])) is False

    def test_model_abnormal_flag_alone_is_positive(self):
        assert ve.decide_signs(self._verdict([], abnormal=True)) is True

    def test_absent_signs_do_not_count(self):
        verdict = {
            "abnormal_event": False,
            "observed_signs": [
                {"sign": "paddling", "present": False},
                {"sign": "drooling", "present": False},
            ],
        }
        assert ve.decide_signs(verdict) is False

    def test_empty_verdict_is_negative(self):
        assert ve.decide_signs({}) is False


class TestRunClaude:
    def _proc(self, stdout, returncode=0):
        return types.SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)

    def _stream(self, result_text, is_error=False):
        lines = [
            json.dumps({"type": "system", "subtype": "init"}),
            json.dumps({"type": "assistant", "message": {"content": []}}),
            json.dumps({"type": "result", "is_error": is_error, "result": result_text}),
        ]
        return "\n".join(lines)

    def test_parses_result_event(self, monkeypatch):
        monkeypatch.setattr(ve.subprocess, "run", lambda *a, **k: self._proc(
            self._stream('{"seen": "no", "confidence": 0.1}')))
        out = ve.run_claude("prompt", "haiku", images=[])
        assert out == {"seen": "no", "confidence": 0.1}

    def test_fenced_result_is_cleaned(self, monkeypatch):
        monkeypatch.setattr(ve.subprocess, "run", lambda *a, **k: self._proc(
            self._stream('```json\n{"seen": "yes", "confidence": 0.9}\n```')))
        assert ve.run_claude("p", "m", images=[])["seen"] == "yes"

    def test_images_are_embedded_as_base64(self, monkeypatch, tmp_path):
        frame = tmp_path / "frame_000_t_1.000s.jpg"
        frame.write_bytes(b"\xff\xd8fakejpg")
        captured = {}

        def fake(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["stdin"] = kwargs.get("input")
            return self._proc(self._stream('{"seen": "no", "confidence": 0.0}'))

        monkeypatch.setattr(ve.subprocess, "run", fake)
        ve.run_claude("p", "m", images=[frame])
        # Regression: persisted transcripts embed the frames and filled a
        # Pi SD card (11 GB in 8 days) — headless calls must not persist.
        assert "--no-session-persistence" in captured["cmd"]
        msg = json.loads(captured["stdin"])
        blocks = msg["message"]["content"]
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "image"
        assert blocks[1]["source"]["media_type"] == "image/jpeg"
        import base64 as b64mod
        assert b64mod.b64decode(blocks[1]["source"]["data"]) == b"\xff\xd8fakejpg"

    def test_login_error_raises_dedicated_error(self, monkeypatch):
        monkeypatch.setattr(ve.subprocess, "run", lambda *a, **k: self._proc(
            self._stream("Not logged in · Please run /login", is_error=True), 1))
        with pytest.raises(ve.ClaudeLoginError):
            ve.run_claude("p", "m", images=[])

    def test_no_result_event_raises(self, monkeypatch):
        monkeypatch.setattr(ve.subprocess, "run",
                            lambda *a, **k: self._proc("segfault or whatever", 1))
        with pytest.raises(RuntimeError):
            ve.run_claude("p", "m", images=[])


class TestClaudeBackendFlow:
    def test_each_batch_makes_exactly_one_confirm_call(self, monkeypatch, tmp_path):
        """The screen tier is gone: its negatives were measured to carry no
        information, so it only cost a call per batch."""
        calls = []

        def fake(prompt, model, images, timeout=600):
            calls.append(model)
            return {"abnormal_event": False, "confidence": 0.1, "observed_signs": []}

        monkeypatch.setattr(ve, "run_claude", fake)
        r = ve.assess_batch_claude(tmp_path, [], ve.get_config(), 1)
        assert r["abnormal_event"] is False
        assert calls == ["claude-fable-5"]
        assert r["screen_verdict"] is None

    def test_positive_confirm_marks_batch(self, monkeypatch, tmp_path):
        def fake(prompt, model, images, timeout=600):
            return {"abnormal_event": True, "confidence": 0.9,
                    "observed_signs": [{"sign": "paddling", "present": True,
                                        "body_region": "legs", "sustained": True}]}

        monkeypatch.setattr(ve, "run_claude", fake)
        r = ve.assess_batch_claude(tmp_path, [], ve.get_config(), 1)
        assert r["abnormal_event"] is True

    def test_confirm_posture_and_note_are_recorded(self, monkeypatch, tmp_path):
        """The event viewer shows this note; it used to come from the blind
        screen model and read "normal walking" on a real seizure."""
        def fake(prompt, model, images, timeout=600):
            return {"abnormal_event": False, "confidence": 0.2,
                    "posture": "lying_lateral", "partially_visible": True,
                    "observed_signs": [], "note": "dog on its side, still"}

        monkeypatch.setattr(ve, "run_claude", fake)
        r = ve.assess_batch_claude(tmp_path, [], ve.get_config(), 1)
        assert r["posture"] == "lying_lateral"
        assert r["partially_visible"] is True
        assert r["note"] == "dog on its side, still"

    def test_confirm_failure_marks_batch_failed(self, monkeypatch, tmp_path):
        def fake(prompt, model, images, timeout=600):
            raise RuntimeError("confirm down")

        monkeypatch.setattr(ve, "run_claude", fake)
        r = ve.assess_batch_claude(tmp_path, [], ve.get_config(), 1)
        assert r["abnormal_event"] is None
        assert "error" in r

    def test_login_error_aborts_immediately(self, monkeypatch, tmp_path):
        def fake(prompt, model, images, timeout=600):
            raise ve.ClaudeLoginError("not logged in")

        monkeypatch.setattr(ve, "run_claude", fake)
        with pytest.raises(ve.ClaudeLoginError):
            ve.assess_batch_claude(tmp_path, [], ve.get_config(), 1)


class TestEndToEnd:
    def test_early_true_survives_later_quiet_batches(self, monkeypatch, synthetic_event_dir):
        def confirm(prompt):
            if _prompt_hits_window(prompt, 20.0, 23.0):
                return {"abnormal_event": True, "confidence": 0.9,
                        "observed_signs": [{"sign": "rhythmic_jerking", "present": True,
                                            "body_region": "whole body", "sustained": True}]}
            return {"abnormal_event": False, "confidence": 0.1, "observed_signs": []}

        monkeypatch.setattr(ve, "run_claude", _fake_run_claude(confirm))
        out = _run_main(monkeypatch, synthetic_event_dir)
        assert out["final_abnormal_event"] is True
        assert out["failed_batches"] == 0
        assert 1 <= out["positive_batches"] < len(out["batches"])

    def test_all_negative_stays_negative(self, monkeypatch, synthetic_event_dir):
        monkeypatch.setattr(ve, "run_claude", _fake_run_claude(
            lambda p: {"abnormal_event": False, "confidence": 0.1,
                       "observed_signs": []}))
        out = _run_main(monkeypatch, synthetic_event_dir)
        assert out["final_abnormal_event"] is False
        assert out["failed_batches"] == 0

    def test_failed_confirm_reports_incomplete(self, monkeypatch, synthetic_event_dir):
        def confirm(prompt):
            if _prompt_hits_window(prompt, 33.0, 37.0):
                raise RuntimeError("simulated failure")
            return {"abnormal_event": False, "confidence": 0.1, "observed_signs": []}

        monkeypatch.setattr(ve, "run_claude", _fake_run_claude(confirm))
        out = _run_main(monkeypatch, synthetic_event_dir)
        assert out["failed_batches"] >= 1
        assert out["final_abnormal_event"] is False
        assert "incomplete" in out["final_reason"]

    def test_analysis_json_contract(self, monkeypatch, synthetic_event_dir):
        monkeypatch.setattr(ve, "run_claude", _fake_run_claude(
            lambda p: {"abnormal_event": False, "confidence": 0.1,
                       "observed_signs": []}))
        out = _run_main(monkeypatch, synthetic_event_dir)
        for key in ("final_abnormal_event", "final_confidence", "backend",
                    "confirm_model", "batch_size", "num_frames",
                    "base_frames", "burst_frames", "failed_batches",
                    "final_reason", "batches"):
            assert key in out
        assert out["backend"] == "claude-cli"
        for b in out["batches"]:
            for key in ("abnormal_event", "confidence", "screen_verdict",
                        "escalated", "observed_signs"):
                assert key in b

    def test_openai_backend_requires_key(self, monkeypatch, synthetic_event_dir):
        monkeypatch.setenv("SEIZUREGUARD_BACKEND", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(sys, "argv", ["verify_event.py", str(synthetic_event_dir)])
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            ve.main()


class TestConfidenceSemantics:
    """Regression: final_confidence used to be the max over ALL analyzed
    batches, so a plainly normal event reported 0.85 and a real alert quoted
    0.85 while its only positive batch scored 0.45."""

    def _run(self, monkeypatch, event_dir, verdicts):
        seq = list(verdicts)

        def fake(prompt, model, images, timeout=600):
            return seq.pop(0) if seq else {
                "abnormal_event": False, "confidence": 0.1, "observed_signs": []}

        monkeypatch.setattr(ve, "run_claude", fake)
        return _run_main(monkeypatch, event_dir)

    def test_negative_event_does_not_report_positive_confidence(
            self, monkeypatch, synthetic_event_dir):
        out = self._run(monkeypatch, synthetic_event_dir, [
            {"abnormal_event": False, "confidence": 0.85, "observed_signs": []}])
        assert out["final_abnormal_event"] is False
        assert out["positive_batches"] == 0

    def test_positive_confidence_comes_from_a_positive_batch(
            self, monkeypatch, synthetic_event_dir):
        out = self._run(monkeypatch, synthetic_event_dir, [
            {"abnormal_event": False, "confidence": 0.85, "observed_signs": []},
            {"abnormal_event": True, "confidence": 0.45,
             "observed_signs": [{"sign": "paddling", "present": True,
                                 "body_region": "legs", "sustained": True}]},
        ])
        assert out["final_abnormal_event"] is True
        assert out["final_confidence"] == 0.45      # not the 0.85 negative batch
        assert out["positive_batches"] == 1


class TestSignDefinitionsInPrompt:
    """The false alarms came from bare sign names read in their everyday
    sense; the prompt must carry the clinical definition and its exclusion."""

    def test_confirm_prompt_defines_signs_with_exclusions(self):
        prompt = ve.confirm_prompt([])
        assert "SUDDEN INVOLUNTARY COLLAPSE" in prompt
        assert "NOT lying down" in prompt
        assert "circling before lying down" in prompt      # disorientation
        assert "NOT motion blur" in prompt                 # drooling
        assert "MUST be flagged" in prompt                 # recall side
        assert "Voluntary behaviour is never a sign" in prompt

    def test_every_sign_has_a_definition(self):
        prompt = ve.confirm_prompt([])
        for sign in ve.ALL_SIGNS:
            assert f"- {sign}:" in prompt, f"{sign} tanimsiz"


class TestBackendOutageClassification:
    def test_quota_error_is_an_outage(self):
        assert ve.is_backend_outage("claude CLI error: You've hit your limit · resets Sep 5")

    def test_auth_errors_are_outages(self):
        assert ve.is_backend_outage("401 OAuth access token has expired")
        assert ve.is_backend_outage("Not logged in · Please run /login")

    def test_parse_failure_is_not_an_outage(self):
        assert ve.is_backend_outage("Expecting ',' delimiter: line 1 column 1009") is False
        assert ve.is_backend_outage(None) is False

    def test_reason_taken_from_failed_batches_only(self):
        batches = [
            {"abnormal_event": False, "confidence": 0.1},
            {"abnormal_event": None, "error": "claude CLI error: You've hit your limit"},
            {"abnormal_event": None, "error": "claude CLI error: You've hit your limit"},
        ]
        assert "hit your limit" in ve.outage_reason(batches)

    def test_no_outage_when_failures_are_local(self):
        batches = [{"abnormal_event": None, "error": "JSON parse failed"}]
        assert ve.outage_reason(batches) is None
