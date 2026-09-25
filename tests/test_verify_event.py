import json
import shutil
import sys
import types
from pathlib import Path

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
        assert [len(b) for b in batches] == [30, 30, 30]
        assert batches[:2] == [items[:30], items[30:60]]
        assert set(x for b in batches for x in b) == set(items)

    def test_short_tail_is_never_a_batch_of_its_own(self):
        """Regression (2026-09-25 22:06): a 3-frame tail, 0.2 s with no
        context, turned a dog's back-roll into an alert. The tail now takes
        the last full batch, overlapping the one before."""
        items = list(range(63))
        batches = list(ve.iter_batches(items, 30))
        assert batches[-1] == items[33:63]
        assert len(batches) == 3

    def test_short_events_and_exact_multiples_are_unchanged(self):
        assert list(ve.iter_batches(list(range(10)), 30)) == [list(range(10))]
        assert [len(b) for b in ve.iter_batches(list(range(60)), 30)] == [30, 30]

    def test_prompt_states_the_duration_rule(self):
        prompt = ve.confirm_prompt([])
        assert "less than about 2 seconds" in prompt
        assert "on-screen clock is not evidence" in prompt

    def test_parse_json_verdict_plain(self):
        assert ve.parse_json_verdict('{"abnormal_event": false, "confidence": 0.1}') == {
            "abnormal_event": False, "confidence": 0.1}

    def test_parse_json_verdict_fenced(self):
        text = 'Here you go:\n```json\n{"abnormal_event": true, "confidence": 0.8}\n```'
        assert ve.parse_json_verdict(text)["abnormal_event"] is True

    def test_parse_json_verdict_prose_wrapped(self):
        text = 'The answer is {"abnormal_event": false, "confidence": 0.2} as requested.'
        assert ve.parse_json_verdict(text)["abnormal_event"] is False

    def test_parse_json_verdict_garbage_raises(self):
        with pytest.raises(ValueError):
            ve.parse_json_verdict("I cannot read these images")


def _confirm_reply(present=(), abnormal=False, indent=None,
                   note='dog on its side, legs "paddling" rhythmically'):
    """A full 10-sign confirm reply the way the model writes it: the note,
    the schema's last member, carries unescaped inner quotes."""
    signs = [{"sign": s, "present": s in present,
              "body_region": "legs" if s in present else "none",
              "sustained": s in present} for s in ve.ALL_SIGNS]
    body = json.dumps({"abnormal_event": abnormal, "confidence": 0.85,
                       "posture": "lying_lateral", "partially_visible": False,
                       "observed_signs": signs}, indent=indent)
    sep = "\n  " if indent else " "
    return body[:-1].rstrip() + f',{sep}"note": "{note}"' + ("\n}" if indent else "}")


def _note_first_reply(present=()):
    """The top-level note, with inner quotes, placed ahead of observed_signs."""
    signs = [{"sign": s, "present": s in present, "body_region": "legs",
              "sustained": False} for s in ve.ALL_SIGNS]
    return ('{"abnormal_event": false, "confidence": 0.6, "posture": "lying_lateral", '
            '"partially_visible": false, "note": "legs "paddling" while lying", '
            f'"observed_signs": {json.dumps(signs)}}}')


class TestTolerantParse:
    """Regression: a strict parse over the whole reply lost confirm verdicts
    to a flaw in the free-text note (live 2026-09-02 and 2026-09-25: both
    attempts failed with "Expecting ',' delimiter", a positive batch among them)."""

    def test_inner_quotes_in_note_keep_a_positive_verdict(self):
        text = _confirm_reply(present=("paddling", "loss_of_posture"), abnormal=True)
        with pytest.raises(json.JSONDecodeError):
            json.loads(text)                       # the realistic failure
        v = ve.parse_json_verdict(text)
        assert v["abnormal_event"] is True
        assert ve.decide_signs(v) is True
        assert len(v["observed_signs"]) == len(ve.ALL_SIGNS)
        assert not v.get("salvaged")

    def test_inner_quotes_in_note_keep_a_negative_verdict(self):
        text = _confirm_reply(indent=2, note='dog "settling" down to rest')
        v = ve.parse_json_verdict(text)
        assert v["abnormal_event"] is False
        assert ve.decide_signs(v) is False

    def test_note_before_the_signs_is_not_cut_into_a_clean_negative(self):
        # A broken note ahead of observed_signs: cutting at it leaves a
        # sign-less verdict that reads as a clean negative while paddling
        # is marked present further on.
        text = _note_first_reply(present=("paddling",))
        v = ve.parse_json_verdict(text)
        assert ve.decide_signs(v) is True
        assert v["salvaged"] is True

    def test_note_before_the_signs_on_a_negative_is_a_failure(self):
        # Nothing positive to salvage, and the cut lost the signs: unverified,
        # never a silent negative.
        with pytest.raises(ValueError):
            ve.parse_json_verdict(_note_first_reply())

    def test_only_the_trailing_note_is_cut(self):
        # Sign entries carrying their own note must not be cut through.
        signs = [{"sign": s, "present": False, "note": "still"} for s in ve.ALL_SIGNS]
        body = json.dumps({"abnormal_event": False, "confidence": 0.4,
                           "observed_signs": signs})
        text = body[:-1] + ', "note": "dog "settling" down"}'
        v = ve.parse_json_verdict(text)
        assert v["abnormal_event"] is False
        assert len(v["observed_signs"]) == len(ve.ALL_SIGNS)
        assert not v.get("salvaged")

    def test_prose_with_a_brace_before_the_object(self):
        # Every '{' is tried, not just the first.
        text = "Checked {all 10 signs}:\n" + json.dumps(
            {"abnormal_event": False, "confidence": 0.3, "observed_signs": []})
        v = ve.parse_json_verdict(text)
        assert v["abnormal_event"] is False
        assert "salvaged" not in v

    def test_trailing_prose_with_braces(self):
        body = json.dumps({"abnormal_event": True, "confidence": 0.7,
                           "observed_signs": []})
        text = body + "\n\n(Signs evaluated: {paddling, jaw_clonus}.)"
        assert ve.parse_json_verdict(text)["abnormal_event"] is True

    def test_nested_sign_object_is_not_taken_for_the_verdict(self):
        # Broken outer object, parseable inner sign entries: returning a sign
        # entry would read as a clean negative.
        text = _confirm_reply(present=("drooling",)).replace(
            '"posture": "lying_lateral"', '"posture": "lying "lateral""')
        with pytest.raises(ValueError):
            ve.parse_json_verdict(text)

    def test_object_without_a_verdict_is_a_failure_not_a_negative(self):
        with pytest.raises(ValueError):
            ve.parse_json_verdict('{"error": "images unreadable"}')

    def test_unparseable_positive_is_salvaged(self):
        # The quote damage sits inside a sign entry, so dropping the note
        # does not help; the hard sign is still readable.
        text = _confirm_reply(present=("paddling",)).replace(
            '"body_region": "legs"', '"body_region": "front "left" leg"')
        v = ve.parse_json_verdict(text)
        assert v["salvaged"] is True
        assert ve.decide_signs(v) is True
        assert v["confidence"] <= 0.1
        assert text[:100] in v["raw_reply"]

    def test_truncated_reply_with_abnormal_flag_is_salvaged(self):
        text = '{"abnormal_event": true, "confidence": 0.9, "observed_signs": [{"sign": "padd'
        v = ve.parse_json_verdict(text)
        assert v["salvaged"] is True
        assert ve.decide_signs(v) is True

    def test_later_positive_verdict_is_not_lost_to_a_quoted_negative(self):
        # The prompt carries a literal no-dog example object; a reply that
        # quotes it before its real answer must not read as a clean negative.
        pos = {"abnormal_event": True, "confidence": 0.8,
               "observed_signs": [{"sign": "paddling", "present": True}]}
        text = ('{"abnormal_event": false, "confidence": 0.0, "observed_signs": [], '
                '"note": "no dog visible"} does not apply - corrected: ' + json.dumps(pos))
        v = ve.parse_json_verdict(text)
        assert ve.decide_signs(v) is True
        assert v["confidence"] == 0.8
        assert v["multiple_verdicts"] is True

    def test_positive_verdict_wins_whatever_its_position(self):
        pos = json.dumps({"abnormal_event": True, "confidence": 0.7, "observed_signs": []})
        neg = json.dumps({"abnormal_event": False, "confidence": 0.9, "observed_signs": []})
        v = ve.parse_json_verdict(pos + "\n" + neg)
        assert v["abnormal_event"] is True
        assert v["multiple_verdicts"] is True

    def test_several_negative_verdicts_use_the_last(self):
        first = json.dumps({"abnormal_event": False, "confidence": 0.2, "observed_signs": []})
        last = json.dumps({"abnormal_event": False, "confidence": 0.6,
                           "observed_signs": [{"sign": "drooling", "present": True}]})
        v = ve.parse_json_verdict(first + " draft; final: " + last)
        assert ve.decide_signs(v) is False
        assert v["confidence"] == 0.6
        assert v["multiple_verdicts"] is True

    def test_later_positive_that_does_not_decode_is_salvaged(self):
        # The quoted negative decodes; the corrected answer carries the usual
        # inner quotes in its note and does not.
        text = ('{"abnormal_event": false, "confidence": 0.0, "observed_signs": [], '
                '"note": "no dog visible"} does not apply - corrected: '
                + _confirm_reply(present=("paddling",)))
        v = ve.parse_json_verdict(text)
        assert ve.decide_signs(v) is True
        assert v["salvaged"] is True

    def test_sign_entries_inside_one_verdict_are_not_extra_verdicts(self):
        v = ve.parse_json_verdict(_confirm_reply(present=("paddling",), abnormal=True))
        assert "multiple_verdicts" not in v

    def test_unsalvageable_failure_keeps_the_raw_reply(self):
        text = _confirm_reply(present=("drooling",)).replace(
            '"body_region": "legs"', '"body_region": "the "muzzle""')
        with pytest.raises(ValueError) as exc:
            ve.parse_json_verdict(text)
        assert "Expecting" in str(exc.value)
        assert text[:1500] in str(exc.value)


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

    @pytest.mark.parametrize("signs", [7, 0.5, True, "paddling", {"paddling": True},
                                       [{"sign": ["paddling"], "present": True}]])
    def test_odd_observed_signs_shape_does_not_raise(self, signs):
        # the parser runs this on every verdict; the shape itself is judged
        # in parse_json_verdict (TestUnreadableSigns)
        ve.decide_signs({"abnormal_event": False, "observed_signs": signs})

    def test_present_entry_without_a_sign_name_still_counts(self):
        verdict = {"abnormal_event": False, "observed_signs": [
            {"present": True, "body_region": "hind legs"},
            {"sign": "drooling", "present": True}]}
        assert ve.decide_signs(verdict) is True

    def test_each_present_entry_without_a_sign_name_counts(self):
        verdict = {"abnormal_event": False, "observed_signs": [
            {"present": True, "body_region": "hind legs"},
            {"sign": ["drooling"], "present": True}]}
        assert ve.decide_signs(verdict) is True

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
            self._stream('{"abnormal_event": false, "confidence": 0.1}')))
        out = ve.run_claude("prompt", "haiku", images=[])
        assert out == {"abnormal_event": False, "confidence": 0.1}

    def test_fenced_result_is_cleaned(self, monkeypatch):
        monkeypatch.setattr(ve.subprocess, "run", lambda *a, **k: self._proc(
            self._stream('```json\n{"abnormal_event": true, "confidence": 0.9}\n```')))
        assert ve.run_claude("p", "m", images=[])["abnormal_event"] is True

    def test_images_are_embedded_as_base64(self, monkeypatch, tmp_path):
        frame = tmp_path / "frame_000_t_1.000s.jpg"
        frame.write_bytes(b"\xff\xd8fakejpg")
        captured = {}

        def fake(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["stdin"] = kwargs.get("input")
            return self._proc(self._stream('{"abnormal_event": false, "confidence": 0.0}'))

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

    def test_parse_offset_containing_429_is_not_an_outage(self):
        # Regression: the bare "429" marker matched JSON offsets.
        err = "Expecting ',' delimiter: line 1 column 1430 (char 1429)"
        assert ve.is_backend_outage(err) is False
        assert ve.is_backend_outage("Expecting ',' delimiter: line 9 column 5 (char 429)",
                                    kind="parse") is False

    def test_bare_429_outside_cli_text_is_not_an_outage(self):
        # "429" as a JSON offset, where only the CLI-text guard stops it.
        err = "Expecting ',' delimiter: line 1 column 430 (char 429)"
        assert ve.is_backend_outage(err) is False
        assert ve.is_backend_outage(err, kind="call") is False

    def test_parse_kind_is_never_an_outage(self):
        err = "Unparseable model reply: I can't authenticate this; quota of frames overloaded"
        assert ve.is_backend_outage(err, kind="parse") is False
        batches = [{"abnormal_event": None, "error": err, "error_kind": "parse"}]
        assert ve.outage_reason(batches) is None

    def test_cli_rate_limits_are_outages(self):
        assert ve.is_backend_outage(
            'claude CLI error: API Error: 429 {"type":"error","error":'
            '{"type":"rate_limit_error"}}', kind="call")
        assert ve.is_backend_outage("Error code: 429 - Too Many Requests", kind="call")
        assert ve.is_backend_outage("claude CLI error: You've hit your limit", kind="call")

    def test_crash_kind_is_not_an_outage(self):
        batches = [{"abnormal_event": None, "error_kind": "crash",
                    "error": "TypeError: 'NoneType' has no attribute 'quota'"}]
        assert ve.outage_reason(batches) is None

    def test_legacy_batches_without_kind_still_classify(self):
        batches = [{"abnormal_event": None,
                    "error": "claude CLI error: You've hit your limit"}]
        assert "hit your limit" in ve.outage_reason(batches)

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


def _cli_reply(result_text):
    """subprocess.run stand-in: the CLI answered with this model text."""
    stdout = json.dumps({"type": "result", "is_error": False, "result": result_text})
    return lambda *a, **k: types.SimpleNamespace(stdout=stdout, stderr="", returncode=0)


class TestParseFailureRetry:
    """Regression: the 2nd attempt resent the identical prompt, so a model
    habit (inner quotes) failed both attempts the same way."""

    def _prompts(self, monkeypatch, first_error):
        prompts = []

        def fake(prompt, model, images, timeout=600):
            prompts.append(prompt)
            if len(prompts) == 1:
                raise first_error
            return {"abnormal_event": False, "confidence": 0.2, "observed_signs": []}

        monkeypatch.setattr(ve, "run_claude", fake)
        r = ve.assess_batch_claude(None, [], ve.get_config(), 1)
        assert r["abnormal_event"] is False
        return prompts

    def test_retry_after_parse_failure_asks_for_repair(self, monkeypatch):
        err = ve.VerdictParseError("Expecting ',' delimiter: line 1 column 1047 (char 1046)",
                                   '{"abnormal_event": false, "note": "a "b""}')
        first, second = self._prompts(monkeypatch, err)
        assert second != first
        assert second.startswith(first)
        assert "Expecting ',' delimiter: line 1 column 1047" in second
        assert "never put double quotes inside string values" in second.lower()
        assert '"note": "a "b""' not in second      # the reason, not the whole reply

    def test_retry_after_call_failure_is_unchanged(self, monkeypatch):
        first, second = self._prompts(monkeypatch, RuntimeError("claude CLI error: boom"))
        assert second == first

    def test_failed_batch_keeps_raw_reply_and_parse_kind(self, monkeypatch):
        text = _confirm_reply(present=("drooling",)).replace(
            '"body_region": "legs"', '"body_region": "the "muzzle""')
        monkeypatch.setattr(ve.subprocess, "run", _cli_reply(text))
        r = ve.assess_batch_claude(None, [], ve.get_config(), 1)
        assert r["abnormal_event"] is None
        assert r["error_kind"] == "parse"
        assert text[:1500] in r["error"]

    def test_prose_reply_is_a_parse_failure_not_an_outage(self, monkeypatch):
        monkeypatch.setattr(ve.subprocess, "run", _cli_reply(
            "I can't authenticate what is happening here - the quota of usable "
            "frames is low and the scene is overloaded, but the dog is thrashing."))
        r = ve.assess_batch_claude(None, [], ve.get_config(), 1)
        assert r["abnormal_event"] is None
        assert r["error_kind"] == "parse"
        assert ve.outage_reason([r]) is None

    def test_cli_outage_is_still_an_outage(self, monkeypatch):
        stdout = json.dumps({"type": "result", "is_error": True,
                             "result": "You've hit your limit · resets Sep 5"})
        monkeypatch.setattr(ve.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
            stdout=stdout, stderr="", returncode=1))
        r = ve.assess_batch_claude(None, [], ve.get_config(), 1)
        assert r["error_kind"] == "call"
        assert "hit your limit" in ve.outage_reason([r])

    def test_multiple_verdicts_mark_the_batch(self, monkeypatch):
        text = (json.dumps({"abnormal_event": False, "confidence": 0.0, "observed_signs": []})
                + " corrected: "
                + json.dumps({"abnormal_event": False, "confidence": 0.8, "observed_signs": [
                    {"sign": "tonic_stiffening", "present": True}]}))
        monkeypatch.setattr(ve.subprocess, "run", _cli_reply(text))
        r = ve.assess_batch_claude(None, [], ve.get_config(), 1)
        assert r["abnormal_event"] is True
        assert r["multiple_verdicts"] is True

    def test_salvaged_positive_marks_the_batch(self, monkeypatch):
        text = _confirm_reply(present=("paddling",)).replace(
            '"body_region": "legs"', '"body_region": "front "left" leg"')
        monkeypatch.setattr(ve.subprocess, "run", _cli_reply(text))
        r = ve.assess_batch_claude(None, [], ve.get_config(), 1)
        assert r["abnormal_event"] is True
        assert r["salvaged"] is True
        assert text[:100] in r["raw_reply"]


UNREADABLE_SIGNS = [7, True, "paddling", {"paddling": True},
                    [{"sign": ["paddling"], "present": True}],
                    [{"sign": None, "present": True}]]


class TestUnreadableSigns:
    """Regression: sign evidence the rule layer cannot read (a sign name in
    a list, observed_signs that is no list) was scored as a clean negative;
    it must fail the batch, which alerts UNVERIFIED, or count as a sign."""

    @pytest.mark.parametrize("signs", UNREADABLE_SIGNS)
    def test_unreadable_signs_on_a_negative_are_a_failure(self, signs):
        text = json.dumps({"abnormal_event": False, "confidence": 0.3, "observed_signs": signs})
        with pytest.raises(ve.VerdictParseError):
            ve.parse_json_verdict(text)

    @pytest.mark.parametrize("signs", UNREADABLE_SIGNS)
    def test_unreadable_signs_on_a_positive_are_kept(self, signs):
        text = json.dumps({"abnormal_event": True, "confidence": 0.3, "observed_signs": signs})
        assert ve.decide_signs(ve.parse_json_verdict(text)) is True

    def test_nameless_present_sign_keeps_the_event_positive(
            self, monkeypatch, synthetic_event_dir):
        reply = json.dumps({"abnormal_event": False, "confidence": 0.3, "observed_signs": [
            {"present": True, "body_region": "hind legs"},
            {"sign": "drooling", "present": True}]})
        monkeypatch.setattr(ve, "run_claude", _fake_run_claude(
            lambda p: ve.parse_json_verdict(reply)))
        out = _run_main(monkeypatch, synthetic_event_dir)
        assert out["final_abnormal_event"] is True
        assert out["failed_batches"] == 0

    def test_unreadable_sign_fails_the_batch_not_a_negative(
            self, monkeypatch, synthetic_event_dir):
        reply = json.dumps({"abnormal_event": False, "confidence": 0.3,
                            "observed_signs": [{"sign": ["paddling"], "present": True}]})
        prompts = []

        def confirm(prompt):
            prompts.append(prompt)
            return ve.parse_json_verdict(reply)

        monkeypatch.setattr(ve, "run_claude", _fake_run_claude(confirm))
        out = _run_main(monkeypatch, synthetic_event_dir)
        assert out["failed_batches"] == len(out["batches"]) > 0
        assert all(b["error_kind"] == "parse" for b in out["batches"])
        assert out["backend_outage"] is None
        assert "never put double quotes" in prompts[1].lower()     # repair retry ran


class TestValueCoercion:
    """Regression: a null or non-numeric confidence crashed verify_event
    after the verdict was in, losing every batch including positives."""

    @pytest.mark.parametrize("raw,expected", [
        (None, 0.0), ("high", 0.0), ([0.5], 0.0), (float("nan"), 0.0),
        ("0.8", 0.8), (1.7, 1.0), (-0.3, 0.0), (0.45, 0.45),
    ])
    def test_claude_confidence_is_coerced(self, monkeypatch, raw, expected):
        monkeypatch.setattr(ve, "run_claude", lambda *a, **k: {
            "abnormal_event": True, "confidence": raw, "observed_signs": []})
        r = ve.assess_batch_claude(None, [], ve.get_config(), 1)
        assert r["abnormal_event"] is True
        assert r["confidence"] == expected

    def test_openai_confidence_is_coerced(self, monkeypatch):
        monkeypatch.setattr(ve, "ask_openai", lambda *a, **k: {
            "abnormal_event": True, "confidence": None, "observed_signs": []})
        r = ve.assess_batch_openai(None, None, [], ve.get_config(), 1)
        assert r["abnormal_event"] is True
        assert r["confidence"] == 0.0


    def test_openai_decode_error_is_a_parse_failure(self, monkeypatch):
        def bad_json(*a, **k):
            raise json.JSONDecodeError("Expecting value", "doc", 429)
        monkeypatch.setattr(ve, "ask_openai", bad_json)
        r = ve.assess_batch_openai(None, None, [], ve.get_config(), 1)
        assert r["error_kind"] == "parse"
        assert ve.outage_reason([r]) is None

class TestBatchGuard:
    def test_unexpected_batch_crash_keeps_other_batches(
            self, monkeypatch, synthetic_event_dir):
        def assess(event_dir, paths, config, bi):
            if bi == 2:
                raise TypeError("unexpected")
            return {"abnormal_event": True, "confidence": 0.9, "screen_verdict": None,
                    "escalated": True, "observed_signs": []}

        monkeypatch.setattr(ve, "assess_batch_claude", assess)
        out = _run_main(monkeypatch, synthetic_event_dir)
        assert out["final_abnormal_event"] is True
        assert out["failed_batches"] == 1
        crashed = out["batches"][1]
        assert crashed["abnormal_event"] is None
        assert crashed["error_kind"] == "crash"
        assert "TypeError" in crashed["error"]
        assert out["backend_outage"] is None


class TestLoginError:
    """Regression: a login error aborted the run before analysis.json was
    written, so the monitor saw a one-off failure and alerted on every event
    separately instead of announcing one outage."""

    def _run(self, monkeypatch, event_dir, assess):
        (event_dir / "analysis.json").unlink(missing_ok=True)
        monkeypatch.setattr(ve, "assess_batch_claude", assess)
        monkeypatch.setattr(sys, "argv", ["verify_event.py", str(event_dir)])
        with pytest.raises(ve.ClaudeLoginError):       # still a non-zero exit
            ve.main()
        return json.loads((event_dir / "analysis.json").read_text())

    def test_login_error_is_written_as_an_outage(self, monkeypatch, synthetic_event_dir):
        calls = []

        def assess(event_dir, paths, config, bi):
            calls.append(bi)
            raise ve.ClaudeLoginError("claude CLI is not logged in - run /login")

        out = self._run(monkeypatch, synthetic_event_dir, assess)
        assert calls == [1]                  # no model call after a login error
        assert len(out["batches"]) > 1
        assert out["failed_batches"] == len(out["batches"])
        assert all(b["abnormal_event"] is None and b["error_kind"] == "call"
                   for b in out["batches"])
        assert out["final_abnormal_event"] is False
        assert "not logged in" in out["backend_outage"]
        assert out["complete"] is True       # a rerun must not resume it

    def test_positive_before_the_login_error_is_kept(self, monkeypatch, synthetic_event_dir):
        def assess(event_dir, paths, config, bi):
            if bi == 1:
                return {"abnormal_event": True, "confidence": 0.9, "screen_verdict": None,
                        "escalated": True, "observed_signs": []}
            raise ve.ClaudeLoginError("claude CLI is not logged in - run /login")

        out = self._run(monkeypatch, synthetic_event_dir, assess)
        assert out["final_abnormal_event"] is True
        assert out["batches"][0]["abnormal_event"] is True
        assert out["failed_batches"] == len(out["batches"]) - 1

    def test_login_reason_wins_over_an_earlier_outage(self, monkeypatch, synthetic_event_dir):
        # The outage notice must say a human has to /login, not to wait out
        # the usage limit the earlier batches hit.
        def assess(event_dir, paths, config, bi):
            if bi <= 4:
                return ve.failed_batch("claude CLI error: You've hit your limit", ve.ERR_CALL)
            raise ve.ClaudeLoginError("claude CLI is not logged in - run /login")

        out = self._run(monkeypatch, synthetic_event_dir, assess)
        limited = sum("hit your limit" in b["error"] for b in out["batches"])
        assert limited > len(out["batches"]) - limited > 0    # the majority reason
        assert "not logged in" in out["backend_outage"]


def test_analysis_json_is_replaced_atomically(monkeypatch, synthetic_event_dir):
    """The monitor reuses an existing analysis.json, so it must never see a
    half-written one: the file is written beside it, then renamed over it."""
    replaced = []
    real_replace = ve.os.replace

    def spy(src, dst):
        replaced.append((Path(src), Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(ve.os, "replace", spy)
    monkeypatch.setattr(ve, "run_claude", _fake_run_claude(
        lambda p: {"abnormal_event": False, "confidence": 0.1, "observed_signs": []}))
    out = _run_main(monkeypatch, synthetic_event_dir)
    target = synthetic_event_dir / "analysis.json"
    assert len(replaced) == len(out["batches"])      # once per batch, see TestResume
    for src, dst in replaced:
        assert dst == target and src.parent == synthetic_event_dir and src != target
        assert not src.exists()
    assert out["complete"] is True
    assert out["final_abnormal_event"] is False


class _Killed(BaseException):
    """Stands in for systemd killing the run: nothing in main() catches it."""


def _batch(abnormal, confidence=0.5):
    signs = [{"sign": "paddling", "present": True}] if abnormal else []
    return {"abnormal_event": abnormal, "confidence": confidence, "screen_verdict": None,
            "escalated": True, "observed_signs": signs}


class TestResume:
    """Regression: analysis.json was written once, after the last batch, so a
    restart mid-run lost every verdict so far, and the rerun could answer
    negative where the killed run had already found a positive."""

    @pytest.fixture
    def event_dir(self, synthetic_event_dir, tmp_path):
        d = tmp_path / synthetic_event_dir.name
        shutil.copytree(synthetic_event_dir, d)
        (d / "analysis.json").unlink(missing_ok=True)
        return d

    def _read(self, event_dir):
        return json.loads((event_dir / "analysis.json").read_text())

    def _kill_after(self, monkeypatch, event_dir, n, verdict_for):
        def assess(event_dir_, paths, config, bi):
            if bi > n:
                raise _Killed()
            return verdict_for(bi)

        monkeypatch.setattr(ve, "assess_batch_claude", assess)
        monkeypatch.setattr(sys, "argv", ["verify_event.py", str(event_dir)])
        with pytest.raises(_Killed):
            ve.main()
        return self._read(event_dir)

    def _rerun(self, monkeypatch, event_dir, verdict_for=lambda bi: _batch(False, 0.9)):
        calls = []

        def assess(event_dir_, paths, config, bi):
            calls.append(bi)
            return verdict_for(bi)

        monkeypatch.setattr(ve, "assess_batch_claude", assess)
        return _run_main(monkeypatch, event_dir), calls

    def test_partial_file_after_every_batch(self, monkeypatch, event_dir):
        out = self._kill_after(monkeypatch, event_dir, 2,
                               lambda bi: _batch(bi == 1, 0.8))
        assert out["complete"] is False
        assert len(out["batches"]) == 2
        # a positive already found is on disk as a positive
        assert out["final_abnormal_event"] is True
        assert out["positive_batches"] == 1 and out["final_confidence"] == 0.8

    def test_partial_negative_says_it_is_incomplete(self, monkeypatch, event_dir):
        out = self._kill_after(monkeypatch, event_dir, 1, lambda bi: _batch(False, 0.9))
        assert out["complete"] is False
        assert out["final_abnormal_event"] is False
        assert "incomplete" in out["final_reason"]

    def test_rerun_resumes_and_keeps_the_positive(self, monkeypatch, event_dir):
        self._kill_after(monkeypatch, event_dir, 2, lambda bi: _batch(bi == 1, 0.8))
        out, calls = self._rerun(monkeypatch, event_dir)
        total = len(out["batches"])
        assert total >= 4
        assert calls == list(range(3, total + 1))     # batches 1-2 not asked again
        assert out["complete"] is True
        assert out["final_abnormal_event"] is True
        assert out["batches"][0]["abnormal_event"] is True

    def test_failed_batches_are_asked_again(self, monkeypatch, event_dir):
        def first(bi):
            return _batch(False, 0.9) if bi == 1 else ve.failed_batch("boom", ve.ERR_CALL)

        self._kill_after(monkeypatch, event_dir, 2, first)
        out, calls = self._rerun(monkeypatch, event_dir)
        assert calls[0] == 2
        assert out["failed_batches"] == 0

    def test_other_frames_start_fresh(self, monkeypatch, event_dir):
        self._kill_after(monkeypatch, event_dir, 2, lambda bi: _batch(bi == 1, 0.8))
        sorted((event_dir / "burst").glob("frame_*.jpg"))[0].unlink()
        out, calls = self._rerun(monkeypatch, event_dir)
        assert calls[:2] == [1, 2]
        assert out["final_abnormal_event"] is False

    def test_other_batch_size_starts_fresh(self, monkeypatch, event_dir):
        # The recorded verdicts cover other frames once the batches are cut
        # differently; reusing them would leave frames never looked at.
        self._kill_after(monkeypatch, event_dir, 2, lambda bi: _batch(bi == 1, 0.8))
        monkeypatch.setattr(ve, "BATCH_SIZE", 10)
        out, calls = self._rerun(monkeypatch, event_dir)
        assert calls[:2] == [1, 2]
        assert out["batch_size"] == 10

    def test_login_error_on_rerun_keeps_a_later_recorded_positive(self, monkeypatch, event_dir):
        # Batch 1 crashed, batch 2 found a positive, then the run was killed.
        # The rerun asks batch 1 again and hits a login error: batch 2's
        # positive must survive it, not turn into a failed batch.
        self._kill_after(monkeypatch, event_dir, 2,
                         lambda bi: (ve.failed_batch("boom", ve.ERR_CRASH) if bi == 1
                                     else _batch(True, 0.8)))

        def login_error(bi):
            raise ve.ClaudeLoginError("Not logged in")

        with pytest.raises(ve.ClaudeLoginError):
            self._rerun(monkeypatch, event_dir, login_error)
        out = self._read(event_dir)
        assert out["complete"] is True
        assert out["batches"][1]["abnormal_event"] is True
        assert out["final_abnormal_event"] is True

    def test_complete_file_starts_fresh(self, monkeypatch, event_dir):
        self._rerun(monkeypatch, event_dir, lambda bi: _batch(bi == 1, 0.8))
        out, calls = self._rerun(monkeypatch, event_dir)
        assert calls == list(range(1, len(out["batches"]) + 1))
        assert out["final_abnormal_event"] is False

    def test_unreadable_partial_file_starts_fresh(self, monkeypatch, event_dir):
        (event_dir / "analysis.json").write_text("{truncated", encoding="utf-8")
        out, calls = self._rerun(monkeypatch, event_dir)
        assert calls == list(range(1, len(out["batches"]) + 1))
