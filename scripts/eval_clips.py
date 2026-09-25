"""Recall/false-positive evaluation over labeled video clips.

Layout:
    data/eval/seizure/*.mp4   clips that contain a seizure-like event
    data/eval/normal/*.mp4    clips of normal dog behavior

Each clip runs through the real extract + verify pipeline; the prediction is
compared against its folder label. Results (per-clip table + sensitivity /
specificity) go to stdout and <eval_root>/results.json.

A clip that was not fully verified (failed batches or a backend outage with
no positive batch, or no analysis.json at all) is "unanalyzed": it is listed
separately and kept out of TP/FN/TN/FP. Counting it as a negative would
inflate specificity with clips nobody looked at.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from extract_event_from_video import extract_event  # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".ogv"}


def evaluate_clip(video, workdir, verify_cmd=None):
    event_dir = Path(workdir) / f"{video.stem}_event"
    stats = extract_event(video, event_dir)
    # The work dir is reused across runs: a verify that writes nothing must
    # leave no analysis.json to read, so the clip becomes unanalyzed instead
    # of being scored from an earlier run.
    (event_dir / "analysis.json").unlink(missing_ok=True)
    cmd = list(verify_cmd) if verify_cmd else [
        sys.executable, str(REPO / "src" / "verify_event.py")]
    subprocess.run(cmd + [str(event_dir)], capture_output=True, timeout=3600)
    analysis = json.loads((event_dir / "analysis.json").read_text())
    predicted = bool(analysis.get("final_abnormal_event"))
    # a positive verdict stands; a negative one only counts if every batch ran
    analyzed = predicted or not (analysis.get("failed_batches")
                                 or analysis.get("backend_outage"))
    return {
        "clip": video.name,
        "predicted": predicted if analyzed else None,
        "analyzed": analyzed,
        "confidence": analysis.get("final_confidence", 0.0),
        "failed_batches": analysis.get("failed_batches", 0),
        "frames": stats["base"] + stats["burst"],
        "window": stats["window"],
    }


def summarize(rows):
    """rows: [{expected: bool, predicted: bool, analyzed: bool, ...}] -> metrics
    dict. Rows with analyzed=False are reported apart, never scored."""
    unanalyzed = [r for r in rows if r.get("analyzed") is False]
    rows = [r for r in rows if r.get("analyzed") is not False]
    tp = sum(1 for r in rows if r["expected"] and r["predicted"])
    fn = sum(1 for r in rows if r["expected"] and not r["predicted"])
    tn = sum(1 for r in rows if not r["expected"] and not r["predicted"])
    fp = sum(1 for r in rows if not r["expected"] and r["predicted"])
    return {
        "clips": len(rows) + len(unanalyzed),
        "unanalyzed": len(unanalyzed),
        "unanalyzed_clips": [r.get("clip") for r in unanalyzed],
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
    }


def main():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", default="data/eval", type=Path)
    args = parser.parse_args()

    rows = []
    for label, expected in (("seizure", True), ("normal", False)):
        folder = args.eval_root / label
        if not folder.is_dir():
            continue
        for video in sorted(folder.iterdir()):
            if video.suffix.lower() not in VIDEO_EXTS:
                continue
            print(f"[EVAL] {label}/{video.name} ...", flush=True)
            try:
                row = evaluate_clip(video, args.eval_root / "work")
            except Exception as e:
                print(f"[WARN] {video.name} failed: {e}", flush=True)
                row = {"clip": video.name, "predicted": None, "analyzed": False,
                       "error": str(e)}
            row["expected"] = expected
            row["label"] = label
            rows.append(row)
            if not row["analyzed"]:
                continue
            mark = "OK " if row["predicted"] == expected else "MISS"
            print(f"[EVAL] {mark} {label}/{video.name}: predicted="
                  f"{row['predicted']} conf={row['confidence']:.2f} "
                  f"failed_batches={row['failed_batches']}", flush=True)

    if not rows:
        print(f"No eval clips found under {args.eval_root}/(seizure|normal)/")
        raise SystemExit(1)

    metrics = summarize(rows)
    out = {"metrics": metrics, "clips": rows}
    (args.eval_root / "results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== RESULTS ===")
    for r in rows:
        if not r["analyzed"]:
            print(f"  UNAN {r['label']:8s} {r['clip']:40s} "
                  f"failed_batches={r.get('failed_batches', '-')} {r.get('error', '')}")
            continue
        mark = "ok  " if r["predicted"] == r["expected"] else "MISS"
        print(f"  {mark} {r['label']:8s} {r['clip']:40s} "
              f"pred={r['predicted']} conf={r['confidence']:.2f}")
    sens = metrics["sensitivity"]
    spec = metrics["specificity"]
    print(f"clips={metrics['clips']}  unanalyzed={metrics['unanalyzed']}  "
          f"sensitivity={sens if sens is None else round(sens, 3)}  "
          f"specificity={spec if spec is None else round(spec, 3)}")
    if metrics["unanalyzed"]:
        print("unanalyzed (not scored):", ", ".join(metrics["unanalyzed_clips"]))
    print("written:", args.eval_root / "results.json")


if __name__ == "__main__":
    main()
