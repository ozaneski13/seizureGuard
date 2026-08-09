# EVAL — First real-footage evaluation

**Date:** 2026-08-09 · **Pipeline:** as of the 2026-08-09 NUDFT/segmentation fixes · **Verify backend:** `claude-cli`
(screen `claude-haiku-4-5`, confirm `claude-sonnet-5`)

## Headline

11 labeled public clips (4 seizure, 7 normal) through the real extract + verify
pipeline: **sensitivity 4/4 (no missed seizures), specificity 6/7** — the one
false positive is the hardest negative in the corpus (sleep twitching), and it
is analyzed below.

Two results matter more than the raw score:

- The **mild GTCS** that its source publication documents as *missed by a
  commercial seizure-detection device* was caught here at confidence 0.62.
- The **Lafora myoclonus** clip is a different semiology than GTCS (brief
  jerks, no convulsive sequence) and scored the highest confidence of the
  corpus (0.78) — evidence the sign-based rule layer is not overfit to one
  seizure presentation. Relevant because the target dog's seizures are
  irregular.

## Method

Each clip runs `scripts/extract_event_from_video.py` (motion-peak sampling:
2 fps base + 10 fps bursts, max 60 s window) and then `src/verify_event.py`
end-to-end against live Claude models — the same code path the monitor uses.
Prediction = `final_abnormal_event` from `analysis.json`, compared against the
clip's folder label. Runner: `scripts/eval_clips.py`; raw output in
`data/eval/results.json` (gitignored with the clips; sources + licenses in
`data/eval/SOURCES.txt`, fetchable via `scripts/fetch_eval_clips.py`).

## Corpus

Seizure clips are supplementary videos from peer-reviewed veterinary
publications (CC BY, via the PMC Open Access S3 bucket); normal clips are
Pexels/Wikimedia Commons footage chosen as false-positive traps: vigorous
play-fighting, the rhythmic scratch reflex, and sleep twitching — the classic
false-alarm sources for motion-based detectors.

## Results

| Clip | Label | Predicted | Confidence |
|---|---|---|---|
| GTCS compilation (Frontiers Vet Sci 2022, 4 events) | seizure | **seizure** | 0.40 |
| GTCS intense (Frontiers Vet Sci 2025) | seizure | **seizure** | 0.65 |
| GTCS mild — missed by a commercial detector (Frontiers 2025) | seizure | **seizure** | 0.62 |
| Lafora myoclonus (PLOS ONE 2017) | seizure | **seizure** | 0.78 |
| Labrador walking (Pexels) | normal | normal | 0.05 |
| Pomeranian walking (Pexels) | normal | normal | 0.05 |
| Puppies play-fighting (Commons) | normal | normal | 0.08 |
| Puppies playing in meadow (Pexels) | normal | normal | 0.08 |
| Puppy playing indoors (Commons) | normal | normal | 0.12 |
| Scratch reflex, Irish Wolfhound (Commons) | normal | normal | 0.12 |
| Sleeping dogs twitching (Commons) | normal | **seizure — FALSE POSITIVE** | 0.30 |

Failed batches: 0 across all clips (no fail-open verdicts — every decision was
a real model verdict, not an error fallback).

### The false positive, dissected

The sleep-twitching clip fired through exactly the recall-first mechanisms
built into the pipeline, per its `analysis.json`:

- The screen tier (haiku) answered **"no" on all 7 batches**; two batches
  escalated anyway because screen confidence hit the ≥ 0.15 escalation
  threshold.
- In one escalated batch the confirm model marked a dog stretching while
  waking as `tonic_stiffening` (hard sign, sustained) — even though its own
  free-text note read "consistent with normal sleep transitions and
  stretching". The sign rule deliberately cannot be vetoed by prose.
- In the other, genuine sleep myoclonus was marked `muscle_twitching` and the
  model's own abnormal flag was set.

Ictal-vs-sleep discrimination is a known hard problem even for human
observers on video. Two mitigating observations: the FP confidence (0.30)
sits **below every true-seizure confidence** (min 0.40) — suggestive, but
n=11 is far too small to tune a threshold on; and the operational cost is one
alert with a peak-frame photo attached, triaged by a human in seconds,
whereas the symmetric error (sleeping through a nocturnal seizure) is the
exact failure the recall-first design refuses to risk. Down-weighting
sleep-context batches is *not* a safe fix: seizures frequently start from
sleep.

## Pose gate (validated separately)

The local pose gate was exercised on real footage outside this table, with
honest mixed results:

- **Synthetic 3 Hz injection** (`scripts/augment_seizure.py` shifting the
  YOLO-detected dog region sinusoidally inside real walking footage) scores
  **0.60** → escalate. The non-uniform-DFT scorer sees planted clonus-band
  rhythm through real camera noise.
- **Real gait scores ~0.50**, above the 0.45 threshold — walking is
  rhythmically coherent, so the gate escalates it. Safe (recall preserved)
  but the cost saving currently applies only to non-rhythmic events.
- **Sleeping dogs**: detection rate 0.82 but mean keypoint confidence 0.22 →
  fail-open escalate, as designed for curled-up dogs.

Improvement path is in FOLLOWUPS.md: per-dog threshold calibration and
phase-coherence features (clonus is phase-locked across limbs; gait
alternates).

## Benchmark context

The Epi-Moni wearable collar reports 74.3% sensitivity in its published
validation; the commercial video device referenced above missed the mild GTCS
in this corpus. n=11 does not support a numeric comparison claim, but the
pipeline is not obviously behind the commercial state of the art on the
material it has seen.

## Caveats — read before quoting the numbers

1. **n=11.** One clip flipping changes sensitivity by 25 points. This is a
   smoke test with real data, not a validation study.
2. **Clinical clips ≠ home footage.** Sources are daylight, close-range,
   often clinic settings. The deployment target is a fixed home camera,
   possibly at night. Focal and absence semiologies are absent from the
   corpus entirely.
3. **Per-clip, not per-hour.** This measures classification of pre-cut
   events, not false alarms per day of continuous monitoring — that number
   needs long-duration footage through the motion trigger.
4. **Confidences are model self-reports**, not calibrated probabilities.
   Compare within this table only.
5. The original run extracted two portrait clips (Labrador, Pomeranian
   walking) through a since-fixed aspect-distorting resize; both were re-run
   with correct aspect after the fix (results above are the clean re-runs;
   the distorted runs were also correctly negative).
