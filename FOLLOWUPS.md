# FOLLOWUPS

Open items and next steps, ordered by value. Status as of 2026-08-09.

## Needs real footage (blocked on data, not code)

- **Re-test on the original camera export of the first captured seizure.**
  The first owner recording (a screen recording of app playback) fired the
  alert, but forensics showed fragile detection: the convulsing dog sits at
  the frame edge partially cropped by app UI, playback has multi-second time
  jumps (camera clock jumped 6 s within 0.3 s of video), and both VLM tiers
  misread the on-side convulsion as "normal repositioning/walking" when
  sampled densely. A direct export from the camera app (no screen recording,
  no WhatsApp re-compression) is the fair test. Deployment guidance from the
  same footage: place the camera so resting spots are centered, and mind the
  mirrored furniture or glass — the dog's reflection is a plausible VLM
  distractor.
  Upside: this video exposed the pose-gate fail-open hole (fixed the same
  day: detection-rate and unscorable-segment fail-open branches).
  A same-frames model A/B then showed haiku+sonnet calling all 6 seizure
  batches normal while fable-5 flagged all 6 with hard signs (conf up to
  0.65) — confirm default switched to fable-5, and the screen tier gained a
  posture channel (lying_lateral always escalates).

- **Calibrate motion thresholds for the actual room.** `MOTION_ON=2.5` /
  `MOTION_OFF=1.0` are calibrated against the synthetic reference video only.
  Run `python src/monitor.py --log-motion` in the real deployment spot (dog
  present, normal day) and adjust for its noise floor.
- **Validate the pose gate on the actual dog.** The YOLO model is fine-tuned
  on Ultralytics Dog-Pose (stock photos, mostly frontal). Check detection
  confidence on real camera angles; if mean_conf stays below 0.30 the gate
  fail-opens permanently (harmless but useless). DeepLabCut
  SuperAnimal-Quadruped is the documented fallback backend.
  *Tried and rejected (2026-08-09):* rotation-augmented retrain
  (`degrees=90`) — hypothesis was "lying dog ≈ rotated standing dog", but
  seizure-frame detection got worse (0.22 vs 0.32) while walking stayed
  1.00; the gap is motion blur + edge cropping + convulsive postures absent
  from stock photos, not orientation. The validated next step is
  fine-tuning on labeled frames from the clinical eval clips and captured
  events themselves.
  *Known limitation (2026-08-09, see EVAL.md):* real gait is rhythmically
  coherent enough to score ~0.50, above the 0.45 threshold, so walking
  escalates; a synthetic 3 Hz injection into the same footage scores 0.60.
  The gate is safe (recall preserved) but only saves VLM cost on
  non-rhythmic events. Improvement path: per-dog threshold calibration on
  captured events, and phase-coherence features (seizure clonus is
  phase-locked across limbs; gait alternates) to widen the separation.
- **Collect hard negatives.** Play, scratching, shaking off water — save event
  dirs the monitor captures during normal life; they become the false-positive
  regression set (AnomalyRuler-style normality rules are the follow-on idea,
  see DATASETS.md rec #2). The public eval corpus already covers play,
  scratch-reflex, and sleep-twitching negatives (EVAL.md); what's missing is
  *this* dog in *this* room.

## 24/7 operation — production on the Pi 5 (since 2026-08-09 evening)

Production runs on a Raspberry Pi 5 as systemd services: `go2rtc.service`
(~/go2rtc, v1.9.14 arm64, same revision as Windows) plus
`seizureguard-mi360.service` / `seizureguard-c700.service`
(~/seizureGuard, `--source rtsp://127.0.0.1:8554/<cam>_sub`). The Pi
consumes the cameras' **subtype=1 substreams** (mi360 640×360, c700
848×480 — the pipeline's working resolution anyway), so no ffmpeg
transcode runs on the Pi and total load is a few percent CPU. Claude CLI
was already installed and logged in on the Pi (`/usr/local/bin/claude`),
so verify is ON end to end. Logs: `journalctl -u seizureguard-mi360`.
Code updates ship via a small scp + service-restart helper.

The Windows path (start-seizureguard.ps1 + Startup shortcut + hourly
watchdog task) was **decommissioned at cutover** — the script stays in the
repo and works if the PC path is ever needed again. Windows keeps go2rtc
(idle unless the PTZ panel/tracker is used) and the PTZ gateway.

Power-cut resilience (verified by a live reboot test 2026-08-09): all
units enabled at boot, hardware watchdog on (systemd RuntimeWatchdog),
journald capped at 300 MB, and a daily 04:10 go2rtc restart timer
(`go2rtc-refresh.timer`, Persistent=true) counters the Xiaomi
serviceToken going stale in memory.

Standing constraints:

- **Telegram alerts LIVE (2026-08-09 late evening):** a private bot →
  owner chat, token in per-service drop-ins
  (`/etc/systemd/system/seizureguard-*.service.d/telegram.conf`, mode
  600). End-to-end verified with a live-frame photo through
  `alerts.send_alert`. Rotation path: BotFather `/revoke`, then re-run
  `~/setup-telegram.sh` on the Pi (prompts secretly, restarts services).
- **Camera session budget:** the Xiaomi cameras break above ~2 concurrent
  sessions. Pi holds one per camera (substream). Windows go2rtc grabs
  another only while its streams are consumed (PTZ panel iframe, tracker,
  scans) — fine, but don't run Windows monitors again in parallel.
- **Disk hygiene (learned 2026-08-17 from a disk alert at 94%):** two
  leaks, both closed. (1) Every headless `claude -p` call persisted a
  transcript *containing the base64 frames* under `~/.claude/projects` —
  ~4,900 verify calls in 8 days = 11 GB; fixed with
  `--no-session-persistence` (regression-tested) and the old transcripts
  deleted. (2) `data/events/` grows ~30 events/day (~3 GB/8 days);
  `scripts/prune_events.py` now runs daily (`seizureguard-prune.timer`,
  04:20): keeps 14 days plus **every verifier-positive event forever**
  (training set), deletes older negatives.
- **CORRECTION — there is no live false-alarm figure yet.** An earlier
  note here read the ~230 captured events' `final_abnormal_event: false`
  as "the verifier rejected them". It did not: `failed_batches` equalled
  the batch count on every one. See the outage below.

## Silent verification outage (found 2026-08-22, the project's worst bug)

Every verify call on the Pi failed from the very first event (2026-08-09
21:04) through 2026-08-22 — 383 events, **zero successfully analyzed** —
with `401 OAuth access token has expired`. Two independent defects let it
run unnoticed for 13 days:

1. **Silent blindness.** `handle_event` treated any result without
   `final_abnormal_event` as negative and stayed quiet, even when every
   batch had failed. `analysis.json` said "treat this result as
   incomplete"; the monitor ignored it. A seizure in that window would
   not have alerted. Fixed: `alert_text_for()` (pure, regression-tested)
   alerts as UNVERIFIED whenever any batch failed and the event was not
   confirmed positive — an unanalyzed batch is not evidence of absence.
2. **A health check that could not fail.** `claude auth status` keeps
   reporting `loggedIn: true` after the token expires, so startup logged
   "verify: on" while nothing worked. Fixed: `verify_probe()` makes one
   real inference call at startup and sends a Telegram warning when
   verification is down.

Operational fix for a headless 24/7 host: a long-lived token
(`claude setup-token`) installed into the service units as
`CLAUDE_CODE_OAUTH_TOKEN` via `~/setup-claude-token.sh` (prompts
silently, validates with a real call, restarts the services) — session
credentials are not durable enough for an unattended machine.

**Lesson worth keeping:** a component that reports health from local
state rather than from doing its actual job will eventually lie. Probe
the work, not the flag.
- Pose gate is NOT active on the Pi (no `SEIZUREGUARD_POSE_PYTHON`):
  every event goes straight to verify. It only ever saved cost, never
  recall. If quota becomes noisy, port it via NCNN export.
- Claude usage: a no-dog motion event still costs a few haiku screen
  calls; the no-dog short-circuit stays deliberately unimplemented
  (fail-open doctrine).

## PTZ dog tracker (built 2026-08-09, disabled pending motor reliability)

`src/tracker.py` keeps the dog horizontally centered on the mi360 via the
local PTZ gateway (`http://127.0.0.1:1985`): YOLO dog detection, 14%-width
deadband, half-gain stepping, self-calibrating pan sign (the gateway's
left/right labels were verified by SSIM, which proves motion but not
direction), strict rate limits (4 s gap, 8 cmds/min), lost-dog homing by
undoing net steps, and a moving-flag file the mi360 monitor honors so
self-commanded pans never read as motion events (stale flags are ignored —
a dead tracker cannot blind the monitor).

Measured on-device: one motor step ≈ 18 px @640 (~2-3°), command-to-settle
≈ 1.5 s, hard end stops on the pan axis. **Wedge history (2026-08-09):**
after ~40 rapid commands the motor controller wedged — set_motor returned
OK with zero movement in all directions, surviving a 3-min cooldown. It
recovered later the same evening after an MIoT power-cycle (siid 2 piid 1
off/on) plus ~20 min; exact cause unproven, so the tracker's gentle rate
limits stay mandatory. Also learned: concurrent commands from the panel
and the API produce gateway 502s (device errors under contention) —
harmless, but the tracker must tolerate them. The camera was successfully
re-aimed at the dog's resting area through the gateway afterwards, and the
monitor captured zero junk events during all pans (2 s sustain + global
-change filtering absorb manual repositioning naturally).

Tracker remains opt-in (`setx SEIZUREGUARD_TRACK 1`, kill pythons once).
Recommended first activation: a supervised daytime trial with the dog in
frame — a mispointed camera is worse than a fixed one aimed at the resting
area, so do not leave it on unattended before one observed session.

## Code, unblocked

- **Grow the eval corpus.** `scripts/fetch_eval_clips.py` +
  `scripts/eval_clips.py` make adding labeled clips one manifest line; the
  2026-08-09 run (EVAL.md) is n=11. Candidates: more PMC supplementary
  videos (focal/absence semiologies are missing entirely), RodEpil subset
  via HTTP range requests (Zenodo zip supports ranges; 133 req/60s limit).
- **Batch the screen tier harder.** Each `claude -p` spawn costs process
  startup; batches of 60 (vs 30) halve the spawn count at slightly higher
  per-call latency. Measure once real events flow.
- **K9-Bench FP probe (manual).** Dataset is gated on HF + YouTube-linked;
  needs a human to accept terms and pull ~10 clips, then
  `python scripts/extract_event_from_video.py` + verify per clip. See
  DATASETS.md rec #5.
- **RodEpil transfer** (DATASETS.md rec #7): pretrain a local classifier on
  the 13k open rodent seizure clips, few-shot fine-tune on captured events.
  Big job; only worth it once real events accumulate.
- **Frigate NVR integration** as capture layer if the prototype graduates to
  a permanent installation (DATASETS.md, Models & Tools).

## Watch list

- WildDog-Videos public release (Zenodo, was under review Jul 2026).
- Cross-species seizure forecasting code release (arXiv 2603.12887).
- Any published accuracy numbers for Furbo Seizure Alert / PetPace Epilepsy
  Insights — the honest-validation gap is this project's differentiator.
