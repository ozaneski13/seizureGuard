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
  **Measured 2026-09-25 on the 4 surviving real events (Pi 5, CPU):**
  mean_conf 0.35 / 0.41 / 0.39, above the 0.30 fail-open floor. Detection
  was 81% and 97% on the two walking false alarms and 32% on the seizure
  clip, where the gate failed open and the event escalated. FA2 would have
  been skipped (score 0.40 < 0.45), and the 08-30 event failed open because
  the dog was too small to detect. Cost is 25-39 s per event. **Decision:
  the gate stays OFF.** A skip means the event is never verified. The only
  real seizure got past the gate only by failing open, and the verifier
  catches it on a single 3 s burst (batch 2). If wanted later, the next
  step is a log-only mode: it records the gate's decision after
  verification and never skips. The prepared venv stays on the Pi
  (`~/seizureGuard/.venv-pose`, 5.7 GB, mostly unused CUDA wheels, plus
  `models/dog-pose.pt`); `rm -rf` both to reclaim the space.
- **Collect hard negatives.** Play, scratching, shaking off water — save event
  dirs the monitor captures during normal life; they become the false-positive
  regression set (AnomalyRuler-style normality rules are the follow-on idea,
  see DATASETS.md rec #2). The public eval corpus already covers play,
  scratch-reflex, and sleep-twitching negatives (EVAL.md); what's missing is
  *this* dog in *this* room.
  **Automated (2026-09-25):** `prune_events.py` now keeps one verified
  negative per camera per day forever. It keeps the one whose batches mark
  the most observed signs present (ties go to the earliest). Events with
  failed batches or no verdict never qualify. That is about 2 events a day,
  roughly 15 GB a year. The 129 negatives lost in the 2026-09 blind spell
  predate this rule.

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
  04:20): keeps the 14 days before the newest event, **every
  verifier-positive event forever** (training set), and one hard-negative
  sample per camera per day forever; deletes the other older negatives.
- **CORRECTION — there is no live false-alarm figure yet.** An earlier
  note here read the ~230 captured events' `final_abnormal_event: false`
  as "the verifier rejected them". It did not: `failed_batches` equalled
  the batch count on every one. See the outage below.
- **Event archive viewer (2026-09-14):** `seizureguard-events.service`
  serves `src/event_server.py` on port 8090 (all interfaces, LAN only,
  **no auth** — anyone on the home network can watch the clips). Grid of
  captured events with thumbnail, date/time, camera, verdict badge and
  `final_confidence`; the detail page plays `event.mp4` with HTTP Range
  (seeking works) and lists the observed signs. Default order is
  positives first, because `final_confidence` is the verifier's
  confidence in its own verdict — a raw confidence sort puts a 0.96
  *negative* on top. Thumbnails are cached in `data/thumbs/` and dropped
  once their event is pruned. `scripts/deploy-pi.ps1` ships and restarts
  it.
- **The SD card still holds a bootable pre-NVMe copy** (frozen
  2026-08-22). `BOOT_ORDER=0x416` falls back to it if the NVMe fails, and
  it would then run that old seizureGuard code without any warning.
  Remove the card or make the boot order NVMe-only once the NVMe is
  trusted.

## Blind for 20 days (2026-09-04 → 2026-09-24)

Both cameras dropped off the Wi-Fi at 2026-09-04 14:44. The Xiaomi cloud
reported `device offline` for both while their 2.4 GHz SSID stayed up,
so the fault was on the camera side. Every go2rtc stream — main and
`_sub` alike — failed with `read udp ... i/o timeout`, both monitors
hung in their reconnect path, and nothing was captured. The cameras were
back on the network by 2026-09-24 and the monitors were restarted that
evening; both report `Monitoring camera ... (verify: on)` again.

Two detection gaps let it run for 20 days:

- `systemctl is-active` kept answering `active` while both monitors had
  stopped logging entirely. "Active" is not "watching".
- The "monitor blind" alert fired once (on 2026-09-04) and never
  repeated. **Fixed 2026-09-24:** `StreamWatchdog` repeats it every 6 h
  while the stream stays dead, and the monitor announces recovery with how
  long it was blind; the same applies to a stream that cannot be opened at
  startup.

And one data loss: `prune_events.py` keeps "the last 14 days" measured
from *now*. With nothing new arriving, it deleted every negative from
before the outage (129 events, 2026-09-15..18), leaving only the 4
positives. **Fixed 2026-09-24:** the window is now measured back from the
newest event, so a blind spell removes nothing (regression-tested).

**Why the monitors never recovered (found 2026-09-24, fixed in
`3c82190`).** The camera drop-out was the trigger, not the reason the
blindness lasted 20 days. When go2rtc answered 404, OpenCV, which had
no backend pinned, fell back from FFMPEG to GStreamer. GStreamer leaked
a pipeline per reconnect (`appsink6637` in the log) and deadlocked in
native code: main thread in `futex_wait`, workers spinning at ~92% CPU,
no log line after 2026-09-04 14:18, and zero go2rtc consumers even after
the cameras returned. The monitors only came back when restarted.

A repeated blind alert (the follow-up above) would not have caught this:
it runs in the Python loop, and the Python loop never ran again. The
fix has two layers:

- stream URLs open with `cv2.CAP_FFMPEG` and 10 s open/read timeouts, so
  a dead stream fails fast instead of falling back (verified on the Pi:
  a 404 stream now fails in 0.0 s);
- `SystemdWatchdog` pings `WATCHDOG=1`, and both monitor units carry a
  drop-in `/etc/systemd/system/seizureguard-*.service.d/watchdog.conf`
  with `WatchdogSec=300` + `NotifyAccess=main`. A process that wedges
  anywhere in native code stops pinging and systemd restarts it within
  five minutes. The window must stay above the slowest unpinged startup
  path (auth check 30 s + verify probe 120 s); `handle_event` pings from
  a keepalive thread because verify can take minutes.

The repeated blind alert (added the same day) covers the case the
watchdog does not: a healthy loop staring at a stream that stays dead.

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

**Resolved 2026-08-22.** A long-lived token now lives in the service
units and the first real verdict in 13 days came back clean (8/8 batches
analyzed, 0 failures). One trap cost an hour and is worth remembering:
**a stale `~/.claude/.credentials.json` silently overrides
`CLAUDE_CODE_OAUTH_TOKEN`** — a brand-new valid token kept reporting
"access token has expired" until the stored file was moved aside. The
tell: an intentionally bogus token produces the same "expired" error
instead of "invalid". `~/setup-claude-token.sh` now clears it first.

Two quality issues surfaced by that first clean run (neither is a safety
risk; both resolved since, listed at the end of "False alarms from
undefined sign vocabulary" below).

## Alert storm during a quota outage (fixed 2026-09-02)

Every motion event produced an "UNVERIFIED" Telegram alert for hours. The
alerts were correct — the fail-open net doing its job — but the cause was
a backend outage, not an event problem: `You've hit your limit · resets
Sep 5` on 46 consecutive batches. During an outage every event fails
identically, so per-event alerts add no information and train the owner to
ignore the one channel that must never be ignored.

Fix, in two pieces:

- `verify_event.is_backend_outage()` separates backend-level failures
  (quota, rate limit, 401/auth, overloaded) from event-level ones (a JSON
  parse error stays event-level), and `analysis.json` carries
  `backend_outage` with the shared reason.
- `monitor.OutageNotifier` announces the outage **once**, then at most
  every 6 h, and each notice reports how many events went unchecked
  meanwhile. Recovery is announced once too. A positive verdict still
  alerts immediately even mid-outage.

### The screen tier cannot be used to save money (measured 2026-09-02)

Two attempts to make the cheap tier filter, both reverted:

1. **Ask the question properly.** The field was renamed `seen` ->
   `abnormal_seen` and the question stated in one sentence with the
   voluntary-behaviour exclusions. On the two false-alarm events this cut
   expensive calls from 7/7 to 2/7 and 3/7 — real savings. But the model
   then reported *certainty of its own answer* in `confidence` on other
   batches (`abnormal_seen: "no"` with `confidence: 0.95`), which sailed
   past the 0.15 escalation threshold, so the gate was still driven by a
   number whose meaning the model changes at will.
2. **Make the gate categorical** (`yes` / `no` / `unsure`, escalate unless
   a clear no). This removed the numeric ambiguity — and on the reference
   seizure the screen model answered **"no" on all six batches**, with
   notes reading "normal purposeful walking". Zero escalations, seizure
   missed completely.

That is the finding: **a "no" from the cheap tier is not evidence of
absence.** It matches the August model A/B, where haiku and sonnet called
every batch of the same seizure normal and only fable flagged it. The tier
was never filtering — it was accidentally answering "yes" to everything,
which is why the pipeline worked.

`should_escalate()` is therefore a documented no-op: it returns True
always, and the docstring carries this measurement so the optimization is
not attempted a third time.

**Dropped entirely on 2026-09-24.** The no-op gate still cost one extra
call per batch (half of all calls) and supplied the note the event viewer
showed as each event's explanation, which on the seizure read "normal
purposeful walking". Each batch now goes straight to the confirm model,
whose own note is stored instead. The only remaining cheap-model call is
the startup health probe (`SEIZUREGUARD_PROBE_MODEL`).

Cost must come from somewhere that cannot cost recall:
- fewer frames per event (base sampling 2 fps -> 1 fps; bursts carry the
  signal and stay untouched) — the cheapest lever, ~30-40% fewer batches
- fewer events (motion threshold calibration for the actual room)
- the local pose gate on the Pi (free, but measured to escalate most real
  footage anyway)
- a stronger screen model, which inverts the economics and is pointless

**Who actually exhausted the quota — measured, after an initial wrong
call.** This log first blamed seizureGuard's own call volume. The event
timeline says otherwise: on the day of the outage seizureGuard's last call
was at 08:56 and the first `hit your limit` error came at 16:30, with
**zero calls in between**. The quota went during those 7.5 hours, to the
owner's interactive Claude use on the same subscription.

The real structural issue is that **a 24/7 safety monitor shares one quota
with interactive work**. Whoever spends it, the monitor is the one that
goes blind, and it cannot ask for priority. seizureGuard's own footprint,
for the record: ~20-55 events/day, ~130-380 batches, **~250-690 calls/day**
(screen + confirm) — real, but not the thing that emptied the bucket that
afternoon. Dropping the screen tier (2026-09-24) halved the call count to
one per batch.

Options, in order of how well they fit a monitor that must not go blind:

1. **Give the monitor its own credential** — a separate subscription or an
   API key with billing, so interactive work and the dog monitor cannot
   starve each other. Costs money; needs a token estimate first (each call
   carries 30 frames at 640 px).
2. **Reduce the monitor's footprint** so it survives on the leftovers:
   base sampling 2 fps -> 1 fps, motion-threshold calibration for the room.
   Cheap, no recall cost, but does not remove the shared-fate problem.
3. **Accept it**, now that an outage announces itself once instead of
   spamming, and the owner knows to check when they have been hammering
   Claude themselves.

## False alarms from undefined sign vocabulary (fixed 2026-08-23)

Within two hours of verification coming back online the system sent two
alerts on a dog that was only walking around. The cause was not the rule
layer or the models' judgment — it was vocabulary. The prompt listed the
ten sign names with no definitions, so they were read in their everyday
sense:

- a dog settling onto the floor -> `loss_of_posture` (a HARD sign, so one
  occurrence alerts), body_region text: "drops from standing to a low
  lateral position"
- circling before lying down -> `disorientation`
- motion blur around the muzzle of a trotting dog -> `drooling`, plus
  `head_tremor` — two soft signs, which the rule layer also treats as
  positive

In both events the model's own free-text note said "coordinated",
"goal-directed", "consistent with normal behavior" while its structured
fields said otherwise. Bare terms invited the everyday reading.

Fix: `SIGN_DEFINITIONS` gives every sign a clinical definition **and its
exclusion** ("loss_of_posture: SUDDEN INVOLUNTARY COLLAPSE ... NOT lying
down, settling to rest, rolling over"), plus a two-sided rule.

**The two-sided rule matters more than the definitions.** The first
attempt ended with a one-sided rule ("if the behaviour is voluntary and
goal-directed, every sign is false"). Re-running all three reference
events showed it silenced both false alarms *and the real seizure*
(6/6 positive batches -> 0/6): the model took the escape hatch and
described convulsive thrashing as "purposeful, controlled walking". The
shipped rule states both directions — what is never a sign, and what MUST
be flagged (on-side rapid limb movement, legs giving way, thrashing
without righting, 2-6 Hz jerking) — with "if torn, flag it".

Validation on real footage, same three events, before and after:

| Event | Before | One-sided rule | Shipped |
|---|---|---|---|
| Walking dog (false alarm 1) | ALERT 1/7 | silent | **silent 0/7** |
| Walking dog (false alarm 2) | ALERT 1/7 | silent | **silent 0/7** |
| Real seizure (owner footage) | ALERT 6/6 | *silent 0/6* | **ALERT 2/6** |

Watch item: the true positive's margin narrowed from 6/6 to 2/6 positive
batches. It still fires (the event rule is a pure OR), and separation from
the negatives is clean, but a subtler seizure has less headroom than
before. Re-check this table whenever the prompt changes.

**Re-check 2026-09-25**, after the screen tier was dropped. The confirm
prompt was unchanged. Runs were on copies on the Pi, with the production
token:

| Event | Runs | Result |
|---|---|---|
| Walking dog (false alarm 1) | 1 | silent 0/7 |
| Walking dog (false alarm 2) | 1 | silent 0/7 |
| Real seizure (owner footage) | 5 | **ALERT in 5/5**, always 1/6, always the same batch |
| Same seizure in grayscale (IR stand-in) | 3 | **ALERT in 3/3**, 1-2/6 |

Run to run, the detection is stable. But it rests on a single window:
batch 2, the 10 fps motion burst in which the dog lies on its side with
rapid limb motion for ~3 s "without rising". Every other batch is judged
voluntary rolling because the dog "repeatedly rights itself". So the thin
margin is not a coin flip. It is one structural point of failure: if a
seizure's burst window is shorter, cropped, or broken by a head lift, no
batch would come back positive. Grayscale did not hurt (it flagged as
often or more). Real IR footage, with its noise and lower contrast, is
still unmeasured. In one grayscale run a batch was lost to malformed
model JSON, which failed both attempts ("Expecting ',' delimiter"). The
event still alerted from its other batches.

Also fixed in the same pass: `final_confidence` is now the confidence of
the *finding* (max over positive batches) instead of the max over all
analyzed batches — a normal event used to report 0.85, and the first false
alarm quoted 0.85 while its only positive batch scored 0.45. Alerts now
also carry how much of the event looked abnormal ("2/7 segments"), which
is the fastest triage signal: the real seizure spans many segments, false
alarms one.

- **Resolved: the screen tier was dropped (2026-09-24).** Original
  finding: **the screen tier no longer filters anything.** On plainly normal
  footage haiku returned `{"seen": "yes", "confidence": 0.05,
  "posture": "standing"}` with a note reading "Normal ambulation
  throughout... no jerking, paddling, stiffening" — so `should_escalate`
  fires on every batch and each event pays for a full confirm pass. The
  field name "seen" is ambiguous (the model appears to answer "did I see
  the dog/frames"). Fix: rename it to something unmistakable
  (`abnormal_seen`) and state the question in one line.
- **Resolved in the same pass (see above): `final_confidence` is now the
  max over positive batches.** Original finding: **`final_confidence` is
  meaningless for negatives.** It is the max
  batch confidence, and the confirm model reports confidence *in its
  verdict*, so a clearly normal event reported 0.85. Alerts only show it
  for positives, so nothing user-facing is wrong, but do not compare it
  across positive/negative events.
- Pose gate is NOT active on the Pi (no `SEIZUREGUARD_POSE_PYTHON`):
  every event goes straight to verify. It only ever saved cost, never
  recall. If quota becomes noisy, port it via NCNN export.
- Claude usage: a no-dog motion event still costs a few confirm calls
  (one per batch); the no-dog short-circuit stays deliberately
  unimplemented (fail-open doctrine).

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
