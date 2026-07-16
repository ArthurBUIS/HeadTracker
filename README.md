# HeadTracker

Prototype: take **one** live video stream and emit **N** live streams — one
per participant in the room — where each output is a **200×200 square that
stays centred on that participant's head**, recomputed every 2 seconds with
smoothed motion and a **stable identity** (each box keeps following its own
person).

Built to drop into
[`portals-projector-agent`](../portals-projector-agent) later: it's
TypeScript, runs in the Electron renderer / browser, consumes an
`HTMLVideoElement` (i.e. a WebRTC `MediaStream`), and produces `MediaStream`s
— the same currency Daily.co uses. It reuses the vision stack portals
already ships (`@vladmandic/face-api` SsdMobilenetv1 on tfjs).

## How it works

Four stages, two clocks:

| Stage | Runs | What it does |
|-------|------|--------------|
| **Detect** | every **0.2–2 s** (adjustable) | coco-ssd detects **bodies**; the head box is the top-centre region of each person box. Body detection works from any angle, so people **turned away from the camera are still tracked**. |
| **Track** | each detection | (1) greedy IoU **+ centre-distance** on the body box, (2) **appearance rescue**, (3) **gallery re-ID** → a stable integer id per participant, with birth (`minHits`) / death (interval-scaled `maxMisses`). **This is the "which box is which" part.** |
| **Smooth + crop** | every render frame (~30 fps) | per-id EMA glides the 200×200 crop toward the head; motion stays smooth between the 2 s detections. |
| **Output** | continuous | one `canvas.captureStream()` per id. |

The smoothing is the continuous-lowpass EMA from VideoStitcher's
`stitcher/person_tracking.py` (`alpha = 1 - exp(-dt/τ)`), generalised to the
measured frame delta. VideoStitcher only ever tracks **one** "closest"
person and has no identity; the tracker here is the new piece.

### Appearance re-identification

Spatial tracking alone can't recover an id once a person is lost — they come
back as a new number (the "one person → 8 heads" bug). So each track also
carries an **appearance descriptor**: an HSV colour histogram of the torso
region (`appearance.ts`) — essentially *what colour their clothes are*.
Chosen over a face embedding (FaceNet) because it works from **any angle**,
including people turned away, and needs no extra model.

Association runs in three phases (`tracker.ts`): (1) spatial on the body
box, (2) **appearance rescue** — matches a track and detection that phase 1
left unmatched (e.g. a big jump out of an occlusion), (3) **gallery re-ID** —
a detection matching a recently-*lost* track's signature **resurrects that
id** instead of minting a new one, so someone who left and returned reclaims
their number. Lost ids linger for `reidMemorySeconds` (default 30 s). All
verified headless, including the contrast case (re-ID off → new id) and that
two differently-dressed people are never merged.

### Losing and regaining a head

When a head is lost, the stream is **not** cut — it's kept alive, frozen on
the head's last position (an unmoving camera on that spot), and the tile is
greyed out and labelled "(lost)". If the person comes back within
`lostStreamLingerSeconds` (default 30 s) the same stream **resumes
seamlessly** — no flicker, no new id — reusing the gallery re-ID above. Only
after lingering that long with no return is the stream finally stopped.

### Detection model

The active detector is **coco-ssd** (`@tensorflow-models/coco-ssd` on tfjs —
already a dependency in portals-projector-agent), chosen over face detection
because it fires on whole bodies and so survives people facing away. The
`HeadDetector` interface is pluggable: a **face-api** detector
(`faceApiDetector.ts`) is kept in the repo for the planned in-page model
selector, and a literal YOLOv8-ONNX or a pose model can drop into the same
slot. The head box is derived from the body box via two tunable fractions
(`headSizeFraction`, `headTopOffsetFraction` in `cocoSsdDetector.ts`).

## Run the demo

```bash
npm install
npm run dev        # http://127.0.0.1:5180  → click “Start webcam”
```

Two source modes:

- **Start webcam** — live camera.
- **Load video file…** — pick any local video; it plays (looping) and is fed
  to the algorithm *as if it were a live stream* (every frame is sampled at
  real-time playback speed). Best way to test against pre-recorded footage.

The **Detection interval** slider (0.2–2 s) re-runs detection more or less
often; it applies live to a running session. The **Appearance re-ID**
checkbox toggles identity recovery by clothing colour — turn it off to see
how much more the ids churn without it.

The left pane is the source; the grid on the right is one 200×200 tracked
stream per detected head, each labelled with its stable id. Walk out of frame
and back to watch ids persist and streams appear/disappear.

> **Open `http://127.0.0.1:5180`** (not `localhost`) if your browser resolves
> `localhost` to IPv6 `::1` and can't connect — a common Windows quirk.

```bash
npm run typecheck  # tsc --noEmit, strict
```

## Using the core (the part that ships to portals)

The algorithm lives in [`src/core`](src/core) and imports nothing
Vite/React/Electron — only DOM + canvas + MediaStream APIs plus an injected
face-api instance.

```ts
import { HeadTrackerEngine } from './core';

// In portals: inject the shared esm-nobundle faceapi from
// src/renderer/utils/faceApi.js instead of a bundled one.
const engine = HeadTrackerEngine.withFaceApi(faceapi, {
  onHeadStreamAdded: ({ id, stream }) => attachToTile(id, stream),
  onHeadStreamRemoved: (id) => removeTile(id),
});
engine.start(videoElement);   // an HTMLVideoElement playing the source
// …
engine.stop();
```

Every timing/threshold is configurable (detection interval, output size,
IoU threshold, confirm/drop counts, EMA time constants, crop padding) — see
`DEFAULT_ENGINE_CONFIG` and the per-stage config types.

## Repository layout

See [progress.md](progress.md) for the dense module map. In short:

```
HeadTracker/
├── index.html              demo harness page
├── src/
│   ├── core/               the reusable algorithm (portals-bound)
│   │   ├── types.ts            shared value types + detector interface
│   │   ├── smoothing.ts        time-constant EMA (from person_tracking.py)
│   │   ├── tracker.ts          identity association (spatial + appearance + gallery)
│   │   ├── appearance.ts       torso colour-histogram descriptors for re-ID
│   │   ├── cocoSsdDetector.ts  coco-ssd body detection → head boxes (active)
│   │   ├── faceApiDetector.ts  face-api faces → head boxes (kept for selector)
│   │   ├── headCrop.ts         per-track smoothed 200×200 crop geometry
│   │   ├── headTrackerEngine.ts orchestrator → N MediaStreams
│   │   └── index.ts            public barrel
│   └── demo/main.ts        webcam → engine → grid of tiles (not shipped)
└── progress.md
```

## Known prototype limits

- Appearance re-ID uses clothing **colour**, so it's weak when people wear
  similar colours or under strong lighting changes. `appearanceThreshold`
  (in `tracker.ts`) trades wrong-merges against churn; a learned embedding
  would be more discriminative behind the same interface.
- **Active crossings can still swap ids.** Phase 1 spatial matching runs
  before appearance, so when two people overlap and part, labels can trade.
  Folding appearance into the phase-1 cost (not just as a rescue) would help.
- The head box is a heuristic slice of the body box (COCO has no "head"
  class), so its size/position can drift with unusual poses or partial bodies
  — tune `headSizeFraction` / `headTopOffsetFraction`, or swap in a
  head/pose-specific model behind the same `HeadDetector` interface.
