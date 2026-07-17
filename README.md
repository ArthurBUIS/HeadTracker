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
| **Detect** | every **0.2–2 s** (adjustable) | MoveNet MultiPose locates each person's **head** from keypoints (nose/eyes/ears), falling back to shoulder geometry when the face isn't visible — so people **turned away from the camera are still tracked**, without the body-box jitter. |
| **Track** | each detection | (1) greedy IoU **+ centre-distance** on the body box, (2) **appearance rescue**, (3) **gallery re-ID** → a stable integer id per participant, with birth (`minHits`) / death (interval-scaled `maxMisses`). **This is the "which box is which" part.** |
| **Smooth + crop** | every render frame (~30 fps) | per-id EMA glides the 200×200 crop toward the head. Position and zoom smooth on **separate** time constants — position snappy, zoom (`sizeSeconds`, long) near-constant so the framing doesn't pulse with head-size noise. |
| **Output** | continuous | one `canvas.captureStream()` per id. |

The smoothing is the continuous-lowpass EMA from VideoStitcher's
`stitcher/person_tracking.py` (`alpha = 1 - exp(-dt/τ)`), generalised to the
measured frame delta. VideoStitcher only ever tracks **one** "closest"
person and has no identity; the tracker here is the new piece.

### Appearance re-identification

Spatial tracking alone can't recover an id once a person is lost — they come
back as a new number (the "one person → 8 heads" bug). So each track also
carries appearance cues:

- **Torso colour histogram** (`appearance.ts`) — an HSV histogram of the
  clothing. Works from **any angle**, including people turned away; the
  always-on backbone.
- **Face embedding** (`faceEmbedding.ts`) — a 128-D descriptor from face-api's
  FaceNet-style FaceRecognitionNet, a *much* stronger cue that separates
  **look-alikes** (same clothes, different face) that colour can't. Only
  present when a face is visible, so it's a **booster** layered on the colour
  backbone, not a replacement. Toggle with **Face re-ID**.

The tracker prefers the face embedding when both a track and a detection have
one (weighted `faceWeight`, 0.85), and falls back to colour otherwise.

Association runs in three phases (`tracker.ts`): (1) **primary** — one
**optimal (Hungarian) assignment** over a cost that **fuses** spatial
overlap/proximity with clothing similarity, gated to plausible pairs;
(2) **appearance rescue** — matches a track and detection phase 1 left
unmatched beyond the spatial gate (a big jump out of an occlusion);
(3) **gallery re-ID** — a detection matching a recently-*lost* track's
signature **resurrects that id** instead of minting a new one, so someone who
left and returned reclaims their number (lingers `reidMemorySeconds`, 30 s).

Fusing appearance into phase 1 (rather than matching spatially first) is what
stops two people **crossing** from swapping ids: within the gate, identity
follows clothing, not whichever detection ended up nearest. Matches to a
mutually-occluding detection also skip the appearance update, so a
neighbour's clothing can't contaminate a signature mid-crossing. All verified
headless, including the crossing swap (prevented with fusion, happens
without), the contamination guard, and that differently-dressed people are
never merged.

### Losing and regaining a head

When a head is lost, the stream is **not** cut — it's kept alive, frozen on
the head's last position (an unmoving camera on that spot), and the tile is
greyed out and labelled "(lost)". If the person comes back within
`lostStreamLingerSeconds` (default 30 s) the same stream **resumes
seamlessly** — no flicker, no new id — reusing the gallery re-ID above. Only
after lingering that long with no return is the stream finally stopped.

### Detection model

The active detector is **MoveNet MultiPose**
(`@tensorflow-models/pose-detection` on tfjs). It locates the head from
actual keypoints, so the head box is far steadier than a slice of a jittery
person box, and it still resolves a head from behind via shoulder geometry
(`moveNetDetector.ts`). It also emits a body box (the keypoint bounding box)
that the tracker associates on and the engine samples for appearance.

The `HeadDetector` interface is pluggable — three implementations live in the
repo for a planned in-page model selector: **MoveNet** (active),
**coco-ssd** body detection (`cocoSsdDetector.ts`), and **face-api**
(`faceApiDetector.ts`). A YOLOv8-ONNX detector could drop into the same slot.

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
checkbox toggles identity recovery by clothing colour; **Face re-ID** adds
the 128-D face descriptor on top (loads face-api models on first enable).
Turn either off to feel how much they help — Face re-ID especially when
people are dressed alike.

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
│   │   ├── tracker.ts          identity association (fused-cost + appearance + gallery)
│   │   ├── assignment.ts       Hungarian optimal min-cost assignment
│   │   ├── appearance.ts       torso colour-histogram descriptors for re-ID
│   │   ├── faceEmbedding.ts    128-D face-descriptor re-ID cue + face→head map
│   │   ├── moveNetDetector.ts  MoveNet pose → head boxes from keypoints (active)
│   │   ├── cocoSsdDetector.ts  coco-ssd body → head boxes (kept for selector)
│   │   ├── faceApiDetector.ts  face-api faces → head boxes (kept for selector)
│   │   ├── headCrop.ts         per-track smoothed 200×200 crop geometry
│   │   ├── headTrackerEngine.ts orchestrator → N MediaStreams
│   │   └── index.ts            public barrel
│   └── demo/main.ts        webcam → engine → grid of tiles (not shipped)
└── progress.md
```

## Known prototype limits

- **Face re-ID only helps when faces are visible.** For people turned away
  it falls back to the colour histogram, which still can't separate
  look-alikes — so a same-clothes crossing where neither face shows can still
  swap. A learned whole-body re-ID embedding would cover that gap behind the
  same interface. `faceWeight` / `appearanceWeight` (in `tracker.ts`) tune
  the cue balance.
- Face embedding adds cost: it runs face-api (detector + landmarks +
  recognition) each detection round. Fine at the room scale here; for many
  people or a fast interval, budget accordingly.
- The facing-away head estimate is shoulder geometry (a fixed rise above the
  shoulder line), so it's approximate for unusual postures — tune
  `shoulderHeadScale` / `shoulderHeadRise` in `moveNetDetector.ts`.
