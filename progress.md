# HeadTracker — repo map

## Purpose
From ONE live video stream, emit N live 200×200 streams — one per participant
— each centred on that person's head, recomputed every 2 s, motion smoothed,
with a **stable identity per stream** (each box tracks its own person). A
TypeScript prototype meant to be lifted into `portals-projector-agent`
(Electron renderer), which already runs `@vladmandic/face-api` + tfjs for
people-counting. The reusable algorithm is `src/core`; `src/demo` +
`index.html` are a throwaway webcam harness proving it works.

## Architecture
```
          source <video> / MediaStream
                     │
   ┌─────────────────┴──────────────────────────── HeadTrackerEngine ──────┐
   │                                                                        │
   │  detect loop (setInterval, 200–2000 ms)  render loop (requestAnimationFrame)
   │        │                                        │                      │
   │        ▼                                        ▼                      │
   │  HeadDetector.detectHeads()            per-id HeadCropSmoother.step()  │
   │  (MoveNetHeadDetector: pose                (EMA glide toward target)   │
   │   keypoints → head + body boxes)                │                      │
   │        │                                        ▼                      │
   │   + sampleAppearance (torso HSV hist)   drawImage(source → 200×200)     │
   │  HeadIdentityTracker.update()                   │                      │
   │  (spatial → appearance → gallery re-ID)         ▼                      │
   │        │  active tracks + removedIds    per-id canvas.captureStream()  │
   │        └──────── reconcileSlots() ──────────────┘   → onHeadStreamAdded/Removed
   └────────────────────────────────────────────────────────────────────────┘
```
Two clocks: detection at the interval sets each track's *target*; the render
loop glides the crop toward it every frame so motion is smooth between
detections.

## Contents
| File | Responsibility | Key symbols |
|------|----------------|-------------|
| `src/core/types.ts` | Shared value types + the injectable detector interface. | `HeadDetection`, `HeadDetector`, `FrameSource`, `FrameSize` |
| `src/core/smoothing.ts` | Time-constant EMA (ported from VideoStitcher `person_tracking.py`, generalised to measured `dt`). | `emaWeightForTimeConstant`, `emaStep`, `clamp` |
| `src/core/tracker.ts` | **Identity: which box is which.** 3-phase association (spatial on body box → appearance rescue → gallery re-ID) + birth/death; `setMaxMisses`, `setGalleryMaxRounds`. | `HeadIdentityTracker`, `TrackedHead`, `TrackerConfig`, `DEFAULT_TRACKER_CONFIG` |
| `src/core/appearance.ts` | Torso HSV colour-histogram descriptor + intersection similarity + EMA blend. DOM-free (operates on RGBA data). | `computeHsvHistogram`, `appearanceSimilarity`, `blendAppearance`, `AppearanceDescriptor` |
| `src/core/moveNetDetector.ts` | **Active detector.** MoveNet pose keypoints → head box (face keypoints, else shoulder geometry) + body box. | `MoveNetHeadDetector`, `PoseDetectorLike`, `DEFAULT_MOVENET_DETECTOR_CONFIG` |
| `src/core/cocoSsdDetector.ts` | Alt detector: coco-ssd body boxes → top-centre head boxes. | `CocoSsdHeadDetector`, `CocoSsdModelLike`, `DEFAULT_COCO_SSD_DETECTOR_CONFIG` |
| `src/core/faceApiDetector.ts` | Alt detector: face-api faces → head boxes; tfjs scope disposal. | `FaceApiHeadDetector`, `FaceApiLike`, `DEFAULT_FACE_API_DETECTOR_CONFIG` |
| `src/core/headCrop.ts` | Per-track smoothed square crop geometry, clamped to frame. | `HeadCropSmoother`, `HeadCropConfig`, `CropRect` |
| `src/core/headTrackerEngine.ts` | Orchestrator: two loops, slot lifecycle, N output `MediaStream`s. | `HeadTrackerEngine`, `HeadStream`, `HeadTrackerEngineConfig`, `HeadTrackerCallbacks` |
| `src/core/index.ts` | Public barrel — the portals-bound module boundary. | (re-exports) |
| `src/demo/main.ts` | Webcam **or looping video file** → engine → grid of tiles. **Not shipped.** | `startWebcam`, `loadVideoFile` |
| `index.html` | Demo page (source pane + output grid; webcam + file-load buttons). | — |

## Data flow
Input: an `HTMLVideoElement` playing the source (webcam in the demo; a
Daily.co track in portals). Output: `HeadStream[]` — each `{ id, stream,
canvas }`, `stream` a 200×200 `captureStream`. Lifecycle events via
`onHeadStreamAdded` / `onHeadStreamRemoved`.

Per detection round (interval, 0.2–2 s): detect → `tracker.update(detections)` returns
`{ active, removedIds }` → `reconcileSlots` creates a canvas+stream+smoother
per new confirmed id, refreshes targets, tears down removed ids. Per render
frame: each slot's smoother EMAs toward its target and draws the crop.

## Dependencies
- **Runtime:** `@tensorflow-models/pose-detection` + `@tensorflow/tfjs`
  (active MoveNet detector); `@tensorflow-models/coco-ssd` and
  `@vladmandic/face-api` (alt detectors for the model selector). Core itself
  imports none of them — only DOM/canvas/MediaStream (Electron-renderer
  safe); models are injected.
- **Dev:** `vite` (demo dev server + `esbuild` bundling), `typescript`
  (strict, `tsc --noEmit`). tsconfig mirrors portals: `es2021`, strict,
  `noUnusedLocals/Parameters`.
- **Model weights:** MoveNet MultiPose Lightning auto-downloads on first load
  (verified in-browser); coco-ssd / face-api weights similar when selected.

## Key decisions & gotchas
- **Pose model for the head, not a body-box heuristic (fixes head jitter).**
  MoveNet MultiPose gives head keypoints (nose/eyes/ears) → head box tracks
  the real head; facing away, it falls back to shoulder geometry (head sits
  `shoulderHeadRise` × shoulder-width above the shoulder line). Emits a body
  box (keypoint bbox) for association + appearance. coco-ssd (body→head
  heuristic) and face-api remain behind `HeadDetector` for the selector.
  FaceNet was rejected as a detector: it's face-*recognition*, needs a
  visible face — useful later only for re-ID.
- **Detector is injected, not imported** by the engine (`HeadDetector`
  interface) so portals reuses its one loaded tfjs engine — the same "one
  engine" concern documented in portals' `src/renderer/utils/faceApi.js`.
  `HeadTrackerEngine.withCocoSsd(...)` / `.withFaceApi(...)` are the wiring
  shortcuts.
- **Associate on the BODY box, not the head box (the anti-churn fix).** The
  head box is derived from a wobbling coco-ssd body box (size AND y both
  scale with body height), so a still person's head box teleports between
  detections and churns ids — one person read as 8. `HeadDetection.bodyBox`
  carries the person box; the tracker matches on it (large, stable overlap)
  and keeps the head centre/size only for the crop. Falls back to the head
  box when no `bodyBox` (face-api).
- **Association = IoU OR centre-distance gate** (`centerDistanceGateFactor`
  2.5 × avg box size), so a box that moved far still matches; kept modest so
  two close people aren't merged.
- **Appearance re-identification (3-phase association).** Each detection gets
  an HSV torso colour histogram (`sampleAppearance` → `computeHsvHistogram`);
  the tracker matches spatially first, then rescues unmatched pairs by
  appearance, then resurrects a **recently-lost id from a gallery** when a
  detection's signature matches — so a person who left/was occluded reclaims
  their number instead of churning. `appearanceThreshold` 0.55 gates both
  appearance phases; gallery lifetime = `reidMemorySeconds` (30) / interval.
  Toggle live via `setAppearanceReid`; when off (or descriptor absent, e.g.
  face-api) phases 2–3 no-op → pure spatial. Verified headless (gallery
  reclaim vs new-id contrast, big-jump rescue, no-merge of different clothes).
  Colour is angle-agnostic (works facing away), unlike a FaceNet embedding.
- **Detection interval is adjustable 200–2000 ms** (`setDetectionInterval`,
  demo slider), default 500. **Coast time is held constant**: the engine sets
  the tracker's `maxMisses = round(trackCoastSeconds / interval)` (default
  coast 2 s) on start and on every interval change, so faster detection
  doesn't shorten the grace before a stream closes.
- **`minHits` gates flicker vs latency.** Default `minHits: 2` ⇒ a stream
  appears on the 2nd consecutive detection. Ids are **never reused** after
  death.
- **Lost heads freeze, they don't cut.** When the tracker drops an id the
  engine marks its slot `lost` rather than removing it: the stream stays
  alive frozen on the last crop (`onHeadStreamLost`), and a return within
  `lostStreamLingerSeconds` (30) resumes the SAME stream/canvas seamlessly
  (`onHeadStreamResumed`) — pairs with gallery re-ID so a returning person
  reclaims both id and stream. Only after lingering out is `removeSlot` /
  `onHeadStreamRemoved` fired. The linger timeout is counted in the render
  loop via accumulated rAF dt.
- **head box → crop.** `headCrop` adds framing via `paddingFactor` (2.0) then
  resamples the square source region to 200×200, so near and far heads are
  framed consistently. Set `paddingFactor` to 1 for a literal fixed-size crop.
- **EMA uses measured `dt`** from rAF timestamps, not a fixed fps, so glide
  speed is right even when the render loop stutters; a long stall snaps
  (weight clamped to 1) instead of overshooting.
- **Position and zoom smooth on separate time constants.** Crop centre uses
  `lockSeconds`/`holdSeconds` (snappy); crop side uses `sizeSeconds` (4 s,
  long) so the noisy per-detection head-size estimate — and the jump when the
  detector switches face↔shoulder — doesn't read as the view zooming in/out.
  Head geometry is initialised on the first observation, so the long size tau
  doesn't cause a slow zoom-in at stream start. Verified headless (zoom swing
  37px→5.6px under alternating head sizes; position still tracks fast).
- **tfjs memory:** the face-api detector wraps each pass in
  `tf.engine().startScope()/endScope()` (from portals' FaceDetection.jsx) to
  avoid a WebGL leak on the interval; coco-ssd manages its own tensors.
- **Two loops, no worker.** Detection is `async` on `setInterval`, guarded by
  a `detecting` flag so a slow pass can't overlap itself; rendering is on
  `requestAnimationFrame`. A missed/paused video frame is skipped safely.

## Entry points
- Demo: `npm run dev` → `index.html` → `src/demo/main.ts` (webcam).
- Library: `import { HeadTrackerEngine } from './core'` →
  `HeadTrackerEngine.withMoveNet(detector, callbacks, config).start(video)`
  (or `.withCocoSsd(model, …)` / `.withFaceApi(faceapi, …)`).
- Verified headless during development (identity persistence + re-ID + gallery,
  MoveNet head geometry, EMA glide + clamp); no committed test suite yet.

## TODO(verify) / next steps
- **Fold appearance into phase-1 cost** so active crossings don't swap ids
  (today appearance is only a rescue/gallery step after spatial matching).
- **Stronger descriptor** than a colour histogram (learned embedding, or
  FaceNet when a face is visible) for similar-clothing / lighting robustness.
- **In-page model selector** (MoveNet ↔ coco-ssd ↔ face-api): all three
  exist behind `HeadDetector`; the demo UI toggle + shared tfjs engine
  wiring are not built yet.
- Kalman/motion prediction to lift the "≤ ~2.5 head-widths per interval" limit.
- Integration shim in portals (Daily track → engine → tiles) — not started.

## Pointers
- Inspiration: `../VideoStitcher/stitcher/person_tracking.py` (single-person
  EMA crop) and `../VideoStitcher/stitcher/segmentation.py` (face/person det).
- Target host: `../portals-projector-agent` — `src/renderer/utils/faceApi.js`,
  `src/renderer/components/CameraDetection/FaceDetection.jsx`.
