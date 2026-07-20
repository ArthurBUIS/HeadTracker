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
   │  (fused Hungarian → rescue → gallery)           ▼                      │
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
| `src/core/tracker.ts` | **Identity: which box is which.** 3-phase association (fused spatial+appearance Hungarian match → appearance rescue → gallery re-ID) + contamination guard + birth/death; `setMaxMisses`, `setGalleryMaxRounds`. | `HeadIdentityTracker`, `TrackedHead`, `TrackerConfig`, `DEFAULT_TRACKER_CONFIG` |
| `src/core/assignment.ts` | Hungarian (Kuhn–Munkres) min-cost assignment; rectangular via padding, forbidden pairs via a disallow cost. | `solveMinCostAssignment`, `NO_ASSIGNMENT` |
| `src/core/faceEmbedding.ts` | 128-D face-descriptor distance/affinity/EMA + face→head box assignment. DOM-free; the embedder itself is injected. | `faceDistance`, `faceAffinity`, `blendFace`, `matchFacesToBoxes`, `FaceDescriptor` |
| `src/core/bodyEmbedding.ts` | Whole-body embedding cosine affinity + EMA. DOM-free; the CNN (MobileNet by default) is injected. | `bodyAffinity`, `blendBody`, `BodyDescriptor` |
| `src/core/appearance.ts` | Torso HSV colour-histogram descriptor + intersection similarity + EMA blend. DOM-free (operates on RGBA data). | `computeHsvHistogram`, `appearanceSimilarity`, `blendAppearance`, `AppearanceDescriptor` |
| `src/core/moveNetDetector.ts` | **Active detector.** MoveNet pose keypoints → head box (face keypoints, else shoulder geometry) + body box. | `MoveNetHeadDetector`, `PoseDetectorLike`, `DEFAULT_MOVENET_DETECTOR_CONFIG` |
| `src/core/cocoSsdDetector.ts` | Alt detector: coco-ssd body boxes → top-centre head boxes. | `CocoSsdHeadDetector`, `CocoSsdModelLike`, `DEFAULT_COCO_SSD_DETECTOR_CONFIG` |
| `src/core/faceApiDetector.ts` | Alt detector: face-api faces → head boxes; tfjs scope disposal. | `FaceApiHeadDetector`, `FaceApiLike`, `DEFAULT_FACE_API_DETECTOR_CONFIG` |
| `src/core/headCrop.ts` | Per-track smoothed square crop geometry, clamped to frame. | `HeadCropSmoother`, `HeadCropConfig`, `CropRect` |
| `src/core/headTrackerEngine.ts` | Orchestrator: two loops, slot lifecycle, N output `MediaStream`s. | `HeadTrackerEngine`, `HeadStream`, `HeadTrackerEngineConfig`, `HeadTrackerCallbacks` |
| `src/core/index.ts` | Public barrel — the portals-bound module boundary. | (re-exports) |
| `src/demo/main.ts` | Webcam **or looping video file** → engine → grid of tiles; explicit **Load models** step locks the cue selection first. **Not shipped.** | `startWebcam`, `loadVideoFile`, `loadSelectedModels` |
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
  (active MoveNet detector); `@tensorflow-models/mobilenet` (body-embedding
  re-ID); `@vladmandic/face-api` **nobundle** build (face-embedding re-ID,
  shares the one tfjs engine — see FaceDetection.jsx in portals);
  `@tensorflow-models/coco-ssd` (alt detector). Core imports none of them —
  only DOM/canvas/MediaStream (Electron-renderer safe); the detector and both
  embedders are injected. All models run on the forced **webgl** backend.
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
  an HSV torso colour histogram (`sampleAppearance` → `computeHsvHistogram`)
  and, when a face is visible and `faceReid` is on, a **128-D face embedding**
  (`attachFaceDescriptors` → injected `FaceEmbedder`, mapped to a head via
  `matchFacesToBoxes`). Phase 1 is a single **optimal (Hungarian) assignment**
  over a cost fusing spatial affinity with a re-ID affinity; phase 2 rescues
  beyond the gate; phase 3 resurrects a **recently-lost id from a gallery**.
  `reidSimilarity` prefers the FACE cue when both sides have one (weight
  `faceWeight` 0.85, threshold via `faceMatchAffinity`) and falls back to
  colour (`appearanceWeight` 0.6, `appearanceThreshold` 0.55). Gallery entries
  store both cues. Toggle live via `setAppearanceReid` / `setFaceReid`; with
  neither cue phase 1 is spatial-only.
- **Three re-ID cues, priority face → body → colour** (`reidSimilarity`
  returns the best shared cue's affinity + `kind`; `reidWeight`/`reidThreshold`
  /`reidRank` pick per-kind constants). Face = most discriminative but needs a
  visible face; **body embedding = any-angle backbone** (works facing away,
  separates look-alikes by texture/pattern, unlike colour); colour = cheap
  fallback. Each cue EMA-updated and stored in the gallery.
- **Body embedding (the look-alike + facing-away fix).** A CNN feature vector
  of the body crop (`attachBodyEmbeddings` → injected `BodyEmbedder`; demo =
  MobileNet deep features, 1280-D, cosine similarity). Runs one forward per
  person per round, time-capped like face. Default embedder is MobileNet
  ImageNet features (baseline); OSNet drops into the same interface. Verified
  headless (facing-away same-clothes crossing kept-with-body / swaps-without;
  gallery look-alike not reclaimed) and in-browser (MobileNet loads + embeds
  1280-D on the shared webgl engine).
- **Face embedding.** face-api FaceRecognitionNet (nobundle, shared engine),
  128-D, the precise booster when a face is visible. Opt-in (heavier). Runs on
  the forced webgl backend (webgpu hung); embed is time-capped so it can't
  stall the loop. Verified headless + in-browser.
- **Three swap defences (why crossings don't trade ids).** (1) Fusing
  appearance into the phase-1 cost + optimal assignment (`assignment.ts`) —
  within the gate identity follows appearance, not the nearest box. (2) A
  match to a detection that overlaps another (`detectionOverlapFreeze` IoU)
  skips the appearance EMA update (also below `appearanceUpdateFloor`), so a
  neighbour's clothing can't contaminate a signature mid-crossing. (3) **Veto**
  (`faceVetoAffinity`/`bodyVetoAffinity`): a spatially-close pair is FORBIDDEN
  when both carry the same strong embedding yet it clearly disagrees — this is
  what stops a track whose own detection dropped for a round from grabbing a
  nearby DIFFERENT person ("stream 1 jumps to person 2"). Verified headless
  (bug reproduced with veto off, fixed with veto on, same-person not blocked). Verified
  headless (crossing swap prevented-with-fusion / happens-without, guard
  preserves re-ID, Hungarian incl. optimal-beats-greedy). Limit: colour can't
  separate look-alikes — a learned/face embedding is the next step.
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
- **Upgrade the body embedder to OSNet** (purpose-trained re-ID via
  onnxruntime-web) — the current MobileNet-features baseline is decent but not
  re-ID-optimised. Drops into the injected `BodyEmbedder`.
- **Offload embedders to a Web Worker + OffscreenCanvas** so per-person CNN
  inference never hitches the render loop (the perf lever).
- **Motion prediction (Kalman)** to disambiguate crossings from trajectory.
- **Rendering polish** (framing/headroom, dead-zone, HQ resampling) — brainstormed, not started.
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
