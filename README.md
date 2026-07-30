# HeadTracker

**One camera in, many head-locked streams out.** HeadTracker takes a single
video (your webcam or a video file) and produces **N live 16:9 streams — one
per person's head** — each following its own head around the frame, with a
stable identity (a box keeps tracking the same head, and gets its own number).

It runs entirely in the browser. Nothing is uploaded — the video, the model,
and all processing stay on your machine.

This branch (`simple-face-pipeline`) is the **demo deliverable**. It uses a
lightweight, privacy-friendly pipeline: a **YOLO head detector** runs a few
times per second, and a **proximity tracker** keeps each head's stream stable
between runs. (A heavier, identity-focused pipeline with face/body re-ID lives
on the `main` branch — see [progress.md](progress.md).)

---

## Quick start

You need **[Node.js](https://nodejs.org) 18 or newer** and a **YOLO head model
file** (`.onnx` — see [Get a model](#get-a-model) below).

```bash
npm install
npm run dev
```

Then open **http://127.0.0.1:5180** in your browser.

> Use `127.0.0.1`, **not** `localhost` — on some Windows setups the browser
> resolves `localhost` to IPv6 and can't connect. `127.0.0.1` always works.

**Browser:** use **Chrome or Edge** (recent version). They support WebGPU,
which runs the model ~4× faster and keeps the video tiles smooth. It also
works in other browsers via CPU, just slower.

---

## Get a model

HeadTracker doesn't bundle a model — you load one yourself (so you can swap in
a better one anytime). The recommended one is **`nano.onnx`**: a small, fast
YOLOv8 head detector that runs on any machine. Models are not committed to the
repo (they're big and gitignored).

**How to get `nano.onnx`:**

1. **Easiest — ask Arthur for the file.** It's ~12 MB; grab it from the shared
   drive / Slack and keep it somewhere handy. Then load it in the demo (see
   below). Done.

2. **Or make it yourself** from the YOLOv8-head weights (`nano.pt`, also from
   Arthur). Export it to ONNX once with [Ultralytics](https://docs.ultralytics.com):

   ```bash
   pip install ultralytics
   yolo export model=nano.pt format=onnx imgsz=640 opset=12 simplify=True
   ```

   That writes `nano.onnx` next to the `.pt`. This is exactly how the file in
   use was produced — any standard Ultralytics YOLOv8 **detection** export
   (input `[1,3,640,640]`, output `[1,4+classes,8400]`) works, so you can point
   it at a different head model the same way.

There's also a **`medium.onnx`** (~99 MB, more accurate, wants a bit more GPU)
and a **YOLOE** option — an open-vocabulary model exported with the text prompt
"head". For YOLOE, pick *YOLOE segmentation — 1-class "head"* in the format
dropdown when you load it.

The first time you load any model the browser spends a few seconds compiling it
(you'll see it "think"); after that it's fast.

---

## Using the demo

1. **Choose the model format** in the dropdown:
   - *YOLOv8 detection (auto classes)* — for `nano.onnx` / `medium.onnx`.
   - *YOLOE segmentation — 1-class "head"* — for a YOLOE "head" export.
2. **Pick the model** — either choose the `.onnx` **file**, or paste a model
   **URL** — then click **Load model**.
3. **Choose a source:**
   - **Start webcam** — your live camera.
   - **Load video file…** — pick any local video; it plays on a loop and is
     fed to the algorithm as if it were live. Best way to test on real footage.

The left pane shows the source. The grid on the right fills with one **320×180
(16:9)** stream per tracked head, arranged in columns of three. Each tile is
labelled with its head's number and current detection score (e.g.
`stream #2 · 87%`).

### The controls

All controls apply **live** — adjust them while it's running.

| Control | What it does |
|---|---|
| **Detection period** (0.2–2 s, default 0.5) | How often the model looks for heads. Shorter = more responsive but heavier. With the `medium`/YOLOE models, stay at 0.5 s or higher. |
| **Extracted box size** (×head, default 2.0) | How much of the frame each stream shows around the head. Larger = more zoomed-out / more context; smaller = tight on the face. |
| **Merge method** (default box overlap) | How the app decides two heads are close enough to share one stream — see [Merging](#merging-close-heads) below. Switching it enables that method's own sliders and greys the other's. |
| **Merge zone (X:9)** | *(head-proximity method)* How aligned two heads must be to merge. Lower = they must nearly coincide; higher = merges more easily. |
| **Box overlap** (merge ≥ / split <) | *(box-overlap method, default 70% / 50%)* Two boxes merge when they overlap by ≥ the *merge* %, and split when overlap falls below the *split* %. Keep merge ≥ split. |
| **Lost duration** (1–10, default 3) | How many **missed detections** a stream is kept (greyed) before it's dropped — see [Behaviours](#what-youll-see-happen). |
| **Disengagement** (checkbox + 1–100, default 5) | Turn on a face check: a detected head showing **no face** for this many rounds is greyed as "disengaged". Loads a small extra model the first time it's enabled. |
| **Score threshold** (0–100%, default 60) | Detections below this confidence are ignored. Raise it if you get spurious boxes; lower it if real heads are missed. |
| **Video speed** (×0.1–×1, default ×0.5) | Slows a loaded video down, handy for watching the tracking behave frame by frame. |

<a id="what-youll-see-happen"></a>
### What you'll see happen

- **A new head isn't streamed instantly.** It must be detected **3 times in a
  row** first — this stops flickers and false positives from spawning junk
  streams.
- **Losing a head doesn't cut its stream.** If a head disappears (turns away,
  walks out, a missed detection), its stream is **held** — frozen on the last
  spot and greyed out — for the number of missed detections set by **Lost
  duration** (default 3). If the head comes back within that window, the same
  stream resumes, no new number. Otherwise the stream closes.
- **Disengaged heads go grey (optional).** With **Disengagement** enabled, a
  head that's still detected but shows **no face** for the set number of rounds
  is greyed with a blue outline and a `(disengaged)` label — the person is
  there but turned away. It lights back up the instant a face reappears.

<a id="merging-close-heads"></a>
### Merging close heads

When two people get close, their two streams **merge into one** that frames
both, then **split** again when they separate. There's an intentional gap
between the merge and split points so streams don't flicker when people stand
near each other. Two methods decide "close" — pick one with **Merge method**:

- **Box overlap** (default) — merge when the two stream boxes **overlap by ≥ the
  merge %** (of the smaller box), split when the overlap drops **below the split
  %**. Defaults: merge ≥ 70%, split < 50%.
- **Head proximity** — merge when one head's centre enters the **inner X:9 core**
  of the other's box, split when it leaves the full 16:9 box. The **Merge zone**
  slider (X) sets how aligned they must be.

---

## Troubleshooting

- **"Can't connect to the server."** Open `http://127.0.0.1:5180`, not
  `localhost`. Make sure `npm run dev` is still running in the terminal.
- **Webcam is black / no permission.** The browser must grant camera access on
  `127.0.0.1` (a secure context). Allow it when prompted.
- **Tiles freeze or it's slow.** Use Chrome or Edge (WebGPU). Try the `nano`
  model, or a longer detection period.
- **Too many / too few boxes.** Adjust the **Score threshold** — raise it to
  cut false detections, lower it to catch missed heads.
- **Streams merge or split too readily.** Tune the **Merge method** controls —
  in box-overlap mode raise the *merge %* (harder to merge) or lower the *split
  %* (harder to split); in head-proximity mode lower **Merge zone** X.
- **First face check hangs briefly.** The round where **Disengagement** is first
  enabled compiles the face model (a few seconds), then runs smoothly.

---

## For developers

The reusable algorithm lives in [`src/core/simple`](src/core/simple) and
depends only on DOM / canvas / MediaStream APIs — no Vite, React, or Electron
— so it can be lifted into `portals-projector-agent` as-is. The onnxruntime
model is **injected**, keeping the core runtime-agnostic.

```
src/core/simple/
├── yolov8HeadDetector.ts   letterbox a frame → run YOLO → head centres + sizes
├── yoloPostprocess.ts      pure decode / NMS / coordinate mapping (unit-tested)
├── proximityTracker.ts     nearest-match, 3-hit confirm, round-based lost + disengaged
├── boxGrouping.ts          hysteretic merge/split (box-overlap OR head-proximity)
├── faceDetector.ts         face-presence gate: interface + face→head mapping
├── faceApiFaceDetector.ts  SsdMobilenetv1 (MobileNet-SSD) impl of that gate
├── simpleFaceEngine.ts     orchestrates the above → one MediaStream per group
└── index.ts                public barrel
```

Every timing/threshold is configurable — see the `DEFAULT_*_CONFIG` objects and
the per-stage config types. [progress.md](progress.md) has the full module map
and the reasoning behind each stage.

```bash
npm run typecheck   # tsc --noEmit, strict
npm run build       # typecheck + production bundle
```
