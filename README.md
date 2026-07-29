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

HeadTracker doesn't ship a model — you load one yourself, so you can swap in a
better one anytime. **Ask Arthur for a model file** (`.onnx`), or point the
demo at a hosted model URL.

Any standard **Ultralytics YOLOv8 head-detection export** works. Two you may be
handed:

| File | Size | Notes |
|------|------|-------|
| `nano.onnx`   | ~12 MB | Fast and light. Good default, works on any machine. |
| `medium.onnx` | ~99 MB | More accurate, needs a bit more GPU. |

There's also a **YOLOE** option (an open-vocabulary model exported with the
prompt "head") — pick the matching format in the dropdown when you load it.

The first time you load a model, the browser spends a few seconds compiling it
(you'll see it "think"). After that it's fast.

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

All sliders apply **live** — adjust them while it's running.

| Control | What it does |
|---|---|
| **Detection period** (0.2–2 s) | How often the model looks for heads. Shorter = more responsive but heavier. 0.5 s is a good balance; with the `medium`/YOLOE models, stay at 0.5 s or higher. |
| **Extracted box size** (×head) | How much of the frame each stream shows around the head. Larger = more zoomed-out / more context; smaller = tight on the face. |
| **Merge zone (X:9)** | How close two heads must get before their streams **merge** into one (see below). Lower = they have to be very close; higher = they merge more easily. |
| **Score threshold** (0–100%) | Detections below this confidence are ignored. Default 60%. Raise it if you get spurious boxes; lower it if real heads are being missed. |
| **Video speed** (×0.1–×1) | Slows a loaded video down, handy for watching the tracking behave frame by frame. |

### What you'll see happen

- **A new head isn't streamed instantly.** It must be detected **3 times in a
  row** first — this stops flickers and false positives from spawning junk
  streams.
- **Losing a head doesn't cut its stream.** If a head disappears (turns away,
  walks out, a missed detection), its stream is **held for 5 seconds**, frozen
  on the last spot and greyed out. If the head comes back within that window,
  the same stream resumes — no new number. After 5 s with no return, the
  stream closes.
- **Close heads merge.** When two people get close enough that one head's
  centre enters the **inner X:9 core** of the other's box, their two streams
  **merge into one** that frames both. They **split apart again** only once a
  centre leaves the other's full 16:9 box. The gap between "merge" and "split"
  is deliberate — it stops streams flickering when people stand near each
  other. The **Merge zone** slider sets how eager that merge is.

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
- **Streams merge or split too readily.** Tune the **Merge zone** slider (lower
  X = harder to merge).

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
├── proximityTracker.ts     nearest-distance matching, 3-hit confirm, 5 s lost hold
├── boxGrouping.ts          hysteretic merge/split of close heads (X:9 merge zone)
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
