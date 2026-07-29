/**
 * Demo harness for the SIMPLE pipeline (embedding-free).
 *
 * A YOLOv8 HEAD detector (ONNX, via onnxruntime-web) runs periodically →
 * ProximityTracker nearest-matches head centres across runs → each tracked
 * head becomes a 300×200 output stream. No embeddings, no re-ID models. This
 * is the demo for the `simple-face-pipeline` branch; it drives
 * `src/core/simple`.
 */

import * as ort from 'onnxruntime-web';

import {
  SimpleFaceEngine,
  Yolov8HeadDetector,
  type Yolov8Runner,
  type SimpleFaceCallbacks,
} from '../core/simple';

// Serve the onnxruntime WASM binaries from the CDN matching the installed
// version, so Vite doesn't have to bundle them.
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';

/** YOLOv8 square input side. Standard Ultralytics export is 640. */
const YOLO_INPUT_SIZE = 640;

const statusEl = document.getElementById('status') as HTMLElement;
const debugEl = document.getElementById('debug') as HTMLElement;
const captionEl = document.getElementById('sourceCaption') as HTMLElement;
const gridEl = document.getElementById('grid') as HTMLElement;
const sourceVideo = document.getElementById('source') as HTMLVideoElement;
const startButton = document.getElementById('start') as HTMLButtonElement;
const loadVideoButton = document.getElementById('loadVideo') as HTMLButtonElement;
const videoFileInput = document.getElementById('videoFile') as HTMLInputElement;
const intervalInput = document.getElementById('interval') as HTMLInputElement;
const intervalLabel = document.getElementById('intervalLabel') as HTMLElement;
const loadModelButton = document.getElementById('loadModel') as HTMLButtonElement;
const modelFileInput = document.getElementById('modelFile') as HTMLInputElement;
const modelUrlInput = document.getElementById('modelUrl') as HTMLInputElement;
const cropSizeInput = document.getElementById('cropSize') as HTMLInputElement;
const cropSizeLabel = document.getElementById('cropSizeLabel') as HTMLElement;
const scoreThresholdInput = document.getElementById('scoreThreshold') as HTMLInputElement;
const scoreThresholdLabel = document.getElementById('scoreThresholdLabel') as HTMLElement;

const tileById = new Map<number, HTMLElement>();

let engine: SimpleFaceEngine | null = null;
let session: ort.InferenceSession | null = null;
let headDetector: Yolov8HeadDetector | null = null;
let currentObjectUrl: string | null = null;
let detectionIntervalMs = Number(intervalInput.value);
let cropPadding = Number(cropSizeInput.value);
let confThreshold = Number(scoreThresholdInput.value) / 100;

function setStatus(text: string): void {
  statusEl.textContent = text;
}
function formatSeconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)} s`;
}

/**
 * Load the YOLOv8 head model — from the chosen local .onnx file, else the URL
 * — and build the detector. The ONNX runtime is wrapped as a `Yolov8Runner`.
 */
async function ensureModelLoaded(): Promise<void> {
  if (headDetector) return;
  const file = modelFileInput.files?.[0];
  const url = modelUrlInput.value.trim();
  // Always create from a buffer (both file and URL). We fetch the URL
  // ourselves rather than handing it to onnxruntime, whose own URL loader
  // trips over HF redirects / external-data files.
  let buffer: Uint8Array;
  if (file) {
    buffer = new Uint8Array(await file.arrayBuffer());
  } else if (url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Could not fetch model (HTTP ${resp.status}).`);
    buffer = new Uint8Array(await resp.arrayBuffer());
  } else {
    throw new Error('Choose a .onnx file or paste a model URL.');
  }
  // Prefer WebGPU: much faster than WASM and runs off the CPU main thread, so
  // the render loop keeps drawing → the tiles show LIVE video, not a frozen
  // frame. Falls back to WASM where WebGPU isn't available.
  session = await ort.InferenceSession.create(buffer, {
    executionProviders: ['webgpu', 'wasm'],
  });
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  const runner: Yolov8Runner = {
    inputSize: YOLO_INPUT_SIZE,
    async run(input) {
      const s = session;
      if (!s) throw new Error('Session not ready');
      const tensor = new ort.Tensor('float32', input, [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]);
      const output = await s.run({ [inputName]: tensor });
      const o = output[outputName];
      return { data: o.data as Float32Array, dims: [...o.dims] };
    },
  };
  headDetector = new Yolov8HeadDetector(runner, { confThreshold });
}

function addTile(id: number, stream: MediaStream): void {
  const tile = document.createElement('div');
  tile.className = 'tile';
  const video = document.createElement('video');
  video.autoplay = true;
  video.playsInline = true;
  video.muted = true;
  video.srcObject = stream;
  const label = document.createElement('div');
  label.className = 'tile-label';
  label.textContent = `stream #${id}`;
  tile.appendChild(video);
  tile.appendChild(label);
  gridEl.appendChild(tile);
  tileById.set(id, tile);
}
function setTileLost(id: number, lost: boolean): void {
  tileById.get(id)?.classList.toggle('lost', lost);
}
/** Set a tile's label with the score(s) of its member head(s). */
function setTileScore(id: number, scores: number[], lost: boolean): void {
  const label = tileById.get(id)?.querySelector('.tile-label');
  if (!label) return;
  const pct = scores.map((s) => `${Math.round(s * 100)}%`).join(', ');
  label.textContent = `stream #${id} · ${pct}${lost ? ' (lost)' : ''}`;
}
function removeTile(id: number): void {
  tileById.get(id)?.remove();
  tileById.delete(id);
}

function teardownCurrentSource(): void {
  if (engine) {
    engine.stop();
    engine = null;
  }
  const prev = sourceVideo.srcObject as MediaStream | null;
  if (prev) {
    prev.getTracks().forEach((t) => t.stop());
    sourceVideo.srcObject = null;
  }
  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
  }
  gridEl.replaceChildren();
  tileById.clear();
}

function startEngineOnSource(): void {
  const callbacks: SimpleFaceCallbacks = {
    onFaceStreamAdded: ({ id, stream }) => addTile(id, stream),
    onFaceStreamLost: (id) => setTileLost(id, true),
    onFaceStreamResumed: (id) => setTileLost(id, false),
    onFaceStreamRemoved: (id) => removeTile(id),
    onStreamScores: (streams) => {
      for (const s of streams) setTileScore(s.id, s.scores, s.lost);
    },
    onDiagnostics: (d) => {
      debugEl.textContent =
        `round ${d.round} · detected ${d.detected} · confirmed ${d.faceCount} ` +
        `(${d.lost} lost, ${d.pending} pending) · streams ${d.groups}`;
    },
  };
  if (!headDetector) throw new Error('Model not loaded');
  engine = new SimpleFaceEngine(headDetector, callbacks, {
    detectionIntervalMs,
    cropPadding,
  });
  engine.start(sourceVideo);
  (window as unknown as { simpleEngine: SimpleFaceEngine }).simpleEngine = engine;
}

async function startWebcam(): Promise<void> {
  startButton.disabled = true;
  loadVideoButton.disabled = true;
  try {
    teardownCurrentSource();
    setStatus('Requesting webcam…');
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    sourceVideo.srcObject = stream;
    sourceVideo.loop = false;
    await sourceVideo.play();
    captionEl.textContent = 'Source stream (webcam)';
    startEngineOnSource();
    setStatus('Running on webcam.');
  } catch (err) {
    setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    startButton.disabled = false;
    loadVideoButton.disabled = false;
  }
}

async function loadVideoFile(file: File): Promise<void> {
  startButton.disabled = true;
  loadVideoButton.disabled = true;
  try {
    teardownCurrentSource();
    setStatus(`Opening “${file.name}”…`);
    currentObjectUrl = URL.createObjectURL(file);
    sourceVideo.src = currentObjectUrl;
    sourceVideo.loop = true;
    await new Promise<void>((resolve, reject) => {
      sourceVideo.onloadeddata = () => resolve();
      sourceVideo.onerror = () => reject(new Error('Could not decode this video file.'));
    });
    await sourceVideo.play();
    captionEl.textContent = `Source: ${file.name} (looping)`;
    startEngineOnSource();
    setStatus(`Running on “${file.name}”.`);
  } catch (err) {
    setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    startButton.disabled = false;
    loadVideoButton.disabled = false;
  }
}

intervalLabel.textContent = formatSeconds(detectionIntervalMs);
intervalInput.addEventListener('input', () => {
  detectionIntervalMs = Number(intervalInput.value);
  intervalLabel.textContent = formatSeconds(detectionIntervalMs);
  engine?.setDetectionInterval(detectionIntervalMs);
});

cropSizeLabel.textContent = `${cropPadding.toFixed(1)}× head`;
cropSizeInput.addEventListener('input', () => {
  cropPadding = Number(cropSizeInput.value);
  cropSizeLabel.textContent = `${cropPadding.toFixed(1)}× head`;
  engine?.setCropPadding(cropPadding);
});

scoreThresholdLabel.textContent = `${scoreThresholdInput.value}%`;
scoreThresholdInput.addEventListener('input', () => {
  confThreshold = Number(scoreThresholdInput.value) / 100;
  scoreThresholdLabel.textContent = `${scoreThresholdInput.value}%`;
  headDetector?.setConfThreshold(confThreshold);
});

loadModelButton.addEventListener('click', () => {
  if (headDetector) return;
  loadModelButton.disabled = true;
  loadModelButton.textContent = 'Loading…';
  setStatus('Loading YOLOv8 head model (ONNX)…');
  ensureModelLoaded()
    .then(() => {
      startButton.disabled = false;
      loadVideoButton.disabled = false;
      loadModelButton.textContent = 'Model loaded ✓';
      setStatus('Model loaded — start a webcam or load a video file.');
    })
    .catch((err) => {
      loadModelButton.disabled = false;
      loadModelButton.textContent = 'Load model';
      setStatus(`Model load failed: ${err instanceof Error ? err.message : String(err)}`);
    });
});

startButton.addEventListener('click', () => {
  void startWebcam();
});
loadVideoButton.addEventListener('click', () => {
  videoFileInput.click();
});
videoFileInput.addEventListener('change', () => {
  const file = videoFileInput.files?.[0];
  if (file) void loadVideoFile(file);
  videoFileInput.value = '';
});
