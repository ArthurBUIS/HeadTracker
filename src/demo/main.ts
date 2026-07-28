/**
 * Demo harness for the SIMPLE face pipeline (embedding-free).
 *
 * face-api SSD MobileNet detects faces periodically → ProximityTracker
 * nearest-matches them across runs → each tracked face becomes a 300×200
 * output stream. No embeddings, no re-ID models. This is the demo for the
 * `simple-face-pipeline` branch; it drives `src/core/simple`.
 */

import * as tf from '@tensorflow/tfjs';
// nobundle face-api shares the app's one tfjs engine (no "two globals" crash).
import * as faceapi from '@vladmandic/face-api/dist/face-api.esm-nobundle.js';

import {
  SimpleFaceEngine,
  type FaceCenterDetector,
  type SimpleFaceCallbacks,
} from '../core/simple';

const FACE_MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model';

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

const tileById = new Map<number, HTMLElement>();

let engine: SimpleFaceEngine | null = null;
let modelLoaded = false;
let currentObjectUrl: string | null = null;
let detectionIntervalMs = Number(intervalInput.value);

function setStatus(text: string): void {
  statusEl.textContent = text;
}
function formatSeconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)} s`;
}

/** face-api SSD face detector → face centres + sizes (no landmarks/descriptors). */
const faceDetector: FaceCenterDetector = {
  async detectFaces(source) {
    const engineTf = (
      faceapi.tf as unknown as { engine: () => { startScope(): void; endScope(): void } }
    ).engine();
    engineTf.startScope();
    try {
      const detections = await faceapi.detectAllFaces(
        source,
        new faceapi.SsdMobilenetv1Options({ minConfidence: 0.3 }),
      );
      return detections.map((d) => ({
        cx: d.box.x + d.box.width / 2,
        cy: d.box.y + d.box.height / 2,
        size: d.box.height,
      }));
    } finally {
      engineTf.endScope();
    }
  },
};

async function ensureModelLoaded(): Promise<void> {
  if (modelLoaded) return;
  try {
    await tf.setBackend('webgl');
  } catch {
    /* fall back to whatever backend is available */
  }
  await tf.ready();
  await faceapi.nets.ssdMobilenetv1.loadFromUri(FACE_MODEL_URL);
  modelLoaded = true;
}

function addTile(id: number, stream: MediaStream): void {
  const tile = document.createElement('div');
  tile.className = 'tile';
  const video = document.createElement('video');
  video.autoplay = true;
  video.playsInline = true;
  video.muted = true;
  video.srcObject = stream;
  video.width = 300;
  video.height = 200;
  const label = document.createElement('div');
  label.className = 'tile-label';
  label.textContent = `face #${id}`;
  tile.appendChild(video);
  tile.appendChild(label);
  gridEl.appendChild(tile);
  tileById.set(id, tile);
}
function setTileLost(id: number, lost: boolean): void {
  const tile = tileById.get(id);
  if (!tile) return;
  tile.classList.toggle('lost', lost);
  const label = tile.querySelector('.tile-label');
  if (label) label.textContent = lost ? `face #${id} (lost)` : `face #${id}`;
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
    onDiagnostics: (d) => {
      debugEl.textContent =
        `round ${d.round} · detected ${d.detected} · nb_faces ${d.faceCount} ` +
        `(${d.lost} lost) · streams ${tileById.size}`;
    },
  };
  engine = new SimpleFaceEngine(faceDetector, callbacks, { detectionIntervalMs });
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

loadModelButton.addEventListener('click', () => {
  if (modelLoaded) return;
  loadModelButton.disabled = true;
  loadModelButton.textContent = 'Loading…';
  setStatus('Loading face-api SSD MobileNet…');
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
