/**
 * Browser demo harness for the head-tracking core.
 *
 * Two source modes, both fed to the SAME engine as a live source:
 *   - a webcam MediaStream, or
 *   - a pre-recorded video FILE, played (and looped) in real time.
 * A playing <video> is a `FrameSource` either way, so the engine treats a
 * file exactly like a live stream — every render frame samples whatever
 * frame the <video> is currently showing, at real-time playback speed.
 *
 * Each emitted 200x200 stream is mounted as a labelled grid tile. This is
 * NOT the code that ships to portals — it's the runnable proof that the
 * src/core algorithm works. In portals the engine would be driven by a
 * Daily.co track and each HeadStream fed to a video tile.
 */

// Pose model: MoveNet MultiPose on tfjs. Importing '@tensorflow/tfjs'
// registers the webgl/cpu backend the model runs on. Pose keypoints give
// the actual head position (steadier than a body-box heuristic) and still
// resolve heads from behind via shoulder geometry.
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';
// The **nobundle** face-api build imports the app's external @tensorflow/tfjs
// instead of inlining its own, so face recognition shares the ONE tfjs engine
// MoveNet already uses (avoids the "two TensorFlow globals" crash).
import * as faceapi from '@vladmandic/face-api/dist/face-api.esm-nobundle.js';

import {
  HeadTrackerEngine,
  type PoseDetectorLike,
  type FaceEmbedder,
  type FaceObservation,
} from '../core';

// face-api hosts the SSD / landmark / recognition weights under /model.
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
const reidInput = document.getElementById('reid') as HTMLInputElement;
const faceReidInput = document.getElementById('faceReid') as HTMLInputElement;

const tileById = new Map<number, HTMLElement>();

let engine: HeadTrackerEngine | null = null;
let poseDetector: poseDetection.PoseDetector | null = null;
let faceModelsLoaded = false;
let currentObjectUrl: string | null = null;
let detectionIntervalMs = Number(intervalInput.value);
let appearanceReid = reidInput.checked;
let faceReid = faceReidInput.checked;

/** face-api face detector + 128-D embedder, wrapping the FaceEmbedder API. */
const faceEmbedder: FaceEmbedder = {
  async embedFaces(source): Promise<FaceObservation[]> {
    const engineTf = (
      faceapi.tf as unknown as { engine: () => { startScope(): void; endScope(): void } }
    ).engine();
    engineTf.startScope();
    try {
      const results = await faceapi
        .detectAllFaces(source, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.3 }))
        .withFaceLandmarks()
        .withFaceDescriptors();
      return results.map((r) => ({
        box: {
          x: r.detection.box.x,
          y: r.detection.box.y,
          width: r.detection.box.width,
          height: r.detection.box.height,
        },
        descriptor: r.descriptor as Float32Array,
      }));
    } finally {
      engineTf.endScope();
    }
  },
};

/** Load the three face-api nets once (detector + landmarks + recognition). */
async function ensureFaceModelsLoaded(): Promise<void> {
  if (faceModelsLoaded) return;
  await tf.ready();
  await faceapi.nets.ssdMobilenetv1.loadFromUri(FACE_MODEL_URL);
  await faceapi.nets.faceLandmark68Net.loadFromUri(FACE_MODEL_URL);
  await faceapi.nets.faceRecognitionNet.loadFromUri(FACE_MODEL_URL);
  faceModelsLoaded = true;
}

/** Turn face re-ID on for the running engine, loading its models on demand. */
async function activateFaceReid(): Promise<void> {
  if (!engine) return;
  try {
    setFaceState('loading…');
    setStatus('Loading face-recognition models…');
    await ensureFaceModelsLoaded();
    if (!engine) return; // may have stopped while loading
    engine.setFaceEmbedder(faceEmbedder);
    engine.setFaceReid(true);
    setFaceState(`active (${tf.getBackend()})`);
    // eslint-disable-next-line no-console
    console.log(`[HeadTracker] face models loaded — tf backend: ${tf.getBackend()}`);
    setStatus('Face re-ID active.');
  } catch (err) {
    setFaceState('FAILED to load');
    // eslint-disable-next-line no-console
    console.error('[HeadTracker] face model load failed:', err);
    setStatus(`Face re-ID unavailable: ${err instanceof Error ? err.message : String(err)}`);
  }
}

function setStatus(text: string): void {
  statusEl.textContent = text;
}

let faceState = 'off';
let lastDiagLine = '';
function renderDebug(): void {
  debugEl.textContent = `face models: ${faceState}   |   ${lastDiagLine || '(waiting for detection…)'}`;
}
function setFaceState(state: string): void {
  faceState = state;
  renderDebug();
}

function formatSeconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)} s`;
}

function addTile(id: number, stream: MediaStream): void {
  const tile = document.createElement('div');
  tile.className = 'tile';

  const video = document.createElement('video');
  video.autoplay = true;
  video.playsInline = true;
  video.muted = true;
  video.srcObject = stream;
  video.width = 200;
  video.height = 200;

  const label = document.createElement('div');
  label.className = 'tile-label';
  label.textContent = `head #${id}`;
  label.dataset.id = String(id);

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
  if (label) label.textContent = lost ? `head #${id} (lost)` : `head #${id}`;
}

function removeTile(id: number): void {
  const tile = tileById.get(id);
  if (tile) {
    tile.remove();
    tileById.delete(id);
  }
}

/** Load the MoveNet pose detector once; idempotent across source switches. */
async function ensureModelLoaded(): Promise<void> {
  if (poseDetector) return;
  setStatus('Loading MoveNet MultiPose model…');
  // Force the webgl backend before any model loads. face-api's landmark /
  // recognition nets are unreliable on webgpu (they can hang, which stalls
  // the detection loop and stops streams appearing); webgl is the tested
  // path for MoveNet AND face-api, and they share this one engine.
  try {
    await tf.setBackend('webgl');
  } catch {
    /* fall back to whatever backend is available */
  }
  await tf.ready();
  poseDetector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
    modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING,
    enableTracking: false,
  });
}

/** Stop any running engine and any previous source (webcam tracks / file URL). */
function teardownCurrentSource(): void {
  if (engine) {
    engine.stop(); // fires onHeadStreamRemoved for every slot → clears tiles
    engine = null;
  }
  const previous = sourceVideo.srcObject as MediaStream | null;
  if (previous) {
    previous.getTracks().forEach((t) => t.stop());
    sourceVideo.srcObject = null;
  }
  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
  }
  // Belt-and-suspenders in case a tile survived a race.
  gridEl.replaceChildren();
  tileById.clear();
}

/** Build the engine on the (already playing) source video and start it. */
function startEngineOnSource(): void {
  if (!poseDetector) throw new Error('Model not loaded');
  engine = HeadTrackerEngine.withMoveNet(
    poseDetector as unknown as PoseDetectorLike,
    {
      onHeadStreamAdded: ({ id, stream }) => {
        addTile(id, stream);
        setStatus(`Tracking ${tileById.size} head(s).`);
      },
      onHeadStreamLost: (id) => setTileLost(id, true),
      onHeadStreamResumed: (id) => setTileLost(id, false),
      onHeadStreamRemoved: (id) => {
        removeTile(id);
        setStatus(`Tracking ${tileById.size} head(s).`);
      },
      onDiagnostics: (d) => {
        lastDiagLine =
          `round ${d.round} · dets ${d.detections} · tracks ${d.activeTracks} · ` +
          `appearance ${d.appearanceReidEnabled ? 'on' : 'off'} · ` +
          `face ${d.faceReidActive ? 'on' : 'off'} · ` +
          `faces ${d.facesDetected}→${d.facesAttached} attached`;
        renderDebug();
        // eslint-disable-next-line no-console
        console.log(`[HeadTracker] ${lastDiagLine}`);
      },
    },
    { detectionIntervalMs, appearanceReid },
  );
  engine.start(sourceVideo);
  if (faceReid) void activateFaceReid();

  // Expose for ad-hoc inspection from the devtools console.
  (window as unknown as { headEngine: HeadTrackerEngine }).headEngine = engine;
}

async function startWebcam(): Promise<void> {
  startButton.disabled = true;
  loadVideoButton.disabled = true;
  try {
    await ensureModelLoaded();
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
    setStatus('Running on webcam. Detection every 2 s; crops glide between detections.');
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
    await ensureModelLoaded();
    teardownCurrentSource();

    setStatus(`Opening “${file.name}”…`);
    currentObjectUrl = URL.createObjectURL(file);
    sourceVideo.src = currentObjectUrl;
    sourceVideo.loop = true; // replay so you can keep watching the algorithm

    // Wait until the first frame is decoded so videoWidth/Height are known
    // before the engine starts sampling.
    await new Promise<void>((resolve, reject) => {
      sourceVideo.onloadeddata = () => resolve();
      sourceVideo.onerror = () => reject(new Error('Could not decode this video file.'));
    });
    await sourceVideo.play();

    captionEl.textContent = `Source: ${file.name} (looping, real time)`;
    startEngineOnSource();
    setStatus(`Running on “${file.name}”, fed to the algorithm as a live stream.`);
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
  // Applies immediately to a running engine; also used for the next start.
  engine?.setDetectionInterval(detectionIntervalMs);
});

reidInput.addEventListener('change', () => {
  appearanceReid = reidInput.checked;
  engine?.setAppearanceReid(appearanceReid);
});

faceReidInput.addEventListener('change', () => {
  faceReid = faceReidInput.checked;
  if (faceReid) {
    void activateFaceReid();
  } else {
    engine?.setFaceReid(false);
    setFaceState('off');
  }
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
  // Reset so choosing the same file again re-fires 'change'.
  videoFileInput.value = '';
});
