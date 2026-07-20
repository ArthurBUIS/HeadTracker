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

// Detector: BodyPix instance segmentation on tfjs. Importing '@tensorflow/tfjs'
// registers the webgl/cpu backend. BodyPix gives a per-person MASK (so the
// body embedding uses only that person's pixels — clean through occlusions)
// plus pose keypoints for the head (face keypoints, or shoulders when facing
// away). MobileNet embeds the masked body crop.
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as bodyPix from '@tensorflow-models/body-pix';
import * as poseDetection from '@tensorflow-models/pose-detection';
// The **nobundle** face-api build imports the app's external @tensorflow/tfjs
// instead of inlining its own, so face recognition shares the ONE tfjs engine
// (avoids the "two TensorFlow globals" crash).
import * as faceapi from '@vladmandic/face-api/dist/face-api.esm-nobundle.js';

import {
  HeadTrackerEngine,
  type BodyPixNet,
  type PoseDetectorLike,
  type FaceEmbedder,
  type FaceObservation,
  type BodyEmbedder,
  type HeadTrackerCallbacks,
} from '../core';

type DetectorChoice = 'bodypix' | 'movenet';
type ReidChoice = 'none' | 'colour' | 'body' | 'face';

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
const loadModelsButton = document.getElementById('loadModels') as HTMLButtonElement;
const detectorRadios = [...document.querySelectorAll<HTMLInputElement>('input[name="detector"]')];
const reidRadios = [...document.querySelectorAll<HTMLInputElement>('input[name="reid"]')];

function checkedValue<T extends string>(radios: HTMLInputElement[], fallback: T): T {
  return (radios.find((r) => r.checked)?.value as T) ?? fallback;
}

const tileById = new Map<number, HTMLElement>();

let engine: HeadTrackerEngine | null = null;
let bodyPixNet: bodyPix.BodyPix | null = null;
let poseDetector: poseDetection.PoseDetector | null = null;
let faceModelsLoaded = false;
let mobilenetModel: mobilenet.MobileNet | null = null;
let modelsLoaded = false;
let currentObjectUrl: string | null = null;
let detectionIntervalMs = Number(intervalInput.value);

// Chosen models, fixed at "Load models" time (the radios lock then).
let detectorChoice: DetectorChoice = 'bodypix';
let reidChoice: ReidChoice = 'body';
// Derived from reidChoice — exactly one cue is attached (or none).
let appearanceReid = false;
let bodyReid = true;
let faceReid = false;

/** Whole-body embedder: MobileNet pooled deep features of a body crop. */
const bodyEmbedder: BodyEmbedder = {
  async embed(cropped): Promise<Float32Array> {
    const model = mobilenetModel;
    if (!model) return new Float32Array();
    const embeddingTensor = tf.tidy(() => model.infer(cropped, true) as tf.Tensor);
    const data = await embeddingTensor.data();
    embeddingTensor.dispose();
    return new Float32Array(data);
  },
};

/** Load MobileNet once (feature extractor for the body embedding). */
async function ensureBodyModelLoaded(): Promise<void> {
  if (mobilenetModel) return;
  await tf.ready();
  mobilenetModel = await mobilenet.load({ version: 2, alpha: 1.0 });
}


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

function setStatus(text: string): void {
  statusEl.textContent = text;
}

let faceState = 'off';
let bodyState = 'off';
let lastDiagLine = '';
function renderDebug(): void {
  debugEl.textContent =
    `face: ${faceState} · body: ${bodyState}   |   ` +
    `${lastDiagLine || '(waiting for detection…)'}`;
}
function setFaceState(state: string): void {
  faceState = state;
  renderDebug();
}
function setBodyState(state: string): void {
  bodyState = state;
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

/**
 * Force the webgl backend before any model loads. face-api's landmark /
 * recognition nets are unreliable on webgpu (they can hang, which stalls the
 * detection loop and stops streams); webgl is the tested path for every model
 * here, and they all share this one engine.
 */
async function ensureBackend(): Promise<void> {
  try {
    await tf.setBackend('webgl');
  } catch {
    /* fall back to whatever backend is available */
  }
  await tf.ready();
}

/** Load whichever detector is selected; idempotent across source switches. */
async function ensureDetectorLoaded(): Promise<void> {
  await ensureBackend();
  if (detectorChoice === 'bodypix') {
    if (bodyPixNet) return;
    setStatus('Loading BodyPix (MobileNetV1) segmentation model…');
    bodyPixNet = await bodyPix.load({
      architecture: 'MobileNetV1',
      outputStride: 16,
      multiplier: 0.75,
      quantBytes: 2,
    });
  } else {
    if (poseDetector) return;
    setStatus('Loading MoveNet MultiPose Lightning model…');
    poseDetector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
      modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING,
      enableTracking: false,
    });
  }
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
  const callbacks: HeadTrackerCallbacks = {
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
        `face ${d.faceReidActive ? 'on' : 'off'} · faces ${d.facesDetected}→${d.facesAttached} · ` +
        `body ${d.bodyReidActive ? 'on' : 'off'} · bodies ${d.bodiesEmbedded}`;
      renderDebug();
      // eslint-disable-next-line no-console
      console.log(`[HeadTracker] ${lastDiagLine}`);
    },
  };
  const config = { detectionIntervalMs, appearanceReid };
  engine =
    detectorChoice === 'bodypix'
      ? HeadTrackerEngine.withBodyPix(bodyPixNet as unknown as BodyPixNet, callbacks, config)
      : HeadTrackerEngine.withMoveNet(poseDetector as unknown as PoseDetectorLike, callbacks, config);
  engine.start(sourceVideo);
  // Models are already loaded (via "Load models"); just wire the cues.
  if (bodyReid) {
    engine.setBodyEmbedder(bodyEmbedder);
    engine.setBodyReid(true);
  }
  if (faceReid) {
    engine.setFaceEmbedder(faceEmbedder);
    engine.setFaceReid(true);
  }

  // Expose for ad-hoc inspection from the devtools console.
  (window as unknown as { headEngine: HeadTrackerEngine }).headEngine = engine;
}

async function startWebcam(): Promise<void> {
  startButton.disabled = true;
  loadVideoButton.disabled = true;
  try {
    await ensureDetectorLoaded();
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
    await ensureDetectorLoaded();
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

/** Enable/disable the detector + re-ID radios and the load button. */
function setModelSelectionLocked(locked: boolean): void {
  for (const r of [...detectorRadios, ...reidRadios]) r.disabled = locked;
  loadModelsButton.disabled = locked;
}

/**
 * Load exactly the models for the chosen detector + re-ID, then lock the
 * selection and enable the source buttons. Configure once, up front, so it's
 * unambiguous which models are running before any video is loaded.
 */
async function loadSelectedModels(): Promise<void> {
  detectorChoice = checkedValue<DetectorChoice>(detectorRadios, 'bodypix');
  reidChoice = checkedValue<ReidChoice>(reidRadios, 'body');
  appearanceReid = reidChoice === 'colour';
  bodyReid = reidChoice === 'body';
  faceReid = reidChoice === 'face';
  setModelSelectionLocked(true);
  loadModelsButton.textContent = 'Loading models…';
  try {
    await ensureDetectorLoaded();
    if (bodyReid) {
      setBodyState('loading…');
      await ensureBodyModelLoaded();
      setBodyState('loaded');
    }
    if (faceReid) {
      setFaceState('loading…');
      await ensureFaceModelsLoaded();
      setFaceState('loaded');
    }
    modelsLoaded = true;
    startButton.disabled = false;
    loadVideoButton.disabled = false;
    loadModelsButton.textContent = 'Models loaded ✓';
    setStatus(
      `Loaded: ${detectorLabel(detectorChoice)} · re-ID ${reidLabel(reidChoice)}. ` +
        'Start a webcam or load a video file.',
    );
  } catch (err) {
    setModelSelectionLocked(false); // let the user retry / change selection
    loadModelsButton.textContent = 'Load models';
    setStatus(`Model load failed: ${err instanceof Error ? err.message : String(err)}`);
    // eslint-disable-next-line no-console
    console.error('[HeadTracker] model load failed:', err);
  }
}

function detectorLabel(d: DetectorChoice): string {
  return d === 'bodypix' ? 'BodyPix segmentation' : 'MoveNet pose';
}
function reidLabel(r: ReidChoice): string {
  return { none: 'none', colour: 'colour histogram', body: 'MobileNet body embedding', face: 'face-api face embedding' }[r];
}

loadModelsButton.addEventListener('click', () => {
  if (!modelsLoaded) void loadSelectedModels();
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
