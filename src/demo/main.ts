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

// Body-detection model: coco-ssd on tfjs — the same pair portals already
// ships. Importing '@tensorflow/tfjs' registers the webgl/cpu backend that
// coco-ssd runs on. Detecting whole bodies (not faces) is what lets the
// tracker follow heads even when people are turned away from the camera.
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

import { HeadTrackerEngine, type CocoSsdModelLike } from '../core';

const statusEl = document.getElementById('status') as HTMLElement;
const captionEl = document.getElementById('sourceCaption') as HTMLElement;
const gridEl = document.getElementById('grid') as HTMLElement;
const sourceVideo = document.getElementById('source') as HTMLVideoElement;
const startButton = document.getElementById('start') as HTMLButtonElement;
const loadVideoButton = document.getElementById('loadVideo') as HTMLButtonElement;
const videoFileInput = document.getElementById('videoFile') as HTMLInputElement;
const intervalInput = document.getElementById('interval') as HTMLInputElement;
const intervalLabel = document.getElementById('intervalLabel') as HTMLElement;
const reidInput = document.getElementById('reid') as HTMLInputElement;

const tileById = new Map<number, HTMLElement>();

let engine: HeadTrackerEngine | null = null;
let cocoSsdModel: cocoSsd.ObjectDetection | null = null;
let currentObjectUrl: string | null = null;
let detectionIntervalMs = Number(intervalInput.value);
let appearanceReid = reidInput.checked;

function setStatus(text: string): void {
  statusEl.textContent = text;
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

  tile.appendChild(video);
  tile.appendChild(label);
  gridEl.appendChild(tile);
  tileById.set(id, tile);
}

function removeTile(id: number): void {
  const tile = tileById.get(id);
  if (tile) {
    tile.remove();
    tileById.delete(id);
  }
}

/** Load the coco-ssd model once; idempotent across source switches. */
async function ensureModelLoaded(): Promise<void> {
  if (cocoSsdModel) return;
  setStatus('Loading coco-ssd body-detection model…');
  await tf.ready();
  cocoSsdModel = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
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
  if (!cocoSsdModel) throw new Error('Model not loaded');
  engine = HeadTrackerEngine.withCocoSsd(
    cocoSsdModel as unknown as CocoSsdModelLike,
    {
      onHeadStreamAdded: ({ id, stream }) => {
        addTile(id, stream);
        setStatus(`Tracking ${tileById.size} head(s).`);
      },
      onHeadStreamRemoved: (id) => {
        removeTile(id);
        setStatus(`Tracking ${tileById.size} head(s).`);
      },
    },
    { detectionIntervalMs, appearanceReid },
  );
  engine.start(sourceVideo);

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
