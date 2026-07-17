/**
 * HeadTrackerEngine — the orchestrator.
 *
 * Wires the four stages together and exposes the result as N live
 * MediaStreams, one per tracked participant:
 *
 *   detect (every detectionIntervalMs)  ─► HeadIdentityTracker
 *                                              │  stable ids
 *                                              ▼
 *   render loop (every rAF frame)  ─► per-id HeadCropSmoother ─► per-id
 *                                     200x200 canvas ─► captureStream()
 *
 * Detection is throttled to `detectionIntervalMs` (2000 ms per the spec).
 * The render loop runs every animation frame so the crop glides smoothly
 * between detections. Each confirmed track owns a hidden 200x200 canvas
 * whose captureStream() is handed to the caller — in portals that stream
 * would feed a Daily.co track or a video tile.
 *
 * Framework-agnostic on purpose: only DOM + canvas + MediaStream APIs,
 * all available in the Electron renderer. No React, no Vite, no Electron
 * imports.
 */

import { computeHsvHistogram, type AppearanceDescriptor } from './appearance';
import { CocoSsdHeadDetector, type CocoSsdModelLike } from './cocoSsdDetector';
import { FaceApiHeadDetector, type FaceApiLike } from './faceApiDetector';
import { matchFacesToBoxes } from './faceEmbedding';
import { MoveNetHeadDetector, type PoseDetectorLike } from './moveNetDetector';
import { HeadCropSmoother, type HeadCropConfig } from './headCrop';
import {
  DEFAULT_TRACKER_CONFIG,
  HeadIdentityTracker,
  type TrackedHead,
  type TrackerConfig,
} from './tracker';
import type {
  Box,
  FaceEmbedder,
  FrameSize,
  HeadDetection,
  HeadDetector,
  FrameSource,
} from './types';

/** Torso sub-region of a body box, sampled for the appearance histogram. */
const TORSO_X_INSET = 0.2; // trim 20% off each side (arms/background)
const TORSO_TOP = 0.2; // start below the head
const TORSO_BOTTOM = 0.6; // stop above the legs
/** Pixels the torso region is downscaled to before histogramming. */
const APPEARANCE_SAMPLE_SIZE = 24;

export interface HeadStream {
  /** Stable participant id (matches the tracker's track id). */
  id: number;
  /** 200x200 live stream centred on this participant's head. */
  stream: MediaStream;
  /** The backing canvas, exposed for direct DOM mounting in demos. */
  canvas: HTMLCanvasElement;
}

export interface HeadTrackerEngineConfig {
  /** Output square size in px. Spec: 200. */
  outputSize: number;
  /**
   * Detection cadence in ms — how often boxes are recomputed. Adjustable
   * live via `setDetectionInterval`; the demo exposes a 200–2000 ms slider.
   */
  detectionIntervalMs: number;
  /**
   * Wall-clock grace a track keeps coasting with no detection before its
   * stream closes. Held roughly constant as the interval changes by
   * deriving the tracker's `maxMisses` = round(coastSeconds / interval).
   */
  trackCoastSeconds: number;
  /**
   * Enable appearance re-identification: compute a colour signature per
   * detection so ids survive crossings/occlusions and returning people
   * reclaim their number. Toggleable live via `setAppearanceReid`.
   */
  appearanceReid: boolean;
  /**
   * Enable FACE-embedding re-identification: when a `FaceEmbedder` is set
   * (via `setFaceEmbedder`), run it each detection round and attach a 128-D
   * descriptor to the matching head — a strong cue that separates
   * look-alikes. No-op when no embedder is set. Toggle via `setFaceReid`.
   */
  faceReid: boolean;
  /**
   * How long (wall-clock) a lost id stays re-identifiable. Converted to
   * tracker gallery rounds using the current interval.
   */
  reidMemorySeconds: number;
  /**
   * When a head is lost, its stream is kept alive frozen on the last crop
   * for this many seconds (so a brief loss doesn't blank it and a return
   * resumes it seamlessly) before it's finally stopped. Aligns with
   * `reidMemorySeconds` by default so re-ID can revive it while it lingers.
   */
  lostStreamLingerSeconds: number;
  /** Frame rate requested from each output canvas's captureStream. */
  outputFps: number;
  tracker: Partial<TrackerConfig>;
  crop: Partial<HeadCropConfig>;
}

/** Detection interval is clamped to this inclusive range (ms). */
export const MIN_DETECTION_INTERVAL_MS = 200;
export const MAX_DETECTION_INTERVAL_MS = 2000;

export const DEFAULT_ENGINE_CONFIG: HeadTrackerEngineConfig = {
  outputSize: 200,
  detectionIntervalMs: 500,
  trackCoastSeconds: 2.0,
  appearanceReid: true,
  faceReid: false,
  reidMemorySeconds: 30,
  lostStreamLingerSeconds: 30,
  outputFps: 30,
  tracker: DEFAULT_TRACKER_CONFIG,
  crop: {},
};

export interface HeadTrackerCallbacks {
  /** A newly-confirmed participant got a stream. Mount it. */
  onHeadStreamAdded?: (head: HeadStream) => void;
  /**
   * A participant's head was lost, but the stream is kept alive frozen on
   * its last position (see `lostStreamLingerSeconds`). Not unmounted — a
   * hint to indicate "lost" in the UI.
   */
  onHeadStreamLost?: (id: number) => void;
  /** A previously-lost head was re-found; its stream resumes tracking. */
  onHeadStreamResumed?: (id: number) => void;
  /** The stream is finally stopped and gone. Unmount it. */
  onHeadStreamRemoved?: (id: number) => void;
}

interface HeadSlot {
  id: number;
  smoother: HeadCropSmoother;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  stream: MediaStream;
  /** Whether the track was matched in the latest detection round. */
  seen: boolean;
  /**
   * 'lost' = the track died but we keep the stream alive, frozen on the
   * last crop, so a brief loss doesn't blank the stream and a return
   * resumes it seamlessly. Purged to 'removed' after lingering too long.
   */
  state: 'active' | 'lost';
  /** Seconds spent in the 'lost' state (drives the linger timeout). */
  lostSeconds: number;
  target: { centerX: number; centerY: number; size: number };
}

export class HeadTrackerEngine {
  private readonly config: HeadTrackerEngineConfig;

  private readonly detector: HeadDetector;

  private readonly tracker: HeadIdentityTracker;

  private readonly slots = new Map<number, HeadSlot>();

  private source: FrameSource | null = null;

  private detectionTimer: ReturnType<typeof setInterval> | null = null;

  private rafHandle: number | null = null;

  private lastFrameTimeMs: number | null = null;

  private detecting = false;

  private running = false;

  /** Reused offscreen canvas for sampling torso pixels (appearance). */
  private appearanceCanvas: HTMLCanvasElement | null = null;

  private appearanceCtx: CanvasRenderingContext2D | null = null;

  /** Optional face detector+embedder for face-based re-ID (injected). */
  private faceEmbedder: FaceEmbedder | null = null;

  constructor(
    detector: HeadDetector,
    private readonly callbacks: HeadTrackerCallbacks = {},
    config: Partial<HeadTrackerEngineConfig> = {},
  ) {
    this.config = { ...DEFAULT_ENGINE_CONFIG, ...config };
    this.config.detectionIntervalMs = this.clampInterval(this.config.detectionIntervalMs);
    this.detector = detector;
    this.tracker = new HeadIdentityTracker(this.config.tracker);
    this.applyCoastToTracker();
    this.applyReidMemoryToTracker();
  }

  /**
   * Convenience constructor for the common case: build the engine on top
   * of a @vladmandic/face-api instance the caller has already loaded.
   */
  static withFaceApi(
    faceapi: FaceApiLike,
    callbacks: HeadTrackerCallbacks = {},
    config: Partial<HeadTrackerEngineConfig> = {},
  ): HeadTrackerEngine {
    return new HeadTrackerEngine(new FaceApiHeadDetector(faceapi), callbacks, config);
  }

  /**
   * Convenience constructor for the body-detection path: build the engine
   * on top of a loaded @tensorflow-models/coco-ssd model. This is the
   * default in the demo — robust to people facing away from the camera.
   */
  static withCocoSsd(
    model: CocoSsdModelLike,
    callbacks: HeadTrackerCallbacks = {},
    config: Partial<HeadTrackerEngineConfig> = {},
  ): HeadTrackerEngine {
    return new HeadTrackerEngine(new CocoSsdHeadDetector(model), callbacks, config);
  }

  /**
   * Convenience constructor for the pose-detection path: build the engine
   * on a loaded @tensorflow-models/pose-detection MoveNet detector. Heads
   * come from actual keypoints (steadier than the coco-ssd body heuristic)
   * and still resolve from behind via shoulder geometry.
   */
  static withMoveNet(
    detector: PoseDetectorLike,
    callbacks: HeadTrackerCallbacks = {},
    config: Partial<HeadTrackerEngineConfig> = {},
  ): HeadTrackerEngine {
    return new HeadTrackerEngine(new MoveNetHeadDetector(detector), callbacks, config);
  }

  /** Begin detecting/tracking against `source` (a playing `<video>`). */
  start(source: FrameSource): void {
    if (this.running) this.stop();
    this.source = source;
    this.running = true;
    this.lastFrameTimeMs = null;

    // Kick one detection immediately so streams appear without waiting a
    // full interval, then settle into the configured cadence.
    void this.runDetectionRound();
    this.startDetectionTimer();

    this.rafHandle = requestAnimationFrame((t) => this.renderLoop(t));
  }

  /**
   * Change the detection cadence on the fly (clamped to
   * [MIN_DETECTION_INTERVAL_MS, MAX_DETECTION_INTERVAL_MS]). The coast
   * time is preserved by rescaling the tracker's miss tolerance. Safe to
   * call before or during a run.
   */
  setDetectionInterval(intervalMs: number): void {
    this.config.detectionIntervalMs = this.clampInterval(intervalMs);
    this.applyCoastToTracker();
    this.applyReidMemoryToTracker();
    if (this.running) this.startDetectionTimer();
  }

  /** Current (clamped) detection interval in ms. */
  getDetectionInterval(): number {
    return this.config.detectionIntervalMs;
  }

  /** Turn appearance re-identification on/off at runtime. */
  setAppearanceReid(enabled: boolean): void {
    this.config.appearanceReid = enabled;
  }

  /** Whether appearance re-identification is currently on. */
  isAppearanceReidEnabled(): boolean {
    return this.config.appearanceReid;
  }

  /** Inject (or clear) the face detector+embedder used for face re-ID. */
  setFaceEmbedder(embedder: FaceEmbedder | null): void {
    this.faceEmbedder = embedder;
  }

  /** Turn face-embedding re-identification on/off at runtime. */
  setFaceReid(enabled: boolean): void {
    this.config.faceReid = enabled;
  }

  /** Whether face re-ID is on AND an embedder is available. */
  isFaceReidActive(): boolean {
    return this.config.faceReid && this.faceEmbedder !== null;
  }

  private clampInterval(intervalMs: number): number {
    if (!Number.isFinite(intervalMs)) return DEFAULT_ENGINE_CONFIG.detectionIntervalMs;
    return Math.min(
      MAX_DETECTION_INTERVAL_MS,
      Math.max(MIN_DETECTION_INTERVAL_MS, Math.round(intervalMs)),
    );
  }

  /** Derive miss tolerance from the target coast time and current interval. */
  private applyCoastToTracker(): void {
    const intervalSeconds = this.config.detectionIntervalMs / 1000;
    this.tracker.setMaxMisses(this.config.trackCoastSeconds / intervalSeconds);
  }

  /** Derive gallery lifetime (rounds) from the re-ID memory time. */
  private applyReidMemoryToTracker(): void {
    const intervalSeconds = this.config.detectionIntervalMs / 1000;
    this.tracker.setGalleryMaxRounds(this.config.reidMemorySeconds / intervalSeconds);
  }

  /** (Re)start the interval timer with the current cadence. */
  private startDetectionTimer(): void {
    if (this.detectionTimer !== null) clearInterval(this.detectionTimer);
    this.detectionTimer = setInterval(() => {
      void this.runDetectionRound();
    }, this.config.detectionIntervalMs);
  }

  /** Stop everything and tear down every output stream. */
  stop(): void {
    this.running = false;
    if (this.detectionTimer !== null) {
      clearInterval(this.detectionTimer);
      this.detectionTimer = null;
    }
    if (this.rafHandle !== null) {
      cancelAnimationFrame(this.rafHandle);
      this.rafHandle = null;
    }
    for (const id of [...this.slots.keys()]) this.removeSlot(id);
    this.source = null;
  }

  /** Snapshot of the current live streams. */
  getHeadStreams(): HeadStream[] {
    return [...this.slots.values()].map((s) => ({
      id: s.id,
      stream: s.stream,
      canvas: s.canvas,
    }));
  }

  private frameSize(): FrameSize | null {
    const source = this.source;
    if (!source) return null;
    const width =
      source instanceof HTMLVideoElement ? source.videoWidth : source.width;
    const height =
      source instanceof HTMLVideoElement ? source.videoHeight : source.height;
    if (!width || !height) return null;
    return { width, height };
  }

  /**
   * Sample the torso region of a detection into a fixed-size buffer and
   * return its HSV colour histogram. Returns undefined when the region is
   * degenerate or the frame can't be read (e.g. a cross-origin taint).
   */
  private sampleAppearance(
    detection: HeadDetection,
    frame: FrameSize,
  ): AppearanceDescriptor | undefined {
    const source = this.source;
    if (!source) return undefined;
    const region = this.torsoRegion(detection, frame);
    if (!region) return undefined;

    if (!this.appearanceCtx) {
      this.appearanceCanvas = document.createElement('canvas');
      this.appearanceCanvas.width = APPEARANCE_SAMPLE_SIZE;
      this.appearanceCanvas.height = APPEARANCE_SAMPLE_SIZE;
      this.appearanceCtx = this.appearanceCanvas.getContext('2d', {
        willReadFrequently: true,
      });
    }
    const ctx = this.appearanceCtx;
    if (!ctx) return undefined;
    try {
      ctx.drawImage(
        source,
        region.x, region.y, region.width, region.height,
        0, 0, APPEARANCE_SAMPLE_SIZE, APPEARANCE_SAMPLE_SIZE,
      );
      const { data } = ctx.getImageData(0, 0, APPEARANCE_SAMPLE_SIZE, APPEARANCE_SAMPLE_SIZE);
      return computeHsvHistogram(data);
    } catch {
      return undefined;
    }
  }

  /** The clamped torso sub-region of a detection's body box. */
  private torsoRegion(detection: HeadDetection, frame: FrameSize): Box | null {
    const box = detection.bodyBox ?? detection;
    let x = box.x + box.width * TORSO_X_INSET;
    let y = box.y + box.height * TORSO_TOP;
    let width = box.width * (1 - 2 * TORSO_X_INSET);
    let height = box.height * (TORSO_BOTTOM - TORSO_TOP);
    x = Math.max(0, Math.min(x, frame.width - 1));
    y = Math.max(0, Math.min(y, frame.height - 1));
    width = Math.min(width, frame.width - x);
    height = Math.min(height, frame.height - y);
    if (width < 2 || height < 2) return null;
    return { x, y, width, height };
  }

  /**
   * Run the injected face embedder and attach each face's 128-D descriptor
   * to the head box that contains it. Failures are swallowed (face re-ID is
   * a booster; detection/tracking must continue without it).
   */
  private async attachFaceDescriptors(detections: HeadDetection[]): Promise<void> {
    const source = this.source;
    if (!source || !this.faceEmbedder || detections.length === 0) return;
    try {
      const faces = await this.faceEmbedder.embedFaces(source);
      if (faces.length === 0) return;
      const headBoxes: Box[] = detections.map((d) => ({
        x: d.x,
        y: d.y,
        width: d.width,
        height: d.height,
      }));
      const faceToHead = matchFacesToBoxes(
        faces.map((f) => f.box),
        headBoxes,
      );
      for (let fi = 0; fi < faces.length; fi += 1) {
        const di = faceToHead[fi];
        if (di >= 0) detections[di].faceDescriptor = faces[fi].descriptor;
      }
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error('[HeadTracker] face embedding failed:', err);
    }
  }

  private async runDetectionRound(): Promise<void> {
    if (!this.running || this.detecting || !this.source) return;
    if (!this.frameSize()) return; // video not ready yet
    this.detecting = true;
    try {
      const frame = this.frameSize();
      const detections = await this.detector.detectHeads(this.source);
      if (this.config.appearanceReid && frame) {
        for (const detection of detections) {
          detection.appearance = this.sampleAppearance(detection, frame);
        }
      }
      if (this.isFaceReidActive()) {
        await this.attachFaceDescriptors(detections);
      }
      const { active, removedIds } = this.tracker.update(detections);
      this.reconcileSlots(active, removedIds);
    } catch (err) {
      // Detection failures must not kill the render loop; log and move on.
      // eslint-disable-next-line no-console
      console.error('[HeadTracker] detection round failed:', err);
    } finally {
      this.detecting = false;
    }
  }

  /**
   * Reconcile slots with the tracker's result. New ids get a slot; active
   * ids refresh their target (and resume if they were lost); ids the
   * tracker dropped are marked LOST — the stream is kept alive, frozen on
   * its last crop, and only purged later by the linger timeout in the
   * render loop.
   */
  private reconcileSlots(active: TrackedHead[], removedIds: number[]): void {
    const activeIds = new Set(active.map((t) => t.id));

    for (const track of active) {
      const existing = this.slots.get(track.id);
      if (existing) {
        existing.target = {
          centerX: track.centerX,
          centerY: track.centerY,
          size: track.size,
        };
        // misses === 0 means this track was matched in this very round;
        // a carried-over (missed) track uses the slower hold constant.
        existing.seen = track.misses === 0;
        if (existing.state === 'lost') this.resumeSlot(existing);
      } else {
        this.addSlot(track);
      }
    }

    // Any slot whose id is no longer active (this round's deaths, plus any
    // still-lingering earlier losses) enters/stays in the lost state.
    for (const slot of this.slots.values()) {
      if (!activeIds.has(slot.id) && slot.state === 'active') {
        this.markSlotLost(slot);
      }
    }
    // removedIds is implied by the loop above; referenced for clarity only.
    void removedIds;
  }

  private markSlotLost(slot: HeadSlot): void {
    slot.state = 'lost';
    slot.lostSeconds = 0;
    slot.seen = false;
    this.callbacks.onHeadStreamLost?.(slot.id);
  }

  private resumeSlot(slot: HeadSlot): void {
    slot.state = 'active';
    slot.lostSeconds = 0;
    this.callbacks.onHeadStreamResumed?.(slot.id);
  }

  private addSlot(track: TrackedHead): void {
    const size = this.config.outputSize;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('[HeadTracker] 2D canvas context unavailable');
    const stream = canvas.captureStream(this.config.outputFps);

    const slot: HeadSlot = {
      id: track.id,
      smoother: new HeadCropSmoother(
        track.centerX,
        track.centerY,
        track.size,
        this.config.crop,
      ),
      canvas,
      ctx,
      stream,
      seen: true,
      state: 'active',
      lostSeconds: 0,
      target: { centerX: track.centerX, centerY: track.centerY, size: track.size },
    };
    this.slots.set(track.id, slot);
    this.callbacks.onHeadStreamAdded?.({ id: track.id, stream, canvas });
  }

  private removeSlot(id: number): void {
    const slot = this.slots.get(id);
    if (!slot) return;
    slot.stream.getTracks().forEach((t) => t.stop());
    this.slots.delete(id);
    this.callbacks.onHeadStreamRemoved?.(id);
  }

  private renderLoop(timestampMs: number): void {
    if (!this.running) return;
    const dtSeconds =
      this.lastFrameTimeMs === null ? 0 : (timestampMs - this.lastFrameTimeMs) / 1000;
    this.lastFrameTimeMs = timestampMs;

    const frame = this.frameSize();
    if (frame && this.source) this.drawAllSlots(frame, dtSeconds);

    this.rafHandle = requestAnimationFrame((t) => this.renderLoop(t));
  }

  private drawAllSlots(frame: FrameSize, dtSeconds: number): void {
    const source = this.source;
    if (!source) return;
    const outputSize = this.config.outputSize;

    for (const slot of this.slots.values()) {
      if (slot.state === 'lost') {
        slot.lostSeconds += dtSeconds;
        if (slot.lostSeconds >= this.config.lostStreamLingerSeconds) {
          this.removeSlot(slot.id);
          continue;
        }
      }
      // A lost slot keeps its frozen target (seen=false), so the crop holds
      // its last position while still sampling the live frame — an unmoving
      // camera on that spot rather than a blanked stream.
      slot.smoother.step(
        slot.target.centerX,
        slot.target.centerY,
        slot.target.size,
        dtSeconds,
        slot.seen,
      );
      const rect = slot.smoother.getCropRect(frame);
      slot.ctx.drawImage(
        source,
        rect.sx,
        rect.sy,
        rect.side,
        rect.side,
        0,
        0,
        outputSize,
        outputSize,
      );
    }
  }
}
