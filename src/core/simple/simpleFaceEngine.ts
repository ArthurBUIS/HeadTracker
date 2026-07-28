/**
 * SimpleFaceEngine — the orchestrator for the embedding-free pipeline.
 *
 *   detect faces (every intervalMs)  →  ProximityTracker (nearest-match)
 *                                            │
 *                                            ▼
 *   render loop (rAF)  →  per-track 300×200 crop  →  captureStream()
 *
 * A face detector is injected (the demo wires face-api SSD). Each tracked
 * face owns a hidden 300×200 canvas whose captureStream() is the output. The
 * crop is a 3:2 rectangle sized from the face box and EMA-glided toward the
 * latest centre, so it moves smoothly between the periodic detections; a lost
 * track keeps its stream frozen on its last spot until the tracker drops it.
 *
 * Framework-agnostic: DOM + canvas + MediaStream only.
 */

import { clamp, emaStep, emaWeightForTimeConstant } from '../smoothing';
import type { FrameSize, FrameSource } from '../types';
import {
  ProximityTracker,
  type FaceObservation,
  type ProximityTrackerConfig,
} from './proximityTracker';

/** Detects faces in a frame, returning each face's centre + size. */
export interface FaceCenterDetector {
  detectFaces(source: FrameSource): Promise<FaceObservation[]>;
}

export interface SimpleFaceEngineConfig {
  /** Output width in px. Spec: 300. */
  outputWidth: number;
  /** Output height in px. Spec: 200. */
  outputHeight: number;
  /** Detection period in ms (the "toggle bar"). */
  detectionIntervalMs: number;
  /** Crop height as a multiple of face size, before clamping. */
  cropPadding: number;
  /** EMA time constant for the crop glide (seconds). */
  smoothSeconds: number;
  /** Output canvas capture frame rate. */
  outputFps: number;
  tracker: Partial<ProximityTrackerConfig>;
}

export const DEFAULT_SIMPLE_ENGINE_CONFIG: SimpleFaceEngineConfig = {
  outputWidth: 300,
  outputHeight: 200,
  detectionIntervalMs: 500,
  cropPadding: 2.6,
  smoothSeconds: 0.4,
  outputFps: 30,
  tracker: {},
};

export const MIN_SIMPLE_INTERVAL_MS = 200;
export const MAX_SIMPLE_INTERVAL_MS = 2000;

export interface FaceStream {
  id: number;
  stream: MediaStream;
  canvas: HTMLCanvasElement;
}

export interface SimpleFaceDiagnostics {
  round: number;
  /** Faces the detector returned this round. */
  detected: number;
  /** Tracks held (active + lost within hysteresis) — the stream count. */
  faceCount: number;
  /** How many held tracks are currently lost. */
  lost: number;
}

export interface SimpleFaceCallbacks {
  onFaceStreamAdded?: (face: FaceStream) => void;
  onFaceStreamLost?: (id: number) => void;
  onFaceStreamResumed?: (id: number) => void;
  onFaceStreamRemoved?: (id: number) => void;
  onDiagnostics?: (d: SimpleFaceDiagnostics) => void;
}

interface FaceSlot {
  id: number;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  stream: MediaStream;
  lost: boolean;
  /** Smoothed crop state, source px. */
  cx: number;
  cy: number;
  cropH: number;
  /** Latest target from the tracker. */
  targetCx: number;
  targetCy: number;
  targetSize: number;
}

export class SimpleFaceEngine {
  private readonly config: SimpleFaceEngineConfig;

  private readonly tracker: ProximityTracker;

  private readonly slots = new Map<number, FaceSlot>();

  private source: FrameSource | null = null;

  private detectionTimer: ReturnType<typeof setInterval> | null = null;

  private rafHandle: number | null = null;

  private lastFrameTimeMs: number | null = null;

  private detecting = false;

  private running = false;

  private roundCounter = 0;

  constructor(
    private readonly detector: FaceCenterDetector,
    private readonly callbacks: SimpleFaceCallbacks = {},
    config: Partial<SimpleFaceEngineConfig> = {},
  ) {
    this.config = { ...DEFAULT_SIMPLE_ENGINE_CONFIG, ...config };
    this.config.detectionIntervalMs = this.clampInterval(this.config.detectionIntervalMs);
    this.tracker = new ProximityTracker(this.config.tracker);
  }

  start(source: FrameSource): void {
    if (this.running) this.stop();
    this.source = source;
    this.running = true;
    this.lastFrameTimeMs = null;
    void this.runDetectionRound();
    this.startDetectionTimer();
    this.rafHandle = requestAnimationFrame((t) => this.renderLoop(t));
  }

  stop(): void {
    this.running = false;
    if (this.detectionTimer !== null) clearInterval(this.detectionTimer);
    this.detectionTimer = null;
    if (this.rafHandle !== null) cancelAnimationFrame(this.rafHandle);
    this.rafHandle = null;
    for (const id of [...this.slots.keys()]) this.removeSlot(id);
    this.source = null;
  }

  setDetectionInterval(intervalMs: number): void {
    this.config.detectionIntervalMs = this.clampInterval(intervalMs);
    if (this.running) this.startDetectionTimer();
  }

  getDetectionInterval(): number {
    return this.config.detectionIntervalMs;
  }

  private clampInterval(ms: number): number {
    if (!Number.isFinite(ms)) return DEFAULT_SIMPLE_ENGINE_CONFIG.detectionIntervalMs;
    return Math.min(MAX_SIMPLE_INTERVAL_MS, Math.max(MIN_SIMPLE_INTERVAL_MS, Math.round(ms)));
  }

  private startDetectionTimer(): void {
    if (this.detectionTimer !== null) clearInterval(this.detectionTimer);
    this.detectionTimer = setInterval(() => {
      void this.runDetectionRound();
    }, this.config.detectionIntervalMs);
  }

  private frameSize(): FrameSize | null {
    const s = this.source;
    if (!s) return null;
    const width = s instanceof HTMLVideoElement ? s.videoWidth : s.width;
    const height = s instanceof HTMLVideoElement ? s.videoHeight : s.height;
    if (!width || !height) return null;
    return { width, height };
  }

  private async runDetectionRound(): Promise<void> {
    if (!this.running || this.detecting || !this.source) return;
    if (!this.frameSize()) return;
    this.detecting = true;
    try {
      const faces = await this.detector.detectFaces(this.source);
      const nowMs = performance.now();
      const { tracks, removed } = this.tracker.update(faces, nowMs);
      for (const id of removed) this.removeSlot(id);

      let lostCount = 0;
      for (const track of tracks) {
        if (track.status === 'lost') lostCount += 1;
        const slot = this.slots.get(track.id);
        if (slot) {
          slot.targetCx = track.cx;
          slot.targetCy = track.cy;
          slot.targetSize = track.size;
          this.setSlotLost(slot, track.status === 'lost');
        } else {
          this.addSlot(track.id, track.cx, track.cy, track.size);
        }
      }

      this.roundCounter += 1;
      this.callbacks.onDiagnostics?.({
        round: this.roundCounter,
        detected: faces.length,
        faceCount: tracks.length,
        lost: lostCount,
      });
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error('[SimpleFace] detection round failed:', err);
    } finally {
      this.detecting = false;
    }
  }

  private addSlot(id: number, cx: number, cy: number, size: number): void {
    const canvas = document.createElement('canvas');
    canvas.width = this.config.outputWidth;
    canvas.height = this.config.outputHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('[SimpleFace] 2D context unavailable');
    const stream = canvas.captureStream(this.config.outputFps);
    const cropH = Math.max(1, size * this.config.cropPadding);
    const slot: FaceSlot = {
      id, canvas, ctx, stream, lost: false,
      cx, cy, cropH, targetCx: cx, targetCy: cy, targetSize: size,
    };
    this.slots.set(id, slot);
    this.callbacks.onFaceStreamAdded?.({ id, stream, canvas });
  }

  private setSlotLost(slot: FaceSlot, lost: boolean): void {
    if (slot.lost === lost) return;
    slot.lost = lost;
    if (lost) this.callbacks.onFaceStreamLost?.(slot.id);
    else this.callbacks.onFaceStreamResumed?.(slot.id);
  }

  private removeSlot(id: number): void {
    const slot = this.slots.get(id);
    if (!slot) return;
    slot.stream.getTracks().forEach((t) => t.stop());
    this.slots.delete(id);
    this.callbacks.onFaceStreamRemoved?.(id);
  }

  private renderLoop(timestampMs: number): void {
    if (!this.running) return;
    const dt = this.lastFrameTimeMs === null ? 0 : (timestampMs - this.lastFrameTimeMs) / 1000;
    this.lastFrameTimeMs = timestampMs;
    const frame = this.frameSize();
    if (frame && this.source) this.drawAll(frame, dt);
    this.rafHandle = requestAnimationFrame((t) => this.renderLoop(t));
  }

  private drawAll(frame: FrameSize, dt: number): void {
    const source = this.source;
    if (!source) return;
    const { outputWidth, outputHeight } = this.config;
    const aspect = outputWidth / outputHeight; // 3:2
    const weight = emaWeightForTimeConstant(dt, this.config.smoothSeconds);

    for (const slot of this.slots.values()) {
      // Glide centre + crop height toward the latest target.
      slot.cx = emaStep(slot.cx, slot.targetCx, weight);
      slot.cy = emaStep(slot.cy, slot.targetCy, weight);
      slot.cropH = emaStep(slot.cropH, Math.max(1, slot.targetSize * this.config.cropPadding), weight);

      // A 3:2 source rectangle around the centre, clamped to the frame.
      let cropH = Math.min(slot.cropH, frame.height);
      let cropW = cropH * aspect;
      if (cropW > frame.width) {
        cropW = frame.width;
        cropH = cropW / aspect;
      }
      const sx = clamp(slot.cx - cropW / 2, 0, frame.width - cropW);
      const sy = clamp(slot.cy - cropH / 2, 0, frame.height - cropH);
      slot.ctx.drawImage(source, sx, sy, cropW, cropH, 0, 0, outputWidth, outputHeight);
    }
  }
}
