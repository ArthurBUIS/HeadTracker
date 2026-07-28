/**
 * SimpleFaceEngine — orchestrator for the embedding-free pipeline.
 *
 *   detect heads (every intervalMs)  →  ProximityTracker (confirm ≥ minHits)
 *                                            │  confirmed tracks
 *                                            ▼
 *   BoxGroupManager (merge close boxes, hysteretic)  →  one stream per GROUP
 *                                            │
 *                                            ▼
 *   render loop (rAF)  →  per-group 16:9 crop  →  captureStream()
 *
 * A group of one is a single head; a merged group frames all its members in
 * one 16:9 crop. Detection runs off the main thread (the injected detector
 * may use WebGPU), so the render loop keeps the tiles live between rounds.
 */

import { clamp, emaStep, emaWeightForTimeConstant } from '../smoothing';
import type { FrameSize, FrameSource } from '../types';
import {
  BoxGroupManager,
  type GroupInput,
  type GroupManagerConfig,
} from './boxGrouping';
import {
  ProximityTracker,
  type FaceObservation,
  type ProximityTrackerConfig,
} from './proximityTracker';

/** Detects heads in a frame, returning each head's centre + size. */
export interface FaceCenterDetector {
  detectFaces(source: FrameSource): Promise<FaceObservation[]>;
}

export interface SimpleFaceEngineConfig {
  /** Output width in px. Spec: 320 (16:9). */
  outputWidth: number;
  /** Output height in px. Spec: 180 (16:9). */
  outputHeight: number;
  /** Detection period in ms (the "toggle bar"). */
  detectionIntervalMs: number;
  /** Crop box height as a multiple of head size. */
  cropPadding: number;
  /** EMA time constant for the crop glide (seconds). */
  smoothSeconds: number;
  /** Output canvas capture frame rate. */
  outputFps: number;
  tracker: Partial<ProximityTrackerConfig>;
  grouping: Partial<GroupManagerConfig>;
}

export const DEFAULT_SIMPLE_ENGINE_CONFIG: SimpleFaceEngineConfig = {
  outputWidth: 320,
  outputHeight: 180,
  detectionIntervalMs: 500,
  cropPadding: 2.6,
  smoothSeconds: 0.4,
  outputFps: 30,
  tracker: {},
  grouping: {},
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
  /** Heads the detector returned this round. */
  detected: number;
  /** Confirmed tracks (active + lost) — before merging. */
  faceCount: number;
  /** Confirmed tracks currently lost. */
  lost: number;
  /** Tracks still awaiting confirmation. */
  pending: number;
  /** Output groups/streams after merging. */
  groups: number;
}

export interface SimpleFaceCallbacks {
  onFaceStreamAdded?: (face: FaceStream) => void;
  onFaceStreamLost?: (id: number) => void;
  onFaceStreamResumed?: (id: number) => void;
  onFaceStreamRemoved?: (id: number) => void;
  onDiagnostics?: (d: SimpleFaceDiagnostics) => void;
}

/** Smoothed crop target for a group, in source px. */
interface CropTarget {
  cx: number;
  cy: number;
  cropH: number;
}

interface GroupSlot {
  id: number;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  stream: MediaStream;
  lost: boolean;
  /** Smoothed crop state. */
  cx: number;
  cy: number;
  cropH: number;
  target: CropTarget;
}

export class SimpleFaceEngine {
  private readonly config: SimpleFaceEngineConfig;

  private readonly tracker: ProximityTracker;

  private readonly groupManager: BoxGroupManager;

  private readonly slots = new Map<number, GroupSlot>();

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
    this.groupManager = new BoxGroupManager(this.config.grouping);
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

  /**
   * Resize the output streams live. Existing slot canvases are resized (the
   * captureStream keeps producing, now at the new resolution); the aspect
   * ratio (width/height) drives the crop shape, so this can change 16:9 ↔
   * other ratios on the fly.
   */
  setOutputSize(width: number, height: number): void {
    this.config.outputWidth = Math.max(1, Math.round(width));
    this.config.outputHeight = Math.max(1, Math.round(height));
    for (const slot of this.slots.values()) {
      slot.canvas.width = this.config.outputWidth;
      slot.canvas.height = this.config.outputHeight;
    }
  }

  private get aspect(): number {
    return this.config.outputWidth / this.config.outputHeight;
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
      const { tracks } = this.tracker.update(faces, nowMs);

      // Each confirmed track's crop box (16:9, sized from head size).
      const inputs: GroupInput[] = [];
      const lostById = new Map<number, boolean>();
      for (const t of tracks) {
        const boxH = Math.max(1, t.size * this.config.cropPadding);
        inputs.push({ id: t.id, cx: t.cx, cy: t.cy, boxW: boxH * this.aspect, boxH });
        lostById.set(t.id, t.status === 'lost');
      }
      const inputById = new Map(inputs.map((i) => [i.id, i]));
      const groups = this.groupManager.update(inputs);
      this.reconcileGroupSlots(groups, inputById, lostById);

      this.roundCounter += 1;
      this.callbacks.onDiagnostics?.({
        round: this.roundCounter,
        detected: faces.length,
        faceCount: tracks.length,
        lost: tracks.filter((t) => t.status === 'lost').length,
        pending: this.tracker.pendingCount,
        groups: groups.length,
      });
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error('[SimpleFace] detection round failed:', err);
    } finally {
      this.detecting = false;
    }
  }

  private reconcileGroupSlots(
    groups: { groupId: number; memberIds: number[] }[],
    inputById: Map<number, GroupInput>,
    lostById: Map<number, boolean>,
  ): void {
    const activeGroupIds = new Set(groups.map((g) => g.groupId));
    for (const id of [...this.slots.keys()]) {
      if (!activeGroupIds.has(id)) this.removeSlot(id);
    }
    for (const group of groups) {
      const target = this.groupCropTarget(group.memberIds, inputById);
      const lost = group.memberIds.every((id) => lostById.get(id));
      const slot = this.slots.get(group.groupId);
      if (slot) {
        slot.target = target;
        this.setSlotLost(slot, lost);
      } else {
        this.addSlot(group.groupId, target);
      }
    }
  }

  /** A 16:9 crop that covers every member box. */
  private groupCropTarget(memberIds: number[], inputById: Map<number, GroupInput>): CropTarget {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const id of memberIds) {
      const b = inputById.get(id);
      if (!b) continue;
      minX = Math.min(minX, b.cx - b.boxW / 2);
      maxX = Math.max(maxX, b.cx + b.boxW / 2);
      minY = Math.min(minY, b.cy - b.boxH / 2);
      maxY = Math.max(maxY, b.cy + b.boxH / 2);
    }
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const bw = maxX - minX;
    const bh = maxY - minY;
    // Grow the short axis so the box is 16:9 and contains both extents.
    const cropH = Math.max(bh, bw / this.aspect);
    return { cx, cy, cropH };
  }

  private addSlot(id: number, target: CropTarget): void {
    const canvas = document.createElement('canvas');
    canvas.width = this.config.outputWidth;
    canvas.height = this.config.outputHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('[SimpleFace] 2D context unavailable');
    const stream = canvas.captureStream(this.config.outputFps);
    const slot: GroupSlot = {
      id, canvas, ctx, stream, lost: false,
      cx: target.cx, cy: target.cy, cropH: target.cropH, target,
    };
    this.slots.set(id, slot);
    this.callbacks.onFaceStreamAdded?.({ id, stream, canvas });
  }

  private setSlotLost(slot: GroupSlot, lost: boolean): void {
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
    const weight = emaWeightForTimeConstant(dt, this.config.smoothSeconds);

    for (const slot of this.slots.values()) {
      slot.cx = emaStep(slot.cx, slot.target.cx, weight);
      slot.cy = emaStep(slot.cy, slot.target.cy, weight);
      slot.cropH = emaStep(slot.cropH, slot.target.cropH, weight);

      let cropH = Math.min(slot.cropH, frame.height);
      let cropW = cropH * this.aspect;
      if (cropW > frame.width) {
        cropW = frame.width;
        cropH = cropW / this.aspect;
      }
      const sx = clamp(slot.cx - cropW / 2, 0, frame.width - cropW);
      const sy = clamp(slot.cy - cropH / 2, 0, frame.height - cropH);
      slot.ctx.drawImage(source, sx, sy, cropW, cropH, 0, 0, outputWidth, outputHeight);
    }
  }
}
