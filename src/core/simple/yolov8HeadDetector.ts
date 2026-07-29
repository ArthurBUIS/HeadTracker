/**
 * YOLOv8 head detector for the simple pipeline — a drop-in replacement for
 * the face detector (same `FaceCenterDetector` interface), returning head
 * centres + sizes. Because it detects HEADS (not faces) it also works when a
 * person is turned away from the camera.
 *
 * The ONNX runtime is INJECTED (`Yolov8Runner`) so the core stays free of
 * onnxruntime-web; the demo wires an `ort.InferenceSession`. This class owns
 * the DOM-side work — letterbox the frame into the model's square input, pack
 * it NCHW/RGB/0–1 — and the pure decode/NMS in `yoloPostprocess.ts`.
 */

import type { FaceCenterDetector } from './simpleFaceEngine';
import type { FaceObservation } from './proximityTracker';
import {
  computeLetterbox,
  decodeYolov8,
  mapDetectionToSource,
  nonMaxSuppression,
} from './yoloPostprocess';
import type { FrameSource } from '../types';

/** Runs one forward pass of a YOLOv8 ONNX model on an NCHW input. */
export interface Yolov8Runner {
  /** Square model input side (e.g. 640). */
  readonly inputSize: number;
  /** Run inference; returns the raw output tensor data + its dims. */
  run(inputNchw: Float32Array): Promise<{ data: Float32Array; dims: number[] }>;
}

export interface Yolov8HeadDetectorConfig {
  /** Minimum class score to keep a detection. */
  confThreshold: number;
  /** IoU threshold for non-maximum suppression. */
  iouThreshold: number;
  /**
   * Number of classes. Omit for a plain detection export (inferred as
   * channels-4). Set to 1 for a YOLOE / segmentation export whose output has
   * extra mask channels that must not be read as classes.
   */
  numClasses?: number;
}

export const DEFAULT_YOLOV8_HEAD_DETECTOR_CONFIG: Yolov8HeadDetectorConfig = {
  confThreshold: 0.1,
  iouThreshold: 0.45,
};

export class Yolov8HeadDetector implements FaceCenterDetector {
  private readonly config: Yolov8HeadDetectorConfig;

  private canvas: HTMLCanvasElement | null = null;

  private ctx: CanvasRenderingContext2D | null = null;

  constructor(
    private readonly runner: Yolov8Runner,
    config: Partial<Yolov8HeadDetectorConfig> = {},
  ) {
    this.config = { ...DEFAULT_YOLOV8_HEAD_DETECTOR_CONFIG, ...config };
  }

  /** Set the minimum detection confidence [0, 1] (live). */
  setConfThreshold(threshold: number): void {
    this.config.confThreshold = Math.min(1, Math.max(0, threshold));
  }

  getConfThreshold(): number {
    return this.config.confThreshold;
  }

  /** Set the class count (1 for a YOLOE/seg export), or undefined to infer. */
  setNumClasses(numClasses: number | undefined): void {
    this.config.numClasses = numClasses;
  }

  async detectFaces(source: FrameSource): Promise<FaceObservation[]> {
    const size = this.runner.inputSize;
    const srcW = source instanceof HTMLVideoElement ? source.videoWidth : source.width;
    const srcH = source instanceof HTMLVideoElement ? source.videoHeight : source.height;
    if (!srcW || !srcH) return [];

    if (!this.ctx) {
      this.canvas = document.createElement('canvas');
      this.canvas.width = size;
      this.canvas.height = size;
      this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    }
    const ctx = this.ctx;
    if (!ctx) return [];

    const lb = computeLetterbox(srcW, srcH, size);
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, size, size);
    ctx.drawImage(source, lb.padX, lb.padY, lb.drawW, lb.drawH);
    const { data: rgba } = ctx.getImageData(0, 0, size, size);

    // Pack NCHW, RGB, normalised to 0–1.
    const input = new Float32Array(3 * size * size);
    const plane = size * size;
    for (let i = 0; i < plane; i += 1) {
      input[i] = rgba[i * 4] / 255;
      input[plane + i] = rgba[i * 4 + 1] / 255;
      input[2 * plane + i] = rgba[i * 4 + 2] / 255;
    }

    const { data, dims } = await this.runner.run(input);
    const decoded = decodeYolov8(data, dims, this.config.confThreshold, this.config.numClasses);
    const kept = nonMaxSuppression(decoded, this.config.iouThreshold);
    return kept.map((det) => {
      const s = mapDetectionToSource(det, lb);
      return { cx: s.cx, cy: s.cy, size: Math.max(s.w, s.h), score: det.score };
    });
  }
}
