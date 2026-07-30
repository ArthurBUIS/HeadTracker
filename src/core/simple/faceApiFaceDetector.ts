/**
 * A `FacePresenceDetector` backed by @vladmandic/face-api's SsdMobilenetv1 —
 * the same MobileNet-SSD face detector portals-projector-agent already ships.
 *
 * It runs ONCE per round on the whole frame (not once per head box): one
 * inference yields every face, and `annotateFacePresence` maps them onto the
 * head boxes. Equivalent to per-box cropping but far cheaper.
 *
 * The `faceapi` module is injected (not imported) so the caller owns which
 * build/tfjs engine is used. Detection is wrapped in a tf scope so the
 * intermediate WebGL tensors are freed every pass (else running on an interval
 * leaks GPU memory).
 */

import type { FrameSource } from '../types';
import type { FaceBox, FacePresenceDetector } from './faceDetector';

/** The slice of the face-api surface this detector calls. */
export interface FaceApiLike {
  SsdMobilenetv1Options: new (opts: { minConfidence: number }) => unknown;
  detectAllFaces: (
    input: FrameSource,
    options: unknown,
  ) => Promise<
    Array<{ box: { x: number; y: number; width: number; height: number }; score: number }>
  >;
  tf: { engine: () => { startScope: () => void; endScope: () => void } };
}

export interface FaceApiFaceDetectorConfig {
  /** SsdMobilenetv1 confidence floor. Matches portals' 0.25. */
  minConfidence: number;
}

export const DEFAULT_FACE_API_FACE_DETECTOR_CONFIG: FaceApiFaceDetectorConfig = {
  minConfidence: 0.25,
};

export class FaceApiFaceDetector implements FacePresenceDetector {
  private readonly config: FaceApiFaceDetectorConfig;

  private detectorOptions: unknown;

  constructor(
    private readonly faceapi: FaceApiLike,
    config: Partial<FaceApiFaceDetectorConfig> = {},
  ) {
    this.config = { ...DEFAULT_FACE_API_FACE_DETECTOR_CONFIG, ...config };
    this.detectorOptions = new faceapi.SsdMobilenetv1Options({
      minConfidence: this.config.minConfidence,
    });
  }

  setMinConfidence(minConfidence: number): void {
    this.config.minConfidence = Math.min(1, Math.max(0, minConfidence));
    this.detectorOptions = new this.faceapi.SsdMobilenetv1Options({
      minConfidence: this.config.minConfidence,
    });
  }

  async detectFaceBoxes(source: FrameSource): Promise<FaceBox[]> {
    const eng = this.faceapi.tf.engine();
    eng.startScope();
    try {
      const detections = await this.faceapi.detectAllFaces(source, this.detectorOptions);
      return detections.map((d) => ({
        cx: d.box.x + d.box.width / 2,
        cy: d.box.y + d.box.height / 2,
        w: d.box.width,
        h: d.box.height,
      }));
    } finally {
      eng.endScope();
    }
  }
}
