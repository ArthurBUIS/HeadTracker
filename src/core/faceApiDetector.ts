/**
 * A HeadDetector backed by @vladmandic/face-api's SsdMobilenetv1 —
 * the exact model portals-projector-agent already loads for
 * people-counting (see src/renderer/utils/faceApi.js + FaceDetection.jsx).
 *
 * Two integration notes carried over from that codebase:
 *   1. Detection is wrapped in tf.engine().startScope()/endScope() so the
 *      intermediate GPU tensors are freed after every pass. Without this,
 *      running detection on an interval leaks WebGL memory.
 *   2. face-api returns a **face** box (tight around the face). A head is
 *      larger, so we expand each box by `headExpansion` about its centre
 *      to approximate the head, then let the crop stage add framing on top.
 *
 * The `faceapi` module is injected rather than imported here, so the
 * caller controls which build/instance is used: the demo passes the
 * self-contained bundle; portals would pass its shared esm-nobundle
 * instance so face detection shares the app's single tfjs engine.
 */

import type { FrameSource, HeadDetection, HeadDetector } from './types';

/** The slice of the face-api surface this detector actually calls. */
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

export interface FaceApiDetectorConfig {
  /** SsdMobilenetv1 confidence floor. Matches portals' 0.25. */
  minConfidence: number;
  /**
   * Multiplier applied to each face box (about its centre) to grow it
   * from a face box to an approximate head box. 1.0 = no growth.
   */
  headExpansion: number;
}

export const DEFAULT_FACE_API_DETECTOR_CONFIG: FaceApiDetectorConfig = {
  minConfidence: 0.25,
  headExpansion: 1.4,
};

export class FaceApiHeadDetector implements HeadDetector {
  private readonly config: FaceApiDetectorConfig;

  private readonly detectorOptions: unknown;

  constructor(
    private readonly faceapi: FaceApiLike,
    config: Partial<FaceApiDetectorConfig> = {},
  ) {
    this.config = { ...DEFAULT_FACE_API_DETECTOR_CONFIG, ...config };
    this.detectorOptions = new faceapi.SsdMobilenetv1Options({
      minConfidence: this.config.minConfidence,
    });
  }

  async detectHeads(source: FrameSource): Promise<HeadDetection[]> {
    const engine = this.faceapi.tf.engine();
    engine.startScope();
    try {
      const detections = await this.faceapi.detectAllFaces(source, this.detectorOptions);
      return detections.map((d) => this.faceBoxToHeadBox(d.box, d.score));
    } finally {
      engine.endScope();
    }
  }

  private faceBoxToHeadBox(
    box: { x: number; y: number; width: number; height: number },
    score: number,
  ): HeadDetection {
    const grow = this.config.headExpansion;
    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;
    const width = box.width * grow;
    const height = box.height * grow;
    return {
      x: cx - width / 2,
      y: cy - height / 2,
      width,
      height,
      score,
    };
  }
}
