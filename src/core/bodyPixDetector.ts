/**
 * A HeadDetector backed by BodyPix instance segmentation.
 *
 * Why segmentation instead of pose-only detection: when one person walks in
 * front of another, their bounding boxes overlap, so the body crop used for
 * the re-ID embedding fills up with the *occluding* person's pixels and the
 * embedding gets contaminated — which is what let a stream jump to the wrong
 * head. BodyPix gives a per-person MASK, so the engine can embed only that
 * person's own pixels and keep their signature clean through an occlusion.
 *
 * For each segmented person this emits:
 *   - a body box (bounding box of the mask),
 *   - a head box from the pose keypoints BodyPix returns (face keypoints, or
 *     shoulder geometry when facing away — so it still works from behind),
 *   - the person mask, carried on the detection for masked embedding.
 *
 * The BodyPix `net` is INJECTED so the core stays decoupled from the model.
 */

import { estimateHeadFromKeypoints, type PoseKeypoint2D } from './poseHead';
import type { Box, FrameSource, HeadDetection, HeadDetector, PersonMask } from './types';

/** One BodyPix pose keypoint: `{ position, score, part }`. */
export interface BodyPixKeypoint {
  position: { x: number; y: number };
  score: number;
  part: string;
}

/** One BodyPix person segmentation. */
export interface BodyPixPersonSegmentation {
  /** Binary mask, row-major, length width*height (1 = this person). */
  data: Uint8Array;
  width: number;
  height: number;
  pose: { keypoints: BodyPixKeypoint[]; score: number };
}

/** The slice of BodyPix's net surface this detector calls. */
export interface BodyPixNet {
  segmentMultiPerson(
    input: FrameSource,
    config?: unknown,
  ): Promise<BodyPixPersonSegmentation[]>;
}

export interface BodyPixDetectorConfig {
  /** Skip a person whose overall pose score is below this. */
  minPoseScore: number;
  /** Ignore keypoints below this confidence (head geometry). */
  minKeypointScore: number;
  faceHeadScale: number;
  shoulderHeadScale: number;
  shoulderHeadRise: number;
  minHeadSize: number;
  /** Optional per-call config forwarded to `segmentMultiPerson`. */
  segmentationConfig?: unknown;
}

export const DEFAULT_BODY_PIX_DETECTOR_CONFIG: BodyPixDetectorConfig = {
  minPoseScore: 0.2,
  minKeypointScore: 0.3,
  faceHeadScale: 2.2,
  shoulderHeadScale: 1.0,
  shoulderHeadRise: 0.6,
  minHeadSize: 24,
};

export class BodyPixHeadDetector implements HeadDetector {
  private readonly config: BodyPixDetectorConfig;

  constructor(
    private readonly net: BodyPixNet,
    config: Partial<BodyPixDetectorConfig> = {},
  ) {
    this.config = { ...DEFAULT_BODY_PIX_DETECTOR_CONFIG, ...config };
  }

  async detectHeads(source: FrameSource): Promise<HeadDetection[]> {
    const people = await this.net.segmentMultiPerson(source, this.config.segmentationConfig);
    const out: HeadDetection[] = [];
    for (const person of people) {
      if ((person.pose?.score ?? 1) < this.config.minPoseScore) continue;
      const keypoints = this.toKeypoints(person.pose?.keypoints ?? []);
      const head = estimateHeadFromKeypoints(keypoints, {
        minKeypointScore: this.config.minKeypointScore,
        faceHeadScale: this.config.faceHeadScale,
        shoulderHeadScale: this.config.shoulderHeadScale,
        shoulderHeadRise: this.config.shoulderHeadRise,
        minHeadSize: this.config.minHeadSize,
      });
      if (!head) continue;
      const bodyBox = maskBoundingBox(person);
      if (!bodyBox) continue;
      out.push({
        x: head.cx - head.size / 2,
        y: head.cy - head.size / 2,
        width: head.size,
        height: head.size,
        score: person.pose?.score ?? 1,
        bodyBox,
        personMask: { data: person.data, width: person.width, height: person.height } as PersonMask,
      });
    }
    return out;
  }

  /**
   * Adapt BodyPix's `{position, score, part}` to the shared keypoint shape.
   * BodyPix (PoseNet) names parts in camelCase (`leftShoulder`) but the shared
   * head geometry uses MoveNet's snake_case (`left_shoulder`) — normalise, or
   * every head lookup silently misses and no heads are detected at all.
   */
  private toKeypoints(keypoints: BodyPixKeypoint[]): PoseKeypoint2D[] {
    return keypoints.map((k) => ({
      x: k.position.x,
      y: k.position.y,
      score: k.score,
      name: camelToSnake(k.part),
    }));
  }
}

/** `leftShoulder` → `left_shoulder`; already-snake names pass through. */
function camelToSnake(name: string): string {
  return name.replace(/([A-Z])/g, '_$1').toLowerCase();
}

/** Tight bounding box of a person's mask, or null if the mask is empty. */
function maskBoundingBox(person: BodyPixPersonSegmentation): Box | null {
  const { data, width, height } = person;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    for (let x = 0; x < width; x += 1) {
      if (data[row + x]) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < minX || maxY < minY) return null;
  return { x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1 };
}
