/**
 * A HeadDetector backed by a pose model (@tensorflow-models/pose-detection,
 * MoveNet MultiPose).
 *
 * Fixes the head-from-body-box heuristic used by cocoSsdDetector: instead of
 * guessing the head as a fixed slice of a jittery person box, MoveNet gives
 * actual head keypoints (nose/eyes/ears) so the head box tracks the real
 * head, with far less vertical wobble.
 *
 * Facing away from the camera the face keypoints drop out, so we fall back
 * to SHOULDER geometry — the head sits a known fraction above the shoulder
 * line — which keeps a head box even from behind (the reason we detect
 * bodies, not faces). A body box is still emitted (the bounding box of the
 * visible keypoints) so the tracker keeps associating on a large stable box
 * and the engine can sample the torso for appearance.
 *
 * The pose detector is INJECTED so the engine stays model-agnostic and
 * portals can share one loaded instance.
 */

import {
  estimateHeadFromKeypoints,
  keypointBoundingBox,
  type HeadGeometryConfig,
} from './poseHead';
import type { FrameSource, HeadDetection, HeadDetector } from './types';

/** One keypoint from pose-detection (pixel coords, optional confidence/name). */
export interface PoseKeypoint {
  x: number;
  y: number;
  score?: number;
  name?: string;
}

/** One detected person. */
export interface Pose {
  keypoints: PoseKeypoint[];
  score?: number;
}

/** The slice of pose-detection's Detector surface this detector calls. */
export interface PoseDetectorLike {
  estimatePoses(input: FrameSource): Promise<Pose[]>;
}

export interface MoveNetDetectorConfig {
  /** Skip a whole pose below this overall confidence. */
  minPoseScore: number;
  /** Ignore individual keypoints below this confidence. */
  minKeypointScore: number;
  /** Head box side = face-keypoint span × this. */
  faceHeadScale: number;
  /** Head box side = shoulder width × this (facing-away fallback). */
  shoulderHeadScale: number;
  /** Head centre rises above the shoulder line by shoulder width × this. */
  shoulderHeadRise: number;
  /** Body box = visible-keypoint bbox padded by this fraction of its size. */
  bodyPadding: number;
  /** Floor on head box side in px, so a tiny/uncertain head still crops. */
  minHeadSize: number;
}

export const DEFAULT_MOVENET_DETECTOR_CONFIG: MoveNetDetectorConfig = {
  minPoseScore: 0.2,
  minKeypointScore: 0.3,
  faceHeadScale: 2.2,
  shoulderHeadScale: 1.0,
  shoulderHeadRise: 0.6,
  bodyPadding: 0.1,
  minHeadSize: 24,
};


export class MoveNetHeadDetector implements HeadDetector {
  private readonly config: MoveNetDetectorConfig;

  constructor(
    private readonly detector: PoseDetectorLike,
    config: Partial<MoveNetDetectorConfig> = {},
  ) {
    this.config = { ...DEFAULT_MOVENET_DETECTOR_CONFIG, ...config };
  }

  async detectHeads(source: FrameSource): Promise<HeadDetection[]> {
    const poses = await this.detector.estimatePoses(source);
    const out: HeadDetection[] = [];
    for (const pose of poses) {
      if ((pose.score ?? 1) < this.config.minPoseScore) continue;
      const head = estimateHeadFromKeypoints(pose.keypoints, this.headGeometry());
      if (!head) continue;
      out.push({
        x: head.cx - head.size / 2,
        y: head.cy - head.size / 2,
        width: head.size,
        height: head.size,
        score: pose.score ?? 1,
        bodyBox:
          keypointBoundingBox(
            pose.keypoints,
            this.config.minKeypointScore,
            this.config.bodyPadding,
          ) ?? undefined,
      });
    }
    return out;
  }

  private headGeometry(): HeadGeometryConfig {
    return {
      minKeypointScore: this.config.minKeypointScore,
      faceHeadScale: this.config.faceHeadScale,
      shoulderHeadScale: this.config.shoulderHeadScale,
      shoulderHeadRise: this.config.shoulderHeadRise,
      minHeadSize: this.config.minHeadSize,
    };
  }
}
