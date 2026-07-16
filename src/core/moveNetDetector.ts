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

import type { Box, FrameSource, HeadDetection, HeadDetector } from './types';

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

const FACE_KEYPOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'];

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
      const head = this.estimateHead(pose);
      if (!head) continue;
      out.push({
        x: head.cx - head.size / 2,
        y: head.cy - head.size / 2,
        width: head.size,
        height: head.size,
        score: pose.score ?? 1,
        bodyBox: this.estimateBodyBox(pose) ?? undefined,
      });
    }
    return out;
  }

  /** Head centre + size from face keypoints, else shoulder geometry. */
  private estimateHead(pose: Pose): { cx: number; cy: number; size: number } | null {
    const named = new Map<string, PoseKeypoint>();
    for (const kp of pose.keypoints) {
      if (kp.name && (kp.score ?? 0) >= this.config.minKeypointScore) named.set(kp.name, kp);
    }

    const face = FACE_KEYPOINTS.map((n) => named.get(n)).filter(
      (kp): kp is PoseKeypoint => kp !== undefined,
    );
    if (face.length >= 2) {
      const xs = face.map((k) => k.x);
      const ys = face.map((k) => k.y);
      const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
      const size = Math.max(this.config.minHeadSize, span * this.config.faceHeadScale);
      const cx = xs.reduce((a, b) => a + b, 0) / xs.length;
      const cy = ys.reduce((a, b) => a + b, 0) / ys.length;
      return { cx, cy, size };
    }

    const ls = named.get('left_shoulder');
    const rs = named.get('right_shoulder');
    if (ls && rs) {
      const shoulderWidth = Math.hypot(ls.x - rs.x, ls.y - rs.y);
      const size = Math.max(this.config.minHeadSize, shoulderWidth * this.config.shoulderHeadScale);
      const cx = (ls.x + rs.x) / 2;
      const cy = (ls.y + rs.y) / 2 - shoulderWidth * this.config.shoulderHeadRise;
      return { cx, cy, size };
    }
    return null; // not enough evidence for a reliable head
  }

  /** Body box = padded bounding box of the confident keypoints. */
  private estimateBodyBox(pose: Pose): Box | null {
    const valid = pose.keypoints.filter((k) => (k.score ?? 0) >= this.config.minKeypointScore);
    if (valid.length < 2) return null;
    const xs = valid.map((k) => k.x);
    const ys = valid.map((k) => k.y);
    const minX = Math.min(...xs);
    const minY = Math.min(...ys);
    const width = Math.max(...xs) - minX;
    const height = Math.max(...ys) - minY;
    const pad = this.config.bodyPadding * Math.max(width, height);
    return {
      x: minX - pad,
      y: minY - pad,
      width: width + 2 * pad,
      height: height + 2 * pad,
    };
  }
}
