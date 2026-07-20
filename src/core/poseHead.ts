/**
 * Head geometry from COCO pose keypoints — shared by the MoveNet and
 * BodyPix detectors (both emit the same 17 keypoints).
 *
 * The head is placed from the face keypoints (nose/eyes/ears) when they're
 * confident, and falls back to SHOULDER geometry when the face isn't visible
 * — so it still resolves a head from behind, which is the whole point of
 * detecting bodies rather than faces.
 */

import type { Box } from './types';

/** A pose keypoint in source-video px (name = COCO joint name). */
export interface PoseKeypoint2D {
  x: number;
  y: number;
  score?: number;
  name?: string;
}

export interface HeadGeometryConfig {
  /** Ignore keypoints below this confidence. */
  minKeypointScore: number;
  /** Head box side = face-keypoint span × this. */
  faceHeadScale: number;
  /** Head box side = shoulder width × this (facing-away fallback). */
  shoulderHeadScale: number;
  /** Head centre rises above the shoulder line by shoulder width × this. */
  shoulderHeadRise: number;
  /** Floor on head box side in px. */
  minHeadSize: number;
}

const FACE_KEYPOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'];

/** Head centre + square side from keypoints, or null if too little evidence. */
export function estimateHeadFromKeypoints(
  keypoints: PoseKeypoint2D[],
  config: HeadGeometryConfig,
): { cx: number; cy: number; size: number } | null {
  const named = new Map<string, PoseKeypoint2D>();
  for (const kp of keypoints) {
    if (kp.name && (kp.score ?? 0) >= config.minKeypointScore) named.set(kp.name, kp);
  }

  const face = FACE_KEYPOINTS.map((n) => named.get(n)).filter(
    (kp): kp is PoseKeypoint2D => kp !== undefined,
  );
  if (face.length >= 2) {
    const xs = face.map((k) => k.x);
    const ys = face.map((k) => k.y);
    const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
    const size = Math.max(config.minHeadSize, span * config.faceHeadScale);
    return {
      cx: xs.reduce((a, b) => a + b, 0) / xs.length,
      cy: ys.reduce((a, b) => a + b, 0) / ys.length,
      size,
    };
  }

  const ls = named.get('left_shoulder');
  const rs = named.get('right_shoulder');
  if (ls && rs) {
    const shoulderWidth = Math.hypot(ls.x - rs.x, ls.y - rs.y);
    const size = Math.max(config.minHeadSize, shoulderWidth * config.shoulderHeadScale);
    return {
      cx: (ls.x + rs.x) / 2,
      cy: (ls.y + rs.y) / 2 - shoulderWidth * config.shoulderHeadRise,
      size,
    };
  }
  return null;
}

/** Padded bounding box of the confident keypoints, or null if < 2. */
export function keypointBoundingBox(
  keypoints: PoseKeypoint2D[],
  minKeypointScore: number,
  padFraction: number,
): Box | null {
  const valid = keypoints.filter((k) => (k.score ?? 0) >= minKeypointScore);
  if (valid.length < 2) return null;
  const xs = valid.map((k) => k.x);
  const ys = valid.map((k) => k.y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const width = Math.max(...xs) - minX;
  const height = Math.max(...ys) - minY;
  const pad = padFraction * Math.max(width, height);
  return { x: minX - pad, y: minY - pad, width: width + 2 * pad, height: height + 2 * pad };
}
