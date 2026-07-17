/**
 * Face-embedding re-identification helpers.
 *
 * The torso colour histogram (appearance.ts) can't separate two people who
 * are dressed alike — the case where crossings still swap ids. A face
 * embedding can: face-api's FaceRecognitionNet (a FaceNet-style network)
 * maps an aligned face to a 128-D vector where the SAME person's faces are
 * close in Euclidean distance and different people are far apart, regardless
 * of clothing.
 *
 * This is a BOOSTER, not a replacement: it only exists for detections whose
 * face is visible and detectable, so people turned away from the camera fall
 * back to the colour histogram. The tracker prefers a face match when both
 * sides have one, else colour.
 *
 * DOM-free: the actual face detection/embedding runs in an injected
 * `FaceEmbedder` (the demo wires face-api); this module only does the vector
 * maths and the face→head box assignment.
 */

import type { Box } from './types';

/** A 128-D face embedding (face-api FaceRecognitionNet output). */
export type FaceDescriptor = Float32Array;

/** Euclidean distance between two descriptors; Infinity if either missing. */
export function faceDistance(
  a: FaceDescriptor | undefined,
  b: FaceDescriptor | undefined,
): number {
  if (!a || !b || a.length !== b.length) return Infinity;
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

/**
 * Similarity in [0, 1] from the distance: 1 at distance 0, 0 once the
 * distance reaches `distanceNorm`. face-api's own same-person threshold is
 * ~0.6, so a `distanceNorm` of ~1.2 puts that at ≈0.5 similarity.
 */
export function faceAffinity(
  a: FaceDescriptor | undefined,
  b: FaceDescriptor | undefined,
  distanceNorm: number,
): number {
  const d = faceDistance(a, b);
  if (!Number.isFinite(d) || distanceNorm <= 0) return 0;
  return Math.max(0, 1 - d / distanceNorm);
}

/** EMA-blend a stored face descriptor toward a fresh one (weight on new). */
export function blendFace(
  previous: FaceDescriptor,
  next: FaceDescriptor,
  weight: number,
): FaceDescriptor {
  const out = new Float32Array(previous.length);
  for (let i = 0; i < previous.length; i += 1) {
    out[i] = (1 - weight) * previous[i] + weight * next[i];
  }
  return out;
}

/** Fraction of `inner` that lies inside `outer` (0–1). */
function containment(inner: Box, outer: Box): number {
  const interW = Math.max(0, Math.min(inner.x + inner.width, outer.x + outer.width) - Math.max(inner.x, outer.x));
  const interH = Math.max(0, Math.min(inner.y + inner.height, outer.y + outer.height) - Math.max(inner.y, outer.y));
  const innerArea = inner.width * inner.height;
  return innerArea <= 0 ? 0 : (interW * interH) / innerArea;
}

/**
 * Assign each face box to the head box that best contains it (a face sits
 * inside its head). Greedy one-to-one by containment; returns, per face, the
 * index of its head box or -1 (`minContainment` default 0.5).
 */
export function matchFacesToBoxes(
  faceBoxes: Box[],
  headBoxes: Box[],
  minContainment = 0.5,
): number[] {
  const pairs: { fi: number; hi: number; score: number }[] = [];
  for (let fi = 0; fi < faceBoxes.length; fi += 1) {
    for (let hi = 0; hi < headBoxes.length; hi += 1) {
      const score = containment(faceBoxes[fi], headBoxes[hi]);
      if (score >= minContainment) pairs.push({ fi, hi, score });
    }
  }
  pairs.sort((p, q) => q.score - p.score);
  const faceToHead = new Array<number>(faceBoxes.length).fill(-1);
  const headTaken = new Array<boolean>(headBoxes.length).fill(false);
  for (const pair of pairs) {
    if (faceToHead[pair.fi] !== -1 || headTaken[pair.hi]) continue;
    faceToHead[pair.fi] = pair.hi;
    headTaken[pair.hi] = true;
  }
  return faceToHead;
}
