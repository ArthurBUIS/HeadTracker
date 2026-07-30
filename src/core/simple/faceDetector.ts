/**
 * Face-presence detection for the simple pipeline — the "disengagement" gate.
 *
 * The head detector says WHERE the heads are; this says which of those heads
 * currently show a FACE. A head with no face for several rounds is treated as
 * "disengaged" (turned away / not attending). Kept behind an injected
 * interface so the core stays free of face-api / tfjs; the demo wires an
 * implementation (`FaceApiFaceDetector`).
 */

import type { FrameSource } from '../types';
import type { FaceObservation } from './proximityTracker';

/** A detected face box, centre + extent in SOURCE pixels. */
export interface FaceBox {
  cx: number;
  cy: number;
  w: number;
  h: number;
}

/** Detects faces in a frame, returning their boxes in source px. */
export interface FacePresenceDetector {
  detectFaceBoxes(source: FrameSource): Promise<FaceBox[]>;
}

/**
 * Annotate each head with whether a detected face sits inside it, in place.
 * A head "has a face" when any face-box centre falls within the head's square
 * (side = head `size`, centred on the head). One face maps to at most one head
 * (its nearest containing head), so two heads don't both claim the same face.
 */
export function annotateFacePresence(
  heads: FaceObservation[],
  faces: FaceBox[],
): FaceObservation[] {
  for (const head of heads) head.hasFace = false;
  for (const face of faces) {
    let best: FaceObservation | null = null;
    let bestDist = Infinity;
    for (const head of heads) {
      const half = head.size / 2;
      if (
        face.cx >= head.cx - half &&
        face.cx <= head.cx + half &&
        face.cy >= head.cy - half &&
        face.cy <= head.cy + half
      ) {
        const d = Math.hypot(face.cx - head.cx, face.cy - head.cy);
        if (d < bestDist) {
          bestDist = d;
          best = head;
        }
      }
    }
    if (best) best.hasFace = true;
  }
  return heads;
}
