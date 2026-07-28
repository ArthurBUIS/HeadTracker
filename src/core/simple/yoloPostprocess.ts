/**
 * Pure YOLOv8 detection post-processing — no DOM, no runtime. Kept separate
 * so the tricky bits (output decoding, NMS, letterbox coordinate mapping) are
 * unit-testable without a model or a browser.
 *
 * Handles a standard Ultralytics YOLOv8 *detection* export: input [1,3,S,S],
 * output either [1, 4+nc, N] (channels-first, the default) or [1, N, 4+nc].
 * Box coords are (cx, cy, w, h) in letterboxed input pixels; there is no
 * separate objectness — the score is the max class score.
 */

/** A detection in a given coordinate space. */
export interface Detection {
  cx: number;
  cy: number;
  w: number;
  h: number;
  score: number;
}

/** Letterbox transform mapping source pixels → SxS model input. */
export interface Letterbox {
  scale: number;
  padX: number;
  padY: number;
  drawW: number;
  drawH: number;
  inputSize: number;
}

/** Compute the letterbox that fits (srcW × srcH) into a square `inputSize`. */
export function computeLetterbox(srcW: number, srcH: number, inputSize: number): Letterbox {
  const scale = Math.min(inputSize / srcW, inputSize / srcH);
  const drawW = srcW * scale;
  const drawH = srcH * scale;
  return {
    scale,
    padX: (inputSize - drawW) / 2,
    padY: (inputSize - drawH) / 2,
    drawW,
    drawH,
    inputSize,
  };
}

/** Map a detection from letterboxed input coords back to source coords. */
export function mapDetectionToSource(det: Detection, lb: Letterbox): Detection {
  return {
    cx: (det.cx - lb.padX) / lb.scale,
    cy: (det.cy - lb.padY) / lb.scale,
    w: det.w / lb.scale,
    h: det.h / lb.scale,
    score: det.score,
  };
}

/**
 * Decode a YOLOv8 detection output tensor into detections (in input coords),
 * keeping only those with max-class-score ≥ `confThreshold`.
 */
export function decodeYolov8(
  data: Float32Array,
  dims: number[],
  confThreshold: number,
): Detection[] {
  if (dims.length !== 3 || dims[0] !== 1) return [];
  // [1, C, N] when C (=4+nc) is the smaller of the two — the usual export.
  const channelsFirst = dims[1] <= dims[2];
  const channels = channelsFirst ? dims[1] : dims[2];
  const anchors = channelsFirst ? dims[2] : dims[1];
  const numClasses = channels - 4;
  if (numClasses < 1) return [];

  const value = channelsFirst
    ? (c: number, a: number) => data[c * anchors + a]
    : (c: number, a: number) => data[a * channels + c];

  const out: Detection[] = [];
  for (let a = 0; a < anchors; a += 1) {
    let best = 0;
    for (let c = 4; c < channels; c += 1) {
      const s = value(c, a);
      if (s > best) best = s;
    }
    if (best < confThreshold) continue;
    out.push({ cx: value(0, a), cy: value(1, a), w: value(2, a), h: value(3, a), score: best });
  }
  return out;
}

/** IoU of two (cx, cy, w, h) boxes. */
function iou(a: Detection, b: Detection): number {
  const ax1 = a.cx - a.w / 2;
  const ay1 = a.cy - a.h / 2;
  const ax2 = a.cx + a.w / 2;
  const ay2 = a.cy + a.h / 2;
  const bx1 = b.cx - b.w / 2;
  const by1 = b.cy - b.h / 2;
  const bx2 = b.cx + b.w / 2;
  const by2 = b.cy + b.h / 2;
  const iw = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1));
  const ih = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const union = a.w * a.h + b.w * b.h - inter;
  return union <= 0 ? 0 : inter / union;
}

/** Greedy non-maximum suppression, highest score first. */
export function nonMaxSuppression(
  detections: Detection[],
  iouThreshold: number,
  maxDetections = 50,
): Detection[] {
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const kept: Detection[] = [];
  for (const det of sorted) {
    if (kept.length >= maxDetections) break;
    if (kept.every((k) => iou(k, det) <= iouThreshold)) kept.push(det);
  }
  return kept;
}
