/**
 * Appearance descriptors for re-identification.
 *
 * Spatial association (tracker.ts) can't tell two people apart once they
 * cross or occlude each other, and it forgets anyone who fully leaves. An
 * appearance descriptor gives each track a *visual signature* so its id can
 * be recovered in those cases.
 *
 * The descriptor is an HSV colour histogram of the person's torso region —
 * essentially "what colour are their clothes". Chosen over a face embedding
 * (e.g. FaceNet) because it works from ANY angle, including people turned
 * away from the camera, which is the whole reason we moved to body
 * detection. It needs no extra model: just pixels already on screen.
 *
 * This module is DOM-free — it operates on raw RGBA pixel data (an
 * ImageData.data buffer the engine hands it) so it stays as portable as the
 * rest of core. Similarity is histogram intersection in [0, 1].
 */

/** L1-normalised HSV histogram; length = H_BINS * S_BINS * V_BINS. */
export type AppearanceDescriptor = Float32Array;

export const H_BINS = 8;
export const S_BINS = 3;
export const V_BINS = 3;
export const APPEARANCE_LENGTH = H_BINS * S_BINS * V_BINS;

/**
 * Build an L1-normalised HSV histogram from an RGBA pixel buffer (the
 * `.data` of an ImageData). Value is included so a black shirt and a white
 * shirt — both near-zero saturation — still land in different bins.
 */
export function computeHsvHistogram(rgba: Uint8ClampedArray): AppearanceDescriptor {
  const hist = new Float32Array(APPEARANCE_LENGTH);
  let count = 0;
  for (let i = 0; i + 3 < rgba.length; i += 4) {
    // Skip fully-transparent pixels (padding from an out-of-frame crop).
    if (rgba[i + 3] === 0) continue;
    const { h, s, v } = rgbToHsv(rgba[i], rgba[i + 1], rgba[i + 2]);
    const hb = Math.min(H_BINS - 1, Math.floor((h / 360) * H_BINS));
    const sb = Math.min(S_BINS - 1, Math.floor(s * S_BINS));
    const vb = Math.min(V_BINS - 1, Math.floor(v * V_BINS));
    hist[(hb * S_BINS + sb) * V_BINS + vb] += 1;
    count += 1;
  }
  if (count > 0) {
    for (let i = 0; i < hist.length; i += 1) hist[i] /= count;
  }
  return hist;
}

/**
 * Histogram-intersection similarity in [0, 1]; 1 = identical distribution.
 * Returns 0 if either descriptor is missing (so callers can treat "no
 * appearance" as "no evidence").
 */
export function appearanceSimilarity(
  a: AppearanceDescriptor | undefined,
  b: AppearanceDescriptor | undefined,
): number {
  if (!a || !b || a.length !== b.length) return 0;
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) sum += Math.min(a[i], b[i]);
  return sum;
}

/**
 * EMA-blend a track's stored descriptor toward a fresh one (weight in
 * [0, 1] on the new observation), keeping it L1-normalised. Lets the
 * signature adapt to slow lighting/pose change without snapping.
 */
export function blendAppearance(
  previous: AppearanceDescriptor,
  next: AppearanceDescriptor,
  weight: number,
): AppearanceDescriptor {
  const out = new Float32Array(previous.length);
  let sum = 0;
  for (let i = 0; i < previous.length; i += 1) {
    out[i] = (1 - weight) * previous[i] + weight * next[i];
    sum += out[i];
  }
  if (sum > 0) {
    for (let i = 0; i < out.length; i += 1) out[i] /= sum;
  }
  return out;
}

function rgbToHsv(r: number, g: number, b: number): { h: number; s: number; v: number } {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const delta = max - min;

  let h = 0;
  if (delta > 0) {
    if (max === rn) h = 60 * (((gn - bn) / delta) % 6);
    else if (max === gn) h = 60 * ((bn - rn) / delta + 2);
    else h = 60 * ((rn - gn) / delta + 4);
  }
  if (h < 0) h += 360;

  const s = max === 0 ? 0 : delta / max;
  return { h, s, v: max };
}
