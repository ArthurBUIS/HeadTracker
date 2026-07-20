/**
 * Whole-body re-identification helpers.
 *
 * The colour histogram can't separate look-alikes and the face embedding
 * only exists when a face is visible — so neither covers the case that hurt
 * most: telling people apart while they're turned AWAY from the camera. A
 * whole-body embedding does: a CNN maps the person's torso/body crop to a
 * feature vector where the same person (from any angle) is close and
 * different people are far, by clothing texture/pattern/shape — not just
 * average colour.
 *
 * This is the any-angle backbone cue (stronger than colour, works facing
 * away). The default embedder is MobileNet deep features (a solid,
 * self-contained baseline); a purpose-trained re-ID model (OSNet) can drop
 * into the same injected `BodyEmbedder` for more discriminative results.
 *
 * DOM-free: the actual CNN runs in an injected embedder; this module only
 * does the vector maths. Similarity is cosine, mapped to [0, 1].
 */

/** A whole-body feature embedding (e.g. MobileNet pooled features). */
export type BodyDescriptor = Float32Array;

/** Cosine similarity clamped to [0, 1]; 0 if either descriptor is missing. */
export function bodyAffinity(
  a: BodyDescriptor | undefined,
  b: BodyDescriptor | undefined,
): number {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na <= 0 || nb <= 0) return 0;
  const cos = dot / (Math.sqrt(na) * Math.sqrt(nb));
  return Math.max(0, Math.min(1, cos));
}

/** EMA-blend a stored body descriptor toward a fresh one (weight on new). */
export function blendBody(
  previous: BodyDescriptor,
  next: BodyDescriptor,
  weight: number,
): BodyDescriptor {
  const out = new Float32Array(previous.length);
  for (let i = 0; i < previous.length; i += 1) {
    out[i] = (1 - weight) * previous[i] + weight * next[i];
  }
  return out;
}
