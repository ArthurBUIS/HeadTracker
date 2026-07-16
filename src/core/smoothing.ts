/**
 * Time-constant EMA smoothing.
 *
 * Ported from VideoStitcher's stitcher/person_tracking.py, which uses
 *     alpha = 1 - exp(-1 / (tau * fps))
 * i.e. a per-frame low-pass weight equivalent to a continuous first-order
 * filter with time constant `tau` seconds. Here we generalise it to the
 * *measured* frame delta `dtSeconds` (from requestAnimationFrame
 * timestamps) instead of assuming a fixed fps, so the glide speed is
 * correct even when the render loop stutters.
 *
 * A larger tau = slower, heavier glide; a smaller tau = snappier.
 */

/**
 * Per-frame EMA weight for a continuous low-pass with time constant
 * `tauSeconds`, given the actual elapsed time `dtSeconds` since the last
 * update. Returns a value in (0, 1]; clamped so a long stall (tab
 * backgrounded) snaps rather than overshoots.
 */
export function emaWeightForTimeConstant(dtSeconds: number, tauSeconds: number): number {
  if (tauSeconds <= 0) return 1;
  if (dtSeconds <= 0) return 0;
  const weight = 1 - Math.exp(-dtSeconds / tauSeconds);
  return Math.min(1, Math.max(0, weight));
}

/** Standard scalar EMA step: `current` moved a fraction `weight` toward `target`. */
export function emaStep(current: number, target: number, weight: number): number {
  return current + (target - current) * weight;
}

/** Clamp `value` into the inclusive range [lo, hi]. */
export function clamp(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, value));
}
