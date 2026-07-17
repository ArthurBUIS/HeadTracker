/**
 * Per-track smoothed crop geometry.
 *
 * Detection gives a fresh target head centre/size only every ~2 s. To make
 * the 200x200 view *glide* instead of jumping every 2 s, each track owns
 * one of these smoothers, stepped every render frame: the crop centre and
 * crop side EMA toward the latest target using the time-constant filter
 * from smoothing.ts (the person_tracking.py technique).
 *
 * Position and size (zoom) are smoothed on SEPARATE time constants, and
 * this matters: the crop centre should track the head quickly, but the crop
 * *side* (which is resampled into the fixed 200x200 output) must be very
 * stable — otherwise the noisy per-detection head-size estimate (and the
 * jump when the detector switches between its face-keypoint and shoulder
 * estimates) shows up as the view zooming in and out. So:
 *   - `lockSeconds` / `holdSeconds` — position, snappy (seen) vs slow (lost).
 *   - `sizeSeconds` — the zoom, deliberately long so it stays near-constant
 *     and only self-calibrates slowly if the person really changes distance.
 */

import { clamp, emaStep, emaWeightForTimeConstant } from './smoothing';
import type { FrameSize } from './types';

export interface HeadCropConfig {
  /** EMA time constant for crop POSITION while the head is detected (s). */
  lockSeconds: number;
  /** EMA time constant for crop POSITION while detection is missing (s). */
  holdSeconds: number;
  /**
   * EMA time constant for crop SIZE / zoom (seconds). Long by design so the
   * output framing holds steady instead of pulsing with head-size noise.
   */
  sizeSeconds: number;
  /**
   * Crop side length as a multiple of head size, before clamping. >1
   * leaves headroom/shoulders around the head. The square source region
   * is then resampled to the fixed 200x200 output, so a near head and a
   * far head are framed consistently.
   */
  paddingFactor: number;
  /** Smallest allowed square crop side, source-video px. */
  minCropSide: number;
}

export const DEFAULT_HEAD_CROP_CONFIG: HeadCropConfig = {
  lockSeconds: 0.5,
  holdSeconds: 1.5,
  sizeSeconds: 4.0,
  paddingFactor: 2.0,
  minCropSide: 80,
};

/** A square region of the source frame to sample, in source-video px. */
export interface CropRect {
  sx: number;
  sy: number;
  side: number;
}

export class HeadCropSmoother {
  private readonly config: HeadCropConfig;

  private centerX: number;

  private centerY: number;

  private side: number;

  constructor(
    initialCenterX: number,
    initialCenterY: number,
    initialHeadSize: number,
    config: Partial<HeadCropConfig> = {},
  ) {
    this.config = { ...DEFAULT_HEAD_CROP_CONFIG, ...config };
    this.centerX = initialCenterX;
    this.centerY = initialCenterY;
    this.side = this.sideForHeadSize(initialHeadSize);
  }

  /**
   * Step the EMA toward the track's current target for one render frame.
   *
   * @param targetCenterX  latest tracked head centre x (source px)
   * @param targetCenterY  latest tracked head centre y (source px)
   * @param targetHeadSize latest tracked head size (source px)
   * @param dtSeconds      wall-clock time since the previous render frame
   * @param isSeen         whether the track was matched in the most recent
   *                       detection round (selects lock vs hold constant)
   */
  step(
    targetCenterX: number,
    targetCenterY: number,
    targetHeadSize: number,
    dtSeconds: number,
    isSeen: boolean,
  ): void {
    const posTau = isSeen ? this.config.lockSeconds : this.config.holdSeconds;
    const posWeight = emaWeightForTimeConstant(dtSeconds, posTau);
    this.centerX = emaStep(this.centerX, targetCenterX, posWeight);
    this.centerY = emaStep(this.centerY, targetCenterY, posWeight);

    // Size uses its own long time constant so the zoom stays steady.
    const sizeWeight = emaWeightForTimeConstant(dtSeconds, this.config.sizeSeconds);
    this.side = emaStep(this.side, this.sideForHeadSize(targetHeadSize), sizeWeight);
  }

  /**
   * The square source region to sample this frame, clamped so it never
   * runs off the frame edges. When the frame is smaller than the desired
   * side, the side shrinks to fit.
   */
  getCropRect(frame: FrameSize): CropRect {
    const maxSide = Math.min(frame.width, frame.height);
    const side = clamp(this.side, this.config.minCropSide, maxSide);
    const sx = clamp(this.centerX - side / 2, 0, frame.width - side);
    const sy = clamp(this.centerY - side / 2, 0, frame.height - side);
    return { sx, sy, side };
  }

  private sideForHeadSize(headSize: number): number {
    return Math.max(this.config.minCropSide, headSize * this.config.paddingFactor);
  }
}
