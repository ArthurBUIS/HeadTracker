/**
 * Per-track smoothed crop geometry.
 *
 * Detection gives a fresh target head centre/size only every ~2 s. To make
 * the 200x200 view *glide* instead of jumping every 2 s, each track owns
 * one of these smoothers, stepped every render frame: the crop centre and
 * crop side EMA toward the latest target using the time-constant filter
 * from smoothing.ts (the person_tracking.py technique).
 *
 * Two time constants, mirroring person_tracking.py's track-vs-drift split:
 *   - `lockSeconds`  — snappier, used while the head is being seen.
 *   - `holdSeconds`  — slower, used on missed detections so a brief
 *                      occlusion doesn't make the view lurch.
 */

import { clamp, emaStep, emaWeightForTimeConstant } from './smoothing';
import type { FrameSize } from './types';

export interface HeadCropConfig {
  /** EMA time constant while the head is actively detected (seconds). */
  lockSeconds: number;
  /** EMA time constant while the head's detection is missing (seconds). */
  holdSeconds: number;
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
    const tau = isSeen ? this.config.lockSeconds : this.config.holdSeconds;
    const weight = emaWeightForTimeConstant(dtSeconds, tau);
    this.centerX = emaStep(this.centerX, targetCenterX, weight);
    this.centerY = emaStep(this.centerY, targetCenterY, weight);
    this.side = emaStep(this.side, this.sideForHeadSize(targetHeadSize), weight);
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
