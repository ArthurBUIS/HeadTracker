/**
 * Shared value types for the head-tracking core.
 *
 * All boxes and centres are in **source-video pixel coordinates** (the
 * natural pixel space of the input frame), never in output-canvas space.
 * The 200x200 output is only ever produced at the final draw step.
 */

/** An axis-aligned box in source-video pixels. */
export interface Box {
  /** Left edge, source-video px. */
  x: number;
  /** Top edge, source-video px. */
  y: number;
  /** Width, source-video px. */
  width: number;
  /** Height, source-video px. */
  height: number;
}

/** A single head detection in source-video pixels (the box is the HEAD). */
export interface HeadDetection extends Box {
  /** Detector confidence in [0, 1]. */
  score: number;
  /**
   * The larger box this head was derived from (the person/body box for
   * coco-ssd). When present the tracker associates identities on THIS box
   * instead of the small head box — body boxes overlap far more reliably
   * between detections, which stops one person from churning into many
   * ids. Absent for detectors whose primary output already is the head
   * (e.g. face-api), in which case the head box is used for association.
   */
  bodyBox?: Box;
  /**
   * Optional appearance signature (an HSV colour histogram of the torso).
   * Filled in by the engine when appearance re-identification is enabled;
   * the tracker uses it to recover an id after a crossing/occlusion or a
   * full exit that spatial association alone can't survive. See
   * `appearance.ts`.
   */
  appearance?: import('./appearance').AppearanceDescriptor;
}

/**
 * Anything a detector / the render loop can sample pixels from. In the
 * Electron renderer and in the browser demo this is a live `<video>`;
 * a `<canvas>` is accepted so callers can pre-rotate/pre-process a frame
 * (mirroring how FaceDetection.jsx draws the rotated frame to a canvas
 * before detecting).
 */
export type FrameSource = HTMLVideoElement | HTMLCanvasElement;

/**
 * Head detector abstraction. The engine depends only on this interface,
 * so portals-projector-agent can inject its already-loaded shared
 * `@vladmandic/face-api` instance instead of the demo's self-contained one.
 */
export interface HeadDetector {
  /** Run one detection pass over the current frame of `source`. */
  detectHeads(source: FrameSource): Promise<HeadDetection[]>;
}

/** Natural pixel dimensions of a frame source. */
export interface FrameSize {
  width: number;
  height: number;
}
