/**
 * A HeadDetector backed by a body detector (@tensorflow-models/coco-ssd).
 *
 * Why a body detector instead of a face detector: face-api only fires when
 * a face is visible, so anyone turned away from the camera vanishes. A
 * person detector fires on the whole body from any angle, so we can follow
 * heads even when people face away — then derive the head region from the
 * top of each person box. This is the workflow VideoStitcher gets from
 * YOLO; coco-ssd is the in-browser equivalent already shipped in
 * portals-projector-agent (`@tensorflow-models/coco-ssd` + tfjs).
 *
 * The head box is a heuristic square anchored near the top-centre of the
 * person box (people don't have a "head" class in COCO). Two fractions —
 * `headSizeFraction` and `headTopOffsetFraction`, both relative to the
 * person-box HEIGHT — control it, so it can be tuned live for a given
 * camera framing. The crop stage (headCrop.ts) then adds context around it.
 *
 * The coco-ssd model is INJECTED (not imported) so the engine stays
 * model-agnostic and portals can pass its already-loaded shared model.
 */

import type { FrameSource, HeadDetection, HeadDetector } from './types';

/** One detection from coco-ssd: `bbox` is [x, y, width, height] in px. */
export interface CocoSsdDetectedObject {
  bbox: [number, number, number, number];
  class: string;
  score: number;
}

/** The slice of coco-ssd's ObjectDetection surface this detector calls. */
export interface CocoSsdModelLike {
  detect(
    input: FrameSource,
    maxNumBoxes?: number,
    minScore?: number,
  ): Promise<CocoSsdDetectedObject[]>;
}

export interface CocoSsdDetectorConfig {
  /** Person-confidence floor passed to coco-ssd. */
  minScore: number;
  /** Max people considered per frame (coco-ssd maxNumBoxes). */
  maxPeople: number;
  /** Head square side as a fraction of the person-box height. */
  headSizeFraction: number;
  /** Head-centre y as a fraction of person-box height below its top edge. */
  headTopOffsetFraction: number;
}

export const DEFAULT_COCO_SSD_DETECTOR_CONFIG: CocoSsdDetectorConfig = {
  minScore: 0.5,
  maxPeople: 20,
  headSizeFraction: 0.18,
  headTopOffsetFraction: 0.12,
};

const PERSON_CLASS = 'person';

export class CocoSsdHeadDetector implements HeadDetector {
  private readonly config: CocoSsdDetectorConfig;

  constructor(
    private readonly model: CocoSsdModelLike,
    config: Partial<CocoSsdDetectorConfig> = {},
  ) {
    this.config = { ...DEFAULT_COCO_SSD_DETECTOR_CONFIG, ...config };
  }

  async detectHeads(source: FrameSource): Promise<HeadDetection[]> {
    const objects = await this.model.detect(
      source,
      this.config.maxPeople,
      this.config.minScore,
    );
    return objects
      .filter((o) => o.class === PERSON_CLASS)
      .map((o) => this.personBoxToHeadBox(o));
  }

  private personBoxToHeadBox(object: CocoSsdDetectedObject): HeadDetection {
    const [x, y, width, height] = object.bbox;
    const side = height * this.config.headSizeFraction;
    const centerX = x + width / 2;
    const centerY = y + height * this.config.headTopOffsetFraction;
    return {
      x: centerX - side / 2,
      y: centerY - side / 2,
      width: side,
      height: side,
      score: object.score,
      // Carry the person box so the tracker associates on it — big, stable
      // overlap between detections keeps a person's id from churning.
      bodyBox: { x, y, width, height },
    };
  }
}
