/**
 * Public surface of the head-tracking core.
 *
 * This is the module boundary intended to be lifted into
 * portals-projector-agent (e.g. under src/renderer/utils/headTracker/).
 * It depends only on DOM/canvas/MediaStream APIs plus an injected
 * face-api instance — nothing Vite/React/Electron specific.
 */

export type {
  HeadDetection,
  HeadDetector,
  FrameSource,
  FrameSize,
} from './types';

export {
  emaWeightForTimeConstant,
  emaStep,
  clamp,
} from './smoothing';

export {
  HeadIdentityTracker,
  DEFAULT_TRACKER_CONFIG,
  type TrackedHead,
  type TrackerConfig,
  type TrackerUpdate,
} from './tracker';

export {
  FaceApiHeadDetector,
  DEFAULT_FACE_API_DETECTOR_CONFIG,
  type FaceApiLike,
  type FaceApiDetectorConfig,
} from './faceApiDetector';

export {
  CocoSsdHeadDetector,
  DEFAULT_COCO_SSD_DETECTOR_CONFIG,
  type CocoSsdModelLike,
  type CocoSsdDetectedObject,
  type CocoSsdDetectorConfig,
} from './cocoSsdDetector';

export {
  MoveNetHeadDetector,
  DEFAULT_MOVENET_DETECTOR_CONFIG,
  type PoseDetectorLike,
  type Pose,
  type PoseKeypoint,
  type MoveNetDetectorConfig,
} from './moveNetDetector';

export {
  computeHsvHistogram,
  appearanceSimilarity,
  blendAppearance,
  APPEARANCE_LENGTH,
  type AppearanceDescriptor,
} from './appearance';

export { solveMinCostAssignment, NO_ASSIGNMENT } from './assignment';

export {
  faceDistance,
  faceAffinity,
  blendFace,
  matchFacesToBoxes,
  type FaceDescriptor,
} from './faceEmbedding';

export type { FaceObservation, FaceEmbedder } from './types';

export {
  HeadCropSmoother,
  DEFAULT_HEAD_CROP_CONFIG,
  type HeadCropConfig,
  type CropRect,
} from './headCrop';

export {
  HeadTrackerEngine,
  DEFAULT_ENGINE_CONFIG,
  MIN_DETECTION_INTERVAL_MS,
  MAX_DETECTION_INTERVAL_MS,
  type HeadStream,
  type HeadTrackerEngineConfig,
  type HeadTrackerCallbacks,
} from './headTrackerEngine';
