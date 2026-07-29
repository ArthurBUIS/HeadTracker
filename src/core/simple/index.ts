/**
 * Public surface of the SIMPLE face pipeline (embedding-free).
 * Face detection → nearest-match proximity tracking → 300×200 streams.
 */

export {
  ProximityTracker,
  DEFAULT_PROXIMITY_TRACKER_CONFIG,
  type FaceObservation,
  type SimpleTrack,
  type ProximityTrackerConfig,
  type ProximityUpdate,
} from './proximityTracker';

export {
  BoxGroupManager,
  DEFAULT_GROUP_MANAGER_CONFIG,
  type GroupInput,
  type Group,
  type GroupManagerConfig,
} from './boxGrouping';

export {
  SimpleFaceEngine,
  DEFAULT_SIMPLE_ENGINE_CONFIG,
  MIN_SIMPLE_INTERVAL_MS,
  MAX_SIMPLE_INTERVAL_MS,
  type FaceCenterDetector,
  type SimpleFaceEngineConfig,
  type FaceStream,
  type SimpleFaceCallbacks,
  type SimpleFaceDiagnostics,
  type StreamScore,
} from './simpleFaceEngine';

export {
  Yolov8HeadDetector,
  DEFAULT_YOLOV8_HEAD_DETECTOR_CONFIG,
  type Yolov8Runner,
  type Yolov8HeadDetectorConfig,
} from './yolov8HeadDetector';

export {
  computeLetterbox,
  mapDetectionToSource,
  decodeYolov8,
  nonMaxSuppression,
  type Detection,
  type Letterbox,
} from './yoloPostprocess';
