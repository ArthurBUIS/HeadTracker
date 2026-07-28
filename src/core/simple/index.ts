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
  SimpleFaceEngine,
  DEFAULT_SIMPLE_ENGINE_CONFIG,
  MIN_SIMPLE_INTERVAL_MS,
  MAX_SIMPLE_INTERVAL_MS,
  type FaceCenterDetector,
  type SimpleFaceEngineConfig,
  type FaceStream,
  type SimpleFaceCallbacks,
  type SimpleFaceDiagnostics,
} from './simpleFaceEngine';
