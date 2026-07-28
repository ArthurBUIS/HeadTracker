/**
 * Proximity tracker — the identity logic for the SIMPLE face pipeline.
 *
 * No embeddings, no Hungarian, no gallery. Each detection round we get a set
 * of face centres; we match them to the existing tracks by repeatedly taking
 * the globally closest (track, detection) pair and removing both from play,
 * until one side is empty. Then:
 *   - leftover detections  → new tracks (a new stream each),
 *   - leftover tracks      → marked "lost": they keep their stream and still
 *     count, and stay matchable, for `lostSeconds` (hysteresis, so one or two
 *     missed detections don't drop a stream); a face reappearing near a lost
 *     track revives it. After `lostSeconds` unmatched, the track is dropped.
 *
 * This unifies the three cases in the spec (count same / up / down): they all
 * fall out of "match nearest, birth the extra detections, age the extra
 * tracks toward death".
 */

/** A detected face: its centre and a size (for framing the crop). */
export interface FaceObservation {
  cx: number;
  cy: number;
  /** Representative face size in source px (e.g. box height). */
  size: number;
}

export interface SimpleTrack {
  /** Stable id, assigned at birth, never reused. */
  id: number;
  cx: number;
  cy: number;
  size: number;
  /** 'active' = matched recently; 'lost' = coasting within the hysteresis. */
  status: 'active' | 'lost';
  /** When the track first went lost (ms), else null. */
  lostSinceMs: number | null;
}

export interface ProximityTrackerConfig {
  /** How long a lost track keeps its stream and stays matchable (seconds). */
  lostSeconds: number;
  /**
   * Optional cap on match distance (source px). A (track, detection) pair
   * farther apart than this is never matched. `Infinity` = no cap (pure
   * nearest-match, as specified).
   */
  maxMatchDistance: number;
}

export const DEFAULT_PROXIMITY_TRACKER_CONFIG: ProximityTrackerConfig = {
  lostSeconds: 5,
  maxMatchDistance: Infinity,
};

export interface ProximityUpdate {
  /** All current tracks (active + still-within-hysteresis lost). */
  tracks: SimpleTrack[];
  /** Ids born this round. */
  added: number[];
  /** Ids dropped this round (lost longer than `lostSeconds`). */
  removed: number[];
}

export class ProximityTracker {
  private readonly config: ProximityTrackerConfig;

  private tracks: SimpleTrack[] = [];

  private nextId = 1;

  constructor(config: Partial<ProximityTrackerConfig> = {}) {
    this.config = { ...DEFAULT_PROXIMITY_TRACKER_CONFIG, ...config };
  }

  setLostSeconds(seconds: number): void {
    this.config.lostSeconds = Math.max(0, seconds);
  }

  /** Number of faces currently held (active + lost within hysteresis). */
  get faceCount(): number {
    return this.tracks.length;
  }

  update(observations: FaceObservation[], nowMs: number): ProximityUpdate {
    const existing = this.tracks;
    const trackUsed = new Array<boolean>(existing.length).fill(false);
    const obsUsed = new Array<boolean>(observations.length).fill(false);

    // Greedy global-nearest matching.
    const pairs: { ti: number; oi: number; d: number }[] = [];
    for (let ti = 0; ti < existing.length; ti += 1) {
      for (let oi = 0; oi < observations.length; oi += 1) {
        pairs.push({ ti, oi, d: distance(existing[ti], observations[oi]) });
      }
    }
    pairs.sort((a, b) => a.d - b.d);
    for (const pair of pairs) {
      if (trackUsed[pair.ti] || obsUsed[pair.oi]) continue;
      if (pair.d > this.config.maxMatchDistance) continue;
      trackUsed[pair.ti] = true;
      obsUsed[pair.oi] = true;
      this.applyMatch(existing[pair.ti], observations[pair.oi]);
    }

    const added: number[] = [];
    const removed: number[] = [];

    // Leftover detections → new tracks.
    const born: SimpleTrack[] = [];
    for (let oi = 0; oi < observations.length; oi += 1) {
      if (obsUsed[oi]) continue;
      const track = this.birth(observations[oi]);
      born.push(track);
      added.push(track.id);
    }

    // Leftover tracks → lost (kept for hysteresis) or dropped when expired.
    const survivors: SimpleTrack[] = [];
    const lostLimitMs = this.config.lostSeconds * 1000;
    for (let ti = 0; ti < existing.length; ti += 1) {
      const track = existing[ti];
      if (trackUsed[ti]) {
        survivors.push(track);
        continue;
      }
      if (track.status !== 'lost') {
        track.status = 'lost';
        track.lostSinceMs = nowMs;
      }
      if (nowMs - (track.lostSinceMs ?? nowMs) <= lostLimitMs) {
        survivors.push(track);
      } else {
        removed.push(track.id);
      }
    }

    this.tracks = [...survivors, ...born];
    return { tracks: this.tracks, added, removed };
  }

  private applyMatch(track: SimpleTrack, obs: FaceObservation): void {
    track.cx = obs.cx;
    track.cy = obs.cy;
    track.size = obs.size;
    track.status = 'active';
    track.lostSinceMs = null;
  }

  private birth(obs: FaceObservation): SimpleTrack {
    const id = this.nextId;
    this.nextId += 1;
    return { id, cx: obs.cx, cy: obs.cy, size: obs.size, status: 'active', lostSinceMs: null };
  }
}

function distance(a: { cx: number; cy: number }, b: { cx: number; cy: number }): number {
  return Math.hypot(a.cx - b.cx, a.cy - b.cy);
}
