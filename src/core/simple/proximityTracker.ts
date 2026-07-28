/**
 * Proximity tracker — the identity logic for the SIMPLE pipeline.
 *
 * No embeddings, no Hungarian, no gallery. Each detection round we get a set
 * of head centres; we match them to the existing tracks by repeatedly taking
 * the globally closest (track, detection) pair and removing both from play,
 * until one side is empty. Then:
 *   - leftover detections  → new (pending) tracks,
 *   - leftover tracks      → a CONFIRMED track goes "lost" (keeps its stream
 *     and stays matchable for `lostSeconds` — hysteresis so a missed
 *     detection or two doesn't drop it); an UNCONFIRMED track is dropped.
 *
 * A new track must be matched `minHits` rounds IN A ROW before it is
 * confirmed and gets a stream — this suppresses false positives. (Missing a
 * round drops an unconfirmed track, so the count really is consecutive.)
 */

export interface FaceObservation {
  cx: number;
  cy: number;
  /** Representative head size in source px (e.g. box height). */
  size: number;
}

export interface SimpleTrack {
  id: number;
  cx: number;
  cy: number;
  size: number;
  /** Consecutive matched rounds (resets by drop-on-miss while unconfirmed). */
  hits: number;
  /** True once `hits >= minHits`; only confirmed tracks are streamed. */
  confirmed: boolean;
  status: 'active' | 'lost';
  lostSinceMs: number | null;
}

export interface ProximityTrackerConfig {
  /** Consecutive detections a new track needs before it's confirmed/streamed. */
  minHits: number;
  /** How long a lost (confirmed) track keeps its stream and stays matchable. */
  lostSeconds: number;
  /** Optional cap on match distance (source px); `Infinity` = no cap. */
  maxMatchDistance: number;
}

export const DEFAULT_PROXIMITY_TRACKER_CONFIG: ProximityTrackerConfig = {
  minHits: 3,
  lostSeconds: 5,
  maxMatchDistance: Infinity,
};

export interface ProximityUpdate {
  /** Confirmed tracks (active + still-within-hysteresis lost). */
  tracks: SimpleTrack[];
  /** Ids that became confirmed this round. */
  added: number[];
  /** Confirmed ids dropped this round (lost longer than `lostSeconds`). */
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

  /** Confirmed faces held (active + lost within hysteresis). */
  get faceCount(): number {
    return this.tracks.filter((t) => t.confirmed).length;
  }

  /** Tracks still awaiting confirmation (not yet streamed). */
  get pendingCount(): number {
    return this.tracks.filter((t) => !t.confirmed).length;
  }

  update(observations: FaceObservation[], nowMs: number): ProximityUpdate {
    const existing = this.tracks;
    const confirmedBefore = new Set(existing.filter((t) => t.confirmed).map((t) => t.id));
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

    // Leftover detections → new pending tracks.
    const born: SimpleTrack[] = [];
    for (let oi = 0; oi < observations.length; oi += 1) {
      if (!obsUsed[oi]) born.push(this.birth(observations[oi]));
    }

    // Leftover tracks: confirmed → lost hysteresis; unconfirmed → dropped.
    const survivors: SimpleTrack[] = [];
    const lostLimitMs = this.config.lostSeconds * 1000;
    for (let ti = 0; ti < existing.length; ti += 1) {
      const track = existing[ti];
      if (trackUsed[ti]) {
        survivors.push(track);
        continue;
      }
      if (!track.confirmed) continue; // drop unconfirmed on a miss (resets the streak)
      if (track.status !== 'lost') {
        track.status = 'lost';
        track.lostSinceMs = nowMs;
      }
      if (nowMs - (track.lostSinceMs ?? nowMs) <= lostLimitMs) survivors.push(track);
    }

    this.tracks = [...survivors, ...born];

    const confirmedNow = this.tracks.filter((t) => t.confirmed);
    const activeIds = new Set(confirmedNow.map((t) => t.id));
    const added = confirmedNow.filter((t) => !confirmedBefore.has(t.id)).map((t) => t.id);
    const removed = [...confirmedBefore].filter((id) => !activeIds.has(id));
    return { tracks: confirmedNow, added, removed };
  }

  private applyMatch(track: SimpleTrack, obs: FaceObservation): void {
    track.cx = obs.cx;
    track.cy = obs.cy;
    track.size = obs.size;
    track.hits += 1;
    track.status = 'active';
    track.lostSinceMs = null;
    if (!track.confirmed && track.hits >= this.config.minHits) track.confirmed = true;
  }

  private birth(obs: FaceObservation): SimpleTrack {
    const id = this.nextId;
    this.nextId += 1;
    return {
      id,
      cx: obs.cx,
      cy: obs.cy,
      size: obs.size,
      hits: 1,
      confirmed: this.config.minHits <= 1,
      status: 'active',
      lostSinceMs: null,
    };
  }
}

function distance(a: { cx: number; cy: number }, b: { cx: number; cy: number }): number {
  return Math.hypot(a.cx - b.cx, a.cy - b.cy);
}
