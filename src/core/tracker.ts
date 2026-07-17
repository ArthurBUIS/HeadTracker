/**
 * Multi-head identity tracker.
 *
 * VideoStitcher's person_tracking.py only ever follows ONE person (the
 * largest blob) and has no notion of identity across frames. This module
 * adds the missing piece the task needs — "know which box is which" — with
 * a lightweight SORT-style association plus appearance re-identification.
 *
 * Association per detection round, in three phases:
 *   1. Primary — a single OPTIMAL (Hungarian) assignment over a cost that
 *      FUSES spatial overlap/proximity with clothing similarity, gated to
 *      spatially-plausible pairs. Fusing appearance here (rather than
 *      matching spatially first) is what stops two people crossing from
 *      swapping ids: within the gate, identity follows clothing, not
 *      whichever detection ended up nearest. Matches to a mutually-
 *      occluding detection don't update the appearance signature, so a
 *      neighbour's clothing can't contaminate it mid-crossing.
 *   2. Appearance rescue — for tracks/detections phase 1 left unmatched,
 *      match by clothing-colour similarity beyond the spatial gate.
 *      Recovers an id that jumped position (big move, brief occlusion).
 *   3. Gallery re-ID — a detection still unmatched is compared against a
 *      gallery of recently-lost tracks; a strong appearance match
 *      RESURRECTS that id instead of minting a new one, so a person who
 *      fully left and came back reclaims their number.
 * Anything still unmatched is born as a new id.
 *
 * Appearance is optional: when detections carry no descriptor (e.g. the
 * face-api path, or when re-ID is toggled off) phases 2–3 no-op and the
 * behaviour is pure spatial tracking.
 */

import {
  appearanceSimilarity,
  blendAppearance,
  type AppearanceDescriptor,
} from './appearance';
import { NO_ASSIGNMENT, solveMinCostAssignment } from './assignment';
import { blendFace, faceAffinity, type FaceDescriptor } from './faceEmbedding';
import type { Box, HeadDetection } from './types';

/** Cost marking a (track, detection) pair as forbidden in the assignment. */
const DISALLOWED_COST = 1e8;

/** A persistently-identified head across detection rounds. */
export interface TrackedHead {
  /** Stable identity, assigned once at birth and never reused. */
  id: number;
  /**
   * Box used for matching next round: the body box when the detector
   * supplies one, else the head box. Large + stable ⇒ resilient ids.
   */
  assocBox: Box;
  /** Head centre x, source-video px (drives the crop). */
  centerX: number;
  /** Head centre y, source-video px (drives the crop). */
  centerY: number;
  /** Head size = max(head width, head height), used to scale the crop. */
  size: number;
  /** How many detection rounds this track has been matched in total. */
  hits: number;
  /** Consecutive detection rounds with no match (drives death). */
  misses: number;
  /** True once `hits >= minHits`; only confirmed tracks get a stream. */
  confirmed: boolean;
  /** EMA of this track's torso colour signature (undefined until observed). */
  appearance?: AppearanceDescriptor;
  /** EMA of this track's 128-D face embedding (undefined until a face seen). */
  faceDescriptor?: FaceDescriptor;
}

/** An id kept in memory after its track died, for gallery re-identification. */
interface GalleryEntry {
  id: number;
  appearance?: AppearanceDescriptor;
  faceDescriptor?: FaceDescriptor;
  /** Rounds since the track died; expired past `galleryMaxRounds`. */
  age: number;
}

export interface TrackerConfig {
  /** Minimum IoU for a spatial match; below this the distance gate decides. */
  iouThreshold: number;
  /** Detection rounds a track must be matched before it is confirmed. */
  minHits: number;
  /**
   * Consecutive missed rounds tolerated before death. The engine sets this
   * from a target coast TIME / the (adjustable) detection interval, so the
   * grace before a stream closes stays roughly constant.
   */
  maxMisses: number;
  /**
   * Fallback spatial gate: also a candidate when centres are within this ×
   * average box size, even at IoU 0. Modest so close people aren't merged.
   */
  centerDistanceGateFactor: number;
  /**
   * Histogram-intersection similarity (0–1) required for an appearance
   * match, in both the rescue and gallery phases. Higher = stricter (fewer
   * wrong merges, more churn); lower = more aggressive re-ID.
   */
  appearanceThreshold: number;
  /** EMA weight on a fresh descriptor when updating a track's signature. */
  appearanceEmaWeight: number;
  /** Rounds a lost id stays in the gallery before it's forgotten. */
  galleryMaxRounds: number;
  /**
   * Weight of appearance vs spatial in the primary (phase-1) match cost,
   * in [0, 1] (spatial weight is 1 − this). High so that when two people
   * cross, identity follows CLOTHING rather than whichever detection ended
   * up nearest — the main swap fix. Applies only when both sides carry an
   * appearance descriptor; otherwise the match is spatial-only.
   */
  appearanceWeight: number;
  /**
   * Detections overlapping another detection by more than this IoU are
   * treated as mutually occluding; a track matched to one does NOT fold in
   * its appearance (would contaminate the signature with a neighbour's
   * clothing mid-crossing).
   */
  detectionOverlapFreeze: number;
  /**
   * Skip the appearance update when the matched detection's similarity to
   * the track's stored signature is below this — a spatial-only match to a
   * different-looking box shouldn't overwrite a good signature.
   */
  appearanceUpdateFloor: number;
  /**
   * Weight of a FACE-embedding affinity vs spatial in the phase-1 cost.
   * Higher than `appearanceWeight` because a face embedding is far more
   * discriminative than clothing colour, so it should dominate when present.
   */
  faceWeight: number;
  /** Distance at which face affinity reaches 0 (face-api same-person ≈0.6). */
  faceDistanceNorm: number;
  /** Min face affinity to accept a face-based rescue / gallery re-ID. */
  faceMatchAffinity: number;
  /** EMA weight on a fresh face descriptor when updating a track. */
  faceEmaWeight: number;
}

export const DEFAULT_TRACKER_CONFIG: TrackerConfig = {
  iouThreshold: 0.3,
  minHits: 2,
  maxMisses: 2,
  centerDistanceGateFactor: 2.5,
  appearanceThreshold: 0.55,
  appearanceEmaWeight: 0.3,
  galleryMaxRounds: 60,
  appearanceWeight: 0.6,
  detectionOverlapFreeze: 0.15,
  appearanceUpdateFloor: 0.3,
  faceWeight: 0.85,
  faceDistanceNorm: 1.2,
  faceMatchAffinity: 0.5,
  faceEmaWeight: 0.2,
};

/** Result of one association round, for the engine to reconcile streams. */
export interface TrackerUpdate {
  /** Tracks confirmed and alive after this round. */
  active: TrackedHead[];
  /** Ids confirmed in a prior round but not active now (died/lost). */
  removedIds: number[];
}

/** The box a detection should be matched on: body box if given, else head. */
function associationBoxOf(detection: HeadDetection): Box {
  return detection.bodyBox ?? detection;
}

function centerOf(box: Box): { cx: number; cy: number } {
  return { cx: box.x + box.width / 2, cy: box.y + box.height / 2 };
}

/** Intersection-over-union of two boxes. */
function intersectionOverUnion(a: Box, b: Box): number {
  const interW = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x));
  const interH = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
  const interArea = interW * interH;
  if (interArea <= 0) return 0;
  const union = a.width * a.height + b.width * b.height - interArea;
  return union <= 0 ? 0 : interArea / union;
}

/** Euclidean distance between two boxes' centres. */
function centerDistance(a: Box, b: Box): number {
  const ca = centerOf(a);
  const cb = centerOf(b);
  return Math.hypot(ca.cx - cb.cx, ca.cy - cb.cy);
}

/** Average of two boxes' larger side — the reference scale for the gate. */
function averageBoxSize(a: Box, b: Box): number {
  return (Math.max(a.width, a.height) + Math.max(b.width, b.height)) / 2;
}

export class HeadIdentityTracker {
  private readonly config: TrackerConfig;

  private tracks: TrackedHead[] = [];

  private gallery: GalleryEntry[] = [];

  private nextId = 1;

  constructor(config: Partial<TrackerConfig> = {}) {
    this.config = { ...DEFAULT_TRACKER_CONFIG, ...config };
  }

  /** Adjust miss tolerance at runtime (engine keeps coast time constant). */
  setMaxMisses(maxMisses: number): void {
    this.config.maxMisses = Math.max(1, Math.round(maxMisses));
  }

  /** Adjust how long lost ids linger for gallery re-ID. */
  setGalleryMaxRounds(rounds: number): void {
    this.config.galleryMaxRounds = Math.max(1, Math.round(rounds));
  }

  /** Ingest one detection round; returns reconciled active tracks + deaths. */
  update(detections: HeadDetection[]): TrackerUpdate {
    const confirmedBefore = new Set(
      this.tracks.filter((t) => t.confirmed).map((t) => t.id),
    );
    const detBoxes = detections.map(associationBoxOf);
    const occludedDet = this.computeOccludedDetections(detBoxes);
    const trackMatched = new Array<boolean>(this.tracks.length).fill(false);
    const detMatched = new Array<boolean>(detections.length).fill(false);

    this.matchPrimary(detections, detBoxes, occludedDet, trackMatched, detMatched);
    this.matchByAppearance(detections, trackMatched, detMatched);
    this.reidFromGallery(detections, detMatched);

    // Unmatched existing tracks age toward death.
    for (let ti = 0; ti < this.tracks.length; ti += 1) {
      if (!trackMatched[ti]) this.tracks[ti].misses += 1;
    }
    // Unmatched detections are born as new tracks.
    for (let di = 0; di < detections.length; di += 1) {
      if (!detMatched[di]) this.tracks.push(this.birth(detections[di]));
    }

    this.retireDeadTracks();
    this.ageGallery();

    const active = this.tracks.filter((t) => t.confirmed);
    const activeIds = new Set(active.map((t) => t.id));
    const removedIds = [...confirmedBefore].filter((id) => !activeIds.has(id));
    return { active, removedIds };
  }

  /** Current confirmed, alive tracks (no mutation). */
  getActiveTracks(): TrackedHead[] {
    return this.tracks.filter((t) => t.confirmed);
  }

  /**
   * Phase 1: optimal match on a combined spatial + appearance cost.
   *
   * Every spatially-gated (track, detection) pair gets a cost fusing box
   * overlap/proximity with clothing similarity; the Hungarian solver then
   * picks the globally cheapest assignment. Fusing appearance in — instead
   * of matching spatially first — is what stops a crossing from handing a
   * track to whichever detection ended up nearest. Matches to a
   * mutually-occluding detection don't update the appearance signature.
   */
  private matchPrimary(
    detections: HeadDetection[],
    detBoxes: Box[],
    occludedDet: boolean[],
    trackMatched: boolean[],
    detMatched: boolean[],
  ): void {
    const nT = this.tracks.length;
    const nD = detections.length;
    if (nT === 0 || nD === 0) return;

    const cost: number[][] = [];
    for (let ti = 0; ti < nT; ti += 1) {
      const trackBox = this.tracks[ti].assocBox;
      const row: number[] = [];
      for (let di = 0; di < nD; di += 1) {
        row.push(this.pairCost(this.tracks[ti], trackBox, detections[di], detBoxes[di]));
      }
      cost.push(row);
    }

    const assignment = solveMinCostAssignment(cost, DISALLOWED_COST);
    for (let ti = 0; ti < nT; ti += 1) {
      const di = assignment[ti];
      if (di === NO_ASSIGNMENT) continue;
      trackMatched[ti] = true;
      detMatched[di] = true;
      const reid = this.reidSimilarity(this.tracks[ti], detections[di]);
      const updateAppearance =
        !occludedDet[di] &&
        (!reid.hasEvidence || reid.sim >= this.config.appearanceUpdateFloor);
      this.applyMatch(this.tracks[ti], detections[di], updateAppearance);
    }
  }

  /** Combined cost of pairing a track with a detection; DISALLOWED if ungated. */
  private pairCost(
    track: TrackedHead,
    trackBox: Box,
    detection: HeadDetection,
    detBox: Box,
  ): number {
    const iou = intersectionOverUnion(trackBox, detBox);
    const distance = centerDistance(trackBox, detBox);
    const gate = this.config.centerDistanceGateFactor * averageBoxSize(trackBox, detBox);
    if (iou < this.config.iouThreshold && distance > gate) return DISALLOWED_COST;

    const spatialAffinity = Math.max(iou, gate > 0 ? Math.max(0, 1 - distance / gate) : 0);
    const reid = this.reidSimilarity(track, detection);
    if (reid.hasEvidence) {
      const w = reid.isFace ? this.config.faceWeight : this.config.appearanceWeight;
      return 1 - (w * reid.sim + (1 - w) * spatialAffinity);
    }
    return 1 - spatialAffinity;
  }

  /**
   * Re-ID affinity between two appearance-carrying things. Prefers the FACE
   * embedding when both sides have one (far more discriminative than colour);
   * otherwise uses the torso colour histogram. `hasEvidence` is false when
   * there's no shared cue (e.g. both faces away, only spatial to go on).
   */
  private reidSimilarity(
    a: { appearance?: AppearanceDescriptor; faceDescriptor?: FaceDescriptor },
    b: { appearance?: AppearanceDescriptor; faceDescriptor?: FaceDescriptor },
  ): { sim: number; isFace: boolean; hasEvidence: boolean } {
    if (a.faceDescriptor && b.faceDescriptor) {
      return {
        sim: faceAffinity(a.faceDescriptor, b.faceDescriptor, this.config.faceDistanceNorm),
        isFace: true,
        hasEvidence: true,
      };
    }
    if (a.appearance && b.appearance) {
      return { sim: appearanceSimilarity(a.appearance, b.appearance), isFace: false, hasEvidence: true };
    }
    return { sim: 0, isFace: false, hasEvidence: false };
  }

  /** Min similarity to accept a rescue / gallery match, per cue type. */
  private reidThreshold(isFace: boolean): number {
    return isFace ? this.config.faceMatchAffinity : this.config.appearanceThreshold;
  }

  /** Flag detections that overlap another detection (mutual occlusion). */
  private computeOccludedDetections(detBoxes: Box[]): boolean[] {
    const occluded = new Array<boolean>(detBoxes.length).fill(false);
    for (let i = 0; i < detBoxes.length; i += 1) {
      for (let j = i + 1; j < detBoxes.length; j += 1) {
        if (intersectionOverUnion(detBoxes[i], detBoxes[j]) > this.config.detectionOverlapFreeze) {
          occluded[i] = true;
          occluded[j] = true;
        }
      }
    }
    return occluded;
  }

  /** Phase 2: rescue spatially-unmatched pairs by appearance (face or colour). */
  private matchByAppearance(
    detections: HeadDetection[],
    trackMatched: boolean[],
    detMatched: boolean[],
  ): void {
    const pairs: { ti: number; di: number; sim: number; isFace: boolean }[] = [];
    for (let ti = 0; ti < this.tracks.length; ti += 1) {
      if (trackMatched[ti]) continue;
      for (let di = 0; di < detections.length; di += 1) {
        if (detMatched[di]) continue;
        const r = this.reidSimilarity(this.tracks[ti], detections[di]);
        if (r.hasEvidence && r.sim >= this.reidThreshold(r.isFace)) {
          pairs.push({ ti, di, sim: r.sim, isFace: r.isFace });
        }
      }
    }
    // Prefer face matches (more reliable), then higher similarity.
    pairs.sort((p, q) => Number(q.isFace) - Number(p.isFace) || q.sim - p.sim);
    for (const pair of pairs) {
      if (trackMatched[pair.ti] || detMatched[pair.di]) continue;
      trackMatched[pair.ti] = true;
      detMatched[pair.di] = true;
      this.applyMatch(this.tracks[pair.ti], detections[pair.di]);
    }
  }

  /** Phase 3: resurrect a lost id from the gallery for an unmatched detection. */
  private reidFromGallery(detections: HeadDetection[], detMatched: boolean[]): void {
    const usedGalleryIds = new Set<number>();
    for (let di = 0; di < detections.length; di += 1) {
      const det = detections[di];
      if (detMatched[di] || (!det.appearance && !det.faceDescriptor)) continue;
      let best: GalleryEntry | null = null;
      let bestScore = -1;
      for (const entry of this.gallery) {
        if (usedGalleryIds.has(entry.id)) continue;
        const r = this.reidSimilarity(entry, det);
        if (!r.hasEvidence || r.sim < this.reidThreshold(r.isFace)) continue;
        // Rank face matches above any colour match, then by similarity.
        const score = (r.isFace ? 1 : 0) + r.sim;
        if (score > bestScore) {
          bestScore = score;
          best = entry;
        }
      }
      if (best) {
        usedGalleryIds.add(best.id);
        this.gallery = this.gallery.filter((e) => e.id !== best!.id);
        this.tracks.push(this.resurrect(best.id, det));
        detMatched[di] = true;
      }
    }
  }

  /** Move dead confirmed tracks (with a signature) into the gallery. */
  private retireDeadTracks(): void {
    const survivors: TrackedHead[] = [];
    for (const track of this.tracks) {
      if (track.misses <= this.config.maxMisses) {
        survivors.push(track);
      } else if (track.confirmed && (track.appearance || track.faceDescriptor)) {
        this.gallery.push({
          id: track.id,
          appearance: track.appearance,
          faceDescriptor: track.faceDescriptor,
          age: 0,
        });
      }
    }
    this.tracks = survivors;
  }

  /** Age gallery entries and drop the expired ones. */
  private ageGallery(): void {
    for (const entry of this.gallery) entry.age += 1;
    this.gallery = this.gallery.filter((e) => e.age <= this.config.galleryMaxRounds);
  }

  private applyMatch(
    track: TrackedHead,
    detection: HeadDetection,
    updateAppearance = true,
  ): void {
    const { cx, cy } = centerOf(detection);
    track.assocBox = associationBoxOf(detection);
    track.centerX = cx;
    track.centerY = cy;
    track.size = Math.max(detection.width, detection.height);
    track.hits += 1;
    track.misses = 0;
    if (updateAppearance) {
      track.appearance = this.mergeAppearance(track.appearance, detection.appearance);
      track.faceDescriptor = this.mergeFace(track.faceDescriptor, detection.faceDescriptor);
    }
    if (!track.confirmed && track.hits >= this.config.minHits) track.confirmed = true;
  }

  private birth(detection: HeadDetection): TrackedHead {
    const id = this.nextId;
    this.nextId += 1;
    return this.makeTrack(id, detection, this.config.minHits <= 1);
  }

  /** Rebuild a track under a reclaimed gallery id (already confirmed). */
  private resurrect(id: number, detection: HeadDetection): TrackedHead {
    return this.makeTrack(id, detection, true);
  }

  private makeTrack(id: number, detection: HeadDetection, confirmed: boolean): TrackedHead {
    const { cx, cy } = centerOf(detection);
    return {
      id,
      assocBox: associationBoxOf(detection),
      centerX: cx,
      centerY: cy,
      size: Math.max(detection.width, detection.height),
      hits: confirmed ? Math.max(1, this.config.minHits) : 1,
      misses: 0,
      confirmed,
      appearance: detection.appearance ? detection.appearance.slice() : undefined,
      faceDescriptor: detection.faceDescriptor ? detection.faceDescriptor.slice() : undefined,
    };
  }

  private mergeAppearance(
    previous: AppearanceDescriptor | undefined,
    next: AppearanceDescriptor | undefined,
  ): AppearanceDescriptor | undefined {
    if (!next) return previous;
    if (!previous) return next.slice();
    return blendAppearance(previous, next, this.config.appearanceEmaWeight);
  }

  private mergeFace(
    previous: FaceDescriptor | undefined,
    next: FaceDescriptor | undefined,
  ): FaceDescriptor | undefined {
    if (!next) return previous;
    if (!previous) return next.slice();
    return blendFace(previous, next, this.config.faceEmaWeight);
  }
}
