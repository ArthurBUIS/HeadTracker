/**
 * Hysteretic grouping of close head boxes — merges tracks into shared output
 * streams. Pure and DOM-free, so it's unit-testable.
 *
 * Two merge methods, both hysteretic:
 *
 *   'proximity' — each box is a 16:9 rectangle. Merge when the centre of one
 *   falls inside the inner `X:9` core of the other (same height, width
 *   narrowed to X/16, X in [5, 16]); unmerge when a centre leaves the other's
 *   FULL 16:9 box. The gap between the inner zone and the full box is the
 *   hysteresis; X=16 collapses it.
 *
 *   'overlap' — merge when two boxes overlap by ≥ `mergeOverlapPct` of the
 *   smaller box's area; unmerge when the overlap drops below
 *   `unmergeOverlapPct`. Keep merge% ≥ unmerge% for a hysteresis band (the
 *   unmerge threshold is clamped ≤ merge threshold to avoid oscillation).
 *
 * The merge relation is per-pair; groups are the connected components of the
 * "merged" links. Each component becomes one output stream, keyed by the
 * smallest track id in it (stable while that member persists).
 */

/** A track's box for grouping: centre + box extent in source px. */
export interface GroupInput {
  id: number;
  cx: number;
  cy: number;
  boxW: number;
  boxH: number;
}

export type MergeMethod = 'proximity' | 'overlap';

export interface GroupManagerConfig {
  /** Which merge rule is active. */
  mergeMethod: MergeMethod;
  /**
   * ('proximity') Width, in ninths, of the inner merge zone (an X:9 rectangle
   * centred in the 16:9 box). Merge when a centre enters this zone; unmerge
   * when it leaves the full box. Clamped to [5, 16]; 16 = whole box (no band).
   */
  mergeWidthUnits: number;
  /** ('overlap') Merge when overlap ≥ this % of the smaller box. [0, 100]. */
  mergeOverlapPct: number;
  /** ('overlap') Unmerge when overlap < this % of the smaller box. [0, 100]. */
  unmergeOverlapPct: number;
}

export const DEFAULT_GROUP_MANAGER_CONFIG: GroupManagerConfig = {
  mergeMethod: 'overlap',
  mergeWidthUnits: 5,
  mergeOverlapPct: 70,
  unmergeOverlapPct: 50,
};

/** One output group: its stable id and the track ids it contains. */
export interface Group {
  groupId: number;
  memberIds: number[];
}

export class BoxGroupManager {
  private readonly config: GroupManagerConfig;

  /** Currently-merged pairs, keyed "min:max". */
  private links = new Set<string>();

  constructor(config: Partial<GroupManagerConfig> = {}) {
    this.config = { ...DEFAULT_GROUP_MANAGER_CONFIG, ...config };
  }

  /** Pick the merge rule ('proximity' or 'overlap'); live. */
  setMergeMethod(method: MergeMethod): void {
    this.config.mergeMethod = method;
  }

  /** Set the inner merge-zone width in ninths (X in [5, 16]); live. */
  setMergeWidthUnits(units: number): void {
    this.config.mergeWidthUnits = Math.min(16, Math.max(5, units));
  }

  /** Set the overlap merge threshold, in % of the smaller box [0, 100]; live. */
  setMergeOverlapPct(pct: number): void {
    this.config.mergeOverlapPct = Math.min(100, Math.max(0, pct));
  }

  /** Set the overlap unmerge threshold, in % of the smaller box [0, 100]; live. */
  setUnmergeOverlapPct(pct: number): void {
    this.config.unmergeOverlapPct = Math.min(100, Math.max(0, pct));
  }

  /** Recompute the merged links (with hysteresis) and return the groups. */
  update(inputs: GroupInput[]): Group[] {
    const byId = new Map(inputs.map((i) => [i.id, i]));

    // Drop links whose tracks no longer exist.
    for (const key of [...this.links]) {
      const [a, b] = key.split(':').map(Number);
      if (!byId.has(a) || !byId.has(b)) this.links.delete(key);
    }

    // Update each pair's link state with hysteresis.
    for (let i = 0; i < inputs.length; i += 1) {
      for (let j = i + 1; j < inputs.length; j += 1) {
        const a = inputs[i];
        const b = inputs[j];
        const key = linkKey(a.id, b.id);
        const merged = this.links.has(key);
        if (this.pairMerges(a, b, merged)) this.links.add(key);
        else this.links.delete(key);
      }
    }

    return this.connectedComponents(inputs);
  }

  /** Should this pair be merged this round, given its current merged state? */
  private pairMerges(a: GroupInput, b: GroupInput, merged: boolean): boolean {
    if (this.config.mergeMethod === 'overlap') {
      const ov = overlapFraction(a, b);
      const mergeFrac = this.config.mergeOverlapPct / 100;
      // Clamp the unmerge threshold ≤ merge threshold so the band never
      // inverts (which would make a pair oscillate merge/split every round).
      const unmergeFrac = Math.min(this.config.unmergeOverlapPct / 100, mergeFrac);
      return merged ? ov >= unmergeFrac : ov >= mergeFrac;
    }
    // 'proximity': merge zone is the inner X:9 core (width narrowed to X/16);
    // unmerge zone is the full 16:9 box. The gap between them is hysteresis.
    if (merged) return centreInBox(a, b, 1) || centreInBox(b, a, 1);
    const mergeWidthScale = this.config.mergeWidthUnits / 16;
    return centreInBox(a, b, mergeWidthScale) || centreInBox(b, a, mergeWidthScale);
  }

  private connectedComponents(inputs: GroupInput[]): Group[] {
    const parent = new Map<number, number>();
    for (const input of inputs) parent.set(input.id, input.id);
    const find = (x: number): number => {
      let root = x;
      while (parent.get(root) !== root) root = parent.get(root) as number;
      return root;
    };
    const union = (x: number, y: number): void => {
      parent.set(find(x), find(y));
    };
    for (const key of this.links) {
      const [a, b] = key.split(':').map(Number);
      union(a, b);
    }

    const byRoot = new Map<number, number[]>();
    for (const input of inputs) {
      const root = find(input.id);
      const members = byRoot.get(root) ?? [];
      members.push(input.id);
      byRoot.set(root, members);
    }
    return [...byRoot.values()].map((memberIds) => ({
      groupId: Math.min(...memberIds),
      memberIds: memberIds.slice().sort((a, b) => a - b),
    }));
  }
}

function linkKey(a: number, b: number): string {
  return a < b ? `${a}:${b}` : `${b}:${a}`;
}

/**
 * Overlap of two boxes as a fraction [0, 1] of the SMALLER box's area
 * (intersection ÷ min area). 1 = the smaller box is fully inside the larger.
 * More intuitive than IoU for "% overlap": two equal boxes sharing half their
 * area read as 0.5, not IoU's 0.33.
 */
function overlapFraction(a: GroupInput, b: GroupInput): number {
  const ix = Math.max(
    0,
    Math.min(a.cx + a.boxW / 2, b.cx + b.boxW / 2) - Math.max(a.cx - a.boxW / 2, b.cx - b.boxW / 2),
  );
  const iy = Math.max(
    0,
    Math.min(a.cy + a.boxH / 2, b.cy + b.boxH / 2) - Math.max(a.cy - a.boxH / 2, b.cy - b.boxH / 2),
  );
  const inter = ix * iy;
  if (inter <= 0) return 0;
  const minArea = Math.min(a.boxW * a.boxH, b.boxW * b.boxH);
  return minArea <= 0 ? 0 : inter / minArea;
}

/**
 * Is `p`'s centre inside `box`, with its width scaled by `widthScale` (height
 * unchanged)? widthScale 1 tests the full box; X/16 tests the inner X:9 core.
 */
function centreInBox(p: GroupInput, box: GroupInput, widthScale: number): boolean {
  const halfW = (box.boxW * widthScale) / 2;
  return (
    p.cx >= box.cx - halfW &&
    p.cx <= box.cx + halfW &&
    p.cy >= box.cy - box.boxH / 2 &&
    p.cy <= box.cy + box.boxH / 2
  );
}
