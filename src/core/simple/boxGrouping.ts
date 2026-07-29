/**
 * Hysteretic grouping of close head boxes — merges tracks into shared output
 * streams. Pure and DOM-free, so it's unit-testable.
 *
 * Each box is a 16:9 rectangle. Two boxes MERGE when the centre of one falls
 * inside the inner `X:9` core of the other's box — same height, but width
 * narrowed to X/16 (X in [5, 16]). They stay merged until a centre leaves the
 * other's FULL 16:9 box, when they split. Because the merge zone (inner X:9)
 * sits inside the unmerge zone (full box), the gap between them is the
 * hysteresis; X=16 collapses that gap to zero.
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

export interface GroupManagerConfig {
  /**
   * Width, in ninths, of the inner merge zone (an X:9 rectangle centred in the
   * 16:9 box). Merge when a centre enters this zone; unmerge when it leaves the
   * full box. Clamped to [5, 16]; 16 means the whole box (no hysteresis band).
   */
  mergeWidthUnits: number;
}

export const DEFAULT_GROUP_MANAGER_CONFIG: GroupManagerConfig = {
  mergeWidthUnits: 9,
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

  /** Set the inner merge-zone width in ninths (X in [5, 16]); live. */
  setMergeWidthUnits(units: number): void {
    this.config.mergeWidthUnits = Math.min(16, Math.max(5, units));
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
        // Merge zone: inner X:9 core (width narrowed to X/16). Unmerge zone:
        // the full 16:9 box (widthScale 1). The gap between them is hysteresis.
        const mergeWidthScale = this.config.mergeWidthUnits / 16;
        if (this.links.has(key)) {
          if (!centreInBox(a, b, 1) && !centreInBox(b, a, 1)) this.links.delete(key);
        } else if (
          centreInBox(a, b, mergeWidthScale) ||
          centreInBox(b, a, mergeWidthScale)
        ) {
          this.links.add(key);
        }
      }
    }

    return this.connectedComponents(inputs);
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
