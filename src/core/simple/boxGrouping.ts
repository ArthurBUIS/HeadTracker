/**
 * Hysteretic grouping of close head boxes — merges tracks into shared output
 * streams. Pure and DOM-free, so it's unit-testable.
 *
 * Two boxes MERGE when they're too close: the centre of one falls inside the
 * other's box. Once merged they stay merged (no flicker) until their centres
 * drift more than `unmergeDistance` px apart, when they split. The in-between
 * band is the hysteresis.
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
  /** Centre distance (px) beyond which a merged pair splits. */
  unmergeDistance: number;
}

export const DEFAULT_GROUP_MANAGER_CONFIG: GroupManagerConfig = {
  unmergeDistance: 200,
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

  setUnmergeDistance(px: number): void {
    this.config.unmergeDistance = Math.max(0, px);
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
        const dist = Math.hypot(a.cx - b.cx, a.cy - b.cy);
        if (this.links.has(key)) {
          if (dist > this.config.unmergeDistance) this.links.delete(key);
        } else if (centreInBox(a, b) || centreInBox(b, a)) {
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

/** Is `p`'s centre inside `box`'s box? */
function centreInBox(p: GroupInput, box: GroupInput): boolean {
  return (
    p.cx >= box.cx - box.boxW / 2 &&
    p.cx <= box.cx + box.boxW / 2 &&
    p.cy >= box.cy - box.boxH / 2 &&
    p.cy <= box.cy + box.boxH / 2
  );
}
