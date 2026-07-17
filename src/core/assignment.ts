/**
 * Minimum-cost assignment (Hungarian / Kuhn–Munkres).
 *
 * The tracker used greedy matching, which can make a locally-cheap but
 * globally-wrong pairing — a big source of identity swaps when two people
 * cross. This solves the assignment optimally over the whole cost matrix
 * instead, so tracks and detections are paired to minimise total cost.
 *
 * O(n^3) on the padded square, which is nothing at room scale (≤ a handful
 * of people). Rectangular inputs are padded internally; pairs at or above
 * `disallowCost` are treated as forbidden and left unassigned.
 */

export const NO_ASSIGNMENT = -1;

/**
 * Solve the min-cost assignment of rows→columns.
 * @param cost  nRows × nCols matrix (finite numbers).
 * @param disallowCost  a pair whose cost is ≥ this is never assigned.
 * @returns array of length nRows; each entry is a column index or NO_ASSIGNMENT.
 */
export function solveMinCostAssignment(cost: number[][], disallowCost = 1e8): number[] {
  const nRows = cost.length;
  if (nRows === 0) return [];
  const nCols = cost[0].length;
  if (nCols === 0) return new Array<number>(nRows).fill(NO_ASSIGNMENT);

  const n = Math.max(nRows, nCols);
  const PAD = 1e9;
  const a: number[][] = [];
  for (let i = 0; i < n; i += 1) {
    const row: number[] = [];
    for (let j = 0; j < n; j += 1) {
      row.push(i < nRows && j < nCols ? cost[i][j] : PAD);
    }
    a.push(row);
  }

  const rowToCol = hungarianSquare(a);
  const result = new Array<number>(nRows).fill(NO_ASSIGNMENT);
  for (let i = 0; i < nRows; i += 1) {
    const j = rowToCol[i];
    if (j >= 0 && j < nCols && cost[i][j] < disallowCost) result[i] = j;
  }
  return result;
}

/**
 * Standard O(n^3) Hungarian on a square matrix (potentials + augmenting
 * path). Returns row→col for a minimum-cost perfect matching.
 */
function hungarianSquare(a: number[][]): number[] {
  const n = a.length;
  const INF = Infinity;
  const u = new Array<number>(n + 1).fill(0);
  const v = new Array<number>(n + 1).fill(0);
  const p = new Array<number>(n + 1).fill(0); // p[j] = row matched to col j
  const way = new Array<number>(n + 1).fill(0);

  for (let i = 1; i <= n; i += 1) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array<number>(n + 1).fill(INF);
    const used = new Array<boolean>(n + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = INF;
      let j1 = -1;
      for (let j = 1; j <= n; j += 1) {
        if (used[j]) continue;
        const cur = a[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j += 1) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }

  const rowToCol = new Array<number>(n).fill(-1);
  for (let j = 1; j <= n; j += 1) {
    if (p[j] >= 1) rowToCol[p[j] - 1] = j - 1;
  }
  return rowToCol;
}
