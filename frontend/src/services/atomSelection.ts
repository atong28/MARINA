/**
 * Geometry for picking atoms on a depiction: a click picks the nearest atom, a
 * drag draws a lasso that picks every atom inside it. Coordinates are the
 * depiction's own canvas units (DepictionGeometry.atoms).
 */

export type Point = [number, number]

/** A drag shorter than this (canvas units at 400 px) counts as a click. */
export const CLICK_TOLERANCE = 6
/** A click this close (canvas units at 400 px) to an atom picks it. */
export const PICK_RADIUS = 16

export function pathLength(points: Point[]): number {
  let total = 0
  for (let i = 1; i < points.length; i++) {
    total += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
  }
  return total
}

/** Even–odd ray casting; the polygon is closed implicitly. */
export function pointInPolygon([x, y]: Point, polygon: Point[]): boolean {
  let inside = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i]
    const [xj, yj] = polygon[j]
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

/** Indices of the atoms inside the lasso. */
export function atomsInPolygon(atoms: number[][], polygon: Point[]): number[] {
  if (polygon.length < 3) return []
  const out: number[] = []
  atoms.forEach((p, i) => {
    if (p && pointInPolygon([p[0], p[1]], polygon)) out.push(i)
  })
  return out
}

/** The atom nearest the point, if any lies within maxDist. */
export function nearestAtom(atoms: number[][], [x, y]: Point, maxDist: number): number | null {
  let best: number | null = null
  let bestD = maxDist
  atoms.forEach((p, i) => {
    if (!p) return
    const d = Math.hypot(p[0] - x, p[1] - y)
    if (d <= bestD) {
      best = i
      bestD = d
    }
  })
  return best
}

/** Bonds with both ends selected, so the highlight reads as a connected region. */
export function bondsWithin(bonds: number[][], selected: Set<number>): number[] {
  const out: number[] = []
  bonds.forEach(([a, b], i) => {
    if (selected.has(a) && selected.has(b)) out.push(i)
  })
  return out
}

/**
 * Apply one gesture to a selection. A click toggles the atom under it; a lasso
 * adds what it encloses, or removes it when `subtract` (Alt held).
 */
export function applyGesture(
  selected: Set<number>,
  atoms: number[][],
  points: Point[],
  scale: number,
  subtract: boolean,
): Set<number> {
  const next = new Set(selected)
  if (points.length === 0) return next
  if (pathLength(points) < CLICK_TOLERANCE * scale) {
    const hit = nearestAtom(atoms, points[points.length - 1], PICK_RADIUS * scale)
    if (hit !== null) {
      if (next.has(hit)) next.delete(hit)
      else next.add(hit)
    }
    return next
  }
  for (const i of atomsInPolygon(atoms, points)) {
    if (subtract) next.delete(i)
    else next.add(i)
  }
  return next
}
