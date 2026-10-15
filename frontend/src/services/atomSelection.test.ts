import { describe, it, expect } from 'vitest'
import {
  applyGesture, atomsInPolygon, bondsWithin, nearestAtom, pathLength, pointInPolygon,
  type Point,
} from './atomSelection'

// Four atoms on a line, 40 units apart, and a fifth well below.
const ATOMS = [[40, 100], [80, 100], [120, 100], [160, 100], [100, 300]]
const SQUARE: Point[] = [[30, 80], [130, 80], [130, 120], [30, 120]]

describe('pointInPolygon', () => {
  it('tells inside from outside', () => {
    expect(pointInPolygon([50, 100], SQUARE)).toBe(true)
    expect(pointInPolygon([150, 100], SQUARE)).toBe(false)
    expect(pointInPolygon([50, 200], SQUARE)).toBe(false)
  })
})

describe('atomsInPolygon', () => {
  it('returns the enclosed atoms', () => {
    expect(atomsInPolygon(ATOMS, SQUARE)).toEqual([0, 1, 2])
  })

  it('needs at least a triangle', () => {
    expect(atomsInPolygon(ATOMS, [[0, 0], [400, 400]])).toEqual([])
  })
})

describe('nearestAtom', () => {
  it('picks the closest atom within reach', () => {
    expect(nearestAtom(ATOMS, [83, 104], 16)).toBe(1)
  })

  it('picks nothing out of reach', () => {
    expect(nearestAtom(ATOMS, [100, 200], 16)).toBeNull()
  })
})

describe('applyGesture', () => {
  const none = new Set<number>()

  it('treats a short path as a click that toggles one atom', () => {
    const on = applyGesture(none, ATOMS, [[81, 101], [82, 102]], 1, false)
    expect([...on]).toEqual([1])
    const off = applyGesture(on, ATOMS, [[81, 101]], 1, false)
    expect([...off]).toEqual([])
  })

  it('adds the lasso contents', () => {
    const sel = applyGesture(new Set([4]), ATOMS, SQUARE, 1, false)
    expect([...sel].sort()).toEqual([0, 1, 2, 4])
  })

  it('removes the lasso contents with subtract', () => {
    const sel = applyGesture(new Set([0, 1, 2, 3]), ATOMS, SQUARE, 1, true)
    expect([...sel]).toEqual([3])
  })

  it('scales the click tolerance with the canvas', () => {
    // 10 units of travel is a lasso at scale 1 but a click at scale 2.
    const path: Point[] = [[80, 100], [90, 100]]
    expect(pathLength(path)).toBe(10)
    expect([...applyGesture(none, ATOMS, path, 2, false)]).toEqual([1])
  })
})

describe('bondsWithin', () => {
  it('keeps bonds whose ends are both selected', () => {
    const bonds = [[0, 1], [1, 2], [2, 3]]
    expect(bondsWithin(bonds, new Set([0, 1, 2]))).toEqual([0, 1])
  })
})
