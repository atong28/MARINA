/**
 * Grid comparison used to decide whether Handsontable needs reloading.
 *
 * This runs on every store update. It replaced a JSON.stringify of the whole
 * 400x7 grid, so correctness of the early exits matters.
 */
import { describe, it, expect } from 'vitest'
import { gridEquals } from './SpreadsheetTable'

describe('gridEquals', () => {
  it('treats identical grids as equal', () => {
    expect(gridEquals([[1, 'a'], [2, '']], [[1, 'a'], [2, '']])).toBe(true)
  })

  it('detects a changed cell', () => {
    expect(gridEquals([[1, 'a']], [[1, 'b']])).toBe(false)
  })

  it('detects a changed row count', () => {
    expect(gridEquals([[1]], [[1], [2]])).toBe(false)
  })

  it('detects a changed column count', () => {
    expect(gridEquals([[1, 2]], [[1]])).toBe(false)
  })

  it('handles empty grids', () => {
    expect(gridEquals([], [])).toBe(true)
  })

  it('distinguishes a number from its string form', () => {
    // Handsontable hands back either, and they must not be conflated.
    expect(gridEquals([[1]], [['1']])).toBe(false)
  })

  it('distinguishes an empty cell from zero', () => {
    expect(gridEquals([['']], [[0]])).toBe(false)
  })

  it('compares a realistic full-size grid', () => {
    const build = () => Array.from({ length: 400 }, () => [0, 0, 0, 0, 0, 0, 0] as (number | string)[])
    const a = build()
    const b = build()
    expect(gridEquals(a, b)).toBe(true)
    b[399][6] = 1
    expect(gridEquals(a, b)).toBe(false)
  })
})
