/**
 * Grid comparison used to decide whether Handsontable needs reloading.
 *
 * This runs on every store update. It replaced a JSON.stringify of the whole
 * 400x7 grid, so correctness of the early exits matters.
 */
import { describe, it, expect } from 'vitest'
import { gridEquals, lastFilledRow, visibleRowCount } from './SpreadsheetTable'

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

/**
 * Sheet height follows the data. Empty and near-full are the two ends that a
 * naive "show as many rows as there is data" would get wrong.
 */
describe('lastFilledRow', () => {
  const empty = () => Array.from({ length: 400 }, () => ['', '', '', '', '', '', ''] as (number | string)[])

  it('reports -1 for an untouched grid', () => {
    expect(lastFilledRow(empty())).toBe(-1)
  })

  it('finds the lowest row holding any value', () => {
    const g = empty()
    g[12][4] = 55.2
    expect(lastFilledRow(g)).toBe(12)
  })

  it('does not stop at a gap', () => {
    const g = empty()
    g[0][0] = 1
    g[40][3] = 7.26
    expect(lastFilledRow(g)).toBe(40)
  })

  it('counts a zero as filled', () => {
    const g = empty()
    g[3][2] = 0
    expect(lastFilledRow(g)).toBe(3)
  })
})

describe('visibleRowCount', () => {
  it('keeps an empty sheet usable', () => {
    expect(visibleRowCount(-1)).toBe(10)
  })

  it('leaves one spare row under the data', () => {
    expect(visibleRowCount(14)).toBe(16)
  })

  it('caps at 30 rows for a long peak list', () => {
    expect(visibleRowCount(399)).toBe(30)
  })
})
