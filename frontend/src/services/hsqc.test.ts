import { describe, it, expect } from 'vitest'
import { hsqcPayload, hsqcRows, incompleteHsqcRows } from './hsqc'

const N = NaN

describe('hsqcRows', () => {
  it('keeps signed rows and drops padding', () => {
    const r = hsqcRows([1.2, 20, 1, 3.4, 70, -1, N, N, N])
    expect(r.unsigned).toBe(false)
    expect(r.rows).toEqual([[1.2, 20, 1], [3.4, 70, -1]])
  })

  it('treats a wholly blank sign column as a shifts-only HSQC', () => {
    const r = hsqcRows([1.2, 20, N, 3.4, 70, N, N, N, N])
    expect(r.unsigned).toBe(true)
    expect(r.rows).toEqual([[1.2, 20, null], [3.4, 70, null]])
  })

  it('is not shifts-only when any row has a sign', () => {
    expect(hsqcRows([1.2, 20, N, 3.4, 70, 1]).unsigned).toBe(false)
  })

  it('is not shifts-only when empty', () => {
    expect(hsqcRows([N, N, N]).unsigned).toBe(false)
  })
})

describe('hsqcPayload', () => {
  it('sends null signs for a shifts-only HSQC', () => {
    expect(hsqcPayload([1.2, 20, N, 3.4, 70, N])).toEqual([1.2, 20, null, 3.4, 70, null])
  })

  it('sends complete rows otherwise', () => {
    expect(hsqcPayload([1.2, 20, 1, N, N, N])).toEqual([1.2, 20, 1])
  })
})

describe('incompleteHsqcRows', () => {
  it('accepts shifts without signs when no row has a sign', () => {
    expect(incompleteHsqcRows([[1.2, 20, null], [3.4, 70, null], [null, null, null]])).toBe(0)
  })

  it('flags rows missing a sign once any row has one', () => {
    expect(incompleteHsqcRows([[1.2, 20, 1], [3.4, 70, null], [5.1, 90, null]])).toBe(2)
  })

  it('always flags a row missing a shift', () => {
    expect(incompleteHsqcRows([[1.2, null, null], [null, 70, 1]])).toBe(2)
  })
})
