import { describe, it, expect } from 'vitest'
import { formatPct } from './BitPanel'

/**
 * The isotonic curve fitted on the MARINA1 test split tops out at 0.9984, so no
 * confidence the backend can produce justifies telling a user a substructure is
 * certainly present. Plain rounding would render that as "100%".
 */
describe('formatPct', () => {
  it('never renders 100% for the highest value the curve can emit', () => {
    expect(formatPct(0.9984)).toBe('99.8%')
  })

  it('keeps a decimal place across the whole saturating range', () => {
    expect(formatPct(0.995)).toBe('99.5%')
    expect(formatPct(0.999)).toBe('99.9%')
  })

  it('rounds to whole percents below the saturating range', () => {
    expect(formatPct(0.885)).toBe('89%')
    expect(formatPct(0.696)).toBe('70%')
    expect(formatPct(0.5)).toBe('50%')
  })

  it('floors rather than rounds near the top, so it never overstates', () => {
    expect(formatPct(0.99989)).toBe('99.9%')
  })

  it('handles the bottom of the range', () => {
    expect(formatPct(0)).toBe('0%')
  })

  it('only reaches 100% at a true 1.0, which the curve never returns', () => {
    expect(formatPct(1)).toBe('100%')
  })
})
