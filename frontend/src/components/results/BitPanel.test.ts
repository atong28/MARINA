import { describe, it, expect } from 'vitest'
import { bitMeta, bitTitle, formatPct } from './BitPanel'
import type { BitExplanation } from '../../services/api'

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

/**
 * A substructure or multiplicity feature carries no radius of its own, so the radius is
 * only known for a feature located in this structure. Every predicted-but-absent bit
 * therefore arrives as -1, and the panel used to render that literally as "r-1" — on
 * NP0332333 that was ~39% of rows, all claiming a radius that does not exist.
 */
function bit(over: Partial<BitExplanation> = {}): BitExplanation {
  return {
    index: 926, fragment_smiles: 'COP', atom_symbol: '', radius: -1,
    raw_confidence: 0.5, confidence: 0.5, band: 'Possible', present: false,
    group: 'uncertain', atoms: [], bonds: [], ...over,
  } as BitExplanation
}

describe('bitMeta', () => {
  it('omits the radius entirely when it is unknown', () => {
    expect(bitMeta(bit({ radius: -1 }))).toBe('#926')
  })

  it('shows the radius when the feature was located', () => {
    expect(bitMeta(bit({ radius: 2, present: true }))).toBe('r2·#926')
  })

  it('keeps radius 0, which is a real radius and not a missing one', () => {
    expect(bitMeta(bit({ radius: 0, present: true }))).toBe('r0·#926')
  })

  it('distinguishes cumulative multiplicity buckets sharing one fragment', () => {
    const a = bitMeta(bit({ index: 926, multiplicity: 1 }))
    const b = bitMeta(bit({ index: 1035, multiplicity: 3 }))
    expect(a).toBe('≥1×·#926')
    expect(b).toBe('≥3×·#1035')
    expect(a).not.toBe(b)
  })

  it('shows radius and multiplicity together when both are known', () => {
    expect(bitMeta(bit({ radius: 2, multiplicity: 3, present: true }))).toBe('r2·≥3×·#926')
  })

  it('leaves Morgan rows untouched, since they carry no multiplicity', () => {
    expect(bitMeta(bit({ radius: 1, multiplicity: null, present: true }))).toBe('r1·#926')
  })
})

describe('bitTitle', () => {
  it('does not mention a radius it does not have', () => {
    expect(bitTitle(bit({ radius: -1 }))).not.toContain('radius')
  })

  it('explains the bucket in words rather than symbols', () => {
    expect(bitTitle(bit({ multiplicity: 3 }))).toContain('appears at least 3')
  })
})
