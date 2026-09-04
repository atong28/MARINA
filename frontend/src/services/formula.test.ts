import { describe, it, expect } from 'vitest'
import {
  validateFormula, tokenizeFormula, buildFormulaConstraints, formulaFilterEntryError,
  toSubscript, fromSubscript,
} from './formula'

describe('subscript conversion', () => {
  it('renders digits as subscripts and round-trips back to ASCII', () => {
    expect(toSubscript('C10H12N2O')).toBe('C₁₀H₁₂N₂O')
    expect(fromSubscript('C₁₀H₁₂N₂O')).toBe('C10H12N2O')
    expect(fromSubscript(toSubscript('CuSO4'))).toBe('CuSO4')
  })
  it('leaves letters untouched', () => {
    expect(toSubscript('NaCl')).toBe('NaCl')
  })
})

describe('validateFormula', () => {
  it('accepts an empty string (formula is optional)', () => {
    expect(validateFormula('').valid).toBe(true)
    expect(validateFormula('   ').valid).toBe(true)
  })

  it('accepts valid Hill-notation formulas', () => {
    expect(validateFormula('C10H12N2O').valid).toBe(true)
    expect(validateFormula('H2O').valid).toBe(true)
    expect(validateFormula('C6H5Cl').valid).toBe(true)   // two-letter element
    expect(validateFormula('CuSO4').valid).toBe(true)
  })

  it('rejects unknown elements', () => {
    const r = validateFormula('C10Xz2')
    expect(r.valid).toBe(false)
    expect(r.error).toContain('Xz')
  })

  it('rejects malformed strings', () => {
    expect(validateFormula('123').valid).toBe(false)   // leading digits, no element
    expect(validateFormula('c10h12').valid).toBe(false) // lowercase-led token
    expect(validateFormula('C-10').valid).toBe(false)
  })
})

describe('tokenizeFormula', () => {
  it('splits into element/count pairs', () => {
    expect(tokenizeFormula('C10H12N2O')).toEqual([
      { symbol: 'C', count: '10' },
      { symbol: 'H', count: '12' },
      { symbol: 'N', count: '2' },
      { symbol: 'O', count: '' },
    ])
  })
})

describe('buildFormulaConstraints', () => {
  it('turns low/high into [min, max]', () => {
    expect(buildFormulaConstraints([{ element: 'C', low: '35', high: '45' }]))
      .toEqual([{ element: 'C', min: 35, max: 45 }])
  })

  it('allows an open-ended bound (blank low or high)', () => {
    expect(buildFormulaConstraints([{ element: 'N', low: '2', high: '' }]))
      .toEqual([{ element: 'N', min: 2 }])
    expect(buildFormulaConstraints([{ element: 'O', low: '', high: '8' }]))
      .toEqual([{ element: 'O', max: 8 }])
  })

  it('skips blank or invalid rows', () => {
    const out = buildFormulaConstraints([
      { element: 'C', low: '10', high: '12' },
      { element: '', low: '', high: '' },        // blank → skipped
      { element: 'Xz', low: '5', high: '9' },    // bad element → skipped
    ])
    expect(out).toEqual([{ element: 'C', min: 10, max: 12 }])
  })
})

describe('formulaFilterEntryError', () => {
  it('accepts a blank row and a valid row', () => {
    expect(formulaFilterEntryError({ element: '', low: '', high: '' })).toBeNull()
    expect(formulaFilterEntryError({ element: 'C', low: '35', high: '45' })).toBeNull()
    expect(formulaFilterEntryError({ element: 'C', low: '35', high: '' })).toBeNull()
  })
  it('rejects unknown elements, bad numbers, and low > high', () => {
    expect(formulaFilterEntryError({ element: 'Xz', low: '1', high: '' })).toContain('Xz')
    expect(formulaFilterEntryError({ element: 'C', low: '-1', high: '' })).toBeTruthy()
    expect(formulaFilterEntryError({ element: 'C', low: '', high: '' })).toBeTruthy()
    expect(formulaFilterEntryError({ element: 'C', low: '10', high: '5' })).toBeTruthy()
  })
})
