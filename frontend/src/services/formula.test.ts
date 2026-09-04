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
  it('turns count ± tolerance into [min, max]', () => {
    expect(buildFormulaConstraints([{ element: 'C', count: '40', tolerance: '5' }]))
      .toEqual([{ element: 'C', min: 35, max: 45 }])
  })

  it('treats blank/zero tolerance as an exact count', () => {
    expect(buildFormulaConstraints([{ element: 'N', count: '2', tolerance: '' }]))
      .toEqual([{ element: 'N', min: 2, max: 2 }])
  })

  it('clamps min at 0 and skips blank or invalid rows', () => {
    const out = buildFormulaConstraints([
      { element: 'O', count: '3', tolerance: '10' },   // min clamps to 0
      { element: '', count: '', tolerance: '' },        // blank → skipped
      { element: 'Xz', count: '5', tolerance: '0' },    // bad element → skipped
    ])
    expect(out).toEqual([{ element: 'O', min: 0, max: 13 }])
  })
})

describe('formulaFilterEntryError', () => {
  it('accepts a blank row and a valid row', () => {
    expect(formulaFilterEntryError({ element: '', count: '', tolerance: '' })).toBeNull()
    expect(formulaFilterEntryError({ element: 'C', count: '40', tolerance: '5' })).toBeNull()
  })
  it('rejects unknown elements and bad counts', () => {
    expect(formulaFilterEntryError({ element: 'Xz', count: '1', tolerance: '' })).toContain('Xz')
    expect(formulaFilterEntryError({ element: 'C', count: '-1', tolerance: '' })).toBeTruthy()
    expect(formulaFilterEntryError({ element: 'C', count: '4', tolerance: '-2' })).toBeTruthy()
  })
})
