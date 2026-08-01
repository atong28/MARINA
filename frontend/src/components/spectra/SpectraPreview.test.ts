/**
 * Scale and tick maths for the spectra preview.
 *
 * The axis directions are the point of these tests: NMR chemical-shift axes
 * must run high → low ppm, and MS must run low → high m/z. Getting either
 * backwards produces a plot that looks plausible and is wrong.
 */
import { describe, it, expect } from 'vitest'
import { linear, padDomain, niceTicks, fmtShift, fmtIntensity } from './SpectraPreview'

describe('linear', () => {
  it('maps the domain onto the range', () => {
    const s = linear(0, 10, 0, 100)
    expect(s(0)).toBe(0)
    expect(s(5)).toBe(50)
    expect(s(10)).toBe(100)
  })

  it('reverses when the domain is given high-to-low', () => {
    // How every ppm axis is built: high shift on the left.
    const s = linear(10, 0, 0, 100)
    expect(s(10)).toBe(0)
    expect(s(0)).toBe(100)
  })

  it('extrapolates outside the domain', () => {
    const s = linear(0, 10, 0, 100)
    expect(s(20)).toBe(200)
    expect(s(-5)).toBe(-50)
  })

  it('does not divide by zero on a degenerate domain', () => {
    expect(Number.isFinite(linear(5, 5, 0, 100)(5))).toBe(true)
  })
})

describe('padDomain', () => {
  it('pads by a fraction of the span', () => {
    expect(padDomain(0, 10, 0.1)).toEqual([-1, 11])
  })

  it('keeps the ordering', () => {
    const [lo, hi] = padDomain(2, 8)
    expect(lo).toBeLessThan(2)
    expect(hi).toBeGreaterThan(8)
  })

  it('gives a single point a usable width', () => {
    const [lo, hi] = padDomain(5, 5)
    expect(hi).toBeGreaterThan(lo)
  })

  it('gives a single zero point a usable width', () => {
    const [lo, hi] = padDomain(0, 0)
    expect(hi).toBeGreaterThan(lo)
    expect(Number.isFinite(lo) && Number.isFinite(hi)).toBe(true)
  })
})

describe('niceTicks', () => {
  it('produces round numbers inside the domain', () => {
    const { values } = niceTicks(0, 10, 5)
    expect(values.length).toBeGreaterThan(1)
    for (const v of values) {
      expect(v).toBeGreaterThanOrEqual(0)
      expect(v).toBeLessThanOrEqual(10)
    }
  })

  it('spaces ticks evenly', () => {
    const { values } = niceTicks(0, 100, 5)
    const gaps = values.slice(1).map((v, i) => v - values[i])
    for (const g of gaps) expect(g).toBeCloseTo(gaps[0], 6)
  })

  it('adds decimals for narrow ranges', () => {
    const { fmt } = niceTicks(0, 1, 5)
    expect(fmt(0.2)).toContain('.')
  })

  it('omits decimals for wide ranges', () => {
    const { fmt } = niceTicks(0, 200, 5)
    expect(fmt(50)).toBe('50')
  })

  it('renders a clean zero rather than -0', () => {
    const { values, fmt } = niceTicks(-10, 10, 5)
    const zero = values.find((v) => Math.abs(v) < 1e-9)
    expect(zero === undefined || !Object.is(zero, -0)).toBe(true)
    expect(fmt(0)).not.toContain('-')
  })

  it('handles a typical proton range', () => {
    const { values } = niceTicks(0, 12, 6)
    expect(values[0]).toBeGreaterThanOrEqual(0)
    expect(values[values.length - 1]).toBeLessThanOrEqual(12)
  })

  it('handles a typical carbon range', () => {
    const { values } = niceTicks(0, 220, 6)
    expect(values.length).toBeGreaterThan(2)
  })
})

describe('formatters', () => {
  it('shows chemical shifts to two decimals', () => {
    expect(fmtShift(5.827)).toBe('5.83')
    expect(fmtShift(78.4)).toBe('78.40')
  })

  it('uses scientific notation for large intensities', () => {
    expect(fmtIntensity(4007870.8)).toMatch(/e\+/)
  })

  it('keeps small intensities readable', () => {
    expect(fmtIntensity(12.5)).toBe('12.50')
  })

  it('formats zero plainly', () => {
    expect(fmtIntensity(0)).toBe('0')
  })

  it('preserves the sign of a negative HSQC intensity', () => {
    // Sign carries the CH2 multiplicity edit; losing it loses the meaning.
    expect(fmtIntensity(-1524417.4).startsWith('-')).toBe(true)
  })
})

describe('ppm axis direction', () => {
  it('places a high shift left of a low one', () => {
    const [lo, hi] = padDomain(1.0, 8.0)
    const x = linear(hi, lo, 0, 500)
    expect(x(8.0)).toBeLessThan(x(1.0))
  })

  it('places a high carbon shift above a low one', () => {
    const [lo, hi] = padDomain(20, 180)
    const y = linear(hi, lo, 0, 400)     // y grows downward in SVG
    expect(y(180)).toBeLessThan(y(20))
  })

  it('places a high m/z right of a low one', () => {
    const [lo, hi] = padDomain(100, 600)
    const x = linear(lo, hi, 0, 500)
    expect(x(600)).toBeGreaterThan(x(100))
  })
})
