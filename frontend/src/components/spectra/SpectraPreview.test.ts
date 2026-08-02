/**
 * Scale, tick and view maths for the spectra preview.
 *
 * The axis directions are the point of these tests: f2 (¹H) must run high → low
 * left→right, f1 (¹³C) low → high top→bottom, and MS low → high m/z. Getting
 * any of them backwards produces a plot that looks plausible and is wrong.
 */
import { describe, it, expect } from 'vitest'
import {
  linear, padDomain, originDomain, niceTicks, fmtShift, fmtIntensity, zoomAxis, viewEquals,
} from './SpectraPreview'

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

describe('originDomain', () => {
  it('reaches 0 exactly for an all-positive spectrum', () => {
    const [lo, hi] = originDomain(2.1, 7.3)
    expect(lo).toBe(0)
    expect(hi).toBeGreaterThan(7.3)
  })

  it('keeps a negative shift visible', () => {
    const [lo, hi] = originDomain(-0.4, 7.3)
    expect(lo).toBeLessThan(-0.4)
    expect(hi).toBeGreaterThan(7.3)
  })

  it('contains the origin for a carbon range', () => {
    const [lo, hi] = originDomain(22.8, 172.4)
    expect(lo).toBe(0)
    expect(hi).toBeGreaterThan(172.4)
  })

  it('gives an all-zero spectrum a usable width', () => {
    const [lo, hi] = originDomain(0, 0)
    expect(hi).toBeGreaterThan(lo)
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

describe('HSQC axis direction', () => {
  // The plot builds its opening window as
  //   { xLeft: hHi, xRight: hLo, yTop: cLo, yBottom: cHi }
  // from originDomain, which is what these reproduce.
  const [hLo, hHi] = originDomain(1.0, 8.0)
  const [cLo, cHi] = originDomain(20, 180)
  const x = linear(hHi, hLo, 0, 500)
  const y = linear(cLo, cHi, 0, 400) // y grows downward in SVG

  it('places a high proton shift left of a low one', () => {
    expect(x(8.0)).toBeLessThan(x(1.0))
  })

  it('places a high carbon shift below a low one', () => {
    expect(y(180)).toBeGreaterThan(y(20))
  })

  it('puts the origin in the top-right corner', () => {
    expect(x(0)).toBeCloseTo(500, 6) // right edge
    expect(y(0)).toBeCloseTo(0, 6) // top edge
  })
})

describe('m/z axis direction', () => {
  it('places a high m/z right of a low one', () => {
    const [lo, hi] = padDomain(100, 600)
    const x = linear(lo, hi, 0, 500)
    expect(x(600)).toBeGreaterThan(x(100))
  })
})

describe('zoomAxis', () => {
  it('holds the cursor position fixed while shrinking the span', () => {
    const [lo, hi] = zoomAxis(0, 10, 4, 0.5, 10)
    expect(hi - lo).toBeCloseTo(5, 6)
    // 4 sits 40% along before and after.
    expect((4 - lo) / (hi - lo)).toBeCloseTo(0.4, 6)
  })

  it('works on a reversed axis', () => {
    const [left, right] = zoomAxis(8, 0, 2, 0.5, -8)
    expect(left).toBeGreaterThan(right) // still reversed
    expect(Math.abs(right - left)).toBeCloseTo(4, 6)
  })

  it('refuses to zoom in past the limit', () => {
    expect(zoomAxis(0, 10, 5, 1e-6, 10)).toEqual([0, 10])
  })

  it('refuses to zoom out past the limit', () => {
    expect(zoomAxis(0, 10, 5, 1e6, 10)).toEqual([0, 10])
  })
})

describe('viewEquals', () => {
  const v = { xLeft: 8, xRight: 0, yTop: 0, yBottom: 180 }

  it('matches an identical window', () => {
    expect(viewEquals(v, { ...v })).toBe(true)
  })

  it('detects a panned window', () => {
    expect(viewEquals(v, { ...v, xLeft: 7.5 })).toBe(false)
  })
})
