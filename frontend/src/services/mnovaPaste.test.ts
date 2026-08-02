/**
 * MestreNova clipboard parsing.
 *
 * The two failure modes worth guarding are silent ones: reading f1 as the
 * proton axis (a plausible, wrong spectrum) and dropping the continuation
 * columns of a long table (a quietly truncated peak list).
 */
import { describe, it, expect } from 'vitest'
import { parseMnovaTable, peakCount, MnovaParseError } from './mnovaPaste'

const H_HEADER = '\tppm\tIntensity\tWidth\tArea\tType\tFlags\tImpurity/Compound\tAnnotation'
const HSQC_HEADER =
  '\tf2 (ppm)\tf1 (ppm)\tIntensity\tWidth f2\tWidth f1\tVolume\tType\tFlags\t' +
  'Impurity/Compound\tAnnotation'

function hRow(i: number, ppm: number, intensity = 1.0) {
  return `${i}\t${ppm}\t${intensity}\t1.5\t0.98\tCompound\t\t\t`
}

function hsqcRow(i: number, f2: number, f1: number, intensity: number) {
  return `${i}\t${f2}\t${f1}\t${intensity}\t0.02\t0.4\t${intensity * 10}\tCompound\t\t\t`
}

describe('1-D tables', () => {
  it('reads shifts and infers ¹H from the shift range', () => {
    const text = [H_HEADER, hRow(1, 7.26), hRow(2, 3.71), hRow(3, 1.24)].join('\n')
    const parsed = parseMnovaTable(text)
    expect(parsed.kind).toBe('h_nmr')
    expect(parsed).toMatchObject({ shifts: [7.26, 3.71, 1.24] })
  })

  it('infers ¹³C when any shift is outside the proton window', () => {
    const text = [H_HEADER, hRow(1, 172.4), hRow(2, 55.1), hRow(3, 14.2)].join('\n')
    const parsed = parseMnovaTable(text)
    expect(parsed.kind).toBe('c_nmr')
  })

  it('treats an all-aliphatic-looking table as ¹H', () => {
    // 14.2 alone is ambiguous; the documented rule is "all inside the window".
    const text = [H_HEADER, hRow(1, 14.2), hRow(2, 2.1)].join('\n')
    expect(parseMnovaTable(text).kind).toBe('h_nmr')
  })

  it('accepts a slightly negative proton shift', () => {
    const text = [H_HEADER, hRow(1, -0.12), hRow(2, 4.5)].join('\n')
    expect(parseMnovaTable(text).kind).toBe('h_nmr')
  })

  it('keeps reading after the first block when the table is continued right', () => {
    const text = [
      `${H_HEADER}${H_HEADER}`,
      `${hRow(1, 7.26)}${hRow(3, 2.05)}`,
      `${hRow(2, 3.71)}${hRow(4, 1.24)}`,
    ].join('\n')
    const parsed = parseMnovaTable(text)
    // Left block in full, then the right block — the order Mnova split them in.
    expect(parsed).toMatchObject({ shifts: [7.26, 3.71, 2.05, 1.24] })
  })

  it('ignores ragged trailing cells in a continued table', () => {
    const text = [
      `${H_HEADER}${H_HEADER}`,
      `${hRow(1, 7.26)}${hRow(3, 2.05)}`,
      hRow(2, 3.71), // right-hand block ran out of peaks
    ].join('\n')
    expect(peakCount(parseMnovaTable(text))).toBe(3)
  })
})

describe('HSQC tables', () => {
  it('maps f2 to ¹H and f1 to ¹³C and keeps the intensity sign', () => {
    const text = [HSQC_HEADER, hsqcRow(1, 3.71, 55.2, -1524417.4), hsqcRow(2, 7.26, 128.9, 4007870.8)].join('\n')
    const parsed = parseMnovaTable(text)
    expect(parsed).toMatchObject({
      kind: 'hsqc',
      peaks: [
        { f2: 3.71, f1: 55.2, intensity: -1524417.4 },
        { f2: 7.26, f1: 128.9, intensity: 4007870.8 },
      ],
    })
  })

  it('is detected even when every carbon shift is small', () => {
    // Range-based inference must not get a vote once f1/f2 headers are present.
    const text = [HSQC_HEADER, hsqcRow(1, 1.2, 14.1, 100)].join('\n')
    expect(parseMnovaTable(text).kind).toBe('hsqc')
  })

  it('reads continuation columns', () => {
    const text = [
      `${HSQC_HEADER}${HSQC_HEADER}`,
      `${hsqcRow(1, 3.71, 55.2, 10)}${hsqcRow(3, 5.02, 72.4, 30)}`,
      `${hsqcRow(2, 7.26, 128.9, 20)}${hsqcRow(4, 1.24, 22.8, -40)}`,
    ].join('\n')
    const parsed = parseMnovaTable(text)
    expect(peakCount(parsed)).toBe(4)
    expect(parsed).toMatchObject({ peaks: [{ f2: 3.71 }, { f2: 7.26 }, { f2: 5.02 }, { f2: 1.24 }] })
  })

  it('drops rows missing any of the three values', () => {
    const text = [HSQC_HEADER, hsqcRow(1, 3.71, 55.2, 10), '2\t7.26\t\t\t\t\t\t\t\t\t'].join('\n')
    expect(peakCount(parseMnovaTable(text))).toBe(1)
  })
})

describe('rejections', () => {
  it('rejects text with no header row', () => {
    expect(() => parseMnovaTable('1\t7.26\t1.0\n2\t3.71\t1.0')).toThrow(MnovaParseError)
  })

  it('rejects an empty clipboard', () => {
    expect(() => parseMnovaTable('')).toThrow(MnovaParseError)
  })

  it('rejects a header with no peaks under it', () => {
    expect(() => parseMnovaTable(H_HEADER)).toThrow(MnovaParseError)
  })

  it('rejects an MS/MS style table rather than guessing', () => {
    expect(() => parseMnovaTable('\tm/z\tIntensity\n1\t345.1234\t1000')).toThrow(MnovaParseError)
  })
})
