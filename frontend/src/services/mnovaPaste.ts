/**
 * Parser for peak tables copied out of MestreNova ("Copy Table").
 *
 * Mnova puts the table on the clipboard as TSV with a header row. Two layouts
 * matter here:
 *
 *   1-D (¹H / ¹³C)   (blank) │ ppm │ Intensity │ Width │ Area │ Type │ …
 *   HSQC             (blank) │ f2 (ppm) │ f1 (ppm) │ Intensity │ Width f2 │ …
 *
 * The leading blank header sits above the peak number. When a table is too
 * long, Mnova continues it in a second (third, …) group of columns to the
 * right, repeating the same headers — so the header row is scanned for *every*
 * occurrence of the anchor column and each occurrence becomes a block, read
 * left block first so peak order survives.
 *
 * f2 is the proton axis and f1 the carbon axis; swapping them silently
 * produces a plausible, wrong spectrum, so the mapping is anchored on the
 * header text rather than on column position.
 *
 * MS/MS tables are deliberately not handled.
 */

export type NmrKind = 'h_nmr' | 'c_nmr' | 'hsqc'

export type MnovaPaste =
  | { kind: 'h_nmr' | 'c_nmr'; shifts: number[] }
  | { kind: 'hsqc'; peaks: HSQCPeak[] }

export interface HSQCPeak {
  /** f2 — proton shift (ppm). */
  f2: number
  /** f1 — carbon shift (ppm). */
  f1: number
  intensity: number
}

/**
 * Shift window used to tell a ¹H table from a ¹³C one, since the two have an
 * identical column layout. A ¹³C table whose every peak fell inside this
 * window would have to be an all-aliphatic compound with no carbonyl, no
 * aromatic and no oxygenated carbon — not something Mnova would produce from a
 * real spectrum.
 */
export const H_SHIFT_MIN = -2
export const H_SHIFT_MAX = 16

export const KIND_LABELS: Record<NmrKind, string> = {
  h_nmr: '¹H NMR',
  c_nmr: '¹³C NMR',
  hsqc: '¹H-¹³C HSQC',
}

// ── Cell helpers ──────────────────────────────────────────────────────────────

function splitRows(text: string): string[][] {
  return text
    .split(/\r\n|\r|\n/)
    .filter((line) => line.trim() !== '')
    .map((line) => line.split('\t').map((c) => c.trim().replace(/^"(.*)"$/, '$1')))
}

function normHeader(cell: string): string {
  return cell.toLowerCase().replace(/\s+/g, ' ').trim()
}

function num(cell: string | undefined): number | null {
  if (cell === undefined || cell === '') return null
  const n = Number(cell)
  return Number.isFinite(n) ? n : null
}

// ── Header location ───────────────────────────────────────────────────────────

interface Layout {
  hsqc: boolean
  /** Index of the anchor (first ppm) column of each repeated block. */
  blocks: number[]
}

function isHsqcAnchor(h: string): boolean {
  return h.startsWith('f2') && h.includes('ppm')
}

function isShiftAnchor(h: string): boolean {
  return h === 'ppm' || h === 'δ (ppm)' || h === 'shift (ppm)'
}

/** Returns the layout for the first row that looks like a Mnova header. */
function findLayout(rows: string[][]): { rowIndex: number; layout: Layout } | null {
  for (let r = 0; r < rows.length; r++) {
    const cells = rows[r].map(normHeader)
    const hsqcBlocks = cells.map((c, i) => (isHsqcAnchor(c) ? i : -1)).filter((i) => i >= 0)
    if (hsqcBlocks.length > 0) return { rowIndex: r, layout: { hsqc: true, blocks: hsqcBlocks } }
    const shiftBlocks = cells.map((c, i) => (isShiftAnchor(c) ? i : -1)).filter((i) => i >= 0)
    if (shiftBlocks.length > 0) return { rowIndex: r, layout: { hsqc: false, blocks: shiftBlocks } }
  }
  return null
}

// ── Public API ────────────────────────────────────────────────────────────────

export class MnovaParseError extends Error {}

/**
 * Parses clipboard text into one NMR modality, inferring which one.
 * Throws MnovaParseError when the text is not a recognisable Mnova table.
 */
export function parseMnovaTable(text: string): MnovaPaste {
  const rows = splitRows(text)
  const found = findLayout(rows)
  if (!found) {
    throw new MnovaParseError(
      'No MestreNova peak table found on the clipboard. Copy the peak table ' +
        'with its header row (1D needs a "ppm" column, HSQC an "f2 (ppm)" column).',
    )
  }

  const { rowIndex, layout } = found
  const body = rows.slice(rowIndex + 1)

  if (layout.hsqc) {
    const peaks: HSQCPeak[] = []
    for (const start of layout.blocks) {
      for (const row of body) {
        const f2 = num(row[start])
        const f1 = num(row[start + 1])
        const intensity = num(row[start + 2])
        if (f2 === null || f1 === null || intensity === null) continue
        peaks.push({ f2, f1, intensity })
      }
    }
    if (peaks.length === 0) throw new MnovaParseError('The HSQC table has no complete peak rows.')
    return { kind: 'hsqc', peaks }
  }

  const shifts: number[] = []
  for (const start of layout.blocks) {
    for (const row of body) {
      const ppm = num(row[start])
      if (ppm === null) continue
      shifts.push(ppm)
    }
  }
  if (shifts.length === 0) throw new MnovaParseError('The peak table has no numeric ppm values.')

  const isProton = shifts.every((s) => s >= H_SHIFT_MIN && s <= H_SHIFT_MAX)
  return { kind: isProton ? 'h_nmr' : 'c_nmr', shifts }
}

/** Peak count of a parse result, for the confirmation notice. */
export function peakCount(parsed: MnovaPaste): number {
  return parsed.kind === 'hsqc' ? parsed.peaks.length : parsed.shifts.length
}
