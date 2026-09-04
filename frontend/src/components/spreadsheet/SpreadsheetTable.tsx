/**
 * Unified spreadsheet for all spectral input types.
 *
 * Columns are grouped by modality — HSQC (3) | ¹H | ¹³C | MS (2) — with a
 * gutter between groups so the modalities read as separate blocks while
 * staying one grid, which keeps the column indices below simple.
 *
 * The component owns its display state; it notifies the parent of changes
 * via onSpectraChange, emitting fixed-length arrays that use NaN to represent
 * empty cells so positions are preserved.
 */
import { useMemo, useCallback, useRef, useState, useEffect } from 'react'
import { HotTable, type HotTableRef } from '@handsontable/react-wrapper'
import { registerAllModules } from 'handsontable/registry'
import 'handsontable/styles/handsontable.min.css'
import 'handsontable/styles/ht-theme-main.min.css'
import Toast, { type ToastMessage } from '../common/Toast'
import { parseMnovaTable, peakCount, KIND_LABELS, MnovaParseError, type MnovaPaste } from '../../services/mnovaPaste'
import './SpreadsheetTable.css'

registerAllModules()

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ValidationSummary {
  hsqcInvalid: number
  hInvalid: number
  cInvalid: number
  msInvalid: number
  msNegInvalid: number
  anyInvalid: boolean
}

export interface SpectraArrays {
  hsqc: number[]
  h_nmr: number[]
  c_nmr: number[]
  mass_spec: number[]
  mass_spec_neg: number[]
}

interface SpreadsheetTableProps {
  hsqc: number[]
  h_nmr: number[]
  c_nmr: number[]
  mass_spec: number[]
  mass_spec_neg: number[]
  /** Emitted as one batch so a single edit causes one store update, not five. */
  onSpectraChange: (data: SpectraArrays) => void
  onValidationChange?: (summary: ValidationSummary) => void
}

// ── Constants ─────────────────────────────────────────────────────────────────

const MAX_ROWS = 400

const COL_HSQC_H = 0
const COL_HSQC_C = 1
const COL_HSQC_I = 2
const COL_H_NMR = 3
const COL_C_NMR = 4
const COL_MS_MZ = 5
const COL_MS_I = 6
const COL_MSNEG_MZ = 7
const COL_MSNEG_I = 8

const COL_HEADERS = [
  'f2 — ¹H (ppm)',
  'f1 — ¹³C (ppm)',
  'Intensity',
  'δ (ppm)',
  'δ (ppm)',
  'm/z',
  'Intensity',
  'm/z',
  'Intensity',
]

const GROUP_HEADERS = [
  { label: '¹H-¹³C HSQC', colspan: 3 },
  { label: '¹H NMR', colspan: 1 },
  { label: '¹³C NMR', colspan: 1 },
  { label: 'MS/MS (Positive)', colspan: 2 },
  { label: 'MS/MS (Negative)', colspan: 2 },
]

/** First column of each modality after the first — these carry the gutter. */
const GROUP_STARTS = new Set([COL_H_NMR, COL_C_NMR, COL_MS_MZ, COL_MSNEG_MZ])

// Height follows the data: enough rows to see everything entered, capped so a
// long peak list does not push the rest of the page off screen.
const MIN_VISIBLE_ROWS = 10
const MAX_VISIBLE_ROWS = 30
// Measured against ht-theme-main; 2px of slack keeps the last row clear of a
// scrollbar if a theme update shifts these by a pixel.
const ROW_HEIGHT = 29
const HEADER_HEIGHT = 30
const HEIGHT_SLACK = 2

// ── Helpers ───────────────────────────────────────────────────────────────────

type Cell = number | string

function parseNum(val: unknown): number | null {
  if (val === '' || val === null || val === undefined) return null
  const n = typeof val === 'number' ? val : parseFloat(String(val))
  return Number.isFinite(n) ? n : null
}

function getNum(arr: number[], idx: number): number | '' {
  if (!arr || idx < 0 || idx >= arr.length) return ''
  const v = arr[idx]
  return typeof v === 'number' && Number.isFinite(v) ? v : ''
}

/**
 * Cell-wise comparison with early exit. This runs on every store update, and
 * JSON.stringify of a 400x7 grid on each keystroke was measurably worse.
 */
export function gridEquals(a: Cell[][], b: Cell[][]): boolean {
  if (a.length !== b.length) return false
  for (let r = 0; r < a.length; r++) {
    const rowA = a[r], rowB = b[r]
    if (rowA.length !== rowB.length) return false
    for (let c = 0; c < rowA.length; c++) {
      if (rowA[c] !== rowB[c]) return false
    }
  }
  return true
}

/** Index of the last row holding any value, or -1 when the grid is empty. */
export function lastFilledRow(grid: Cell[][]): number {
  for (let r = grid.length - 1; r >= 0; r--) {
    if (grid[r].some((v) => v !== '' && v !== null && v !== undefined)) return r
  }
  return -1
}

/** Rows to show: the data plus one spare, floored and capped. */
export function visibleRowCount(lastRow: number): number {
  return Math.min(MAX_VISIBLE_ROWS, Math.max(MIN_VISIBLE_ROWS, lastRow + 2))
}

// ── Component ─────────────────────────────────────────────────────────────────

function SpreadsheetTable({
  hsqc, h_nmr, c_nmr, mass_spec, mass_spec_neg,
  onSpectraChange,
  onValidationChange,
}: SpreadsheetTableProps) {
  const hotRef = useRef<HotTableRef>(null)
  const isInternalRef = useRef(false)
  const lastSyncRef = useRef<Cell[][] | null>(null)
  const toastSeq = useRef(0)
  const [toast, setToast] = useState<ToastMessage | null>(null)
  const [history, setHistory] = useState({ canUndo: false, canRedo: false })
  const [validation, setValidation] = useState<Omit<ValidationSummary, 'anyInvalid'>>({
    hsqcInvalid: 0, hInvalid: 0, cInvalid: 0, msInvalid: 0, msNegInvalid: 0,
  })

  // Build the 2-D table data from the flat arrays coming from the store.
  const tableData = useMemo<Cell[][]>(() => {
    const rows: Cell[][] = []
    for (let i = 0; i < MAX_ROWS; i++) {
      rows.push([
        getNum(hsqc, i * 3),
        getNum(hsqc, i * 3 + 1),
        getNum(hsqc, i * 3 + 2),
        getNum(h_nmr, i),
        getNum(c_nmr, i),
        getNum(mass_spec, i * 2),
        getNum(mass_spec, i * 2 + 1),
        getNum(mass_spec_neg, i * 2),
        getNum(mass_spec_neg, i * 2 + 1),
      ])
    }
    return rows
  }, [hsqc, h_nmr, c_nmr, mass_spec, mass_spec_neg])

  const columnDefs = useMemo(() =>
    COL_HEADERS.map((_, i) => ({
      data: i,
      type: 'numeric' as const,
      editor: 'numeric' as const,
      // Narrow enough that all 9 columns (incl. negative MS/MS) fit a standard
      // laptop; stretchH widens them when there's room. Overflow scrolls otherwise.
      width: 96,
      allowEmpty: true,
      className: GROUP_STARTS.has(i) ? 'ht-modality-start' : undefined,
    })),
  [])

  const tableHeight =
    HEADER_HEIGHT * 2 + visibleRowCount(lastFilledRow(tableData)) * ROW_HEIGHT + HEIGHT_SLACK

  const showToast = useCallback((text: string, tone: ToastMessage['tone']) => {
    setToast({ id: ++toastSeq.current, text, tone })
  }, [])

  const dismissToast = useCallback(() => setToast(null), [])

  // Parse grid data into flat arrays, validate, and emit to parent.
  const extractAndEmit = useCallback((grid: Cell[][]) => {
    isInternalRef.current = true
    lastSyncRef.current = grid.map((r) => [...r])

    const hsqcOut  = new Array<number>(MAX_ROWS * 3).fill(NaN)
    const hOut     = new Array<number>(MAX_ROWS).fill(NaN)
    const cOut     = new Array<number>(MAX_ROWS).fill(NaN)
    const msOut    = new Array<number>(MAX_ROWS * 2).fill(NaN)
    const msNegOut = new Array<number>(MAX_ROWS * 2).fill(NaN)

    let hsqcInvalid = 0, hInvalid = 0, cInvalid = 0, msInvalid = 0, msNegInvalid = 0

    grid.forEach((row, i) => {
      const h  = parseNum(row[COL_HSQC_H]), hc = parseNum(row[COL_HSQC_C]), hi = parseNum(row[COL_HSQC_I])
      const hn = parseNum(row[COL_H_NMR])
      const cn = parseNum(row[COL_C_NMR])
      const mz = parseNum(row[COL_MS_MZ]), mi = parseNum(row[COL_MS_I])
      const nmz = parseNum(row[COL_MSNEG_MZ]), nmi = parseNum(row[COL_MSNEG_I])

      hsqcOut[i * 3]      = h  ?? NaN
      hsqcOut[i * 3 + 1]  = hc ?? NaN
      hsqcOut[i * 3 + 2]  = hi ?? NaN
      hOut[i]             = hn ?? NaN
      cOut[i]             = cn ?? NaN
      msOut[i * 2]        = mz ?? NaN
      msOut[i * 2 + 1]    = mi ?? NaN
      msNegOut[i * 2]     = nmz ?? NaN
      msNegOut[i * 2 + 1] = nmi ?? NaN

      // Validate: partial HSQC rows (some but not all three filled) are invalid.
      const hsqcFilled = [h, hc, hi].filter((v) => v !== null).length
      if (hsqcFilled > 0 && hsqcFilled < 3) hsqcInvalid++

      // Validate: partial MS rows (m/z without intensity or vice-versa) are invalid.
      if ([mz, mi].filter((v) => v !== null).length === 1) msInvalid++
      if ([nmz, nmi].filter((v) => v !== null).length === 1) msNegInvalid++

      if (hn !== null && !Number.isFinite(hn)) hInvalid++
      if (cn !== null && !Number.isFinite(cn)) cInvalid++
    })

    const summary: ValidationSummary = {
      hsqcInvalid, hInvalid, cInvalid, msInvalid, msNegInvalid,
      anyInvalid: (hsqcInvalid + hInvalid + cInvalid + msInvalid + msNegInvalid) > 0,
    }
    setValidation({ hsqcInvalid, hInvalid, cInvalid, msInvalid, msNegInvalid })
    onValidationChange?.(summary)

    onSpectraChange({ hsqc: hsqcOut, h_nmr: hOut, c_nmr: cOut, mass_spec: msOut, mass_spec_neg: msNegOut })

    setTimeout(() => { isInternalRef.current = false }, 0)
  }, [onSpectraChange, onValidationChange])

  /** Mirrors Handsontable's undo stack into render state for the buttons. */
  const refreshHistory = useCallback(() => {
    const hot = hotRef.current?.hotInstance
    if (!hot) return
    setHistory({ canUndo: hot.isUndoAvailable(), canRedo: hot.isRedoAvailable() })
  }, [])

  const handleAfterChange = useCallback((changes: unknown[] | null, source: string) => {
    if (!changes || source === 'loadData' || !hotRef.current?.hotInstance) return
    const grid = hotRef.current.hotInstance.getData() as Cell[][]
    extractAndEmit(grid)
    refreshHistory()
  }, [extractAndEmit, refreshHistory])

  // Sync Handsontable when store data changes externally.
  useEffect(() => {
    if (isInternalRef.current || !hotRef.current?.hotInstance) return
    const changed =
      lastSyncRef.current === null || !gridEquals(tableData, lastSyncRef.current)
    if (changed) {
      // Replacing the data source drops the undo stack, so the buttons have to
      // be re-read rather than assumed unchanged.
      hotRef.current.hotInstance.loadData(tableData)
      lastSyncRef.current = tableData.map((r) => [...r])
      refreshHistory()
    }
  }, [tableData, refreshHistory])

  /**
   * Applies a rewritten grid as one batch of cell edits.
   *
   * setDataAtCell rather than loadData: loadData replaces the data source and
   * is invisible to the undo stack, so a paste or condense could not be undone.
   * One call becomes one undo step, and diffing first keeps that step to the
   * cells that actually moved. afterChange carries the result to the store.
   */
  const commitGrid = useCallback((next: Cell[][]) => {
    const hot = hotRef.current?.hotInstance
    if (!hot) return
    const current = hot.getData() as Cell[][]
    const changes: [number, number, Cell][] = []
    for (let r = 0; r < next.length; r++) {
      for (let c = 0; c < next[r].length; c++) {
        if ((current[r]?.[c] ?? '') !== next[r][c]) changes.push([r, c, next[r][c]])
      }
    }
    if (changes.length) hot.setDataAtCell(changes)
  }, [])

  const handleUndo = useCallback(() => {
    hotRef.current?.hotInstance?.undo()
    refreshHistory()
  }, [refreshHistory])

  const handleRedo = useCallback(() => {
    hotRef.current?.hotInstance?.redo()
    refreshHistory()
  }, [refreshHistory])

  const readGrid = useCallback((): Cell[][] | null => {
    const hot = hotRef.current?.hotInstance
    if (!hot) return null
    return (hot.getData() as Cell[][]).map((r) => r.map((v) => v ?? '') as Cell[])
  }, [])

  const handleCondense = useCallback(() => {
    const data = readGrid()
    if (!data) return

    const condense = (cols: number[]) => {
      let write = 0
      for (let read = 0; read < data.length; read++) {
        const allEmpty = cols.every((c) => parseNum(data[read][c]) === null)
        if (!allEmpty) {
          if (write !== read) {
            cols.forEach((c) => { data[write][c] = data[read][c]; data[read][c] = '' })
          }
          write++
        }
      }
    }

    condense([COL_HSQC_H, COL_HSQC_C, COL_HSQC_I])
    condense([COL_H_NMR])
    condense([COL_C_NMR])
    condense([COL_MS_MZ, COL_MS_I])
    condense([COL_MSNEG_MZ, COL_MSNEG_I])

    commitGrid(data)
  }, [readGrid, commitGrid])

  /**
   * Replaces the columns of one modality with a parsed Mnova table.
   * Returns how many peaks were written (the table may overflow MAX_ROWS).
   */
  const applyMnovaPaste = useCallback((parsed: MnovaPaste): number => {
    const data = readGrid()
    if (!data) return 0

    const cols = parsed.kind === 'hsqc'
      ? [COL_HSQC_H, COL_HSQC_C, COL_HSQC_I]
      : [parsed.kind === 'h_nmr' ? COL_H_NMR : COL_C_NMR]
    data.forEach((row) => cols.forEach((c) => { row[c] = '' }))

    const written = Math.min(peakCount(parsed), MAX_ROWS)
    for (let i = 0; i < written; i++) {
      if (parsed.kind === 'hsqc') {
        const p = parsed.peaks[i]
        data[i][COL_HSQC_H] = p.f2
        data[i][COL_HSQC_C] = p.f1
        data[i][COL_HSQC_I] = p.intensity
      } else {
        data[i][cols[0]] = parsed.shifts[i]
      }
    }

    commitGrid(data)
    return written
  }, [readGrid, commitGrid])

  const handleMnovaPaste = useCallback(async () => {
    let text: string
    try {
      text = await navigator.clipboard.readText()
    } catch {
      showToast(
        'Could not read the clipboard. Allow clipboard access for this site, then try again.',
        'error',
      )
      return
    }

    let parsed: MnovaPaste
    try {
      parsed = parseMnovaTable(text)
    } catch (err) {
      showToast(err instanceof MnovaParseError ? err.message : String(err), 'error')
      return
    }

    const total = peakCount(parsed)
    const written = applyMnovaPaste(parsed)
    const overflow = total > written ? ` (first ${written} of ${total} — the sheet holds ${MAX_ROWS} rows)` : ''
    showToast(`Pasted ${KIND_LABELS[parsed.kind]} — ${written} peaks${overflow}`, 'info')
  }, [applyMnovaPaste, showToast])

  const anyInvalid = (validation.hsqcInvalid + validation.hInvalid + validation.cInvalid + validation.msInvalid + validation.msNegInvalid) > 0

  return (
    <div className="spreadsheet-table">
      <Toast message={toast} onDismiss={dismissToast} />
      <div className="spreadsheet-table__toolbar">
        {anyInvalid ? (
          <span className="spreadsheet-table__validation-error">
            Incomplete rows — HSQC: {validation.hsqcInvalid} · H: {validation.hInvalid} · C: {validation.cInvalid} · MS+: {validation.msInvalid} · MS−: {validation.msNegInvalid}
          </span>
        ) : (
          <span className="spreadsheet-table__validation-ok">No validation errors</span>
        )}
        <div className="spreadsheet-table__actions">
          <button
            className="spreadsheet-table__btn"
            onClick={handleUndo}
            disabled={!history.canUndo}
            title="Undo (Ctrl+Z)"
          >
            ↶ Undo
          </button>
          <button
            className="spreadsheet-table__btn"
            onClick={handleRedo}
            disabled={!history.canRedo}
            title="Redo (Ctrl+Y)"
          >
            ↷ Redo
          </button>
          <span className="spreadsheet-table__divider" />
          <button className="spreadsheet-table__btn" onClick={handleMnovaPaste}>
            Paste NMR Table from MestreNova
          </button>
          <button className="spreadsheet-table__btn" onClick={handleCondense}>
            Condense rows
          </button>
        </div>
      </div>
      <HotTable
        ref={hotRef}
        data={tableData}
        columns={columnDefs}
        colHeaders={COL_HEADERS}
        nestedHeaders={[GROUP_HEADERS, COL_HEADERS]}
        rowHeaders
        height={tableHeight}
        width="100%"
        stretchH="all"
        afterChange={handleAfterChange}
        afterGetColHeader={(col: number, TH: HTMLTableCellElement) => {
          if (GROUP_STARTS.has(col)) TH.classList.add('ht-modality-start')
        }}
        licenseKey="non-commercial-and-evaluation"
        themeName="ht-theme-main"
        copyPaste={{ pasteMode: 'overwrite' } as never}
        fillHandle={false}
      />
    </div>
  )
}

export default SpreadsheetTable
