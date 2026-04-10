/**
 * Unified spreadsheet for all spectral input types.
 *
 * Columns: HSQC H | HSQC C | HSQC Int | H NMR | C NMR | MS m/z | MS Int
 *
 * The component owns its display state; it notifies the parent of changes
 * via the four onXxxChange callbacks, emitting fixed-length arrays that use
 * NaN to represent empty cells so positions are preserved.
 */
import { useMemo, useCallback, useRef, useState, useEffect } from 'react'
import { HotTable, type HotTableRef } from '@handsontable/react-wrapper'
import { registerAllModules } from 'handsontable/registry'
import 'handsontable/styles/handsontable.min.css'
import 'handsontable/styles/ht-theme-main.min.css'
import './SpreadsheetTable.css'

registerAllModules()

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ValidationSummary {
  hsqcInvalid: number
  hInvalid: number
  cInvalid: number
  msInvalid: number
  anyInvalid: boolean
}

interface SpreadsheetTableProps {
  hsqc: number[]
  h_nmr: number[]
  c_nmr: number[]
  mass_spec: number[]
  onHSQCChange: (data: number[]) => void
  onHNMRChange: (data: number[]) => void
  onCNMRChange: (data: number[]) => void
  onMassSpecChange: (data: number[]) => void
  onValidationChange?: (summary: ValidationSummary) => void
}

// ── Constants ─────────────────────────────────────────────────────────────────

const MAX_ROWS = 400
const COL_HEADERS = [
  'HSQC H (ppm)',
  'HSQC C (ppm)',
  'HSQC Intensity',
  'H NMR (ppm)',
  'C NMR (ppm)',
  'MS m/z',
  'MS Intensity',
]

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

// ── Component ─────────────────────────────────────────────────────────────────

function SpreadsheetTable({
  hsqc, h_nmr, c_nmr, mass_spec,
  onHSQCChange, onHNMRChange, onCNMRChange, onMassSpecChange,
  onValidationChange,
}: SpreadsheetTableProps) {
  const hotRef = useRef<HotTableRef>(null)
  const isInternalRef = useRef(false)
  const lastSyncRef = useRef<Cell[][] | null>(null)
  const [validation, setValidation] = useState<Omit<ValidationSummary, 'anyInvalid'>>({
    hsqcInvalid: 0, hInvalid: 0, cInvalid: 0, msInvalid: 0,
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
      ])
    }
    return rows
  }, [hsqc, h_nmr, c_nmr, mass_spec])

  const columnDefs = useMemo(() =>
    COL_HEADERS.map((_, i) => ({
      data: i,
      type: 'numeric' as const,
      editor: 'numeric' as const,
      width: 130,
      allowEmpty: true,
    })),
  [])

  const colWidth = useMemo(
    () => columnDefs.reduce((s, c) => s + (c.width || 130), 0) + 50,
    [columnDefs],
  )
  const tableHeight = 30 + 10 * 24 // header + 10 visible rows

  // Parse grid data into flat arrays, validate, and emit to parent.
  const extractAndEmit = useCallback((grid: Cell[][]) => {
    isInternalRef.current = true
    lastSyncRef.current = grid.map((r) => [...r])

    const hsqcOut = new Array<number>(MAX_ROWS * 3).fill(NaN)
    const hOut    = new Array<number>(MAX_ROWS).fill(NaN)
    const cOut    = new Array<number>(MAX_ROWS).fill(NaN)
    const msOut   = new Array<number>(MAX_ROWS * 2).fill(NaN)

    let hsqcInvalid = 0, hInvalid = 0, cInvalid = 0, msInvalid = 0

    grid.forEach((row, i) => {
      const h  = parseNum(row[0]), hc = parseNum(row[1]), hi = parseNum(row[2])
      const hn = parseNum(row[3])
      const cn = parseNum(row[4])
      const mz = parseNum(row[5]), mi = parseNum(row[6])

      hsqcOut[i * 3]     = h  ?? NaN
      hsqcOut[i * 3 + 1] = hc ?? NaN
      hsqcOut[i * 3 + 2] = hi ?? NaN
      hOut[i]            = hn ?? NaN
      cOut[i]            = cn ?? NaN
      msOut[i * 2]       = mz ?? NaN
      msOut[i * 2 + 1]   = mi ?? NaN

      // Validate: partial HSQC rows (some but not all three filled) are invalid.
      const hsqcFilled = [h, hc, hi].filter((v) => v !== null).length
      if (hsqcFilled > 0 && hsqcFilled < 3) hsqcInvalid++

      // Validate: partial MS rows (m/z without intensity or vice-versa) are invalid.
      const msFilled = [mz, mi].filter((v) => v !== null).length
      if (msFilled === 1) msInvalid++

      if (hn !== null && !Number.isFinite(hn)) hInvalid++
      if (cn !== null && !Number.isFinite(cn)) cInvalid++
    })

    const summary: ValidationSummary = {
      hsqcInvalid, hInvalid, cInvalid, msInvalid,
      anyInvalid: (hsqcInvalid + hInvalid + cInvalid + msInvalid) > 0,
    }
    setValidation({ hsqcInvalid, hInvalid, cInvalid, msInvalid })
    onValidationChange?.(summary)

    onHSQCChange(hsqcOut)
    onHNMRChange(hOut)
    onCNMRChange(cOut)
    onMassSpecChange(msOut)

    setTimeout(() => { isInternalRef.current = false }, 0)
  }, [onHSQCChange, onHNMRChange, onCNMRChange, onMassSpecChange, onValidationChange])

  const handleAfterChange = useCallback((changes: unknown[] | null, source: string) => {
    if (!changes || source === 'loadData' || !hotRef.current?.hotInstance) return
    const grid = hotRef.current.hotInstance.getData() as Cell[][]
    extractAndEmit(grid)
  }, [extractAndEmit])

  // Sync Handsontable when store data changes externally.
  useEffect(() => {
    if (isInternalRef.current || !hotRef.current?.hotInstance) return
    const changed =
      lastSyncRef.current === null ||
      JSON.stringify(tableData) !== JSON.stringify(lastSyncRef.current)
    if (changed) {
      hotRef.current.hotInstance.loadData(tableData)
      lastSyncRef.current = tableData.map((r) => [...r])
    }
  }, [tableData])

  const handleCondense = useCallback(() => {
    if (!hotRef.current?.hotInstance) return
    const hot = hotRef.current.hotInstance
    const data = (hot.getData() as Cell[][]).map((r) => r.map((v) => v ?? '') as Cell[])

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

    condense([0, 1, 2])
    condense([3])
    condense([4])
    condense([5, 6])

    isInternalRef.current = true
    extractAndEmit(data)
    hot.loadData(data)
    lastSyncRef.current = data.map((r) => [...r])
    setTimeout(() => { isInternalRef.current = false }, 0)
  }, [extractAndEmit])

  const anyInvalid = (validation.hsqcInvalid + validation.hInvalid + validation.cInvalid + validation.msInvalid) > 0

  return (
    <div className="spreadsheet-table">
      <div className="spreadsheet-table__toolbar">
        {anyInvalid ? (
          <span className="spreadsheet-table__validation-error">
            Incomplete rows — HSQC: {validation.hsqcInvalid} · H: {validation.hInvalid} · C: {validation.cInvalid} · MS: {validation.msInvalid}
          </span>
        ) : (
          <span className="spreadsheet-table__validation-ok">No validation errors</span>
        )}
        <button className="spreadsheet-table__condense-btn" onClick={handleCondense}>
          Condense rows
        </button>
      </div>
      <HotTable
        ref={hotRef}
        data={tableData}
        columns={columnDefs}
        colHeaders={COL_HEADERS}
        rowHeaders
        height={tableHeight}
        width={colWidth}
        afterChange={handleAfterChange}
        licenseKey="non-commercial-and-evaluation"
        themeName="ht-theme-main"
        copyPaste={{ pasteMode: 'overwrite' } as never}
        fillHandle={false}
      />
    </div>
  )
}

export default SpreadsheetTable
