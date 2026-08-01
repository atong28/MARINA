/**
 * Live preview of whatever spectra are currently in the spreadsheet.
 *
 * Re-renders on every spreadsheet edit or paste, since it reads the same
 * store arrays the spreadsheet writes. Panels appear only for the data types
 * that are actually present.
 *
 * Axis conventions follow standard spectroscopic practice:
 *   • NMR chemical-shift axes always run high → low ppm (left→right, and
 *     top→bottom for the HSQC carbon axis), so HSQC has its origin bottom-right.
 *   • MS runs m/z low → high left→right, intensity normalized to base peak.
 */
import { useMemo, useState } from 'react'
import HelpButton from '../common/HelpButton'
import { HELP } from '../../helpContent'
import './SpectraPreview.css'

// ── Palette (validated: see dataviz palette, light surface) ───────────────────

const SERIES_POS = '#2a78d6' // slot 1 — CH / CH₃ (positive HSQC phase)
const SERIES_NEG = '#eb6834' // slot 2 — CH₂ (negative HSQC phase)
const GRID = '#e1e0d9'
const AXIS = '#c3c2b7'
const SURFACE = '#ffffff'

// ── Scales & ticks ────────────────────────────────────────────────────────────

export type Scale = (v: number) => number

/** Maps [d0, d1] onto [r0, r1]. Pass d0 > d1 for a reversed (ppm) axis. */
export function linear(d0: number, d1: number, r0: number, r1: number): Scale {
  const span = d1 - d0 || 1
  return (v) => r0 + ((v - d0) / span) * (r1 - r0)
}

export function padDomain(lo: number, hi: number, frac = 0.05): [number, number] {
  if (lo === hi) {
    const d = Math.abs(lo) * 0.05 || 0.5
    return [lo - d, hi + d]
  }
  const p = (hi - lo) * frac
  return [lo - p, hi + p]
}

export interface Ticks {
  values: number[]
  fmt: (v: number) => string
}

export function niceTicks(lo: number, hi: number, target = 5): Ticks {
  const raw = (hi - lo) / target
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const norm = raw / mag
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag
  const values: number[] = []
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) {
    values.push(Math.abs(t) < step * 1e-9 ? 0 : t)
  }
  const decimals = Math.max(0, -Math.floor(Math.log10(step)))
  return { values, fmt: (v) => v.toFixed(decimals) }
}

// ── Formatting ────────────────────────────────────────────────────────────────

export function fmtShift(v: number): string {
  return v.toFixed(2)
}

export function fmtIntensity(v: number): string {
  if (v === 0) return '0'
  return Math.abs(v) >= 1e4 || Math.abs(v) < 1e-2 ? v.toExponential(2) : v.toFixed(2)
}

// ── Tooltip ───────────────────────────────────────────────────────────────────

interface Hover {
  /** Anchor position in CSS pixels, relative to the plot wrapper. */
  cx: number
  cy: number
  lines: string[]
}

function Tooltip({ hover }: { hover: Hover | null }) {
  if (!hover) return null
  return (
    <div className="spectra__tooltip" style={{ left: hover.cx, top: hover.cy }}>
      {hover.lines.map((l) => (
        <div key={l}>{l}</div>
      ))}
    </div>
  )
}

/** Converts a mouse event into viewBox coordinates plus the CSS-px scale factor. */
function toViewBox(e: React.MouseEvent<SVGRectElement>, vbWidth: number) {
  const rect = e.currentTarget.ownerSVGElement!.getBoundingClientRect()
  const scale = rect.width / vbWidth
  const cx = e.clientX - rect.left
  const cy = e.clientY - rect.top
  return { vx: cx / scale, vy: cy / scale, cx, cy }
}

// ── Axes ──────────────────────────────────────────────────────────────────────

interface Rect {
  x0: number
  x1: number
  y0: number
  y1: number
}

interface AxesProps {
  rect: Rect
  height: number
  x: Scale
  xTicks: Ticks
  xLabel: string
  y?: Scale
  yTicks?: Ticks
  yLabel?: string
}

function Axes({ rect, height, x, xTicks, xLabel, y, yTicks, yLabel }: AxesProps) {
  const { x0, x1, y0, y1 } = rect
  return (
    <g>
      {xTicks.values.map((t) => (
        <g key={`x${t}`}>
          <line x1={x(t)} x2={x(t)} y1={y0} y2={y1} stroke={GRID} strokeWidth={1} />
          <text className="spectra__tick" x={x(t)} y={y1 + 13} textAnchor="middle">
            {xTicks.fmt(t)}
          </text>
        </g>
      ))}
      {y && yTicks?.values.map((t) => (
        <g key={`y${t}`}>
          <line x1={x0} x2={x1} y1={y(t)} y2={y(t)} stroke={GRID} strokeWidth={1} />
          <text className="spectra__tick" x={x0 - 6} y={y(t) + 3.5} textAnchor="end">
            {yTicks.fmt(t)}
          </text>
        </g>
      ))}
      <line x1={x0} x2={x1} y1={y1} y2={y1} stroke={AXIS} strokeWidth={1} />
      {y && <line x1={x0} x2={x0} y1={y0} y2={y1} stroke={AXIS} strokeWidth={1} />}
      <text className="spectra__axis-label" x={(x0 + x1) / 2} y={height - 3} textAnchor="middle">
        {xLabel}
      </text>
      {yLabel && (
        <text
          className="spectra__axis-label"
          transform={`translate(10 ${(y0 + y1) / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          {yLabel}
        </text>
      )}
    </g>
  )
}

// ── HSQC: 2-D scatter, both ppm axes reversed ────────────────────────────────

const HSQC_W = 400
const HSQC_H = 350

interface HSQCPoint {
  h: number
  c: number
  i: number
}

function HSQCPlot({ points }: { points: HSQCPoint[] }) {
  const [hover, setHover] = useState<Hover | null>(null)

  const view = useMemo(() => {
    const rect: Rect = { x0: 52, x1: HSQC_W - 14, y0: 12, y1: HSQC_H - 32 }
    const [hLo, hHi] = padDomain(
      Math.min(...points.map((p) => p.h)),
      Math.max(...points.map((p) => p.h)),
    )
    const [cLo, cHi] = padDomain(
      Math.min(...points.map((p) => p.c)),
      Math.max(...points.map((p) => p.c)),
    )
    // Reversed: high ppm at left (x) and at top (y).
    const x = linear(hHi, hLo, rect.x0, rect.x1)
    const y = linear(cHi, cLo, rect.y0, rect.y1)
    return {
      rect,
      x,
      y,
      xTicks: niceTicks(hLo, hHi, 5),
      yTicks: niceTicks(cLo, cHi, 6),
      placed: points.map((p) => ({ ...p, px: x(p.h), py: y(p.c) })),
    }
  }, [points])

  const hasNegative = points.some((p) => p.i < 0)
  const hasPositive = points.some((p) => p.i >= 0)
  const showLegend = hasNegative && hasPositive

  const handleMove = (e: React.MouseEvent<SVGRectElement>) => {
    const { vx, vy, cx, cy } = toViewBox(e, HSQC_W)
    let best: (typeof view.placed)[number] | null = null
    let bestD = 24 * 24
    for (const p of view.placed) {
      const d = (p.px - vx) ** 2 + (p.py - vy) ** 2
      if (d < bestD) {
        bestD = d
        best = p
      }
    }
    setHover(
      best
        ? {
            cx,
            cy,
            lines: [
              `δH ${fmtShift(best.h)} ppm · δC ${fmtShift(best.c)} ppm`,
              `intensity ${fmtIntensity(best.i)}${best.i < 0 ? ' (CH₂)' : ''}`,
            ],
          }
        : null,
    )
  }

  return (
    <figure className="spectra__card spectra__card--hsqc">
      <figcaption className="spectra__title">
        HSQC <span className="spectra__count">{points.length} cross-peaks</span>
      </figcaption>
      {showLegend && (
        <div className="spectra__legend">
          <span className="spectra__legend-item">
            <span className="spectra__swatch" style={{ background: SERIES_POS }} />
            CH / CH₃ (positive)
          </span>
          <span className="spectra__legend-item">
            <span className="spectra__swatch" style={{ background: SERIES_NEG }} />
            CH₂ (negative)
          </span>
        </div>
      )}
      <div className="spectra__plot">
        <svg viewBox={`0 0 ${HSQC_W} ${HSQC_H}`} role="img" aria-label="HSQC cross-peak map">
          <Axes
            rect={view.rect}
            height={HSQC_H}
            x={view.x}
            xTicks={view.xTicks}
            xLabel="¹H (ppm)"
            y={view.y}
            yTicks={view.yTicks}
            yLabel="¹³C (ppm)"
          />
          {view.placed.map((p, idx) => (
            <circle
              key={idx}
              cx={p.px}
              cy={p.py}
              r={4}
              fill={p.i < 0 ? SERIES_NEG : SERIES_POS}
              stroke={SURFACE}
              strokeWidth={2}
            />
          ))}
          <rect
            x={view.rect.x0}
            y={view.rect.y0}
            width={view.rect.x1 - view.rect.x0}
            height={view.rect.y1 - view.rect.y0}
            fill="transparent"
            onMouseMove={handleMove}
            onMouseLeave={() => setHover(null)}
          />
        </svg>
        <Tooltip hover={hover} />
      </div>
    </figure>
  )
}

// ── 1-D peak list (¹H / ¹³C): sticks on a reversed ppm axis ──────────────────

const PEAK_W = 700
const PEAK_H = 130

function PeakListPlot({ title, shifts }: { title: string; shifts: number[] }) {
  const [hover, setHover] = useState<Hover | null>(null)

  const view = useMemo(() => {
    const rect: Rect = { x0: 14, x1: PEAK_W - 14, y0: 16, y1: PEAK_H - 32 }
    const [lo, hi] = padDomain(Math.min(...shifts), Math.max(...shifts))
    const x = linear(hi, lo, rect.x0, rect.x1) // reversed: high ppm at left
    return { rect, x, xTicks: niceTicks(lo, hi, 7), placed: shifts.map((s) => ({ s, px: x(s) })) }
  }, [shifts])

  const handleMove = (e: React.MouseEvent<SVGRectElement>) => {
    const { vx, cx, cy } = toViewBox(e, PEAK_W)
    let best: number | null = null
    let bestD = 12
    for (const p of view.placed) {
      const d = Math.abs(p.px - vx)
      if (d < bestD) {
        bestD = d
        best = p.s
      }
    }
    setHover(best === null ? null : { cx, cy, lines: [`δ ${fmtShift(best)} ppm`] })
  }

  return (
    <figure className="spectra__card">
      <figcaption className="spectra__title">
        {title} <span className="spectra__count">{shifts.length} peaks</span>
      </figcaption>
      <div className="spectra__plot">
        <svg viewBox={`0 0 ${PEAK_W} ${PEAK_H}`} role="img" aria-label={`${title} peak list`}>
          <Axes
            rect={view.rect}
            height={PEAK_H}
            x={view.x}
            xTicks={view.xTicks}
            xLabel="δ (ppm)"
          />
          {view.placed.map((p, idx) => (
            <line
              key={idx}
              x1={p.px}
              x2={p.px}
              y1={view.rect.y1}
              y2={view.rect.y0}
              stroke={SERIES_POS}
              strokeWidth={1.5}
            />
          ))}
          <rect
            x={view.rect.x0}
            y={view.rect.y0}
            width={view.rect.x1 - view.rect.x0}
            height={view.rect.y1 - view.rect.y0}
            fill="transparent"
            onMouseMove={handleMove}
            onMouseLeave={() => setHover(null)}
          />
        </svg>
        <Tooltip hover={hover} />
      </div>
    </figure>
  )
}

// ── MS/MS: sticks on an ascending m/z axis, normalized to base peak ──────────

const MS_W = 700
const MS_H = 230

interface MSPeak {
  mz: number
  intensity: number
}

function MassSpecPlot({ peaks }: { peaks: MSPeak[] }) {
  const [hover, setHover] = useState<Hover | null>(null)

  const view = useMemo(() => {
    const rect: Rect = { x0: 46, x1: MS_W - 14, y0: 14, y1: MS_H - 32 }
    const [lo, hi] = padDomain(
      Math.min(...peaks.map((p) => p.mz)),
      Math.max(...peaks.map((p) => p.mz)),
    )
    const base = Math.max(...peaks.map((p) => Math.abs(p.intensity))) || 1
    const x = linear(lo, hi, rect.x0, rect.x1) // ascending m/z, left → right
    const y = linear(0, 100, rect.y1, rect.y0)
    return {
      rect,
      x,
      y,
      base,
      xTicks: niceTicks(lo, hi, 6),
      yTicks: { values: [0, 25, 50, 75, 100], fmt: (v: number) => v.toFixed(0) },
      placed: peaks.map((p) => ({ ...p, px: x(p.mz), rel: (Math.abs(p.intensity) / base) * 100 })),
    }
  }, [peaks])

  const handleMove = (e: React.MouseEvent<SVGRectElement>) => {
    const { vx, cx, cy } = toViewBox(e, MS_W)
    let best: (typeof view.placed)[number] | null = null
    let bestD = 12
    for (const p of view.placed) {
      const d = Math.abs(p.px - vx)
      if (d < bestD) {
        bestD = d
        best = p
      }
    }
    setHover(
      best
        ? {
            cx,
            cy,
            lines: [
              `m/z ${best.mz.toFixed(4)}`,
              `${best.rel.toFixed(1)}% base · ${fmtIntensity(best.intensity)}`,
            ],
          }
        : null,
    )
  }

  return (
    <figure className="spectra__card">
      <figcaption className="spectra__title">
        MS/MS <span className="spectra__count">{peaks.length} peaks</span>
      </figcaption>
      <div className="spectra__plot">
        <svg viewBox={`0 0 ${MS_W} ${MS_H}`} role="img" aria-label="Mass spectrum">
          <Axes
            rect={view.rect}
            height={MS_H}
            x={view.x}
            xTicks={view.xTicks}
            xLabel="m/z"
            y={view.y}
            yTicks={view.yTicks}
            yLabel="Rel. intensity (%)"
          />
          {view.placed.map((p, idx) => (
            <line
              key={idx}
              x1={p.px}
              x2={p.px}
              y1={view.rect.y1}
              y2={view.y(p.rel)}
              stroke={SERIES_POS}
              strokeWidth={1.5}
            />
          ))}
          <rect
            x={view.rect.x0}
            y={view.rect.y0}
            width={view.rect.x1 - view.rect.x0}
            height={view.rect.y1 - view.rect.y0}
            fill="transparent"
            onMouseMove={handleMove}
            onMouseLeave={() => setHover(null)}
          />
        </svg>
        <Tooltip hover={hover} />
      </div>
    </figure>
  )
}

// ── Container ─────────────────────────────────────────────────────────────────

interface SpectraPreviewProps {
  hsqc: number[]
  h_nmr: number[]
  c_nmr: number[]
  mass_spec: number[]
}

function SpectraPreview({ hsqc, h_nmr, c_nmr, mass_spec }: SpectraPreviewProps) {
  // The store arrays are NaN-padded to preserve spreadsheet row positions;
  // a row only counts once every column it needs is filled.
  const hsqcPoints = useMemo(() => {
    const out: HSQCPoint[] = []
    for (let i = 0; i + 2 < hsqc.length; i += 3) {
      const [h, c, v] = [hsqc[i], hsqc[i + 1], hsqc[i + 2]]
      if (Number.isFinite(h) && Number.isFinite(c) && Number.isFinite(v)) out.push({ h, c, i: v })
    }
    return out
  }, [hsqc])

  const msPeaks = useMemo(() => {
    const out: MSPeak[] = []
    for (let i = 0; i + 1 < mass_spec.length; i += 2) {
      const [mz, v] = [mass_spec[i], mass_spec[i + 1]]
      if (Number.isFinite(mz) && Number.isFinite(v)) out.push({ mz, intensity: v })
    }
    return out
  }, [mass_spec])

  const hShifts = useMemo(() => h_nmr.filter(Number.isFinite), [h_nmr])
  const cShifts = useMemo(() => c_nmr.filter(Number.isFinite), [c_nmr])

  const hasHSQC = hsqcPoints.length > 0
  const hasWide = hShifts.length > 0 || cShifts.length > 0 || msPeaks.length > 0
  if (!hasHSQC && !hasWide) return null

  // HSQC is near-square and the rest are wide and short, so they get their own
  // column rather than a uniform grid that would stretch every card to match.
  return (
    <section>
      <h3 className="spectra__heading">
        Spectra preview
        <HelpButton content={HELP.spectral.preview} placement="right" />
      </h3>
      <div className={`spectra${hasHSQC && hasWide ? ' spectra--split' : ''}`}>
        {hasHSQC && <HSQCPlot points={hsqcPoints} />}
        {hasWide && (
          <div className="spectra__stack">
            {hShifts.length > 0 && <PeakListPlot title="¹H NMR" shifts={hShifts} />}
            {cShifts.length > 0 && <PeakListPlot title="¹³C NMR" shifts={cShifts} />}
            {msPeaks.length > 0 && <MassSpecPlot peaks={msPeaks} />}
          </div>
        )}
      </div>
    </section>
  )
}

export default SpectraPreview
