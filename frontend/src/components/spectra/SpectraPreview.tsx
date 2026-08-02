/**
 * Live preview of whatever spectra are currently in the spreadsheet.
 *
 * Re-renders on every spreadsheet edit or paste, since it reads the same
 * store arrays the spreadsheet writes. Panels appear only for the data types
 * that are actually present.
 *
 * Axis conventions follow standard spectroscopic practice:
 *   • HSQC — f2 (¹H) descends left→right, f1 (¹³C) ascends top→bottom, and the
 *     axes are drawn on the bottom and the right, so the origin sits in the
 *     top-right corner.
 *   • ¹H / ¹³C — shift descends left→right.
 *   • Every NMR view opens on a window that contains 0 ppm, so a spectrum is
 *     never shown floating on an axis that hides where the origin is.
 *   • MS runs m/z low → high left→right, intensity normalized to base peak.
 *
 * All plots are drag-to-pan and wheel-to-zoom; double-click (or the reset
 * button) returns to the opening view.
 */
import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import HelpButton from '../common/HelpButton'
import { HELP } from '../../helpContent'
import './SpectraPreview.css'

// ── Palette (validated: see dataviz palette, light surface) ───────────────────

const SERIES_BLUE = '#2a78d6' // slot 1 — negative HSQC phase (CH₂), 1-D sticks
const SERIES_RED = '#e34948' // slot 8 — positive HSQC phase (CH / CH₃)
const GRID = '#e1e0d9'
const AXIS = '#c3c2b7'

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

/**
 * Like padDomain, but the window always contains 0 and is not padded past it.
 * A ppm axis that starts at the first peak makes a cluster of shifts look like
 * it spans the whole spectrum; anchoring on the origin keeps the scale honest.
 */
export function originDomain(lo: number, hi: number, frac = 0.05): [number, number] {
  const l = Math.min(0, lo)
  const h = Math.max(0, hi)
  if (l === h) return [0, 1] // every shift is exactly 0
  const p = (h - l) * frac
  return [l === 0 ? 0 : l - p, h === 0 ? 0 : h + p]
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

/** Ticks for an axis whose endpoints may be in either order. */
function axisTicks(a: number, b: number, target: number): Ticks {
  return niceTicks(Math.min(a, b), Math.max(a, b), target)
}

// ── Formatting ────────────────────────────────────────────────────────────────

export function fmtShift(v: number): string {
  return v.toFixed(2)
}

export function fmtIntensity(v: number): string {
  if (v === 0) return '0'
  return Math.abs(v) >= 1e4 || Math.abs(v) < 1e-2 ? v.toExponential(2) : v.toFixed(2)
}

// ── Pan & zoom ────────────────────────────────────────────────────────────────

interface Rect {
  x0: number
  x1: number
  y0: number
  y1: number
}

/**
 * The visible window, named by where each bound lands on screen rather than by
 * magnitude — so a reversed ppm axis (xLeft > xRight) needs no special cases in
 * the pan and zoom maths.
 */
export interface View {
  xLeft: number
  xRight: number
  yTop: number
  yBottom: number
}

export function viewEquals(a: View, b: View): boolean {
  return (
    a.xLeft === b.xLeft && a.xRight === b.xRight && a.yTop === b.yTop && a.yBottom === b.yBottom
  )
}

/** Zoom bounds, as a multiple of the opening span. */
const MIN_SPAN_FRAC = 0.002
const MAX_SPAN_FRAC = 50

/** Scales one axis about `center`, refusing steps that leave the zoom bounds. */
export function zoomAxis(
  lo: number,
  hi: number,
  center: number,
  factor: number,
  baseSpan: number,
): [number, number] {
  const span = Math.abs((hi - lo) * factor)
  const limit = Math.abs(baseSpan) || 1
  if (span < limit * MIN_SPAN_FRAC || span > limit * MAX_SPAN_FRAC) return [lo, hi]
  return [center + (lo - center) * factor, center + (hi - center) * factor]
}

interface PlotViewOpts {
  /** ¹³C is a data axis on HSQC; on the 1-D plots the vertical is fixed. */
  panY?: boolean
  onHover?: (vx: number, vy: number, cx: number, cy: number) => void
  onLeave?: () => void
}

/**
 * Drag-to-pan / wheel-to-zoom over an SVG viewBox. `base` must be memoised —
 * a new identity means new data, which resets the view.
 */
function usePlotView(base: View, rect: Rect, vbWidth: number, opts: PlotViewOpts = {}) {
  const { panY = false, onHover, onLeave } = opts
  const [view, setView] = useState<View>(base)
  const [dragging, setDragging] = useState(false)
  const svgRef = useRef<SVGSVGElement | null>(null)
  const dragRef = useRef<{ cx: number; cy: number; from: View } | null>(null)

  useEffect(() => setView(base), [base])

  /** Converts a client point into viewBox units. */
  const toViewBox = useCallback(
    (clientX: number, clientY: number) => {
      const box = svgRef.current!.getBoundingClientRect()
      const scale = box.width / vbWidth || 1
      const cx = clientX - box.left
      const cy = clientY - box.top
      return { vx: cx / scale, vy: cy / scale, cx, cy, scale }
    },
    [vbWidth],
  )

  // React routes wheel through a passive root listener, so preventDefault from
  // an onWheel prop is ignored and the page scrolls along with the zoom.
  useEffect(() => {
    const svg = svgRef.current
    if (!svg) return
    const onWheel = (e: WheelEvent) => {
      const { vx, vy } = toViewBox(e.clientX, e.clientY)
      if (vx < rect.x0 || vx > rect.x1 || vy < rect.y0 || vy > rect.y1) return
      e.preventDefault()
      const factor = Math.exp(e.deltaY * 0.0015)
      setView((v) => {
        const dx = v.xLeft + ((vx - rect.x0) / (rect.x1 - rect.x0)) * (v.xRight - v.xLeft)
        const [xLeft, xRight] = zoomAxis(v.xLeft, v.xRight, dx, factor, base.xRight - base.xLeft)
        if (!panY) return { ...v, xLeft, xRight }
        const dy = v.yTop + ((vy - rect.y0) / (rect.y1 - rect.y0)) * (v.yBottom - v.yTop)
        const [yTop, yBottom] = zoomAxis(v.yTop, v.yBottom, dy, factor, base.yBottom - base.yTop)
        return { xLeft, xRight, yTop, yBottom }
      })
    }
    svg.addEventListener('wheel', onWheel, { passive: false })
    return () => svg.removeEventListener('wheel', onWheel)
  }, [base, rect, panY, toViewBox])

  const handlePointerDown = useCallback(
    (e: React.PointerEvent<SVGRectElement>) => {
      e.currentTarget.setPointerCapture(e.pointerId)
      dragRef.current = { cx: e.clientX, cy: e.clientY, from: view }
      setDragging(true)
    },
    [view],
  )

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<SVGRectElement>) => {
      const { vx, vy, cx, cy, scale } = toViewBox(e.clientX, e.clientY)
      const drag = dragRef.current
      if (!drag) {
        onHover?.(vx, vy, cx, cy)
        return
      }
      const dvx = (e.clientX - drag.cx) / scale
      const dvy = (e.clientY - drag.cy) / scale
      const sx = ((drag.from.xRight - drag.from.xLeft) / (rect.x1 - rect.x0)) * dvx
      const sy = ((drag.from.yBottom - drag.from.yTop) / (rect.y1 - rect.y0)) * dvy
      setView({
        xLeft: drag.from.xLeft - sx,
        xRight: drag.from.xRight - sx,
        yTop: panY ? drag.from.yTop - sy : drag.from.yTop,
        yBottom: panY ? drag.from.yBottom - sy : drag.from.yBottom,
      })
    },
    [rect, panY, toViewBox, onHover],
  )

  const endDrag = useCallback((e: React.PointerEvent<SVGRectElement>) => {
    if (!dragRef.current) return
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId)
    }
    dragRef.current = null
    setDragging(false)
  }, [])

  const handlePointerLeave = useCallback(() => {
    if (!dragRef.current) onLeave?.()
  }, [onLeave])

  const reset = useCallback(() => setView(base), [base])

  return {
    svgRef,
    view,
    dragging,
    reset,
    isDefault: viewEquals(view, base),
    x: linear(view.xLeft, view.xRight, rect.x0, rect.x1),
    y: linear(view.yTop, view.yBottom, rect.y0, rect.y1),
    /** Spread onto the transparent overlay rect covering the plot area. */
    bind: {
      onPointerDown: handlePointerDown,
      onPointerMove: handlePointerMove,
      onPointerUp: endDrag,
      onPointerCancel: endDrag,
      onPointerLeave: handlePointerLeave,
      onDoubleClick: reset,
    },
  }
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

function ResetButton({ onClick }: { onClick: () => void }) {
  return (
    <button type="button" className="spectra__reset" onClick={onClick}>
      Reset view
    </button>
  )
}

// ── Axes ──────────────────────────────────────────────────────────────────────

interface AxesProps {
  rect: Rect
  width: number
  height: number
  x: Scale
  xTicks: Ticks
  xLabel: string
  y?: Scale
  yTicks?: Ticks
  yLabel?: string
  /** HSQC draws its shift axis on the right, opposite the reversed ¹H axis. */
  ySide?: 'left' | 'right'
}

function Axes({
  rect, width, height, x, xTicks, xLabel, y, yTicks, yLabel, ySide = 'left',
}: AxesProps) {
  const { x0, x1, y0, y1 } = rect
  const right = ySide === 'right'
  const yAxisX = right ? x1 : x0
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
          <text
            className="spectra__tick"
            x={right ? x1 + 6 : x0 - 6}
            y={y(t) + 3.5}
            textAnchor={right ? 'start' : 'end'}
          >
            {yTicks.fmt(t)}
          </text>
        </g>
      ))}
      <line x1={x0} x2={x1} y1={y1} y2={y1} stroke={AXIS} strokeWidth={1} />
      {y && <line x1={yAxisX} x2={yAxisX} y1={y0} y2={y1} stroke={AXIS} strokeWidth={1} />}
      <text className="spectra__axis-label" x={(x0 + x1) / 2} y={height - 3} textAnchor="middle">
        {xLabel}
      </text>
      {yLabel && (
        <text
          className="spectra__axis-label"
          transform={
            right
              ? `translate(${width - 6} ${(y0 + y1) / 2}) rotate(90)`
              : `translate(10 ${(y0 + y1) / 2}) rotate(-90)`
          }
          textAnchor="middle"
        >
          {yLabel}
        </text>
      )}
    </g>
  )
}

// ── HSQC: 2-D scatter, origin top-right ──────────────────────────────────────

const HSQC_W = 400
const HSQC_H = 350
const HSQC_RECT: Rect = { x0: 16, x1: HSQC_W - 52, y0: 12, y1: HSQC_H - 32 }

interface HSQCPoint {
  h: number
  c: number
  i: number
}

function HSQCPlot({ points }: { points: HSQCPoint[] }) {
  const [hover, setHover] = useState<Hover | null>(null)
  const clipId = useId()

  const base = useMemo<View>(() => {
    const [hLo, hHi] = originDomain(
      Math.min(...points.map((p) => p.h)),
      Math.max(...points.map((p) => p.h)),
    )
    const [cLo, cHi] = originDomain(
      Math.min(...points.map((p) => p.c)),
      Math.max(...points.map((p) => p.c)),
    )
    // ¹H descends left→right; ¹³C ascends top→bottom. Origin: top-right.
    return { xLeft: hHi, xRight: hLo, yTop: cLo, yBottom: cHi }
  }, [points])

  const placedRef = useRef<{ px: number; py: number; p: HSQCPoint }[]>([])

  const handleHover = useCallback((vx: number, vy: number, cx: number, cy: number) => {
    let best: HSQCPoint | null = null
    let bestD = 24 * 24
    for (const { px, py, p } of placedRef.current) {
      const d = (px - vx) ** 2 + (py - vy) ** 2
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
  }, [])

  const plot = usePlotView(base, HSQC_RECT, HSQC_W, {
    panY: true,
    onHover: handleHover,
    onLeave: () => setHover(null),
  })

  const placed = points.map((p) => ({ px: plot.x(p.h), py: plot.y(p.c), p }))
  placedRef.current = placed

  const hasNegative = points.some((p) => p.i < 0)
  const hasPositive = points.some((p) => p.i >= 0)
  const showLegend = hasNegative && hasPositive

  return (
    <figure className="spectra__card spectra__card--hsqc">
      <figcaption className="spectra__title">
        ¹H-¹³C HSQC <span className="spectra__count">{points.length} cross-peaks</span>
        {!plot.isDefault && <ResetButton onClick={plot.reset} />}
      </figcaption>
      {showLegend && (
        <div className="spectra__legend">
          <span className="spectra__legend-item">
            <span className="spectra__swatch" style={{ background: SERIES_RED }} />
            CH / CH₃ (positive)
          </span>
          <span className="spectra__legend-item">
            <span className="spectra__swatch" style={{ background: SERIES_BLUE }} />
            CH₂ (negative)
          </span>
        </div>
      )}
      <div className="spectra__plot">
        <svg
          ref={plot.svgRef}
          viewBox={`0 0 ${HSQC_W} ${HSQC_H}`}
          role="img"
          aria-label="HSQC cross-peak map"
        >
          <defs>
            <clipPath id={clipId}>
              <rect
                x={HSQC_RECT.x0}
                y={HSQC_RECT.y0}
                width={HSQC_RECT.x1 - HSQC_RECT.x0}
                height={HSQC_RECT.y1 - HSQC_RECT.y0}
              />
            </clipPath>
          </defs>
          <Axes
            rect={HSQC_RECT}
            width={HSQC_W}
            height={HSQC_H}
            x={plot.x}
            xTicks={axisTicks(plot.view.xLeft, plot.view.xRight, 5)}
            xLabel="f2 — ¹H (ppm)"
            y={plot.y}
            yTicks={axisTicks(plot.view.yTop, plot.view.yBottom, 6)}
            yLabel="f1 — ¹³C (ppm)"
            ySide="right"
          />
          <g clipPath={`url(#${clipId})`}>
            {placed.map(({ px, py, p }, idx) => (
              <circle key={idx} cx={px} cy={py} r={3} fill={p.i < 0 ? SERIES_BLUE : SERIES_RED} />
            ))}
          </g>
          <rect
            className={`spectra__surface${plot.dragging ? ' spectra__surface--dragging' : ''}`}
            x={HSQC_RECT.x0}
            y={HSQC_RECT.y0}
            width={HSQC_RECT.x1 - HSQC_RECT.x0}
            height={HSQC_RECT.y1 - HSQC_RECT.y0}
            {...plot.bind}
          />
        </svg>
        <Tooltip hover={plot.dragging ? null : hover} />
      </div>
    </figure>
  )
}

// ── 1-D peak list (¹H / ¹³C): sticks on a reversed ppm axis ──────────────────

const PEAK_W = 700
const PEAK_H = 130
const PEAK_RECT: Rect = { x0: 14, x1: PEAK_W - 14, y0: 16, y1: PEAK_H - 32 }

function PeakListPlot({ title, shifts }: { title: string; shifts: number[] }) {
  const [hover, setHover] = useState<Hover | null>(null)
  const clipId = useId()

  const base = useMemo<View>(() => {
    const [lo, hi] = originDomain(Math.min(...shifts), Math.max(...shifts))
    return { xLeft: hi, xRight: lo, yTop: 0, yBottom: 1 }
  }, [shifts])

  const placedRef = useRef<{ px: number; s: number }[]>([])

  const handleHover = useCallback((vx: number, _vy: number, cx: number, cy: number) => {
    let best: number | null = null
    let bestD = 12
    for (const { px, s } of placedRef.current) {
      const d = Math.abs(px - vx)
      if (d < bestD) {
        bestD = d
        best = s
      }
    }
    setHover(best === null ? null : { cx, cy, lines: [`δ ${fmtShift(best)} ppm`] })
  }, [])

  const plot = usePlotView(base, PEAK_RECT, PEAK_W, {
    onHover: handleHover,
    onLeave: () => setHover(null),
  })

  const placed = shifts.map((s) => ({ px: plot.x(s), s }))
  placedRef.current = placed

  return (
    <figure className="spectra__card">
      <figcaption className="spectra__title">
        {title} <span className="spectra__count">{shifts.length} peaks</span>
        {!plot.isDefault && <ResetButton onClick={plot.reset} />}
      </figcaption>
      <div className="spectra__plot">
        <svg
          ref={plot.svgRef}
          viewBox={`0 0 ${PEAK_W} ${PEAK_H}`}
          role="img"
          aria-label={`${title} peak list`}
        >
          <defs>
            <clipPath id={clipId}>
              <rect
                x={PEAK_RECT.x0}
                y={PEAK_RECT.y0}
                width={PEAK_RECT.x1 - PEAK_RECT.x0}
                height={PEAK_RECT.y1 - PEAK_RECT.y0}
              />
            </clipPath>
          </defs>
          <Axes
            rect={PEAK_RECT}
            width={PEAK_W}
            height={PEAK_H}
            x={plot.x}
            xTicks={axisTicks(plot.view.xLeft, plot.view.xRight, 7)}
            xLabel="δ (ppm)"
          />
          <g clipPath={`url(#${clipId})`}>
            {placed.map(({ px }, idx) => (
              <line
                key={idx}
                x1={px}
                x2={px}
                y1={PEAK_RECT.y1}
                y2={PEAK_RECT.y0}
                stroke={SERIES_BLUE}
                strokeWidth={1.5}
              />
            ))}
          </g>
          <rect
            className={`spectra__surface${plot.dragging ? ' spectra__surface--dragging' : ''}`}
            x={PEAK_RECT.x0}
            y={PEAK_RECT.y0}
            width={PEAK_RECT.x1 - PEAK_RECT.x0}
            height={PEAK_RECT.y1 - PEAK_RECT.y0}
            {...plot.bind}
          />
        </svg>
        <Tooltip hover={plot.dragging ? null : hover} />
      </div>
    </figure>
  )
}

// ── MS/MS: sticks on an ascending m/z axis, normalized to base peak ──────────

const MS_W = 700
const MS_H = 230
const MS_RECT: Rect = { x0: 46, x1: MS_W - 14, y0: 14, y1: MS_H - 32 }
const MS_Y_TICKS: Ticks = { values: [0, 25, 50, 75, 100], fmt: (v) => v.toFixed(0) }

interface MSPeak {
  mz: number
  intensity: number
}

function MassSpecPlot({ peaks }: { peaks: MSPeak[] }) {
  const [hover, setHover] = useState<Hover | null>(null)
  const clipId = useId()

  const base = useMemo<View>(() => {
    const [lo, hi] = padDomain(
      Math.min(...peaks.map((p) => p.mz)),
      Math.max(...peaks.map((p) => p.mz)),
    )
    return { xLeft: lo, xRight: hi, yTop: 100, yBottom: 0 }
  }, [peaks])

  const relative = useMemo(() => {
    const strongest = Math.max(...peaks.map((p) => Math.abs(p.intensity))) || 1
    return peaks.map((p) => ({ ...p, rel: (Math.abs(p.intensity) / strongest) * 100 }))
  }, [peaks])

  const placedRef = useRef<{ px: number; p: (typeof relative)[number] }[]>([])

  const handleHover = useCallback((vx: number, _vy: number, cx: number, cy: number) => {
    let best: (typeof relative)[number] | null = null
    let bestD = 12
    for (const { px, p } of placedRef.current) {
      const d = Math.abs(px - vx)
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
  }, [])

  const plot = usePlotView(base, MS_RECT, MS_W, {
    onHover: handleHover,
    onLeave: () => setHover(null),
  })

  const placed = relative.map((p) => ({ px: plot.x(p.mz), p }))
  placedRef.current = placed

  return (
    <figure className="spectra__card">
      <figcaption className="spectra__title">
        MS/MS (Positive) <span className="spectra__count">{peaks.length} peaks</span>
        {!plot.isDefault && <ResetButton onClick={plot.reset} />}
      </figcaption>
      <div className="spectra__plot">
        <svg ref={plot.svgRef} viewBox={`0 0 ${MS_W} ${MS_H}`} role="img" aria-label="Mass spectrum">
          <defs>
            <clipPath id={clipId}>
              <rect
                x={MS_RECT.x0}
                y={MS_RECT.y0}
                width={MS_RECT.x1 - MS_RECT.x0}
                height={MS_RECT.y1 - MS_RECT.y0}
              />
            </clipPath>
          </defs>
          <Axes
            rect={MS_RECT}
            width={MS_W}
            height={MS_H}
            x={plot.x}
            xTicks={axisTicks(plot.view.xLeft, plot.view.xRight, 6)}
            xLabel="m/z"
            y={plot.y}
            yTicks={MS_Y_TICKS}
            yLabel="Rel. intensity (%)"
          />
          <g clipPath={`url(#${clipId})`}>
            {placed.map(({ px, p }, idx) => (
              <line
                key={idx}
                x1={px}
                x2={px}
                y1={MS_RECT.y1}
                y2={plot.y(p.rel)}
                stroke={SERIES_BLUE}
                strokeWidth={1.5}
              />
            ))}
          </g>
          <rect
            className={`spectra__surface${plot.dragging ? ' spectra__surface--dragging' : ''}`}
            x={MS_RECT.x0}
            y={MS_RECT.y0}
            width={MS_RECT.x1 - MS_RECT.x0}
            height={MS_RECT.y1 - MS_RECT.y0}
            {...plot.bind}
          />
        </svg>
        <Tooltip hover={plot.dragging ? null : hover} />
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
