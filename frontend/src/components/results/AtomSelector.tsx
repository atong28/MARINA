import { useRef, useState } from 'react'
import { DepictionGeometry } from '../../services/api'
import { applyGesture, bondsWithin, type Point } from '../../services/atomSelection'
import './AtomSelector.css'

/**
 * Picks a region of the depicted molecule: click an atom to toggle it, drag a
 * lasso to add every atom inside it, Alt-drag to remove them. Drawn in the
 * depiction's own canvas units like FragmentOverlay, so it lines up at any size.
 */

const HALF_WIDTH = 9

interface AtomSelectorProps {
  geometry: DepictionGeometry
  selected: Set<number>
  onChange: (next: Set<number>) => void
}

function AtomSelector({ geometry, selected, onChange }: AtomSelectorProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [path, setPath] = useState<Point[] | null>(null)
  const { size, atoms, bonds } = geometry
  const unit = size / 400
  const r = HALF_WIDTH * unit

  const toCanvas = (e: React.PointerEvent): Point | null => {
    const svg = svgRef.current
    const ctm = svg?.getScreenCTM()
    if (!svg || !ctm) return null
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse())
    return [p.x, p.y]
  }

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return
    const p = toCanvas(e)
    if (!p) return
    e.currentTarget.setPointerCapture(e.pointerId)
    setPath([p])
  }

  const onPointerMove = (e: React.PointerEvent) => {
    if (!path) return
    const p = toCanvas(e)
    if (!p) return
    const last = path[path.length - 1]
    if (Math.hypot(p[0] - last[0], p[1] - last[1]) >= 2 * unit) setPath([...path, p])
  }

  const onPointerUp = (e: React.PointerEvent) => {
    if (!path) return
    const p = toCanvas(e)
    const points = p ? [...path, p] : path
    setPath(null)
    onChange(applyGesture(selected, atoms, points, unit, e.altKey))
  }

  const selectedBonds = bondsWithin(bonds, selected)

  return (
    <svg
      ref={svgRef}
      className="atom-selector"
      viewBox={`0 0 ${size} ${size}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={() => setPath(null)}
      role="application"
      aria-label="Select atoms: click to toggle, drag to lasso, Alt-drag to remove"
    >
      <g className="atom-selector__region" strokeWidth={2 * r} strokeLinecap="round">
        {selectedBonds.map((b) => {
          const [i, j] = bonds[b]
          return (
            <line key={`b${b}`} x1={atoms[i][0]} y1={atoms[i][1]} x2={atoms[j][0]} y2={atoms[j][1]} />
          )
        })}
        {[...selected].map((a) => atoms[a] && (
          <circle key={`a${a}`} cx={atoms[a][0]} cy={atoms[a][1]} r={r} stroke="none" />
        ))}
      </g>
      {path && path.length > 1 && (
        <polygon
          className="atom-selector__lasso"
          points={path.map((p) => p.join(',')).join(' ')}
          strokeWidth={1.5 * unit}
          strokeDasharray={`${4 * unit} ${3 * unit}`}
        />
      )}
    </svg>
  )
}

export default AtomSelector
