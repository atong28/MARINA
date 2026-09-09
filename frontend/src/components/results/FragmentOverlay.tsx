import { useId } from 'react'
import { DepictionGeometry, FragmentOccurrence } from '../../services/api'
import './FragmentOverlay.css'

/**
 * Draws one substructure's occurrences over the candidate's depiction.
 *
 * The backend reports where each atom lands on the image, so the highlight is
 * drawn here as SVG shapes over the <img> rather than baked into a second
 * rendering. Every occurrence keeps its own outline and can be singled out by
 * hovering, which a flat highlight of the union could not offer when
 * occurrences overlap. Coordinates are in the depiction's own pixel units and
 * scale with the image through the viewBox, so the overlay stays aligned at any
 * display size and over either depiction (plain or similarity map), since both
 * share one fit.
 */

// Stroke geometry for a 400 px canvas; scaled by the geometry's actual size.
const HALF_WIDTH = 11
const OUTLINE = 2.5

interface ShapeProps {
  occ: FragmentOccurrence
  geometry: DepictionGeometry
  /** Half the band width: bond strokes are twice this, atom discs this radius. */
  r: number
}

/** A bit's footprint as round-capped bond bands plus a disc per atom. */
function Shape({ occ, geometry, r }: ShapeProps) {
  const { atoms, bonds } = geometry
  return (
    <g strokeWidth={2 * r} strokeLinecap="round">
      {occ.bonds.map((b) => {
        const [i, j] = bonds[b] ?? []
        const p = atoms[i]
        const q = atoms[j]
        if (!p || !q) return null
        return <line key={`b${b}`} x1={p[0]} y1={p[1]} x2={q[0]} y2={q[1]} />
      })}
      {occ.atoms.map((a) => {
        const p = atoms[a]
        if (!p) return null
        return <circle key={`a${a}`} cx={p[0]} cy={p[1]} r={r} stroke="none" />
      })}
    </g>
  )
}

interface FragmentOverlayProps {
  geometry: DepictionGeometry
  occurrences: FragmentOccurrence[]
}

function FragmentOverlay({ geometry, occurrences }: FragmentOverlayProps) {
  // useId's delimiters are not safe inside url(#…); keep only plain characters.
  const id = useId().replace(/[^a-zA-Z0-9_-]/g, '')
  const { size } = geometry
  const unit = size / 400
  const r = HALF_WIDTH * unit
  const outline = OUTLINE * unit

  return (
    <svg
      className="fragment-overlay"
      viewBox={`0 0 ${size} ${size}`}
      aria-hidden="true"
    >
      {occurrences.map((occ, k) => {
        const maskId = `${id}-m${k}`
        return (
          <g key={k} className="fragment-overlay__occ">
            {/*
              The outline is the footprint drawn slightly larger, masked by the
              footprint itself so only a ring around the edge remains. Drawing a
              stroked outline directly would leave every bond's edges visible
              inside the band.
            */}
            <mask id={maskId} maskUnits="userSpaceOnUse" x={0} y={0} width={size} height={size}>
              <rect width={size} height={size} fill="#fff" />
              <g fill="#000" stroke="#000">
                <Shape occ={occ} geometry={geometry} r={r} />
              </g>
            </mask>
            <g className="fragment-overlay__outline" mask={`url(#${maskId})`}>
              <Shape occ={occ} geometry={geometry} r={r + outline} />
            </g>
            <g className="fragment-overlay__fill">
              <Shape occ={occ} geometry={geometry} r={r} />
            </g>
          </g>
        )
      })}
    </svg>
  )
}

export default FragmentOverlay
