import { memo, ReactNode, useMemo } from 'react'
import './MoleculeViewer.css'

interface MoleculeViewerProps {
  /** Plain line drawing, as SVG markup. Always the top image. */
  plainSvg?: string
  /** Similarity-map colour underlay (PNG data URI), shown beneath the drawing when set. */
  mapSrc?: string
  smiles: string
  /** Extra layers stacked above the drawing, e.g. a substructure overlay. */
  children?: ReactNode
}

/** Both raw SVG markup and data URIs are loaded through <img>. */
function toSrc(depiction: string): string {
  if (depiction.startsWith('data:image')) return depiction
  return `data:image/svg+xml,${encodeURIComponent(depiction)}`
}

/**
 * Depictions are rendered through <img>: markup injected with
 * dangerouslySetInnerHTML would execute any <script> or event handler it
 * contained, whereas an SVG loaded as an image is inert by specification.
 *
 * The similarity map is a colour wash drawn *under* the line drawing rather
 * than a second picture of the molecule, so the skeleton stays vector-crisp
 * in both modes and cannot shift when the map is toggled — every layer is
 * drawn to the same square with the server's one fit.
 */
function MoleculeViewer({ plainSvg, mapSrc, smiles, children }: MoleculeViewerProps) {
  const src = useMemo(() => (plainSvg ? toSrc(plainSvg) : null), [plainSvg])
  const map = useMemo(() => (mapSrc ? toSrc(mapSrc) : null), [mapSrc])

  return (
    <div className="molecule-viewer">
      {src ? (
        <div className="molecule-viewer__stack">
          {map && <img className="molecule-viewer__layer" src={map} alt="" />}
          <img className="molecule-viewer__layer" src={src} alt={smiles} />
          {children}
        </div>
      ) : (
        <div className="molecule-viewer__placeholder">
          <span className="molecule-viewer__smiles">{smiles}</span>
        </div>
      )}
    </div>
  )
}

export default memo(MoleculeViewer)
