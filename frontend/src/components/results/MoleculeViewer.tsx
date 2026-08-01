import { memo, useMemo } from 'react'
import './MoleculeViewer.css'

interface MoleculeViewerProps {
  svg?: string
  smiles: string
}

/**
 * Depictions arrive as either a data URI (highlighted, PNG) or raw SVG markup
 * (plain line drawing). Both are rendered through <img>: markup injected with
 * dangerouslySetInnerHTML would execute any <script> or event handler it
 * contained, whereas an SVG loaded as an image is inert by specification.
 */
function MoleculeViewer({ svg, smiles }: MoleculeViewerProps) {
  const src = useMemo(() => {
    if (!svg) return null
    if (svg.startsWith('data:image')) return svg
    return `data:image/svg+xml,${encodeURIComponent(svg)}`
  }, [svg])

  return (
    <div className="molecule-viewer">
      {src ? (
        <img className="molecule-viewer__img" src={src} alt={smiles} />
      ) : (
        <div className="molecule-viewer__placeholder">
          <span className="molecule-viewer__smiles">{smiles}</span>
        </div>
      )}
    </div>
  )
}

export default memo(MoleculeViewer)
