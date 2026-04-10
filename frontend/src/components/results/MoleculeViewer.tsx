import { memo, useMemo } from 'react'
import './MoleculeViewer.css'

interface MoleculeViewerProps {
  svg?: string
  smiles: string
}

function MoleculeViewer({ svg, smiles }: MoleculeViewerProps) {
  const content = useMemo(() => {
    if (!svg) {
      return (
        <div className="molecule-viewer__placeholder">
          <span className="molecule-viewer__smiles">{smiles}</span>
        </div>
      )
    }
    if (svg.startsWith('data:image')) {
      return <img className="molecule-viewer__img" src={svg} alt={smiles} />
    }
    return <div className="molecule-viewer__svg" dangerouslySetInnerHTML={{ __html: svg }} />
  }, [svg, smiles])

  return <div className="molecule-viewer">{content}</div>
}

export default memo(MoleculeViewer)
