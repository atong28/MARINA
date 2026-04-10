import { memo, useMemo } from 'react'
import { ResultCard as ResultCardType } from '../../services/api'
import MoleculeViewer from './MoleculeViewer'
import './ResultCard.css'

interface ResultCardProps {
  result: ResultCardType
  position: number
  /** When true, renders a "Remove" button instead of database links */
  isCustom?: boolean
  onRemove?: () => void
}

function ResultCard({ result, position, isCustom = false, onRemove }: ResultCardProps) {
  const { database_links } = result

  const cosine = useMemo(
    () => result.cosine_similarity ?? result.similarity,
    [result.cosine_similarity, result.similarity],
  )
  const tanimoto = result.tanimoto_similarity

  const hasLinks = !isCustom && Boolean(
    database_links?.coconut || database_links?.lotus || database_links?.npmrd
  )

  return (
    <div className={`result-card${isCustom ? ' result-card--custom' : ''}`}>
      <div className="result-card__header">
        <span className="result-card__rank">{isCustom ? 'Custom' : `#${position}`}</span>
        <div className="result-card__badges">
          <span className="result-card__badge result-card__badge--cosine">
            C: {cosine.toFixed(3)}
          </span>
          {typeof tanimoto === 'number' && (
            <span className="result-card__badge result-card__badge--tanimoto">
              T: {tanimoto.toFixed(3)}
            </span>
          )}
        </div>
      </div>

      <div className="result-card__name-section">
        <h3 className="result-card__name" title={result.name || result.smiles}>
          {result.name || result.smiles}
        </h3>
        {result.exact_mass !== undefined && result.exact_mass !== null && (
          <div className="result-card__mass">
            Exact mass: {result.exact_mass.toFixed(4)} Da
          </div>
        )}
      </div>

      <MoleculeViewer svg={result.svg || result.plain_svg} smiles={result.smiles} />

      <div className="result-card__footer">
        {isCustom ? (
          onRemove && (
            <button className="result-card__remove-btn" onClick={onRemove}>
              ✕ Remove
            </button>
          )
        ) : hasLinks ? (
          <div className="result-card__links">
            {database_links.coconut && (
              <a href={database_links.coconut} target="_blank" rel="noopener noreferrer" className="result-card__link result-card__link--coconut">
                COCONUT
              </a>
            )}
            {database_links.lotus && (
              <a href={database_links.lotus} target="_blank" rel="noopener noreferrer" className="result-card__link result-card__link--lotus">
                LOTUS
              </a>
            )}
            {database_links.npmrd && (
              <a href={database_links.npmrd} target="_blank" rel="noopener noreferrer" className="result-card__link result-card__link--npmrd">
                NPMRD
              </a>
            )}
          </div>
        ) : (
          <span className="result-card__local-note">Local dataset only</span>
        )}
      </div>
    </div>
  )
}

export default memo(ResultCard)
