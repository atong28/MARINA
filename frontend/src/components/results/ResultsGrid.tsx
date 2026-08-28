import { ResultCard as ResultCardType } from '../../services/api'
import type { CustomResult } from '../../store/store'
import ResultCard from './ResultCard'
import './ResultsGrid.css'

interface ResultsGridProps {
  results: ResultCardType[]
  customResults?: CustomResult[]
  onRemoveCustom?: (id: string) => void
  /** Reference fingerprint from the last search; drives the substructure panel. */
  predFp?: number[] | null
  modelId?: string
  /** How the results were produced; controls the retrieval-only caption. */
  resultSource?: 'prediction' | 'smiles-search' | null
}

function ResultsGrid({
  results, customResults = [], onRemoveCustom, predFp, modelId, resultSource,
}: ResultsGridProps) {
  if (results.length === 0 && customResults.length === 0) return null

  return (
    <div className="results-grid">
      {customResults.length > 0 && (
        <section className="results-grid__section">
          <h2 className="results-grid__heading">
            Custom cards <span className="results-grid__count">({customResults.length})</span>
          </h2>
          <div className="results-grid__list">
            {customResults.map((entry, i) => (
              <ResultCard
                key={entry.id}
                result={entry.card}
                position={i + 1}
                isCustom
                onRemove={onRemoveCustom ? () => onRemoveCustom(entry.id) : undefined}
                predFp={predFp}
                modelId={modelId}
              />
            ))}
          </div>
        </section>
      )}

      {results.length > 0 && (
        <section className="results-grid__section">
          <h2 className="results-grid__heading">
            Results <span className="results-grid__count">({results.length})</span>
          </h2>
          {resultSource === 'prediction' && (
            <p className="results-grid__note">
              Results are retrieved from the MARINA retrieval database. The model does not
              predict novel structures; use SMILES search to see a proposed structure's
              closest matches.
            </p>
          )}
          <div className="results-grid__list">
            {results.map((result, i) => (
              <ResultCard
                key={`${result.index}-${result.smiles}`}
                result={result}
                position={i + 1}
                predFp={predFp}
                modelId={modelId}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

export default ResultsGrid
