import { ResultCard as ResultCardType } from '../../services/api'
import ResultCard from './ResultCard'
import './ResultsGrid.css'

interface ResultsGridProps {
  results: ResultCardType[]
  customResults?: ResultCardType[]
  onRemoveCustom?: (index: number) => void
}

function ResultsGrid({ results, customResults = [], onRemoveCustom }: ResultsGridProps) {
  if (results.length === 0 && customResults.length === 0) return null

  return (
    <div className="results-grid">
      {customResults.length > 0 && (
        <section className="results-grid__section">
          <h2 className="results-grid__heading">
            Custom cards <span className="results-grid__count">({customResults.length})</span>
          </h2>
          <div className="results-grid__list">
            {customResults.map((result, i) => (
              <ResultCard
                key={`custom-${i}-${result.smiles}`}
                result={result}
                position={i + 1}
                isCustom
                onRemove={onRemoveCustom ? () => onRemoveCustom(i) : undefined}
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
          <div className="results-grid__list">
            {results.map((result, i) => (
              <ResultCard
                key={`${result.index}-${result.smiles}`}
                result={result}
                position={i + 1}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

export default ResultsGrid
