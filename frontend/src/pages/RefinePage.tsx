import { useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api, RefineMetric, RefineResponse } from '../services/api'
import { bondsWithin } from '../services/atomSelection'
import { loadSearch, updateSearch, type RefineOpSpec } from '../services/refineHandoff'
import { useAppStore } from '../store/store'
import FragmentOverlay from '../components/results/FragmentOverlay'
import FragmentThumb from '../components/results/FragmentThumb'
import MoleculeViewer from '../components/results/MoleculeViewer'
import ResultsGrid from '../components/results/ResultsGrid'
import './RefinePage.css'

/**
 * A secondary search in its own tab: the original prediction with regions of
 * retrieved compounds deleted or held fixed, retrieved again at full scale.
 *
 * The edits are shown down the side — each region on the compound it was taken
 * from — so what was removed or accepted stays in view next to the new cards.
 * Expanding a card works as on the main page, scored against the edited query; a
 * further delete / hold from one opens another tab that builds on these edits.
 * The search itself comes from the opening tab via refineHandoff.
 */

const MODE_LABEL = { delete: 'Deleted', hold: 'Held fixed' } as const

function OpCard({ op, step, smiles, removed }: {
  op: RefineOpSpec
  step: number
  smiles?: string
  removed?: number
}) {
  const occurrences = useMemo(() => {
    if (!op.geometry) return []
    return [{ atoms: op.atoms, bonds: bondsWithin(op.geometry.bonds, new Set(op.atoms)) }]
  }, [op])

  return (
    <article className={`refine-op refine-op--${op.mode}`}>
      <header className="refine-op__header">
        <span className="refine-op__step">{step}</span>
        <span className={`refine-op__mode refine-op__mode--${op.mode}`}>{MODE_LABEL[op.mode]}</span>
        <span className="refine-op__source" title={op.sourceName || op.smiles}>
          {op.sourceLabel}{op.sourceName ? ` · ${op.sourceName}` : ''}
        </span>
      </header>
      <div className="refine-op__figure">
        <MoleculeViewer plainSvg={op.plainSvg} smiles={op.smiles}>
          {op.geometry && <FragmentOverlay geometry={op.geometry} occurrences={occurrences} />}
        </MoleculeViewer>
      </div>
      <footer className="refine-op__footer">
        {smiles && (
          <span className="refine-op__frag">
            <FragmentThumb fragmentSmiles={smiles} atomSymbol="" alt={smiles} />
          </span>
        )}
        <span className="refine-op__meta">
          {op.atoms.length} atoms
          {removed !== undefined && ` · accounts for ${removed} fragment occurrences`}
        </span>
      </footer>
    </article>
  )
}

function RefinePage() {
  const { id = '' } = useParams()
  const spec = useMemo(() => loadSearch(id), [id])
  const parent = useMemo(() => (spec?.parentId ? loadSearch(spec.parentId) : null), [spec])
  const [metric, setMetric] = useState<RefineMetric>(spec?.metric ?? 'cosine')

  const results = useAppStore((s) => s.results)
  const predictedFp = useAppStore((s) => s.predictedFp)
  const activeModelId = useAppStore((s) => s.activeModelId)
  const setPredictResults = useAppStore((s) => s.setPredictResults)
  const setRefineContext = useAppStore((s) => s.setRefineContext)

  const search = useQuery<RefineResponse>({
    queryKey: ['refine-search', id, metric],
    queryFn: () => api.refineSearch({
      pred_fp: spec!.predFp,
      operations: spec!.operations.map(({ smiles, atoms, mode }) => ({ smiles, atoms, mode })),
      metric,
      k: spec!.k,
      model_id: spec!.modelId,
    }),
    enabled: Boolean(spec),
    staleTime: Infinity,
    retry: false,
  })

  // Hand the results to the shared result grid, ranked against the edited query,
  // and record the edits so a further one from this tab builds on them.
  useEffect(() => {
    if (!spec || !search.data) return
    setPredictResults(search.data.results, search.data.query_fp, { id: search.data.model_id ?? spec.modelId })
    setRefineContext({
      searchId: spec.id, basePredFp: spec.predFp, operations: spec.operations,
      metric, k: spec.k, modelId: search.data.model_id ?? spec.modelId,
    })
  }, [spec, search.data, metric, setPredictResults, setRefineContext])

  useEffect(() => {
    const n = spec?.operations.length ?? 0
    document.title = `MARINA · Secondary search (${n} edit${n === 1 ? '' : 's'})`
  }, [spec])

  const changeMetric = (m: RefineMetric) => {
    setMetric(m)
    updateSearch(id, { metric: m })
  }

  if (!spec) {
    return (
      <div className="refine-page refine-page--missing">
        <h1>Search not found</h1>
        <p>
          This secondary search is no longer stored in this browser (the 30 most recent are kept). Run it again
          from the results page.
        </p>
        <a href="/">Back to MARINA</a>
      </div>
    )
  }

  const data = search.data
  return (
    <div className="refine-page">
      <header className="refine-page__header">
        <a className="refine-page__brand" href="/" target="_blank" rel="noopener">MARINA</a>
        <h1 className="refine-page__title">Secondary search</h1>
        <p className="refine-page__subtitle">
          The original prediction with {spec.operations.length} region{spec.operations.length === 1 ? '' : 's'}
          {' '}deleted or held fixed, retrieved again.
          {parent && <> Builds on <a href={`/refine/${parent.id}`}>the previous search</a>.</>}
        </p>
      </header>

      <div className="refine-page__layout">
        <aside className="refine-page__side" aria-label="Edits">
          <h2 className="refine-page__heading">Edits</h2>
          {spec.operations.map((op, i) => (
            <OpCard
              key={i}
              op={op}
              step={i + 1}
              smiles={data?.operations[i]?.selection_smiles}
              removed={data?.operations[i]?.removed_occurrences}
            />
          ))}

          <div className="refine-page__metric" role="radiogroup" aria-label="Scoring">
            {(['cosine', 'containment'] as RefineMetric[]).map((m) => (
              <label key={m} className={metric === m ? 'is-active' : ''}>
                <input type="radio" name="refine-page-metric" checked={metric === m} onChange={() => changeMetric(m)} />
                {m === 'cosine' ? 'Cosine' : 'Containment'}
              </label>
            ))}
          </div>

          {data && (
            <>
              <p className="refine-page__note">
                Confident predicted fragments: {data.confident_before} → {data.confident_after} after the edits.
              </p>
              {data.unexplained.length > 0 && (
                <details className="refine-page__unexplained" open>
                  <summary>Left unexplained ({data.unexplained.length})</summary>
                  <p className="refine-page__note">
                    Fragments the prediction still expects — what the rest of the true structure should contain.
                  </p>
                  <div className="refine-page__frags">
                    {data.unexplained.map((u) => (
                      <figure key={u.index} className="refine-page__fragcell" title={u.fragment_smiles}>
                        <FragmentThumb fragmentSmiles={u.fragment_smiles} atomSymbol="" alt={u.fragment_smiles} />
                        <figcaption>
                          {u.multiplicity > 1 ? `≥${u.multiplicity}× · ` : ''}{Math.round(u.confidence * 100)}%
                        </figcaption>
                      </figure>
                    ))}
                  </div>
                </details>
              )}
            </>
          )}
        </aside>

        <main className="refine-page__main">
          {search.isFetching && <p className="refine-page__status">Searching…</p>}
          {search.isError && <p className="refine-page__error">{(search.error as Error).message}</p>}
          {data && !search.isFetching && (
            <ResultsGrid
              results={results}
              predFp={predictedFp}
              modelId={activeModelId ?? undefined}
              resultSource={null}
            />
          )}
        </main>
      </div>
    </div>
  )
}

export default RefinePage
