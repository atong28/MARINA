import { useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api, RefineMetric, RefineMode, RefineOperationSummary, RefineResponse } from '../services/api'
import { bondsWithin } from '../services/atomSelection'
import { loadSearch, updateSearch, type RefineOpSpec } from '../services/refineHandoff'
import { useAppStore } from '../store/store'
import AtomSelector from '../components/results/AtomSelector'
import FragmentOverlay from '../components/results/FragmentOverlay'
import FragmentThumb from '../components/results/FragmentThumb'
import MoleculeViewer from '../components/results/MoleculeViewer'
import ResultsGrid from '../components/results/ResultsGrid'
import './RefinePage.css'

/**
 * A secondary search in its own tab: the original prediction with regions of
 * retrieved compounds deleted from the query or required in every result,
 * retrieved again at full scale.
 *
 * The edits are listed down the side, each region drawn on the compound it came
 * from, and stay editable there: re-select a region, switch it between delete and
 * require, or remove it, then re-run. Expanding a card works as on the main page,
 * scored against the edited query; a further edit from one opens another tab that
 * builds on these. The search itself comes from the opening tab via refineHandoff.
 */

const MODE_LABEL: Record<RefineMode, string> = { delete: 'Deleted', require: 'Required' }

/** What the server sees of an edit, to tell whether the draft differs from the last run. */
function opsKey(ops: RefineOpSpec[]): string {
  return JSON.stringify(ops.map(({ smiles, atoms, mode }) => [smiles, [...atoms].sort((a, b) => a - b), mode]))
}

interface OpCardProps {
  op: RefineOpSpec
  step: number
  /** Server summary from the last run, when the edits are unchanged since. */
  summary?: RefineOperationSummary
  editing: boolean
  onEdit: () => void
  onChange: (op: RefineOpSpec) => void
  onRemove: () => void
}

function OpCard({ op, step, summary, editing, onEdit, onChange, onRemove }: OpCardProps) {
  const occurrences = useMemo(() => {
    if (!op.geometry) return []
    return [{ atoms: op.atoms, bonds: bondsWithin(op.geometry.bonds, new Set(op.atoms)) }]
  }, [op])
  const selected = useMemo(() => new Set(op.atoms), [op.atoms])

  return (
    <article className={`refine-op refine-op--${op.mode}${editing ? ' is-editing' : ''}`}>
      <header className="refine-op__header">
        <span className="refine-op__step">{step}</span>
        <span className={`refine-op__mode refine-op__mode--${op.mode}`}>{MODE_LABEL[op.mode]}</span>
        <span className="refine-op__source" title={op.sourceName || op.smiles}>
          {op.sourceLabel}{op.sourceName ? ` · ${op.sourceName}` : ''}
        </span>
      </header>
      <div className="refine-op__figure">
        <MoleculeViewer plainSvg={op.plainSvg} smiles={op.smiles}>
          {op.geometry && (editing
            ? <AtomSelector
                geometry={op.geometry}
                selected={selected}
                onChange={(next) => onChange({ ...op, atoms: [...next].sort((a, b) => a - b) })}
              />
            : <FragmentOverlay geometry={op.geometry} occurrences={occurrences} />)}
        </MoleculeViewer>
      </div>
      {editing && (
        <p className="refine-op__hint">Click atoms to toggle them; drag a lasso to add, Alt-drag to remove.</p>
      )}
      <div className="refine-op__controls">
        <div className="refine-op__modes" role="radiogroup" aria-label={`Edit ${step}`}>
          {(['delete', 'require'] as RefineMode[]).map((m) => (
            <label key={m} className={op.mode === m ? 'is-active' : ''}
                   title={m === 'delete' ? 'Subtract this region from the query'
                     : 'Keep only compounds that contain this region as a substructure'}>
              <input type="radio" checked={op.mode === m} onChange={() => onChange({ ...op, mode: m })} />
              {m === 'delete' ? 'Delete' : 'Require'}
            </label>
          ))}
        </div>
        {op.geometry && (
          <button type="button" className="refine-op__btn" onClick={onEdit}>
            {editing ? 'Done' : 'Edit region'}
          </button>
        )}
        <button type="button" className="refine-op__btn refine-op__btn--remove" onClick={onRemove}>Remove</button>
      </div>
      <footer className="refine-op__footer">
        {summary?.selection_smiles && !editing && (
          <span className="refine-op__frag">
            <FragmentThumb fragmentSmiles={summary.selection_smiles} atomSymbol="" alt={summary.selection_smiles} />
          </span>
        )}
        <span className="refine-op__meta">
          {op.atoms.length} atoms
          {summary && op.mode === 'delete' && ` · accounts for ${summary.removed_occurrences} fragment occurrences`}
          {summary && op.mode === 'require' && summary.matches != null &&
            ` · ${summary.matches.toLocaleString()} compounds in the bank contain it`}
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
  // `committed` produced the results on screen; `draft` is what the sidebar is editing.
  const [committed, setCommitted] = useState<RefineOpSpec[]>(spec?.operations ?? [])
  const [draft, setDraft] = useState<RefineOpSpec[]>(spec?.operations ?? [])
  const [editing, setEditing] = useState<number | null>(null)

  const results = useAppStore((s) => s.results)
  const predictedFp = useAppStore((s) => s.predictedFp)
  const activeModelId = useAppStore((s) => s.activeModelId)
  const setPredictResults = useAppStore((s) => s.setPredictResults)
  const setRefineContext = useAppStore((s) => s.setRefineContext)

  const committedKey = opsKey(committed)
  const dirty = opsKey(draft) !== committedKey
  const draftValid = draft.length > 0 && draft.every((o) => o.atoms.length > 0)

  const search = useQuery<RefineResponse>({
    queryKey: ['refine-search', id, metric, committedKey],
    queryFn: () => api.refineSearch({
      pred_fp: spec!.predFp,
      operations: committed.map(({ smiles, atoms, mode }) => ({ smiles, atoms, mode })),
      metric,
      k: spec!.k,
      model_id: spec!.modelId,
    }),
    enabled: Boolean(spec) && committed.length > 0,
    staleTime: Infinity,
    retry: false,
  })

  // Hand the results to the shared result grid, ranked against the edited query,
  // and record the edits so a further one from this tab builds on them.
  useEffect(() => {
    if (!spec || !search.data) return
    setPredictResults(search.data.results, search.data.query_fp, { id: search.data.model_id ?? spec.modelId })
    setRefineContext({
      searchId: spec.id, basePredFp: spec.predFp, operations: committed,
      metric, k: spec.k, modelId: search.data.model_id ?? spec.modelId,
    })
  }, [spec, search.data, committed, metric, setPredictResults, setRefineContext])

  useEffect(() => {
    const n = committed.length
    document.title = `MARINA · Secondary search (${n} edit${n === 1 ? '' : 's'})`
  }, [committed])

  const changeMetric = (m: RefineMetric) => {
    setMetric(m)
    updateSearch(id, { metric: m })
  }

  const rerun = () => {
    if (!draftValid) return
    setEditing(null)
    setCommitted(draft)
    updateSearch(id, { operations: draft })
  }

  const revert = () => {
    setEditing(null)
    setDraft(committed)
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
  const nDelete = committed.filter((o) => o.mode === 'delete').length
  const nRequire = committed.length - nDelete
  return (
    <div className="refine-page">
      <header className="refine-page__header">
        <a className="refine-page__brand" href="/" target="_blank" rel="noopener">MARINA</a>
        <h1 className="refine-page__title">Secondary search</h1>
        <p className="refine-page__subtitle">
          The original prediction
          {nDelete > 0 && <> with {nDelete} region{nDelete === 1 ? '' : 's'} deleted</>}
          {nRequire > 0 && <>{nDelete > 0 ? ',' : ''} keeping only compounds that contain {nRequire === 1 ? 'the required region' : `all ${nRequire} required regions`}</>}
          , retrieved again.
          {parent && <> Builds on <a href={`/refine/${parent.id}`}>the previous search</a>.</>}
        </p>
      </header>

      <div className="refine-page__layout">
        <aside className="refine-page__side" aria-label="Edits">
          <h2 className="refine-page__heading">Edits</h2>
          {draft.map((op, i) => (
            <OpCard
              key={i}
              op={op}
              step={i + 1}
              summary={!dirty ? data?.operations[i] : undefined}
              editing={editing === i}
              onEdit={() => setEditing(editing === i ? null : i)}
              onChange={(next) => setDraft(draft.map((o, j) => (j === i ? next : o)))}
              onRemove={() => {
                setEditing(null)
                setDraft(draft.filter((_, j) => j !== i))
              }}
            />
          ))}
          {draft.length === 0 && <p className="refine-page__note">No edits left; keep at least one to search.</p>}

          {dirty && (
            <div className="refine-page__rerun">
              <button type="button" className="refine-page__rerun-btn" onClick={rerun} disabled={!draftValid}
                      title={draftValid ? 'Run the search with the edits as they are now' : 'Every edit needs at least one atom'}>
                Re-run search
              </button>
              <button type="button" className="refine-page__revert" onClick={revert}>Revert</button>
            </div>
          )}

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
              {data.require_matches != null && (
                <p className="refine-page__note">
                  {data.require_matches.toLocaleString()} compound{data.require_matches === 1 ? '' : 's'} in the
                  bank contain{data.require_matches === 1 ? 's' : ''} every required region.
                </p>
              )}
              {nDelete > 0 && (
                <p className="refine-page__note">
                  Confident predicted fragments: {data.confident_before} → {data.confident_after} after the deletions.
                </p>
              )}
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
          {data && !search.isFetching && data.results.length === 0 && (
            <p className="refine-page__status">No compound in the bank contains every required region.</p>
          )}
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
