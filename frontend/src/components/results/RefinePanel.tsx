import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, DepictionGeometry, RefineMetric, RefineMode, ResultCard } from '../../services/api'
import { hasOperation, saveSearch, type RefineOpSpec } from '../../services/refineHandoff'
import { useAppStore } from '../../store/store'
import './RefinePanel.css'

/**
 * Hybrid compounds: delete a region of this candidate from the query, or require
 * it in every result, and retrieve again in a new browser tab. A hybrid A+B's retrievals are
 * often all A-like, so B never shows; taking A out of the query lets the rest of
 * the prediction speak. Inside a search tab, a further deletion builds on that
 * tab's deletions. The arithmetic is server-side (backend/app/refine.py); the tab is
 * pages/RefinePage.tsx.
 */

const MAX_CORE_POOL = 50
const MIN_K = 10

const METRIC_HINT: Record<RefineMetric, string> = {
  cosine: 'Cosine: compounds that look like what is left. Best when the missing part exists as a compound of its own.',
  containment:
    'Containment: compounds that contain what is left, without penalising their other parts. ' +
    'Use when the missing part only occurs inside larger molecules.',
}

function coverageLabel(coverage: number, found: number, total: number): string {
  return coverage >= 1 ? `All ${total}` : `${found} of ${total}`
}

interface RefinePanelProps {
  card: ResultCard
  label: string
  /** The fingerprint this page's results were ranked against (the original prediction on the main page). */
  predFp: number[]
  modelId?: string
  selection: Set<number>
  onSelection: (next: Set<number>) => void
  /** Null until the depiction geometry has loaded; atoms cannot be picked before. */
  geometry: DepictionGeometry | null
}

function RefinePanel({ card, label, predFp, modelId, selection, onSelection, geometry }: RefinePanelProps) {
  const results = useAppStore((s) => s.results)
  const context = useAppStore((s) => s.refineContext)
  const [metric, setMetric] = useState<RefineMetric>(context?.metric ?? 'cosine')
  const [blockedHref, setBlockedHref] = useState<string | null>(null)

  const pool = useMemo(() => results.slice(0, MAX_CORE_POOL).map((r) => r.smiles), [results])
  const cores = useQuery({
    queryKey: ['shared-core', pool, card.smiles],
    queryFn: () => api.sharedCore({ smiles: pool, target: card.smiles }),
    enabled: pool.length >= 2,
    staleTime: Infinity,
    retry: false,
  })

  const chain = context?.operations ?? []
  const atoms = [...selection].sort((a, b) => a - b)
  const duplicate = atoms.length > 0 && hasOperation(chain, card.smiles, atoms)

  const open = (mode: RefineMode) => {
    if (!atoms.length || duplicate) return
    const op: RefineOpSpec = {
      smiles: card.smiles, atoms, mode,
      sourceLabel: context ? `step ${chain.length} · ${label}` : `original · ${label}`,
      sourceName: card.name ?? '',
      plainSvg: card.plain_svg,
      geometry,
    }
    const id = saveSearch({
      // Edits always apply to the original prediction; a search tab passes its own on.
      predFp: context?.basePredFp ?? predFp,
      operations: [...chain, op],
      metric,
      k: context?.k ?? Math.max(MIN_K, results.length),
      modelId: context?.modelId ?? modelId,
      parentId: context?.searchId,
    })
    const href = `/refine/${id}`
    const win = window.open(href, '_blank')
    setBlockedHref(win ? null : href)
  }

  return (
    <div className="refine">
      <p className="refine__intro">
        Select a region of this compound — for example the part every retrieval agrees on. Delete it to search
        for what the rest of the prediction describes, or require it so that every result contains it. The
        search opens in a new tab.
      </p>

      {context && (
        <p className="refine__context">
          Builds on this tab’s {chain.length} edit{chain.length === 1 ? '' : 's'}: the new tab applies them
          and this one.
        </p>
      )}

      <section className="refine__section">
        <h4 className="refine__heading">1. Select a region</h4>
        <p className="refine__hint">
          {geometry
            ? 'Click atoms on the structure to toggle them; drag a lasso to add atoms, Alt-drag to remove.'
            : 'Loading the structure…'}
        </p>
        <div className="refine__selection">
          <span>{selection.size} atom{selection.size === 1 ? '' : 's'} selected</span>
          {selection.size > 0 && (
            <button type="button" className="refine__link" onClick={() => onSelection(new Set())}>Clear</button>
          )}
        </div>

        {pool.length >= 2 && (
          <div className="refine__cores">
            <span className="refine__cores-title">Shared core of the top {pool.length} retrievals:</span>
            {cores.isLoading && <span className="refine__muted">searching…</span>}
            {cores.isError && <span className="refine__muted">unavailable</span>}
            {cores.data && cores.data.cores.length === 0 && <span className="refine__muted">none found</span>}
            {cores.data?.cores.map((c) => (
              <button
                key={c.smarts}
                type="button"
                className="refine__chip"
                disabled={!c.atoms || !geometry}
                onClick={() => c.atoms && onSelection(new Set(c.atoms))}
                title={c.atoms
                  ? `Select this core (${c.num_atoms} atoms) on this compound${c.partial ? ' — search hit its time limit, a larger core may exist' : ''}`
                  : 'This compound does not contain this core'}
              >
                {coverageLabel(c.coverage, c.molecules_with_core, cores.data.molecules)} · {c.num_atoms} atoms
                {c.partial ? '*' : ''}
              </button>
            ))}
          </div>
        )}
      </section>

      <section className="refine__section">
        <h4 className="refine__heading">2. Search in a new tab</h4>
        <div className="refine__metric" role="radiogroup" aria-label="Scoring">
          {(['cosine', 'containment'] as RefineMetric[]).map((m) => (
            <label key={m} title={METRIC_HINT[m]} className={metric === m ? 'is-active' : ''}>
              <input type="radio" name="refine-metric" checked={metric === m} onChange={() => setMetric(m)} />
              {m === 'cosine' ? 'Cosine' : 'Containment'}
            </label>
          ))}
        </div>
        <div className="refine__actions">
          <button
            type="button"
            className="refine__btn refine__btn--delete"
            disabled={!atoms.length || duplicate}
            onClick={() => open('delete')}
            title="Remove the selected region from the query and retrieve again in a new tab"
          >
            Delete selection &amp; search ↗
          </button>
          <button
            type="button"
            className="refine__btn"
            disabled={!atoms.length || duplicate}
            onClick={() => open('require')}
            title="Retrieve again in a new tab, keeping only compounds that contain the selected substructure"
          >
            Require substructure &amp; search ↗
          </button>
        </div>
        {duplicate && <p className="refine__hint">This region is already one of this tab’s edits.</p>}
        {blockedHref && (
          <p className="refine__error">
            The browser blocked the new tab. <a href={blockedHref} target="_blank" rel="noopener">Open the search</a>.
          </p>
        )}
      </section>
    </div>
  )
}

export default RefinePanel
