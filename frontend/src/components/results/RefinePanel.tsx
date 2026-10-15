import { useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  api, RefineMetric, RefineMode, RefineOperation, RefineResponse, ResultCard,
} from '../../services/api'
import { cardKey, RefineRun, useAppStore } from '../../store/store'
import FragmentThumb from './FragmentThumb'
import MoleculeViewer from './MoleculeViewer'
import './RefinePanel.css'

/**
 * Hybrid compounds: delete a region of this candidate from the query, or hold it
 * fixed, and retrieve again. A hybrid A+B's retrievals are often all A-like, so B
 * never shows; taking A out of the query lets the rest of the prediction speak.
 * The arithmetic is server-side (backend/app/refine.py).
 */

const MAX_CORE_POOL = 50
const REFINE_K = 10

const METRIC_HINT: Record<RefineMetric, string> = {
  cosine: 'Cosine: compounds that look like what is left. Best when the missing part exists as a compound of its own.',
  containment:
    'Containment: compounds that contain what is left, without penalising their other parts. ' +
    'Use when the missing part only occurs inside larger molecules.',
}

const MODE_LABEL: Record<RefineMode, string> = { delete: 'Deleted', hold: 'Held fixed' }

function coverageLabel(coverage: number, found: number, total: number): string {
  return coverage >= 1 ? `All ${total}` : `${found} of ${total}`
}

interface RefinePanelProps {
  card: ResultCard
  label: string
  predFp: number[]
  modelId?: string
  selection: Set<number>
  onSelection: (next: Set<number>) => void
  /** False until the depiction geometry has loaded; atoms cannot be picked before. */
  geometryReady: boolean
  /** Run to show expanded (opened from the workspace). */
  focusRunId?: string
}

function RefinePanel({
  card, label, predFp, modelId, selection, onSelection, geometryReady, focusRunId,
}: RefinePanelProps) {
  const results = useAppStore((s) => s.results)
  const heldFragments = useAppStore((s) => s.heldFragments)
  const refineRuns = useAppStore((s) => s.refineRuns)
  const addHeldFragment = useAppStore((s) => s.addHeldFragment)
  const addRefineRun = useAppStore((s) => s.addRefineRun)
  const openDetail = useAppStore((s) => s.openDetail)
  const [metric, setMetric] = useState<RefineMetric>('cosine')

  const pool = useMemo(() => results.slice(0, MAX_CORE_POOL).map((r) => r.smiles), [results])
  const cores = useQuery({
    queryKey: ['shared-core', pool, card.smiles],
    queryFn: () => api.sharedCore({ smiles: pool, target: card.smiles }),
    enabled: pool.length >= 2,
    staleTime: Infinity,
    retry: false,
  })

  const activeHolds = heldFragments.filter((h) => h.enabled)
  const myRuns = useMemo(
    () => refineRuns.filter((r) => cardKey(r.source) === cardKey(card)).slice().reverse(),
    [refineRuns, card],
  )

  // A held region that is also the region being searched is applied once, not twice.
  const isSameRegion = (h: { source: ResultCard; atoms: number[] }, atoms: number[]) =>
    cardKey(h.source) === cardKey(card) && h.atoms.length === atoms.length && h.atoms.every((a, i) => a === atoms[i])

  const search = useMutation<RefineResponse, Error, { mode: RefineMode; atoms: number[] }>({
    mutationFn: ({ mode, atoms }) => {
      const operations: RefineOperation[] = [
        ...activeHolds
          .filter((h) => !isSameRegion(h, atoms))
          .map((h) => ({ smiles: h.source.smiles, atoms: h.atoms, mode: 'hold' as const })),
        { smiles: card.smiles, atoms, mode },
      ]
      return api.refineSearch({ pred_fp: predFp, operations, metric, k: REFINE_K, model_id: modelId })
    },
    onSuccess: (res, { mode, atoms }) => {
      const own = res.operations[res.operations.length - 1]
      if (mode === 'hold') {
        if (!heldFragments.some((h) => isSameRegion(h, atoms))) {
          addHeldFragment({ source: card, sourceLabel: label, atoms, fragmentSmiles: own?.selection_smiles ?? '' })
        }
        onSelection(new Set())
      }
      addRefineRun({
        seq: refineRuns.reduce((m, r) => Math.max(m, r.seq), 0) + 1,
        source: card,
        sourceLabel: label,
        mode,
        metric: res.metric,
        atoms,
        operations: res.operations,
        heldApplied: activeHolds.filter((h) => !isSameRegion(h, atoms)).map((h) => h.fragmentSmiles),
        results: res.results,
        unexplained: res.unexplained,
        confidentBefore: res.confident_before,
        confidentAfter: res.confident_after,
      })
    },
  })

  const run = (mode: RefineMode) => {
    const atoms = [...selection].sort((a, b) => a - b)
    if (atoms.length) search.mutate({ mode, atoms })
  }

  const openResult = (r: RefineRun, result: ResultCard, i: number) =>
    openDetail({
      card: result,
      label: `R${r.seq} · #${i + 1}`,
      scoreLabel: r.metric === 'containment' ? 'Containment' : 'Cosine (refined)',
    })

  return (
    <div className="refine">
      <p className="refine__intro">
        Select a region of this compound — for example the part every retrieval agrees on — then delete it to
        search for what the rest of the prediction describes, or hold it fixed to compare compounds only on
        what lies beyond it.
      </p>

      <section className="refine__section">
        <h4 className="refine__heading">1. Select a region</h4>
        <p className="refine__hint">
          {geometryReady
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
                disabled={!c.atoms || !geometryReady}
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
        <h4 className="refine__heading">2. Search</h4>
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
            disabled={!selection.size || search.isPending}
            onClick={() => run('delete')}
            title="Remove the selected region from the query and retrieve again"
          >
            Delete selection &amp; search
          </button>
          <button
            type="button"
            className="refine__btn"
            disabled={!selection.size || search.isPending}
            onClick={() => run('hold')}
            title="Accept the region as present: keep it in the workspace and compare compounds on what lies beyond it"
          >
            Hold fixed &amp; search
          </button>
        </div>
        {activeHolds.length > 0 && (
          <p className="refine__hint">
            {activeHolds.length} held fragment{activeHolds.length === 1 ? '' : 's'} from the workspace
            {' '}apply to every search.
          </p>
        )}
        {search.isPending && <p className="refine__muted">Searching…</p>}
        {search.isError && <p className="refine__error">{search.error.message}</p>}
      </section>

      {myRuns.length > 0 && (
        <section className="refine__section">
          <h4 className="refine__heading">Searches from this compound</h4>
          {myRuns.map((r, i) => (
            <RunView
              key={r.id}
              run={r}
              defaultOpen={focusRunId ? r.id === focusRunId : i === 0}
              onOpen={(res, k) => openResult(r, res, k)}
            />
          ))}
        </section>
      )}
    </div>
  )
}

interface RunViewProps {
  run: RefineRun
  defaultOpen: boolean
  onOpen: (card: ResultCard, i: number) => void
}

export function RunView({ run, defaultOpen, onOpen }: RunViewProps) {
  const own = run.operations[run.operations.length - 1]
  return (
    <details className="refine-run" open={defaultOpen}>
      <summary className="refine-run__summary">
        <span className="refine-run__seq">R{run.seq}</span>
        <span className={`refine-run__mode refine-run__mode--${run.mode}`}>{MODE_LABEL[run.mode]}</span>
        {own?.selection_smiles && (
          <span className="refine-run__frag">
            <FragmentThumb fragmentSmiles={own.selection_smiles} atomSymbol="" alt={own.selection_smiles} />
          </span>
        )}
        <span className="refine-run__meta">
          {run.metric === 'containment' ? 'containment' : 'cosine'}
          {run.heldApplied.length > 0 && ` · +${run.heldApplied.length} held`}
          {' · '}confident bits {run.confidentBefore} → {run.confidentAfter}
        </span>
      </summary>

      <ol className="refine-run__results">
        {run.results.map((res, i) => (
          <li key={cardKey(res)}>
            <button type="button" className="refine-run__result" onClick={() => onOpen(res, i)}>
              <span className="refine-run__rank">#{i + 1}</span>
              <span className="refine-run__thumb">
                <MoleculeViewer plainSvg={res.plain_svg} smiles={res.smiles} />
              </span>
              <span className="refine-run__name" title={res.name || res.smiles}>{res.name || res.smiles}</span>
              <span className="refine-run__score">{res.similarity.toFixed(3)}</span>
            </button>
          </li>
        ))}
      </ol>

      {run.unexplained.length > 0 && (
        <details className="refine-run__unexplained">
          <summary>Left unexplained ({run.unexplained.length} confident fragments)</summary>
          <p className="refine__hint">
            Substructures the prediction still expects after the edit — what the rest of the true structure
            should contain.
          </p>
          <div className="refine-run__frags">
            {run.unexplained.map((u) => (
              <figure key={u.index} className="refine-run__fragcell" title={u.fragment_smiles}>
                <FragmentThumb fragmentSmiles={u.fragment_smiles} atomSymbol="" alt={u.fragment_smiles} />
                <figcaption>
                  {u.multiplicity > 1 ? `≥${u.multiplicity}× · ` : ''}{Math.round(u.confidence * 100)}%
                </figcaption>
              </figure>
            ))}
          </div>
        </details>
      )}
    </details>
  )
}

export default RefinePanel
