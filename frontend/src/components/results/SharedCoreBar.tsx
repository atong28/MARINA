import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../../services/api'
import { bondsWithin } from '../../services/atomSelection'
import type { CoreOverlay } from './ResultCard'
import './SharedCoreBar.css'

/**
 * The largest substructure the retrieved compounds share, drawn on every card
 * that has it. Computed on request (a maximum-common-substructure search over up
 * to 50 compounds takes seconds), at several coverage levels down to half the
 * results: a larger core shared by fewer compounds, or a smaller one shared by all.
 */

const MAX_POOL = 50

interface SharedCoreBarProps {
  smiles: string[]
  /** One overlay per compound (null where it lacks the core), or null when nothing is highlighted. */
  onOverlays: (overlays: Array<CoreOverlay | null> | null) => void
}

function SharedCoreBar({ smiles, onOverlays }: SharedCoreBarProps) {
  const pool = useMemo(() => smiles.slice(0, MAX_POOL), [smiles])
  const [requested, setRequested] = useState(false)
  // null until the first result arrives; 'off' once the user turns the highlight off.
  const [selected, setSelected] = useState<number | 'off' | null>(null)

  const query = useQuery({
    queryKey: ['shared-core-matches', pool],
    queryFn: () => api.sharedCore({ smiles: pool, matches: true }),
    enabled: requested && pool.length >= 2,
    staleTime: Infinity,
    retry: false,
  })

  // New results: start over.
  useEffect(() => {
    setRequested(false)
    setSelected(null)
  }, [pool])

  // Highlight the largest core by default once found.
  useEffect(() => {
    if (query.data && selected === null && query.data.cores.length) setSelected(0)
  }, [query.data, selected])

  useEffect(() => {
    const data = query.data
    const core = typeof selected === 'number' ? data?.cores[selected] : undefined
    if (!data || !core?.matches || !data.geometries) {
      onOverlays(null)
      return
    }
    onOverlays(core.matches.map((copies, i) => {
      const geometry = data.geometries?.[i]
      if (!copies || !geometry) return null
      return {
        geometry,
        occurrences: copies.map((atoms) => ({ atoms, bonds: bondsWithin(geometry.bonds, new Set(atoms)) })),
      }
    }))
  }, [query.data, selected, onOverlays])

  if (pool.length < 2) return null

  const total = query.data?.molecules ?? pool.length
  return (
    <div className="shared-core">
      <span className="shared-core__title" title="Largest substructure the retrieved compounds share, complete rings only">
        Shared core
      </span>
      {!requested && (
        <button type="button" className="shared-core__find" onClick={() => setRequested(true)}>
          Find in the top {pool.length}
        </button>
      )}
      {requested && query.isFetching && <span className="shared-core__muted">searching…</span>}
      {query.isError && <span className="shared-core__muted">unavailable</span>}
      {query.data && query.data.cores.length === 0 && <span className="shared-core__muted">none found</span>}
      {query.data && query.data.cores.length > 0 && (
        <>
          <button
            type="button"
            className={`shared-core__chip${selected === 'off' ? ' is-active' : ''}`}
            onClick={() => setSelected('off')}
          >
            Off
          </button>
          {query.data.cores.map((c, i) => (
            <button
              key={c.smarts}
              type="button"
              className={`shared-core__chip${selected === i ? ' is-active' : ''}`}
              onClick={() => setSelected(i)}
              title={c.partial ? 'The search hit its time limit; a larger core may exist' : undefined}
            >
              {c.coverage >= 1 ? `All ${total}` : `${c.molecules_with_core} of ${total}`} · {c.num_atoms} atoms
              {c.partial ? '*' : ''}
            </button>
          ))}
        </>
      )}
    </div>
  )
}

export default SharedCoreBar
