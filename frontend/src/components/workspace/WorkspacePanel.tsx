import { useState } from 'react'
import { useAppStore } from '../../store/store'
import FragmentThumb from '../results/FragmentThumb'
import './WorkspacePanel.css'

/**
 * The hybrid-compound workspace: regions held fixed (applied to every refined
 * search while ticked) and every refined search of this page session. Sits above
 * the compound dialog so a held fragment can be glanced at while working on
 * another compound. Cleared by a new search, since both depend on its fingerprint.
 */
function WorkspacePanel() {
  const held = useAppStore((s) => s.heldFragments)
  const runs = useAppStore((s) => s.refineRuns)
  const toggleHeld = useAppStore((s) => s.toggleHeldFragment)
  const removeHeld = useAppStore((s) => s.removeHeldFragment)
  const removeRun = useAppStore((s) => s.removeRefineRun)
  const clearWorkspace = useAppStore((s) => s.clearWorkspace)
  const openDetail = useAppStore((s) => s.openDetail)
  // Starts collapsed so it never covers the dialog; the collapsed bar still shows
  // what is held, which is what needs to stay in view.
  const [open, setOpen] = useState(false)

  if (held.length === 0 && runs.length === 0) return null

  const summary = `${held.length} held · ${runs.length} search${runs.length === 1 ? '' : 'es'}`

  if (!open) {
    return (
      <button type="button" className="workspace__pill" onClick={() => setOpen(true)} title="Open the workspace">
        <span className="workspace__pill-label">Workspace</span>
        {held.map((h) => (
          <span key={h.id} className={`workspace__pill-thumb${h.enabled ? '' : ' is-off'}`} title={h.fragmentSmiles}>
            <FragmentThumb fragmentSmiles={h.fragmentSmiles} atomSymbol="" alt={h.fragmentSmiles} />
          </span>
        ))}
        <span className="workspace__pill-count">{summary}</span>
      </button>
    )
  }

  return (
    <aside className="workspace" aria-label="Workspace">
      <header className="workspace__header">
        <h3>Workspace</h3>
        <span className="workspace__summary">{summary}</span>
        <button type="button" className="workspace__icon" onClick={() => setOpen(false)} aria-label="Collapse">–</button>
      </header>

      <div className="workspace__body">
        <section>
          <h4 className="workspace__heading">Held fixed</h4>
          {held.length === 0 ? (
            <p className="workspace__empty">
              Nothing held. Use “Hold fixed &amp; search” on a compound to keep a region here.
            </p>
          ) : (
            <>
              <p className="workspace__note">Ticked regions are applied to every refined search.</p>
              <ul className="workspace__list">
                {held.map((h) => (
                  <li key={h.id} className={`workspace__item${h.enabled ? '' : ' is-off'}`}>
                    <input
                      type="checkbox"
                      checked={h.enabled}
                      onChange={() => toggleHeld(h.id)}
                      aria-label={`Apply ${h.fragmentSmiles}`}
                    />
                    <span className="workspace__thumb">
                      <FragmentThumb fragmentSmiles={h.fragmentSmiles} atomSymbol="" alt={h.fragmentSmiles} />
                    </span>
                    <button
                      type="button"
                      className="workspace__text"
                      onClick={() => openDetail({ card: h.source, label: h.sourceLabel })}
                      title={`${h.fragmentSmiles}\nfrom ${h.source.name || h.source.smiles}`}
                    >
                      <span className="workspace__frag">{h.fragmentSmiles}</span>
                      <span className="workspace__source">
                        {h.atoms.length} atoms · {h.sourceLabel} {h.source.name || ''}
                      </span>
                    </button>
                    <button type="button" className="workspace__icon" onClick={() => removeHeld(h.id)} aria-label="Remove">✕</button>
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>

        {runs.length > 0 && (
          <section>
            <h4 className="workspace__heading">Searches</h4>
            <ul className="workspace__list">
              {runs.slice().reverse().map((r) => {
                const own = r.operations[r.operations.length - 1]
                const top = r.results[0]
                return (
                  <li key={r.id} className="workspace__item">
                    <span className={`workspace__mode workspace__mode--${r.mode}`}>R{r.seq}</span>
                    <span className="workspace__thumb">
                      {own?.selection_smiles && (
                        <FragmentThumb fragmentSmiles={own.selection_smiles} atomSymbol="" alt={own.selection_smiles} />
                      )}
                    </span>
                    <button
                      type="button"
                      className="workspace__text"
                      onClick={() => openDetail({ card: r.source, label: r.sourceLabel, runId: r.id })}
                    >
                      <span className="workspace__frag">
                        {r.mode === 'delete' ? 'Deleted' : 'Held'} on {r.sourceLabel} · {r.metric}
                        {r.heldApplied.length > 0 ? ` · +${r.heldApplied.length} held` : ''}
                      </span>
                      <span className="workspace__source">
                        top: {top ? (top.name || top.smiles) : '—'}
                      </span>
                    </button>
                    <button type="button" className="workspace__icon" onClick={() => removeRun(r.id)} aria-label="Remove">✕</button>
                  </li>
                )
              })}
            </ul>
          </section>
        )}
      </div>

      <footer className="workspace__footer">
        <button type="button" className="workspace__clear" onClick={clearWorkspace}>Clear workspace</button>
      </footer>
    </aside>
  )
}

export default WorkspacePanel
