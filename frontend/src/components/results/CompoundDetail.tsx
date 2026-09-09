import { useCallback, useEffect, useRef, useState } from 'react'
import { BitExplanation, DepictionGeometry, ResultCard as ResultCardType } from '../../services/api'
import { useAppStore } from '../../store/store'
import BitPanel from './BitPanel'
import FragmentOverlay from './FragmentOverlay'
import MoleculeViewer from './MoleculeViewer'
import { pickDepiction } from './ResultCard'
import './CompoundDetail.css'

/**
 * Expanded view of one candidate, as an overlay above the results grid.
 *
 * Kept as an overlay rather than a route so the search results underneath stay
 * mounted — reopening a compound after closing costs nothing, and the browser
 * Back button closes it via a pushed history entry.
 */

const COSINE_HINT =
  'Cosine similarity between the query fingerprint and this compound’s fingerprint.'
const TANIMOTO_HINT =
  'Generalised Tanimoto similarity. The query fingerprint holds predicted ' +
  'probabilities rather than 0/1 bits, so this is not the usual binary Tanimoto ' +
  'coefficient and is not comparable to one.'

interface CompoundDetailProps {
  result: ResultCardType
  position: number
  isCustom?: boolean
  predFp: number[]
  modelId?: string
  onClose: () => void
}

function CompoundDetail({
  result, position, isCustom = false, predFp, modelId, onClose,
}: CompoundDetailProps) {
  const [selected, setSelected] = useState<BitExplanation | null>(null)
  const [geometry, setGeometry] = useState<DepictionGeometry | null>(null)
  const closeRef = useRef<HTMLButtonElement>(null)
  const highlightEnabled = useAppStore((s) => s.highlightEnabled)

  const { database_links } = result
  const cosine = result.cosine_similarity ?? result.similarity
  const tanimoto = result.tanimoto_similarity
  const hasLinks = Boolean(
    database_links?.coconut || database_links?.lotus || database_links?.npmrd,
  )

  const npc = result.npclassifier
  const npcTiers: Array<[string, string[]]> = npc
    ? [['Pathway', npc.pathway], ['Superclass', npc.superclass], ['Class', npc.npclass]]
    : []

  // Esc closes, and the overlay owns focus while it is open.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    closeRef.current?.focus()
    const prevOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prevOverflow
    }
  }, [onClose])

  // Push a history entry so the browser Back button closes the overlay rather
  // than navigating away from the results the user just waited for.
  useEffect(() => {
    window.history.pushState({ marinaDetail: true }, '')
    const onPop = () => onClose()
    window.addEventListener('popstate', onPop)
    return () => {
      window.removeEventListener('popstate', onPop)
      // Closing via Esc or the button leaves our entry on the stack; drop it so
      // Back does not have to be pressed twice.
      if (window.history.state?.marinaDetail) window.history.back()
    }
  }, [onClose])

  const onBackdrop = useCallback(
    (e: React.MouseEvent) => { if (e.target === e.currentTarget) onClose() },
    [onClose],
  )

  // The same layers the card behind this overlay is showing, toggle included;
  // a selected bit is drawn over them rather than replacing them, so the map
  // toggle keeps working while a substructure is picked out.
  const layers = pickDepiction(result, highlightEnabled)
  const occurrences = selected?.occurrences ?? []

  return (
    <div className="compound-detail__backdrop" onClick={onBackdrop} role="presentation">
      <div
        className="compound-detail"
        role="dialog"
        aria-modal="true"
        aria-label={result.name || result.smiles}
      >
        <header className="compound-detail__header">
          <div className="compound-detail__title-block">
            <span className="compound-detail__rank">{isCustom ? 'Custom' : `#${position}`}</span>
            <h2 className="compound-detail__name">{result.name || result.smiles}</h2>
          </div>
          <button
            ref={closeRef}
            type="button"
            className="compound-detail__close"
            onClick={onClose}
            aria-label="Close"
          >
            ✕
          </button>
        </header>

        <div className="compound-detail__body">
          <div className="compound-detail__left">
            <div className="compound-detail__figure">
              <MoleculeViewer plainSvg={layers.plain} mapSrc={layers.map} smiles={result.smiles}>
                {geometry && occurrences.length > 0 && (
                  <FragmentOverlay geometry={geometry} occurrences={occurrences} />
                )}
              </MoleculeViewer>
            </div>

            {selected ? (
              <div className="compound-detail__caption">
                Showing <code>{selected.fragment_smiles || `${selected.atom_symbol} atom`}</code>
                {' '}(bit #{selected.index}) — {occurrences.length} occurrence
                {occurrences.length === 1 ? '' : 's'}
                {occurrences.length > 1 ? '; hover one to single it out' : ''}.{' '}
                <button type="button" className="compound-detail__clear" onClick={() => setSelected(null)}>
                  Clear
                </button>
              </div>
            ) : (
              <div className="compound-detail__caption compound-detail__caption--muted">
                Select a substructure on the right to locate it here.
              </div>
            )}

            <dl className="compound-detail__facts">
              <div><dt title={COSINE_HINT}>Cosine</dt><dd>{cosine.toFixed(3)}</dd></div>
              {typeof tanimoto === 'number' && (
                <div><dt title={TANIMOTO_HINT}>Tanimoto</dt><dd>{tanimoto.toFixed(3)}</dd></div>
              )}
              {result.exact_mass != null && (
                <div><dt>Exact mass</dt><dd>{result.exact_mass.toFixed(4)} Da</dd></div>
              )}
            </dl>

            {npc && (
              <div className="compound-detail__npclass">
                <h4 className="compound-detail__npclass-title">
                  NPClassifier
                  {npc.isglycoside && (
                    <span className="compound-detail__npclass-glyco">glycoside</span>
                  )}
                </h4>
                {npcTiers.some(([, values]) => values.length > 0) ? (
                  <dl className="compound-detail__facts">
                    {npcTiers.map(([label, values]) => (
                      <div key={label}>
                        <dt>{label}</dt>
                        <dd>{values.length ? values.join(', ') : '—'}</dd>
                      </div>
                    ))}
                  </dl>
                ) : (
                  <div className="compound-detail__caption compound-detail__caption--muted">
                    Not classified.
                  </div>
                )}
              </div>
            )}

            <code className="compound-detail__smiles">{result.smiles}</code>

            {!isCustom && hasLinks && (
              <div className="compound-detail__links">
                {database_links.coconut && (
                  <a href={database_links.coconut} target="_blank" rel="noopener noreferrer"
                     className="result-card__link result-card__link--coconut">COCONUT</a>
                )}
                {database_links.lotus && (
                  <a href={database_links.lotus} target="_blank" rel="noopener noreferrer"
                     className="result-card__link result-card__link--lotus">LOTUS</a>
                )}
                {database_links.npmrd && (
                  <a href={database_links.npmrd} target="_blank" rel="noopener noreferrer"
                     className="result-card__link result-card__link--npmrd">NPMRD</a>
                )}
              </div>
            )}
          </div>

          <div className="compound-detail__right">
            <h3 className="compound-detail__section-title">Substructures</h3>
            <BitPanel
              smiles={result.smiles}
              predFp={predFp}
              modelId={modelId}
              onSelect={setSelected}
              onGeometry={setGeometry}
              selectedIndex={selected?.index ?? null}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

export default CompoundDetail
