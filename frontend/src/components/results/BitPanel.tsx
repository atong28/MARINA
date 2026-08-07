import { memo, useCallback, useEffect, useState } from 'react'
import { api, BitExplanation, BitGroup, BitExplainResponse } from '../../services/api'
import './BitPanel.css'

/**
 * Per-bit substructure breakdown for one candidate structure.
 *
 * Ordered by disagreement rather than by confidence: a molecule has ~56 bits
 * predicted above 0.9, so a confidence-sorted list is nearly all confident
 * matches. The short list of confident bits this structure *lacks* is what
 * actually separates one candidate from another, so it leads.
 */

const GROUP_LABEL: Record<BitGroup, string> = {
  missing: 'Predicted, but not in this structure',
  match: 'Predicted and present',
  unexpected: 'Present, but not predicted',
  uncertain: 'Uncertain',
}

const GROUP_HINT: Record<BitGroup, string> = {
  missing: 'The model expects these substructures with confidence, but this candidate does not contain them.',
  match: 'The model expects these and this candidate has them.',
  unexpected: 'This candidate contains these, but the model did not predict them.',
  uncertain: 'The model is not confident either way.',
}

const GROUP_ORDER: BitGroup[] = ['missing', 'match', 'unexpected', 'uncertain']

const CALIBRATION_HINT =
  'Calibrated against the MARINA1 test split, so the percentage is the measured ' +
  'fraction of the time a substructure predicted at this confidence is really present.'

const RAW_HINT =
  'This model ships no calibration curve, so these are raw model outputs. They ' +
  'overstate presence — a bit shown at 82% is present about 70% of the time.'

/** Radius-0 bits have no fragment SMILES; they are a bare atom. */
function bitLabel(bit: BitExplanation): string {
  return bit.fragment_smiles || `${bit.atom_symbol} atom`
}

/**
 * The fitted curve tops out at 0.9984, so nothing in the data supports telling a
 * user a substructure is certainly present. Plain rounding would turn that into
 * "100%", so the top of the range keeps a decimal place instead.
 */
export function formatPct(p: number): string {
  const pct = p * 100
  if (pct >= 99.5) return `${Math.floor(pct * 10) / 10}%`
  return `${Math.round(pct)}%`
}

interface BitPanelProps {
  smiles: string
  predFp: number[]
  modelId?: string
  /** Called with the atoms/bonds to light up, or null to clear the highlight. */
  onSelect: (bit: BitExplanation | null) => void
  selectedIndex: number | null
}

function BitPanel({ smiles, predFp, modelId, onSelect, selectedIndex }: BitPanelProps) {
  const [data, setData] = useState<BitExplainResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    api
      .explainBits({ smiles, pred_fp: predFp, model_id: modelId, limit: 60, include_fragment_svg: true })
      .then((res) => { if (!cancelled) setData(res) })
      .catch((err) => { if (!cancelled) setError(err instanceof Error ? err.message : String(err)) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [smiles, predFp, modelId])

  const toggle = useCallback(
    (bit: BitExplanation) => {
      onSelect(selectedIndex === bit.index ? null : bit)
    },
    [onSelect, selectedIndex],
  )

  if (loading) return <div className="bit-panel__status">Analysing substructures…</div>
  if (error) return <div className="bit-panel__status bit-panel__status--error">{error}</div>
  if (!data || data.bits.length === 0) {
    return <div className="bit-panel__status">No substructures to report.</div>
  }

  const hidden = data.total_available - data.total_shown

  return (
    <div className="bit-panel">
      <div
        className={`bit-panel__calibration${data.calibrated ? '' : ' bit-panel__calibration--raw'}`}
        title={data.calibrated ? CALIBRATION_HINT : RAW_HINT}
      >
        {data.calibrated ? 'Calibrated confidence' : 'Uncalibrated — confidences overstate presence'}
      </div>

      {GROUP_ORDER.map((group) => {
        const bits = data.bits.filter((b) => b.group === group)
        if (bits.length === 0) return null
        return (
          <section key={group} className={`bit-panel__group bit-panel__group--${group}`}>
            <h4 className="bit-panel__group-title" title={GROUP_HINT[group]}>
              {GROUP_LABEL[group]}
              <span className="bit-panel__group-count">{data.totals[group]}</span>
            </h4>
            <ul className="bit-panel__list">
              {bits.map((bit) => (
                <li key={bit.index}>
                  <button
                    type="button"
                    className={`bit-panel__row${selectedIndex === bit.index ? ' bit-panel__row--selected' : ''}`}
                    onClick={() => toggle(bit)}
                    // A bit this structure lacks has nowhere to be highlighted.
                    disabled={!bit.present}
                    title={
                      bit.present
                        ? `Show where this sits in the structure (bit ${bit.index}, radius ${bit.radius})`
                        : `Not present in this structure (bit ${bit.index}, radius ${bit.radius})`
                    }
                  >
                    {bit.fragment_svg ? (
                      <img
                        className="bit-panel__thumb"
                        src={`data:image/svg+xml,${encodeURIComponent(bit.fragment_svg)}`}
                        alt={bitLabel(bit)}
                      />
                    ) : (
                      <span className="bit-panel__thumb bit-panel__thumb--empty" />
                    )}
                    <span className="bit-panel__labels">
                      <code className="bit-panel__frag">{bitLabel(bit)}</code>
                      <span className="bit-panel__meta">r{bit.radius}·#{bit.index}</span>
                    </span>
                    <span className="bit-panel__band">{bit.band}</span>
                    <span className="bit-panel__pct">{formatPct(bit.confidence)}</span>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )
      })}

      {hidden > 0 && (
        <div className="bit-panel__truncated">
          + {hidden} more not shown
        </div>
      )}

      {/*
        36 distinct bits carry the label "CCCCC", so the same string can appear
        as both present and absent. Without this note that reads as a
        contradiction rather than as two different environments.
      */}
      <p className="bit-panel__footnote">
        The formula is an abbreviated label — each bit also encodes bond orders and
        substitution around its centre, so the same text can name different bits
        (shown as <code>r</code>adius·<code>#</code>index).
      </p>
    </div>
  )
}

export default memo(BitPanel)
