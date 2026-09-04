import { useAppStore } from '../../store/store'
import { formulaFilterEntryError } from '../../services/formula'
import HelpButton from './HelpButton'
import { HELP } from '../../helpContent'
import './FormulaFilter.css'

/**
 * Retrieval filter: constrain candidates by per-element atom counts, each as an
 * inclusive range (element between low and high). Either bound may be left blank
 * for no bound on that side. Applied server-side after retrieval, intersected with
 * the MW range filter. Blank rows are ignored.
 */
function FormulaFilter() {
  const entries = useAppStore((s) => s.formulaFilter)
  const setEntries = useAppStore((s) => s.setFormulaFilter)

  const update = (i: number, patch: Partial<(typeof entries)[number]>) =>
    setEntries(entries.map((e, j) => (j === i ? { ...e, ...patch } : e)))
  const remove = (i: number) => setEntries(entries.filter((_, j) => j !== i))
  const add = () => setEntries([...entries, { element: '', low: '', high: '' }])

  return (
    <div className="formula-filter">
      <span className="formula-filter__label">
        Retrieval formula filter (atom counts)
        <HelpButton content={HELP.spectral.formulaFilter} placement="right" />
      </span>

      {entries.map((e, i) => {
        const err = formulaFilterEntryError(e)
        return (
          <div key={i} className="formula-filter__row">
            <input
              className={`formula-filter__el${err ? ' formula-input--invalid' : ''}`}
              placeholder="El"
              value={e.element}
              spellCheck={false}
              autoCapitalize="off"
              autoCorrect="off"
              onChange={(ev) => update(i, { element: ev.target.value })}
            />
            <span className="formula-filter__pm">between</span>
            <input
              type="number" min="0" step="1"
              placeholder="low"
              value={e.low}
              onChange={(ev) => update(i, { low: ev.target.value })}
            />
            <span className="formula-filter__pm">and</span>
            <input
              type="number" min="0" step="1"
              placeholder="high"
              value={e.high}
              onChange={(ev) => update(i, { high: ev.target.value })}
            />
            <button
              className="formula-filter__remove"
              onClick={() => remove(i)}
              title="Remove this constraint"
              aria-label="Remove constraint"
            >
              ×
            </button>
            {err && <span className="formula-filter__error">{err}</span>}
          </div>
        )
      })}

      <button className="formula-filter__add" onClick={add}>+ Add element</button>
    </div>
  )
}

export default FormulaFilter
