import { useAppStore } from '../../store/store'
import { validateFormula, tokenizeFormula } from '../../services/formula'
import HelpButton from './HelpButton'
import { HELP } from '../../helpContent'
import './FormulaInput.css'

/**
 * Optional molecular-formula input. The <input> holds plain ASCII (e.g.
 * "C10H12N2O"); a live preview renders the counts as real subscripts and flags
 * an invalid formula. Formula is a descriptor, not a spectrum, so it never
 * satisfies the "at least one spectral input" requirement on its own.
 */
function FormulaInput() {
  const formula = useAppStore((s) => s.formula)
  const setFormula = useAppStore((s) => s.setFormula)

  const trimmed = formula.trim()
  const validation = validateFormula(formula)
  const tokens = tokenizeFormula(formula)

  return (
    <div className="main-page__mw-row">
      <label className="main-page__label">
        <span className="main-page__label-text">
          Molecular formula
          <HelpButton content={HELP.spectral.formula} placement="right" />
        </span>
        <input
          type="text"
          placeholder="Optional, e.g. C10H12N2O"
          value={formula}
          spellCheck={false}
          autoCapitalize="off"
          autoCorrect="off"
          className={trimmed && !validation.valid ? 'formula-input--invalid' : undefined}
          onChange={(e) => setFormula(e.target.value)}
        />
      </label>
      {trimmed && (
        <div className="formula-input__feedback">
          {validation.valid ? (
            <span className="formula-input__preview" aria-label="parsed formula">
              {tokens.map((t, i) => (
                <span key={i}>
                  {t.symbol}
                  {t.count && <sub>{t.count}</sub>}
                </span>
              ))}
            </span>
          ) : (
            <span className="formula-input__error">{validation.error}</span>
          )}
        </div>
      )}
    </div>
  )
}

export default FormulaInput
