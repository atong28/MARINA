import { useAppStore } from '../../store/store'
import { validateFormula, toSubscript, fromSubscript } from '../../services/formula'
import HelpButton from './HelpButton'
import { HELP } from '../../helpContent'
import './FormulaInput.css'

/**
 * Optional molecular-formula input, WYSIWYG: digits are shown as subscripts right
 * in the field (C₁₀H₁₂N₂O) while the store keeps plain ASCII. Formula is a
 * descriptor, not a spectrum, so it never satisfies the "≥1 spectral input" rule.
 */
function FormulaInput() {
  const formula = useAppStore((s) => s.formula)
  const setFormula = useAppStore((s) => s.setFormula)

  const validation = validateFormula(formula)
  const invalid = formula.trim() !== '' && !validation.valid

  return (
    <label className="main-page__label formula-input">
      <span className="main-page__label-text">
        Molecular formula
        <HelpButton content={HELP.spectral.formula} placement="right" />
      </span>
      <input
        type="text"
        className={`formula-input__field${invalid ? ' formula-input--invalid' : ''}`}
        placeholder="Optional, e.g. C10H12N2O"
        // Show subscripts; store ASCII. The map is 1:1 so the caret stays put.
        value={toSubscript(formula)}
        spellCheck={false}
        autoCapitalize="off"
        autoCorrect="off"
        onChange={(e) => setFormula(fromSubscript(e.target.value))}
      />
      {invalid && <span className="formula-input__error">{validation.error}</span>}
    </label>
  )
}

export default FormulaInput
