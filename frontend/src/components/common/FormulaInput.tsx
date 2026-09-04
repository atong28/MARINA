import { useEffect, useRef } from 'react'
import { useAppStore } from '../../store/store'
import { validateFormula } from '../../services/formula'
import HelpButton from './HelpButton'
import { HELP } from '../../helpContent'
import './FormulaInput.css'

/**
 * Optional molecular-formula input. A contentEditable field renders count digits as
 * real <sub> elements — browser-native subscripts at proper size and position, so
 * nothing is tiny or clipped — while the store keeps plain ASCII ("C10H12N2O").
 * Formula is a descriptor, not a spectrum, so it never satisfies the "≥1 spectral
 * input" rule.
 */

/** ASCII formula → HTML with digit runs wrapped in real subscripts. */
function toHtml(ascii: string): string {
  const esc = ascii.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  return esc.replace(/[0-9]+/g, (d) => `<sub>${d}</sub>`)
}

/** Plain-text caret offset within a contentEditable element. */
function caretOffset(el: HTMLElement): number {
  const sel = window.getSelection()
  if (!sel || sel.rangeCount === 0) return 0
  const range = sel.getRangeAt(0)
  const pre = range.cloneRange()
  pre.selectNodeContents(el)
  pre.setEnd(range.endContainer, range.endOffset)
  return pre.toString().length
}

/** Restore the caret to a plain-text offset after re-rendering the HTML. */
function setCaret(el: HTMLElement, offset: number): void {
  const sel = window.getSelection()
  if (!sel) return
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT)
  let remaining = offset
  let node = walker.nextNode()
  while (node) {
    const len = node.textContent?.length ?? 0
    if (remaining <= len) {
      const range = document.createRange()
      range.setStart(node, remaining)
      range.collapse(true)
      sel.removeAllRanges()
      sel.addRange(range)
      return
    }
    remaining -= len
    node = walker.nextNode()
  }
  const range = document.createRange()
  range.selectNodeContents(el)
  range.collapse(false)
  sel.removeAllRanges()
  sel.addRange(range)
}

function FormulaInput() {
  const formula = useAppStore((s) => s.formula)
  const setFormula = useAppStore((s) => s.setFormula)
  const ref = useRef<HTMLDivElement>(null)

  // Sync the DOM when the store changes from outside typing (example load / reset).
  useEffect(() => {
    const el = ref.current
    if (el && (el.textContent ?? '') !== formula) {
      el.innerHTML = toHtml(formula)
    }
  }, [formula])

  const onInput = () => {
    const el = ref.current
    if (!el) return
    const ascii = el.textContent ?? ''
    const off = caretOffset(el)
    el.innerHTML = toHtml(ascii)      // re-wrap digits as they're typed
    setCaret(el, off)
    setFormula(ascii)
  }

  const validation = validateFormula(formula)
  const invalid = formula.trim() !== '' && !validation.valid

  return (
    <div className="main-page__label">
      <span className="main-page__label-text">
        Molecular formula
        <HelpButton content={HELP.spectral.formula} placement="right" />
      </span>
      <div
        ref={ref}
        className={`formula-input__field${invalid ? ' formula-input--invalid' : ''}`}
        contentEditable
        suppressContentEditableWarning
        role="textbox"
        spellCheck={false}
        data-placeholder="Optional, e.g. C10H12N2O"
        onInput={onInput}
      />
      {invalid && <span className="formula-input__error">{validation.error}</span>}
    </div>
  )
}

export default FormulaInput
