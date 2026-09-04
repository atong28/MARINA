// Molecular-formula parsing, validation, and subscript rendering for the formula input.
// Mirrors the backend's Hill-notation parse (src/modules/data/formula.py): an element
// symbol (one uppercase + optional lowercase) followed by an optional count. Unknown
// elements fold into the model's '*' catch-all server-side, but we reject typos up front.

// Real element symbols (periodic table, H..Og).
export const ELEMENTS = new Set([
  'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
  'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
  'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
  'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
  'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
  'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
  'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
  'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
  'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
  'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
  'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
  'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
])

// WYSIWYG subscripting: the input shows counts as Unicode subscripts while the
// store keeps plain ASCII ("C10H12N2O"). Convert on the way in/out of the field.
const SUB = '₀₁₂₃₄₅₆₇₈₉'
export function toSubscript(ascii: string): string {
  return ascii.replace(/[0-9]/g, (d) => SUB[+d])
}
export function fromSubscript(text: string): string {
  return text.replace(/[₀-₉]/g, (c) => String(SUB.indexOf(c)))
}

const TOKEN_RE = /([A-Z][a-z]?)(\d*)/g
// The whole string must be element+optional-count groups, nothing else.
const SHAPE_RE = /^([A-Z][a-z]?\d*)+$/

export interface FormulaToken { symbol: string; count: string }

/** Split a formula into (symbol, count) tokens for rendering. */
export function tokenizeFormula(formula: string): FormulaToken[] {
  return [...formula.trim().matchAll(TOKEN_RE)]
    .filter((m) => m[1])
    .map((m) => ({ symbol: m[1], count: m[2] }))
}

// ── Atom-count filter helpers ─────────────────────────────────────────────────

import type { FormulaConstraint } from './api'

export interface FilterEntryInput { element: string; low: string; high: string }

function intOrNull(s: string): number | null {
  const t = s.trim()
  return t === '' ? null : Number(t)
}

/** Validation for one atom-count filter row (element between low..high, inclusive).
 *  Null = ok; a fully blank row is ok (ignored). Either bound may be left blank. */
export function formulaFilterEntryError(e: FilterEntryInput): string | null {
  const el = e.element.trim()
  const low = e.low.trim()
  const high = e.high.trim()
  if (el === '' && low === '' && high === '') return null   // blank row, ignored
  if (!ELEMENTS.has(el)) return `Unknown element “${el || '?'}”.`
  const lo = intOrNull(low)
  const hi = intOrNull(high)
  if (lo !== null && (!Number.isInteger(lo) || lo < 0)) return 'Low must be a non-negative integer.'
  if (hi !== null && (!Number.isInteger(hi) || hi < 0)) return 'High must be a non-negative integer.'
  if (lo === null && hi === null) return 'Enter a low and/or high bound.'
  if (lo !== null && hi !== null && lo > hi) return 'Low cannot exceed high.'
  return null
}

/** Convert filter rows to backend constraints. A blank bound means unbounded on that side. */
export function buildFormulaConstraints(entries: FilterEntryInput[]): FormulaConstraint[] {
  const out: FormulaConstraint[] = []
  for (const e of entries) {
    if (formulaFilterEntryError(e) !== null) continue
    const el = e.element.trim()
    const lo = intOrNull(e.low)
    const hi = intOrNull(e.high)
    if (!el || (lo === null && hi === null)) continue
    const c: FormulaConstraint = { element: el }
    if (lo !== null) c.min = lo
    if (hi !== null) c.max = hi
    out.push(c)
  }
  return out
}

export interface FormulaValidation { valid: boolean; error?: string }

/** Empty is valid (formula is optional). Otherwise: valid shape + real elements. */
export function validateFormula(raw: string): FormulaValidation {
  const f = raw.trim()
  if (!f) return { valid: true }
  if (!SHAPE_RE.test(f)) {
    return { valid: false, error: 'Not a valid formula — use element symbols and counts, e.g. C10H12N2O.' }
  }
  for (const { symbol } of tokenizeFormula(f)) {
    if (!ELEMENTS.has(symbol)) return { valid: false, error: `Unknown element “${symbol}”.` }
  }
  return { valid: true }
}
