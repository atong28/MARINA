import { create } from 'zustand'
import { ResultCard } from '../services/api'

type ResultSource = 'prediction' | 'smiles-search' | null

interface AppState {
  // The checkpoint the server used for the current results. The manual model
  // picker is gone: /predict auto-selects the best model for the supplied inputs
  // and reports which one back. Reused for follow-up bit-explain / custom-SMILES
  // scoring so they share the fingerprint space that produced these results.
  activeModelId: string | null
  activeModelName: string | null

  // Spectral input data
  hsqc: number[]
  h_nmr: number[]
  c_nmr: number[]
  mass_spec: number[]
  mass_spec_neg: number[]
  mw: number | null
  // Molecular formula (Hill notation, ASCII e.g. "C10H12N2O"). A descriptor input,
  // not a spectrum, so it never satisfies the "at least one spectral input" rule.
  formula: string

  // SMILES search
  smilesInput: string

  // Retrieval MW filter
  retrievalMwMin: number | null
  retrievalMwMax: number | null

  // Retrieval atom-count filter: per-element {count ± tolerance} constraints.
  // Stored as raw input strings; converted to min/max at request time.
  formulaFilter: FormulaFilterEntry[]

  // Whether result cards show the similarity-map depiction or the plain one.
  // Purely a display choice: every card carries both, so this never re-queries.
  highlightEnabled: boolean

  // Results and fingerprints from the last search
  results: ResultCard[]
  predictedFp: number[] | null
  queryFp: number[] | null
  resultSource: ResultSource

  // Custom SMILES result cards (scored against the current session FP).
  // Each carries a stable id so removal keys off identity, not list position.
  customResults: CustomResult[]

  // Actions
  setHSQC: (data: number[]) => void
  setHNMR: (data: number[]) => void
  setCNMR: (data: number[]) => void
  setMassSpec: (data: number[]) => void
  setMassSpecNeg: (data: number[]) => void
  /** Writes all spreadsheet modalities in one update, so an edit re-renders once. */
  setSpectra: (data: {
    hsqc: number[]
    h_nmr: number[]
    c_nmr: number[]
    mass_spec: number[]
    mass_spec_neg: number[]
  }) => void
  setMW: (mw: number | null) => void
  setFormula: (formula: string) => void
  setSmilesInput: (smiles: string) => void
  setRetrievalMwRange: (min: number | null, max: number | null) => void
  setFormulaFilter: (entries: FormulaFilterEntry[]) => void
  setHighlightEnabled: (enabled: boolean) => void

  setPredictResults: (results: ResultCard[], predictedFp: number[] | null,
                      model?: { id?: string | null; name?: string | null }) => void
  setSmilesResults: (results: ResultCard[], queryFp: number[] | null,
                     model?: { id?: string | null; name?: string | null }) => void

  addCustomResult: (result: ResultCard) => void
  removeCustomResult: (id: string) => void
  clearCustomResults: () => void
}

export interface CustomResult {
  id: string
  card: ResultCard
}

/** One row of the atom-count filter UI (raw input strings). */
export interface FormulaFilterEntry {
  element: string
  count: string
  tolerance: string
}

let customIdSeq = 0

const HIGHLIGHT_STORAGE_KEY = 'marina.highlightEnabled'

/** Defaults to on, so only an explicit "false" turns highlighting off. */
function readStoredHighlight(): boolean {
  try { return localStorage.getItem(HIGHLIGHT_STORAGE_KEY) !== 'false' } catch { return true }
}

function writeStoredHighlight(enabled: boolean) {
  try { localStorage.setItem(HIGHLIGHT_STORAGE_KEY, String(enabled)) } catch { /* ignore */ }
}

export const useAppStore = create<AppState>((set) => ({
  activeModelId: null,
  activeModelName: null,

  hsqc: [],
  h_nmr: [],
  c_nmr: [],
  mass_spec: [],
  mass_spec_neg: [],
  mw: null,
  formula: '',

  smilesInput: '',

  retrievalMwMin: null,
  retrievalMwMax: null,
  formulaFilter: [],

  highlightEnabled: readStoredHighlight(),

  results: [],
  predictedFp: null,
  queryFp: null,
  resultSource: null,

  customResults: [],

  setHSQC: (data) => set({ hsqc: data }),
  setHNMR: (data) => set({ h_nmr: data }),
  setCNMR: (data) => set({ c_nmr: data }),
  setMassSpec: (data) => set({ mass_spec: data }),
  setMassSpecNeg: (data) => set({ mass_spec_neg: data }),
  setSpectra: (data) => set(data),
  setMW: (mw) => set({ mw }),
  setFormula: (formula) => set({ formula }),
  setSmilesInput: (smiles) => set({ smilesInput: smiles }),
  setRetrievalMwRange: (min, max) => set({ retrievalMwMin: min, retrievalMwMax: max }),
  setFormulaFilter: (formulaFilter) => set({ formulaFilter }),

  setHighlightEnabled: (enabled) => {
    writeStoredHighlight(enabled)
    set({ highlightEnabled: enabled })
  },

  setPredictResults: (results, predictedFp, model) =>
    set({
      results, predictedFp, queryFp: null, resultSource: 'prediction', customResults: [],
      activeModelId: model?.id ?? null, activeModelName: model?.name ?? null,
    }),

  setSmilesResults: (results, queryFp, model) =>
    set({
      results, queryFp, predictedFp: null, resultSource: 'smiles-search', customResults: [],
      activeModelId: model?.id ?? null, activeModelName: model?.name ?? null,
    }),

  addCustomResult: (result) =>
    set((state) => ({
      customResults: [...state.customResults, { id: `custom-${++customIdSeq}`, card: result }],
    })),

  removeCustomResult: (id) =>
    set((state) => ({ customResults: state.customResults.filter((r) => r.id !== id) })),

  clearCustomResults: () => set({ customResults: [] }),
}))
