import { create } from 'zustand'
import { ModelInfo, ResultCard } from '../services/api'

type ResultSource = 'prediction' | 'smiles-search' | null

interface AppState {
  // Model selection
  availableModels: ModelInfo[] | null
  defaultModelId: string | null
  selectedModelId: string | null

  // Spectral input data
  hsqc: number[]
  h_nmr: number[]
  c_nmr: number[]
  mass_spec: number[]
  mw: number | null

  // SMILES search
  smilesInput: string

  // Retrieval MW filter
  retrievalMwMin: number | null
  retrievalMwMax: number | null

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
  setAvailableModels: (models: ModelInfo[], defaultId: string) => void
  initializeModelSelection: (models: ModelInfo[], defaultId: string) => void
  setSelectedModelId: (id: string) => void

  setHSQC: (data: number[]) => void
  setHNMR: (data: number[]) => void
  setCNMR: (data: number[]) => void
  setMassSpec: (data: number[]) => void
  /** Writes all four modalities in one update, so an edit re-renders once. */
  setSpectra: (data: {
    hsqc: number[]
    h_nmr: number[]
    c_nmr: number[]
    mass_spec: number[]
  }) => void
  setMW: (mw: number | null) => void
  setSmilesInput: (smiles: string) => void
  setRetrievalMwRange: (min: number | null, max: number | null) => void
  setHighlightEnabled: (enabled: boolean) => void

  setPredictResults: (results: ResultCard[], predictedFp: number[] | null) => void
  setSmilesResults: (results: ResultCard[], queryFp: number[] | null) => void

  addCustomResult: (result: ResultCard) => void
  removeCustomResult: (id: string) => void
  clearCustomResults: () => void
}

export interface CustomResult {
  id: string
  card: ResultCard
}

let customIdSeq = 0

const MODEL_STORAGE_KEY = 'marina.selectedModelId'
const HIGHLIGHT_STORAGE_KEY = 'marina.highlightEnabled'

function readStoredModelId(): string | null {
  try { return localStorage.getItem(MODEL_STORAGE_KEY) } catch { return null }
}

function writeStoredModelId(id: string) {
  try { localStorage.setItem(MODEL_STORAGE_KEY, id) } catch { /* ignore */ }
}

/** Defaults to on, so only an explicit "false" turns highlighting off. */
function readStoredHighlight(): boolean {
  try { return localStorage.getItem(HIGHLIGHT_STORAGE_KEY) !== 'false' } catch { return true }
}

function writeStoredHighlight(enabled: boolean) {
  try { localStorage.setItem(HIGHLIGHT_STORAGE_KEY, String(enabled)) } catch { /* ignore */ }
}

export const useAppStore = create<AppState>((set) => ({
  availableModels: null,
  defaultModelId: null,
  selectedModelId: null,

  hsqc: [],
  h_nmr: [],
  c_nmr: [],
  mass_spec: [],
  mw: null,

  smilesInput: '',

  retrievalMwMin: null,
  retrievalMwMax: null,

  highlightEnabled: readStoredHighlight(),

  results: [],
  predictedFp: null,
  queryFp: null,
  resultSource: null,

  customResults: [],

  setAvailableModels: (models, defaultId) =>
    set({ availableModels: models, defaultModelId: defaultId }),

  initializeModelSelection: (models, defaultId) => {
    const stored = readStoredModelId()
    const knownIds = new Set(models.map((m) => m.id))
    const selected = stored && knownIds.has(stored) ? stored : defaultId
    writeStoredModelId(selected)
    set({ availableModels: models, defaultModelId: defaultId, selectedModelId: selected })
  },

  setSelectedModelId: (id) => {
    writeStoredModelId(id)
    set({ selectedModelId: id })
  },

  setHSQC: (data) => set({ hsqc: data }),
  setHNMR: (data) => set({ h_nmr: data }),
  setCNMR: (data) => set({ c_nmr: data }),
  setMassSpec: (data) => set({ mass_spec: data }),
  setSpectra: (data) => set(data),
  setMW: (mw) => set({ mw }),
  setSmilesInput: (smiles) => set({ smilesInput: smiles }),
  setRetrievalMwRange: (min, max) => set({ retrievalMwMin: min, retrievalMwMax: max }),

  setHighlightEnabled: (enabled) => {
    writeStoredHighlight(enabled)
    set({ highlightEnabled: enabled })
  },

  setPredictResults: (results, predictedFp) =>
    set({ results, predictedFp, queryFp: null, resultSource: 'prediction', customResults: [] }),

  setSmilesResults: (results, queryFp) =>
    set({ results, queryFp, predictedFp: null, resultSource: 'smiles-search', customResults: [] }),

  addCustomResult: (result) =>
    set((state) => ({
      customResults: [...state.customResults, { id: `custom-${++customIdSeq}`, card: result }],
    })),

  removeCustomResult: (id) =>
    set((state) => ({ customResults: state.customResults.filter((r) => r.id !== id) })),

  clearCustomResults: () => set({ customResults: [] }),
}))
