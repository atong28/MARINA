/**
 * Store behaviour that the UI depends on.
 */
import { describe, it, expect, beforeEach } from 'vitest'
import { useAppStore } from './store'
import type { ResultCard } from '../services/api'

const card = (smiles: string): ResultCard => ({
  index: 1,
  smiles,
  similarity: 0.5,
  database_links: {},
  retrieved_molecule_fp_indices: [],
})

beforeEach(() => {
  useAppStore.setState({
    hsqc: [], h_nmr: [], c_nmr: [], mass_spec: [], mw: null,
    results: [], customResults: [], predictedFp: null, queryFp: null, resultSource: null,
  })
})

describe('setSpectra', () => {
  it('writes all modalities in one update', () => {
    useAppStore.getState().setSpectra({
      hsqc: [1, 2, 3], h_nmr: [4], c_nmr: [5], mass_spec: [6, 7], mass_spec_neg: [8, 9],
    })
    const s = useAppStore.getState()
    expect(s.hsqc).toEqual([1, 2, 3])
    expect(s.h_nmr).toEqual([4])
    expect(s.c_nmr).toEqual([5])
    expect(s.mass_spec).toEqual([6, 7])
    expect(s.mass_spec_neg).toEqual([8, 9])
  })

  it('notifies subscribers once per edit, not once per modality', () => {
    let updates = 0
    const unsub = useAppStore.subscribe(() => { updates++ })
    useAppStore.getState().setSpectra({
      hsqc: [1], h_nmr: [2], c_nmr: [3], mass_spec: [4], mass_spec_neg: [5],
    })
    unsub()
    expect(updates).toBe(1)
  })
})

describe('custom results', () => {
  it('assigns a distinct id to each card', () => {
    const { addCustomResult } = useAppStore.getState()
    addCustomResult(card('CCO'))
    addCustomResult(card('CCO'))
    const ids = useAppStore.getState().customResults.map((r) => r.id)
    expect(new Set(ids).size).toBe(2)
  })

  it('removes by identity, not list position', () => {
    // Regression: index-based removal made the wrong card disappear once an
    // earlier one had already been removed.
    const { addCustomResult, removeCustomResult } = useAppStore.getState()
    addCustomResult(card('A'))
    addCustomResult(card('B'))
    addCustomResult(card('C'))

    const middle = useAppStore.getState().customResults[1].id
    removeCustomResult(middle)

    const left = useAppStore.getState().customResults.map((r) => r.card.smiles)
    expect(left).toEqual(['A', 'C'])
  })

  it('ignores removal of an unknown id', () => {
    useAppStore.getState().addCustomResult(card('A'))
    useAppStore.getState().removeCustomResult('nope')
    expect(useAppStore.getState().customResults).toHaveLength(1)
  })

  it('clears custom cards when a new prediction lands', () => {
    useAppStore.getState().addCustomResult(card('A'))
    useAppStore.getState().setPredictResults([], [0.1, 0.2])
    expect(useAppStore.getState().customResults).toEqual([])
    expect(useAppStore.getState().resultSource).toBe('prediction')
  })

  it('clears custom cards when a SMILES search lands', () => {
    useAppStore.getState().addCustomResult(card('A'))
    useAppStore.getState().setSmilesResults([], [0.3])
    expect(useAppStore.getState().customResults).toEqual([])
    expect(useAppStore.getState().resultSource).toBe('smiles-search')
  })

  it('keeps only one fingerprint kind at a time', () => {
    useAppStore.getState().setPredictResults([], [0.1])
    expect(useAppStore.getState().queryFp).toBeNull()
    useAppStore.getState().setSmilesResults([], [0.2])
    expect(useAppStore.getState().predictedFp).toBeNull()
  })
})

describe('highlightEnabled', () => {
  it('defaults to on', () => {
    expect(useAppStore.getState().highlightEnabled).toBe(true)
  })

  it('round-trips through the setter', () => {
    useAppStore.getState().setHighlightEnabled(false)
    expect(useAppStore.getState().highlightEnabled).toBe(false)
    useAppStore.getState().setHighlightEnabled(true)
    expect(useAppStore.getState().highlightEnabled).toBe(true)
  })

  it('leaves results untouched — it is a display choice, not a query', () => {
    useAppStore.getState().setPredictResults([card('A')], [0.1])
    useAppStore.getState().setHighlightEnabled(false)
    const s = useAppStore.getState()
    expect(s.results.map((r) => r.smiles)).toEqual(['A'])
    expect(s.predictedFp).toEqual([0.1])
  })
})

describe('hybrid workspace', () => {
  const run = {
    seq: 1, source: card('CCO'), sourceLabel: '#1', mode: 'delete' as const, metric: 'cosine' as const,
    atoms: [0, 1], operations: [], heldApplied: [], results: [], unexplained: [],
    confidentBefore: 10, confidentAfter: 6,
  }

  beforeEach(() => {
    useAppStore.setState({ detail: null, heldFragments: [], refineRuns: [] })
  })

  it('holds fragments enabled, toggles and removes them', () => {
    const s = useAppStore.getState()
    const h = s.addHeldFragment({ source: card('CCO'), sourceLabel: '#1', atoms: [0], fragmentSmiles: 'C' })
    expect(useAppStore.getState().heldFragments).toHaveLength(1)
    expect(h.enabled).toBe(true)
    s.toggleHeldFragment(h.id)
    expect(useAppStore.getState().heldFragments[0].enabled).toBe(false)
    s.removeHeldFragment(h.id)
    expect(useAppStore.getState().heldFragments).toHaveLength(0)
  })

  it('keeps refined searches with distinct ids', () => {
    const s = useAppStore.getState()
    const a = s.addRefineRun(run)
    const b = s.addRefineRun({ ...run, seq: 2 })
    expect(a.id).not.toBe(b.id)
    s.removeRefineRun(a.id)
    expect(useAppStore.getState().refineRuns.map((r) => r.id)).toEqual([b.id])
  })

  it('is cleared, with the open compound, by a new search', () => {
    const s = useAppStore.getState()
    s.addHeldFragment({ source: card('CCO'), sourceLabel: '#1', atoms: [0], fragmentSmiles: 'C' })
    s.addRefineRun(run)
    s.openDetail({ card: card('CCO'), label: '#1' })
    s.setPredictResults([card('CCN')], [0.1, 0.2])
    const after = useAppStore.getState()
    expect(after.heldFragments).toEqual([])
    expect(after.refineRuns).toEqual([])
    expect(after.detail).toBeNull()
  })
})

describe('cardKey', () => {
  it('distinguishes cards by index and structure', async () => {
    const { cardKey } = await import('./store')
    expect(cardKey(card('CCO'))).not.toBe(cardKey(card('CCN')))
    expect(cardKey(card('CCO'))).toBe(cardKey({ ...card('CCO') }))
  })
})
