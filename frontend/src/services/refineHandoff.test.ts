import { describe, it, expect } from 'vitest'
import {
  decodeFp, encodeFp, hasOperation, loadSearch, MAX_STORED, saveSearch, updateSearch,
} from './refineHandoff'

/** In-memory Storage stand-in (the test environment is node, without localStorage). */
function memoryStorage(quota = Infinity) {
  const map = new Map<string, string>()
  return {
    get length() { return map.size },
    key: (i: number) => [...map.keys()][i] ?? null,
    getItem: (k: string) => map.get(k) ?? null,
    setItem: (k: string, v: string) => {
      const used = [...map.values()].reduce((n, s) => n + s.length, 0)
      if (used + v.length > quota) throw new Error('QuotaExceededError')
      map.set(k, v)
    },
    removeItem: (k: string) => { map.delete(k) },
    map,
  }
}

const spec = (predFp: number[]) => ({
  predFp,
  operations: [{ smiles: 'CCO', atoms: [0, 1], sourceLabel: 'original #1', sourceName: '' }],
  metric: 'cosine' as const,
  k: 10,
})

describe('fingerprint encoding', () => {
  it('round-trips at float32 precision', () => {
    const fp = [0, 1, 0.5, 0.123456789, 1e-7]
    const back = decodeFp(encodeFp(fp))
    back.forEach((v, i) => expect(v).toBeCloseTo(fp[i], 6))
  })

  it('handles a full-size fingerprint', () => {
    const fp = Array.from({ length: 16384 }, (_, i) => (i % 97) / 97)
    expect(decodeFp(encodeFp(fp))).toHaveLength(16384)
  })
})

describe('saveSearch / loadSearch', () => {
  it('stores a search another tab can read back', () => {
    const storage = memoryStorage()
    const id = saveSearch(spec([0.25, 0.75]), storage)
    const back = loadSearch(id, storage)
    expect(back?.operations[0].atoms).toEqual([0, 1])
    expect(back?.predFp).toEqual([0.25, 0.75])
    expect(back?.id).toBe(id)
  })

  it('returns null for an unknown or corrupt id', () => {
    const storage = memoryStorage()
    expect(loadSearch('nope', storage)).toBeNull()
    storage.setItem('marina.refine.bad', '{not json')
    expect(loadSearch('bad', storage)).toBeNull()
  })

  it('keeps at most MAX_STORED searches, dropping the oldest', () => {
    const storage = memoryStorage()
    const ids = Array.from({ length: MAX_STORED + 5 }, () => saveSearch(spec([0.1]), storage))
    expect(storage.length).toBe(MAX_STORED)
    expect(loadSearch(ids[ids.length - 1], storage)).not.toBeNull()
  })

  it('makes room when the quota is hit', () => {
    const one = JSON.stringify(spec([0.1])).length + 200
    const storage = memoryStorage(one * 5)
    let last = ''
    for (let i = 0; i < 12; i++) last = saveSearch(spec([0.1]), storage)
    expect(loadSearch(last, storage)).not.toBeNull()
  })

  it('persists a metric change', () => {
    const storage = memoryStorage()
    const id = saveSearch(spec([0.1]), storage)
    updateSearch(id, { metric: 'containment' }, storage)
    expect(loadSearch(id, storage)?.metric).toBe('containment')
  })
})

describe('hasOperation', () => {
  it('matches the same region on the same compound only', () => {
    const ops = [{ smiles: 'CCO', atoms: [0, 1] }]
    expect(hasOperation(ops, 'CCO', [0, 1])).toBe(true)
    expect(hasOperation(ops, 'CCO', [0])).toBe(false)
    expect(hasOperation(ops, 'CCN', [0, 1])).toBe(false)
  })
})
