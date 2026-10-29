/**
 * Hand a secondary search (delete or require regions, then retrieve again) to a new
 * browser tab.
 *
 * The opening tab writes the search — the original predicted fingerprint plus
 * every edit so far — to localStorage under a fresh id and opens /refine/<id>;
 * the new tab reads it back and runs the search itself, so reloading it works and
 * a search started inside it can extend its own edits. localStorage, not
 * sessionStorage: a tab opened with window.open does not reliably share the
 * opener's sessionStorage.
 *
 * The fingerprint is 16,384 probabilities. Stored as base64 Float32 it is ~87 KB
 * rather than ~300 KB of JSON; the oldest searches are dropped past MAX_STORED so
 * the origin's quota is never the limit.
 */
import type { DepictionGeometry, RefineMetric, RefineMode, RefineOperation } from './api'

const PREFIX = 'marina.refine.'
export const MAX_STORED = 30

/** One edit, plus what the new tab needs to draw it. */
export interface RefineOpSpec extends RefineOperation {
  mode: RefineMode
  /** Where the region was selected, e.g. "original #1" or "step 1 · #3". */
  sourceLabel: string
  sourceName: string
  /** The source compound's line drawing and atom positions, to show the region on. */
  plainSvg?: string
  geometry?: DepictionGeometry | null
}

export interface RefineSearchSpec {
  id: string
  createdAt: number
  modelId?: string
  /** The original prediction; every edit is applied to it server-side. */
  predFp: number[]
  operations: RefineOpSpec[]
  metric: RefineMetric
  k: number
  /** The search this one was started from, when it was started inside a search tab. */
  parentId?: string
}

interface StoredSpec extends Omit<RefineSearchSpec, 'predFp'> {
  predFp: string
}

type KV = Pick<Storage, 'getItem' | 'setItem' | 'removeItem' | 'key' | 'length'>

export function encodeFp(fp: number[]): string {
  const bytes = new Uint8Array(new Float32Array(fp).buffer)
  let bin = ''
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  }
  return btoa(bin)
}

export function decodeFp(encoded: string): number[] {
  const bin = atob(encoded)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return Array.from(new Float32Array(bytes.buffer))
}

function newId(): string {
  const rand = typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID().slice(0, 8)
    : Math.random().toString(36).slice(2, 10)
  return `${Date.now().toString(36)}${rand}`
}

function storedIds(storage: KV): Array<{ id: string; createdAt: number }> {
  const out: Array<{ id: string; createdAt: number }> = []
  for (let i = 0; i < storage.length; i++) {
    const key = storage.key(i)
    if (!key?.startsWith(PREFIX)) continue
    try {
      const { createdAt } = JSON.parse(storage.getItem(key) ?? '{}') as StoredSpec
      out.push({ id: key.slice(PREFIX.length), createdAt: createdAt ?? 0 })
    } catch {
      out.push({ id: key.slice(PREFIX.length), createdAt: 0 })
    }
  }
  return out.sort((a, b) => a.createdAt - b.createdAt)
}

/** Drop the oldest searches so at most `keep` remain. */
function prune(storage: KV, keep: number) {
  const ids = storedIds(storage)
  for (const { id } of ids.slice(0, Math.max(0, ids.length - keep))) storage.removeItem(PREFIX + id)
}

function write(storage: KV, spec: RefineSearchSpec) {
  const value = JSON.stringify({ ...spec, predFp: encodeFp(spec.predFp) } satisfies StoredSpec)
  try {
    storage.setItem(PREFIX + spec.id, value)
  } catch {
    // Quota: make room by keeping only the newest few, then try once more.
    prune(storage, 3)
    storage.setItem(PREFIX + spec.id, value)
  }
}

export function saveSearch(
  spec: Omit<RefineSearchSpec, 'id' | 'createdAt'>,
  storage: KV = localStorage,
): string {
  const full: RefineSearchSpec = { ...spec, id: newId(), createdAt: Date.now() }
  prune(storage, MAX_STORED - 1)
  write(storage, full)
  return full.id
}

export function loadSearch(id: string, storage: KV = localStorage): RefineSearchSpec | null {
  const raw = storage.getItem(PREFIX + id)
  if (!raw) return null
  try {
    const stored = JSON.parse(raw) as StoredSpec
    // Searches saved before "require" existed carry no mode, or the retired "hold".
    const operations = stored.operations.map((o) => ({ ...o, mode: o.mode === 'require' ? 'require' : 'delete' } as RefineOpSpec))
    return { ...stored, operations, predFp: decodeFp(stored.predFp) }
  } catch {
    return null
  }
}

/** Persist a change made in the search tab (scoring, edited regions), so a reload keeps it. */
export function updateSearch(
  id: string, patch: Partial<Pick<RefineSearchSpec, 'metric' | 'operations'>>, storage: KV = localStorage,
) {
  const spec = loadSearch(id, storage)
  if (spec) write(storage, { ...spec, ...patch })
}

/** True when an identical region is already one of the edits. */
export function hasOperation(ops: RefineOperation[], smiles: string, atoms: number[]): boolean {
  return ops.some((o) =>
    o.smiles === smiles && o.atoms.length === atoms.length && o.atoms.every((a, i) => a === atoms[i]))
}
