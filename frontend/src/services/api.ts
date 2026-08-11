import { useQuery, useMutation, UseQueryOptions, UseMutationOptions } from '@tanstack/react-query'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface SpectralInput {
  hsqc?: number[]
  h_nmr?: number[]
  c_nmr?: number[]
  mass_spec?: number[]
  mw?: number
}

export interface PredictRequest {
  raw: SpectralInput
  k?: number
  mw_min?: number
  mw_max?: number
  model_id?: string
  /** Client-generated id used to poll this request's queue position. */
  request_id?: string
}

export interface QueuePosition {
  /** "queued" behind others, "running" on a worker, "unknown" once finished. */
  state: 'queued' | 'running' | 'unknown'
  /** 1-based place in the waiting line; 0 when already running. */
  position: number
  /** Total jobs ahead of this one, running ones included. */
  ahead: number
  queued: number
  workers: number
  waited_seconds?: number
}

export interface QueueSnapshot {
  pooled: boolean
  workers: number
  capacity: number
  running: number
  queued: number
  in_flight: number
}

export interface UsageStatsResponse {
  queries_total: number
  by_kind: Record<string, number>
  counting_since: number
}

export interface DatabaseLinks {
  coconut?: string
  lotus?: string
  npmrd?: string
}

export interface ResultCard {
  index: number
  smiles: string
  similarity: number
  cosine_similarity?: number
  tanimoto_similarity?: number
  svg?: string
  plain_svg?: string
  name?: string
  primary_link?: string
  database_links: DatabaseLinks
  retrieved_molecule_fp_indices: number[]
  exact_mass?: number
}

export interface PredictResponse {
  results: ResultCard[]
  total_count: number
  offset: number
  limit: number
  pred_fp?: number[]
}

export interface SmilesSearchRequest {
  smiles: string
  k?: number
  mw_min?: number
  mw_max?: number
  model_id?: string
}

export interface SmilesSearchResponse {
  results: ResultCard[]
  total_count: number
  offset: number
  limit: number
  query_smiles: string
  query_fp?: number[]
}

export interface HealthResponse {
  status: string
  model_loaded: boolean
  uptime_seconds: number
  /** False when the server renders plain depictions only (HIGHLIGHT_ENABLED=false
   *  or no RDKit), so cards carry no similarity map to switch to. */
  highlight_available?: boolean
}

export interface ModelInfo {
  id: string
  root: string
  type: string
  default: boolean
  loaded: boolean
  display_name?: string
}

export interface ModelsResponse {
  models: ModelInfo[]
  default_model_id: string
}

export interface CustomSmilesCardRequest {
  smiles: string
  reference_fp: number[]
  model_id?: string
}

export interface CustomSmilesCardResponse {
  result: ResultCard
}

/**
 * How a predicted bit relates to the candidate being displayed.
 * - `missing`    – model is confident, this structure lacks it (most diagnostic)
 * - `match`      – model is confident and the structure has it
 * - `unexpected` – structure has it but the model did not call it
 * - `uncertain`  – model is unsure and the structure lacks it
 */
export type BitGroup = 'missing' | 'match' | 'unexpected' | 'uncertain'

export interface BitExplanation {
  index: number
  fragment_smiles: string
  atom_symbol: string
  radius: number
  raw_confidence: number
  confidence: number
  band: string
  present: boolean
  group: BitGroup
  atoms: number[]
  bonds: number[]
  /** Drawing of the substructure, present only when requested. */
  fragment_svg?: string | null
}

export interface BitExplainRequest {
  smiles: string
  pred_fp: number[]
  model_id?: string
  limit?: number
  include_fragment_svg?: boolean
}

export interface BitExplainResponse {
  smiles: string
  /** False when the model ships no calibration curve — confidences are then raw. */
  calibrated: boolean
  bits: BitExplanation[]
  totals: Record<BitGroup, number>
  total_shown: number
  total_available: number
}

export interface BitHighlightRequest {
  smiles: string
  atoms: number[]
  bonds: number[]
}

export interface BitHighlightResponse {
  smiles: string
  svg?: string
}

// ── HTTP client ───────────────────────────────────────────────────────────────

/**
 * True for the rejection fetch produces when a request is aborted.
 *
 * Starting a new search cancels the in-flight one, so without this the mutation
 * that got cancelled leaves an error behind and the UI reports "The user aborted
 * a request" over a perfectly good result.
 */
export function isAbortError(err: unknown): boolean {
  return err instanceof Error && err.name === 'AbortError'
}

async function fetchJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options?.headers },
  })

  if (!response.ok) {
    let message = `HTTP ${response.status}`
    try {
      const body = await response.json()
      message = body.detail || body.error || message
    } catch { /* non-JSON error body */ }
    throw new Error(message)
  }

  return response.json()
}

// ── Abort controller per logical operation ────────────────────────────────────

type AbortKey = 'predict' | 'smilesSearch' | 'customSmilesCard'

const controllers = new Map<AbortKey, AbortController>()

function makeController(key: AbortKey): AbortController {
  controllers.get(key)?.abort()
  const ctrl = new AbortController()
  controllers.set(key, ctrl)
  return ctrl
}

export function cancelInFlight(key: AbortKey) {
  controllers.get(key)?.abort()
}

// ── API object ────────────────────────────────────────────────────────────────

export const api = {
  health: () => fetchJson<HealthResponse>('/health'),

  models: () => fetchJson<ModelsResponse>('/models'),

  predict: (data: PredictRequest): Promise<PredictResponse> => {
    const { signal } = makeController('predict')
    return fetchJson('/predict', { method: 'POST', body: JSON.stringify(data), signal })
  },

  smilesSearch: (data: SmilesSearchRequest): Promise<SmilesSearchResponse> => {
    const { signal } = makeController('smilesSearch')
    return fetchJson('/smiles-search', { method: 'POST', body: JSON.stringify(data), signal })
  },

  customSmilesCard: (data: CustomSmilesCardRequest): Promise<CustomSmilesCardResponse> => {
    const { signal } = makeController('customSmilesCard')
    return fetchJson('/custom-smiles-card', { method: 'POST', body: JSON.stringify(data), signal })
  },

  explainBits: (data: BitExplainRequest): Promise<BitExplainResponse> =>
    fetchJson('/fingerprints/explain', { method: 'POST', body: JSON.stringify(data) }),

  highlightBit: (data: BitHighlightRequest): Promise<BitHighlightResponse> =>
    fetchJson('/fingerprints/highlight', { method: 'POST', body: JSON.stringify(data) }),

  queue: () => fetchJson<QueueSnapshot>('/queue'),

  queuePosition: (requestId: string) =>
    fetchJson<QueuePosition>(`/queue/${encodeURIComponent(requestId)}`),

  stats: () => fetchJson<UsageStatsResponse>('/stats'),
}

// ── React Query hooks ─────────────────────────────────────────────────────────

export function useHealth(options?: Partial<UseQueryOptions<HealthResponse>>) {
  return useQuery<HealthResponse>({
    queryKey: ['health'],
    queryFn: api.health,
    // Poll fast only while the model is still loading; once it is ready there
    // is nothing to watch, so back off instead of hammering every 5 s forever.
    refetchInterval: (query) => (query.state.data?.model_loaded ? 60_000 : 5_000),
    refetchIntervalInBackground: false,
    retry: 1,
    retryDelay: 1000,
    ...options,
  })
}

/**
 * Polls this request's place in line while a prediction is in flight.
 * Disabled when there is no active request, so it costs nothing when idle.
 */
export function useQueuePosition(requestId: string | null) {
  return useQuery<QueuePosition>({
    queryKey: ['queue', requestId],
    queryFn: () => api.queuePosition(requestId as string),
    enabled: Boolean(requestId),
    refetchInterval: 1500,
    // The queue moves while the tab is hidden too; keep the count honest.
    refetchIntervalInBackground: true,
    retry: false,
    gcTime: 0,
  })
}

export function useUsageStats(options?: Partial<UseQueryOptions<UsageStatsResponse>>) {
  return useQuery<UsageStatsResponse>({
    queryKey: ['stats'],
    queryFn: api.stats,
    staleTime: 60_000,
    refetchInterval: 120_000,
    refetchIntervalInBackground: false,
    retry: false,
    ...options,
  })
}

export function useModels(options?: Partial<UseQueryOptions<ModelsResponse>>) {
  return useQuery<ModelsResponse>({
    queryKey: ['models'],
    queryFn: api.models,
    staleTime: 5 * 60 * 1000,
    ...options,
  })
}

export function usePredict(options?: UseMutationOptions<PredictResponse, Error, PredictRequest>) {
  return useMutation<PredictResponse, Error, PredictRequest>({
    mutationFn: api.predict,
    ...options,
  })
}

export function useSmilesSearch(options?: UseMutationOptions<SmilesSearchResponse, Error, SmilesSearchRequest>) {
  return useMutation<SmilesSearchResponse, Error, SmilesSearchRequest>({
    mutationFn: api.smilesSearch,
    ...options,
  })
}

export function useCustomSmilesCard(options?: UseMutationOptions<CustomSmilesCardResponse, Error, CustomSmilesCardRequest>) {
  return useMutation<CustomSmilesCardResponse, Error, CustomSmilesCardRequest>({
    mutationFn: api.customSmilesCard,
    ...options,
  })
}
