import { useState, useRef, useCallback, useEffect } from 'react'
import {
  usePredict, useSmilesSearch, useHealth, useCustomSmilesCard, cancelInFlight, isAbortError,
  type SpectralInput, type PredictRequest, type SmilesSearchRequest,
} from '../services/api'
import { useAppStore } from '../store/store'
import { getAvailableExamples, loadExample, type ExampleMeta } from '../services/exampleLoader'
import StatusIndicator from '../components/common/StatusIndicator'
import HighlightToggle from '../components/common/HighlightToggle'
import HelpButton from '../components/common/HelpButton'
import FormulaInput from '../components/common/FormulaInput'
import FormulaFilter from '../components/common/FormulaFilter'
import { validateFormula, buildFormulaConstraints, formulaFilterEntryError } from '../services/formula'
import QueueStatus from '../components/common/QueueStatus'
import UsageCounter from '../components/common/UsageCounter'
import SpreadsheetTable, { type ValidationSummary } from '../components/spreadsheet/SpreadsheetTable'
import SpectraPreview from '../components/spectra/SpectraPreview'
import ResultsGrid from '../components/results/ResultsGrid'
import { HELP } from '../helpContent'
import './MainPage.css'

type Tab = 'spectral' | 'smiles'

// /docs is only routed when the edge proxy runs with EXPOSE_API_VIA_NGINX=true;
// otherwise the link falls through the SPA catch-all to a blank page.
const SHOW_API_DOCS = import.meta.env.VITE_SHOW_API_DOCS === 'true'

/** Opaque handle for queue tracking. randomUUID needs a secure context. */
function newRequestId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID()
  return `req-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
}

function isFiniteNum(v: number | null | undefined): v is number {
  return v !== null && v !== undefined && Number.isFinite(v)
}

function filterValid(arr: number[]): number[] {
  return arr.filter((v) => Number.isFinite(v))
}

function MainPage() {
  const [tab, setTab] = useState<Tab>('spectral')
  const [k, setK] = useState(10)
  const [hasInvalidSpreadsheet, setHasInvalidSpreadsheet] = useState(false)
  const [customSmilesInput, setCustomSmilesInput] = useState('')
  const [customError, setCustomError] = useState<string | null>(null)
  const [activeRequestId, setActiveRequestId] = useState<string | null>(null)

  // Example loading
  const [availableExamples, setAvailableExamples] = useState<ExampleMeta[]>([])
  const [selectedExampleStem, setSelectedExampleStem] = useState('')
  const [isLoadingExample, setIsLoadingExample] = useState(false)

  const {
    activeModelId, activeModelName,
    hsqc, h_nmr, c_nmr, mass_spec, mass_spec_neg, mw, formula,
    smilesInput,
    retrievalMwMin, retrievalMwMax, formulaFilter,
    results, predictedFp, queryFp, resultSource,
    customResults,
    setPredictResults, setSmilesResults,
    setHSQC, setHNMR, setCNMR, setMassSpec, setMassSpecNeg, setSpectra, setMW,
    setSmilesInput,
    setRetrievalMwRange,
    addCustomResult, removeCustomResult,
  } = useAppStore()

  const { data: health } = useHealth()

  // Sequence refs ensure only the latest in-flight request writes to results.
  const seqRef    = useRef(0)
  const predSeq   = useRef<number | null>(null)
  const smilesSeq = useRef<number | null>(null)

  const predictMutation = usePredict({
    onSuccess: (data) => {
      if (predSeq.current !== seqRef.current) return
      setPredictResults(data.results, data.pred_fp ?? null,
        { id: data.model_id, name: data.model_display_name })
      setCustomError(null)
    },
    // Stop polling for a place in line once the request is no longer in flight.
    onSettled: () => setActiveRequestId(null),
  })

  const smilesSearchMutation = useSmilesSearch({
    onSuccess: (data) => {
      if (smilesSeq.current !== seqRef.current) return
      setSmilesResults(data.results, data.query_fp ?? null,
        { id: data.model_id, name: data.model_display_name })
      setCustomError(null)
    },
  })

  const customCardMutation = useCustomSmilesCard({
    onSuccess: (data) => {
      if (data.result) addCustomResult(data.result)
      setCustomSmilesInput('')
      setCustomError(null)
    },
    onError: (err) => setCustomError(isAbortError(err) ? null : err.message),
  })

  const mwRangeInvalid =
    (isFiniteNum(retrievalMwMin) && retrievalMwMin < 0) ||
    (isFiniteNum(retrievalMwMax) && retrievalMwMax < 0) ||
    (isFiniteNum(retrievalMwMin) && isFiniteNum(retrievalMwMax) && retrievalMwMin > retrievalMwMax)

  // A prediction needs at least one spectral modality; MW/formula alone are only
  // descriptors and retrieve nothing meaningful (the backend rejects it too).
  const hasSpectralInput =
    filterValid(hsqc).length > 0 || filterValid(h_nmr).length > 0 ||
    filterValid(c_nmr).length > 0 || filterValid(mass_spec).length > 0 ||
    filterValid(mass_spec_neg).length > 0

  const formulaInvalid = formula.trim() !== '' && !validateFormula(formula).valid
  const formulaFilterInvalid = formulaFilter.some((e) => formulaFilterEntryError(e) !== null)

  const handlePredict = useCallback(() => {
    cancelInFlight('smilesSearch')
    cancelInFlight('predict')
    seqRef.current += 1
    predSeq.current = seqRef.current

    const requestId = newRequestId()
    setActiveRequestId(requestId)

    const raw: SpectralInput = {}
    const validHSQC = filterValid(hsqc)
    const validHNMR = filterValid(h_nmr)
    const validCNMR = filterValid(c_nmr)
    const validMS    = filterValid(mass_spec)
    const validMSNeg = filterValid(mass_spec_neg)
    if (validHSQC.length)  raw.hsqc          = validHSQC
    if (validHNMR.length)  raw.h_nmr         = validHNMR
    if (validCNMR.length)  raw.c_nmr         = validCNMR
    if (validMS.length)    raw.mass_spec     = validMS
    if (validMSNeg.length) raw.mass_spec_neg = validMSNeg
    if (isFiniteNum(mw))   raw.mw            = mw
    if (formula.trim() && validateFormula(formula).valid) raw.formula = formula.trim()

    // No model_id: the server auto-selects the best checkpoint for these inputs.
    const payload: PredictRequest = { raw, k, request_id: requestId }
    if (isFiniteNum(retrievalMwMin)) payload.mw_min = retrievalMwMin
    if (isFiniteNum(retrievalMwMax)) payload.mw_max = retrievalMwMax
    const constraints = buildFormulaConstraints(formulaFilter)
    if (constraints.length) payload.formula_filter = constraints

    predictMutation.mutate(payload)
  }, [hsqc, h_nmr, c_nmr, mass_spec, mass_spec_neg, mw, formula, k, retrievalMwMin, retrievalMwMax, formulaFilter, predictMutation])

  const handleSmilesSearch = useCallback(() => {
    const trimmed = smilesInput.trim()
    if (!trimmed) return
    cancelInFlight('predict')
    cancelInFlight('smilesSearch')
    seqRef.current += 1
    smilesSeq.current = seqRef.current

    const payload: SmilesSearchRequest = { smiles: trimmed, k }
    if (isFiniteNum(retrievalMwMin)) payload.mw_min = retrievalMwMin
    if (isFiniteNum(retrievalMwMax)) payload.mw_max = retrievalMwMax
    const constraints = buildFormulaConstraints(formulaFilter)
    if (constraints.length) payload.formula_filter = constraints

    smilesSearchMutation.mutate(payload)
  }, [smilesInput, k, retrievalMwMin, retrievalMwMax, formulaFilter, smilesSearchMutation])

  const handleAddCustom = useCallback(() => {
    const trimmed = customSmilesInput.trim()
    if (!trimmed) return

    const refFp = resultSource === 'prediction' ? predictedFp : queryFp
    if (!refFp || refFp.length === 0) {
      setCustomError('No reference fingerprint available for the current session.')
      return
    }

    setCustomError(null)
    cancelInFlight('customSmilesCard')
    customCardMutation.mutate({
      smiles: trimmed,
      reference_fp: refFp,
      // Score against the same checkpoint that produced the current results.
      model_id: activeModelId ?? undefined,
    })
  }, [customSmilesInput, resultSource, predictedFp, queryFp, activeModelId, customCardMutation])

  // Stable identity: SpreadsheetTable memoises its emit callback on this.
  const handleValidationChange = useCallback(
    (s: ValidationSummary) => setHasInvalidSpreadsheet(s.anyInvalid),
    [],
  )

  // Populate example list on mount
  useEffect(() => {
    getAvailableExamples().then(setAvailableExamples).catch(console.error)
  }, [])

  const handleLoadExample = useCallback(async () => {
    if (!selectedExampleStem) return
    setIsLoadingExample(true)
    try {
      const data = await loadExample(selectedExampleStem)
      setHSQC(data.hsqc ?? [])
      setHNMR(data.h_nmr ?? [])
      setCNMR(data.c_nmr ?? [])
      setMassSpec(data.mass_spec ?? [])
      setMassSpecNeg(data.mass_spec_neg ?? [])
      setMW(data.mw ?? null)
    } catch (err) {
      console.error('Failed to load example:', err)
    } finally {
      setIsLoadingExample(false)
    }
  }, [selectedExampleStem, setHSQC, setHNMR, setCNMR, setMassSpec, setMassSpecNeg, setMW])

  const isPending = predictMutation.isPending || smilesSearchMutation.isPending
  // A cancelled request is not a failure worth showing — see isAbortError.
  const rawError = predictMutation.error ?? smilesSearchMutation.error
  const searchError = rawError && !isAbortError(rawError) ? rawError : null
  const hasResults = results.length > 0

  return (
    <div className="main-page">
      {/* ── Header ── */}
      <header className="main-page__header">
        <div className="main-page__header-row">
          <div className="main-page__header-controls">
            <span className="main-page__model-indicator">
              Model: {activeModelName ?? 'auto'}
            </span>
            <HelpButton content={HELP.controls.model} placement="bottom" />
            <StatusIndicator health={health} />
            <HelpButton content={HELP.controls.status} placement="bottom" />
            <HighlightToggle />
            <UsageCounter />
          </div>
          {SHOW_API_DOCS && (
            <a className="main-page__api-link" href="/docs" target="_blank" rel="noopener noreferrer">
              API Docs
            </a>
          )}
        </div>
        <h1 className="main-page__title">MARINA</h1>
        <p className="main-page__subtitle">
          Molecular structure retrieval from NMR and mass spectral data
        </p>
      </header>

      {/* ── Tabs ── */}
      <div className="main-page__tabs">
        <button
          className={`main-page__tab${tab === 'spectral' ? ' active' : ''}`}
          onClick={() => setTab('spectral')}
        >
          Spectral Data
          <HelpButton content={HELP.tabs.spectral} placement="bottom" />
        </button>
        <button
          className={`main-page__tab${tab === 'smiles' ? ' active' : ''}`}
          onClick={() => setTab('smiles')}
        >
          SMILES Search
          <HelpButton content={HELP.tabs.smiles} placement="bottom" />
        </button>
      </div>

      {/* ── Spectral panel ── */}
      {tab === 'spectral' && (
        <section className="main-page__panel">
          <div className="main-page__panel-description">
            <p>
              Enter spectral data in the spreadsheet. ¹H-¹³C HSQC requires all three columns per
              row (H shift, C shift, intensity). MS/MS requires both m/z and intensity per row.
              <HelpButton content={HELP.spectral.spreadsheet} placement="right" />
            </p>
          </div>

          <SpreadsheetTable
            hsqc={hsqc}
            h_nmr={h_nmr}
            c_nmr={c_nmr}
            mass_spec={mass_spec}
            mass_spec_neg={mass_spec_neg}
            onSpectraChange={setSpectra}
            onValidationChange={handleValidationChange}
          />

          <SpectraPreview hsqc={hsqc} h_nmr={h_nmr} c_nmr={c_nmr} mass_spec={mass_spec} mass_spec_neg={mass_spec_neg} />

          <div className="main-page__mw-row">
            <label className="main-page__label">
              <span className="main-page__label-text">
                Molecular weight (Da)
                <HelpButton content={HELP.spectral.mw} placement="right" />
              </span>
              <input
                type="number"
                step="0.01"
                placeholder="Optional"
                value={mw ?? ''}
                onChange={(e) => setMW(e.target.value ? parseFloat(e.target.value) : null)}
              />
            </label>
          </div>

          <FormulaInput />

          <MWRangeFilter
            min={retrievalMwMin}
            max={retrievalMwMax}
            onChange={setRetrievalMwRange}
            invalid={mwRangeInvalid}
          />

          <FormulaFilter />

          <div className="main-page__action-row">
            <label className="main-page__label main-page__label--inline">
              Results
              <HelpButton content={HELP.spectral.resultsCount} placement="top" />
              <input
                type="number"
                min="1"
                max="50"
                value={k}
                onChange={(e) => setK(Math.max(1, Math.min(50, parseInt(e.target.value) || 10)))}
              />
            </label>
            <button
              className="main-page__btn main-page__btn--primary"
              onClick={handlePredict}
              disabled={
                predictMutation.isPending ||
                !health?.model_loaded ||
                hasInvalidSpreadsheet ||
                mwRangeInvalid ||
                formulaInvalid ||
                formulaFilterInvalid ||
                !hasSpectralInput
              }
              title={
                hasInvalidSpreadsheet
                  ? 'Fix incomplete rows in the spreadsheet first.'
                  : mwRangeInvalid
                  ? 'MW filter range is invalid.'
                  : formulaInvalid
                  ? 'Fix the molecular formula first.'
                  : formulaFilterInvalid
                  ? 'Fix the formula filter first.'
                  : !hasSpectralInput
                  ? 'Enter at least one spectral input (HSQC, ¹H, ¹³C, or MS).'
                  : undefined
              }
            >
              {predictMutation.isPending ? 'Predicting…' : 'Predict Structure'}
            </button>
            <HelpButton content={HELP.spectral.predict} placement="top" />
            {hasInvalidSpreadsheet && (
              <span className="main-page__warning">
                Incomplete rows — fix or use Condense rows.
              </span>
            )}
          </div>

          {/* Example loader */}
          {availableExamples.length > 0 && (
            <div className="main-page__example-row">
              <select
                className="main-page__example-select"
                value={selectedExampleStem}
                onChange={(e) => setSelectedExampleStem(e.target.value)}
                disabled={isLoadingExample}
              >
                <option value="">Load an example…</option>
                {availableExamples.map((ex) => (
                  <option key={ex.stem} value={ex.stem}>
                    {ex.name}
                  </option>
                ))}
              </select>
              <button
                className="main-page__btn main-page__btn--secondary"
                onClick={handleLoadExample}
                disabled={!selectedExampleStem || isLoadingExample}
              >
                {isLoadingExample ? 'Loading…' : 'Load'}
              </button>
              <HelpButton content={HELP.spectral.examples} placement="top" />
            </div>
          )}
        </section>
      )}

      {/* ── SMILES panel ── */}
      {tab === 'smiles' && (
        <section className="main-page__panel">
          <label className="main-page__label main-page__label--block">
            <span className="main-page__label-text">
              SMILES string
              <HelpButton content={HELP.smiles.input} placement="right" />
            </span>
            <input
              className="main-page__smiles-input"
              type="text"
              placeholder="e.g. CC(C)CCO"
              value={smilesInput}
              onChange={(e) => setSmilesInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSmilesSearch()}
            />
          </label>

          <MWRangeFilter
            min={retrievalMwMin}
            max={retrievalMwMax}
            onChange={setRetrievalMwRange}
            invalid={mwRangeInvalid}
          />

          <FormulaFilter />

          <div className="main-page__action-row">
            <label className="main-page__label main-page__label--inline">
              Results
              <HelpButton content={HELP.spectral.resultsCount} placement="top" />
              <input
                type="number"
                min="1"
                max="50"
                value={k}
                onChange={(e) => setK(Math.max(1, Math.min(50, parseInt(e.target.value) || 10)))}
              />
            </label>
            <button
              className="main-page__btn main-page__btn--primary"
              onClick={handleSmilesSearch}
              disabled={
                smilesSearchMutation.isPending ||
                !health?.model_loaded ||
                !smilesInput.trim() ||
                mwRangeInvalid ||
                formulaFilterInvalid
              }
            >
              {smilesSearchMutation.isPending ? 'Searching…' : 'Search by SMILES'}
            </button>
            <HelpButton content={HELP.smiles.search} placement="top" />
          </div>
        </section>
      )}

      {/* ── Search error ── */}
      {searchError && (
        <div className="main-page__error">
          <strong>Error:</strong> {searchError.message}
        </div>
      )}

      {/* ── Loading state ── */}
      {isPending && (
        <div className="main-page__loading-block">
          <div className="main-page__loading">
            <span className="main-page__spinner" />
            {predictMutation.isPending ? 'Running prediction…' : 'Searching…'}
          </div>
          {predictMutation.isPending && <QueueStatus requestId={activeRequestId} />}
        </div>
      )}

      {/* ── Custom SMILES (shown once results exist) ── */}
      {hasResults && (
        <section className="main-page__panel main-page__custom-panel">
          <h3 className="main-page__custom-title">
            Score a custom SMILES
            <HelpButton content={HELP.custom.panel} placement="right" />
          </h3>
          <p className="main-page__custom-description">
            Enter any SMILES to score it against the current{' '}
            {resultSource === 'prediction' ? 'predicted' : 'query'} fingerprint and add a
            comparison card.
          </p>
          <div className="main-page__action-row">
            <input
              className="main-page__smiles-input main-page__custom-input"
              type="text"
              placeholder="e.g. C1=CC=CC=C1"
              value={customSmilesInput}
              onChange={(e) => setCustomSmilesInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAddCustom()}
            />
            <button
              className="main-page__btn main-page__btn--secondary"
              onClick={handleAddCustom}
              disabled={customCardMutation.isPending || !customSmilesInput.trim()}
            >
              {customCardMutation.isPending ? 'Scoring…' : 'Add card'}
            </button>
          </div>
          {customError && (
            <div className="main-page__error main-page__error--inline">{customError}</div>
          )}
        </section>
      )}

      {/* ── Results ── */}
      <ResultsGrid
        results={results}
        customResults={customResults}
        onRemoveCustom={removeCustomResult}
        // Only prediction results carry per-bit confidences. A SMILES search
        // scores against the query molecule's own binary fingerprint, where
        // every present bit is exactly 1.0 — there is no confidence to report,
        // and the calibration curve was fitted on predicted probabilities.
        predFp={resultSource === 'prediction' ? predictedFp : null}
        modelId={activeModelId ?? undefined}
        resultSource={resultSource}
      />
    </div>
  )
}

// ── MW range filter sub-component ─────────────────────────────────────────────

interface MWRangeFilterProps {
  min: number | null
  max: number | null
  onChange: (min: number | null, max: number | null) => void
  invalid: boolean
}

function parseMW(val: string): number | null {
  if (!val.trim()) return null
  const n = parseFloat(val)
  return Number.isFinite(n) && n > 0 ? n : null
}

function MWRangeFilter({ min, max, onChange, invalid }: MWRangeFilterProps) {
  return (
    <div className="mw-filter">
      <span className="mw-filter__label">
        Retrieval MW filter (Da)
        <HelpButton content={HELP.spectral.mwFilter} placement="right" />
      </span>
      <div className="mw-filter__inputs">
        <input
          type="number"
          step="0.01"
          placeholder="Min"
          value={min ?? ''}
          onChange={(e) => onChange(parseMW(e.target.value), max)}
        />
        <span className="mw-filter__dash">–</span>
        <input
          type="number"
          step="0.01"
          placeholder="Max"
          value={max ?? ''}
          onChange={(e) => onChange(min, parseMW(e.target.value))}
        />
      </div>
      {invalid ? (
        <span className="mw-filter__error">Must satisfy 0 ≤ min ≤ max.</span>
      ) : (
        <span className="mw-filter__hint">Leave blank for no bound.</span>
      )}
    </div>
  )
}

export default MainPage
