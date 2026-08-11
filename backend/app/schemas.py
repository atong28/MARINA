"""
Pydantic request and response models for all API endpoints.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from app.config import (
    MAX_FP_LENGTH, MAX_HSQC_PEAKS, MAX_MS_PEAKS, MAX_NMR_PEAKS, MAX_SMILES_LENGTH,
    MAX_TOP_K, DEFAULT_TOP_K,
)

# Peak counts are converted to flat-array lengths: HSQC is triplets, MS is pairs.
_MAX_HSQC_LEN = MAX_HSQC_PEAKS * 3
_MAX_MS_LEN   = MAX_MS_PEAKS * 2


# ── Spectral data input ───────────────────────────────────────────────────────

class SpectralInput(BaseModel):
    """Combined spectral data payload for /predict."""
    hsqc:      Optional[List[float]] = Field(None, max_length=_MAX_HSQC_LEN,
                                             description="HSQC triplets [H1,C1,I1, ...]")
    h_nmr:     Optional[List[float]] = Field(None, max_length=MAX_NMR_PEAKS,
                                             description="¹H NMR shifts (ppm)")
    c_nmr:     Optional[List[float]] = Field(None, max_length=MAX_NMR_PEAKS,
                                             description="¹³C NMR shifts (ppm)")
    mass_spec: Optional[List[float]] = Field(None, max_length=_MAX_MS_LEN,
                                             description="Mass spec pairs [m/z,I, ...]")
    mw:        Optional[float]       = Field(None, gt=0, description="Molecular weight (Da)")

    @field_validator("hsqc")
    @classmethod
    def hsqc_triplets(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None and len(v) % 3 != 0:
            raise ValueError("hsqc must contain triplets (length divisible by 3)")
        return v

    @field_validator("mass_spec")
    @classmethod
    def mass_spec_pairs(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None and len(v) % 2 != 0:
            raise ValueError("mass_spec must contain pairs (length divisible by 2)")
        return v


# ── Predict ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    raw:       SpectralInput       = Field(..., description="Spectral inputs")
    # Client-generated handle so the browser can poll GET /api/queue/{request_id}
    # for its place in line while this request is still open.
    request_id: Optional[str]      = Field(None, max_length=64,
                                           description="Opaque id for queue tracking")
    k:         int                 = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    model_id:  Optional[str]       = Field(None)
    mw_min:    Optional[float]     = Field(None, gt=0, description="Min MW filter (Da)")
    mw_max:    Optional[float]     = Field(None, gt=0, description="Max MW filter (Da)")


# ── SMILES search ─────────────────────────────────────────────────────────────

class SmilesSearchRequest(BaseModel):
    smiles:   str              = Field(..., min_length=1, max_length=MAX_SMILES_LENGTH,
                                        description="Query SMILES")
    k:        int              = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    model_id: Optional[str]   = Field(None)
    mw_min:   Optional[float] = Field(None, gt=0)
    mw_max:   Optional[float] = Field(None, gt=0)


# ── Fingerprints ──────────────────────────────────────────────────────────────

class FingerprintIndicesRequest(BaseModel):
    smiles:   str            = Field(..., min_length=1, max_length=MAX_SMILES_LENGTH)
    model_id: Optional[str] = Field(None)


class FingerprintIndicesResponse(BaseModel):
    smiles:     str              = Field(..., description="Input SMILES")
    fp_indices: Optional[List[int]] = Field(None, description="Active bit indices (sorted)")


class BitExplainRequest(BaseModel):
    smiles:   str           = Field(..., min_length=1, max_length=MAX_SMILES_LENGTH,
                                     description="Candidate structure to locate bits on")
    pred_fp:  List[float]   = Field(..., min_length=1, max_length=MAX_FP_LENGTH,
                                     description="Predicted fingerprint from /predict or /smiles-search")
    model_id: Optional[str] = Field(None)
    limit:    int           = Field(60, ge=1, le=500, description="Max rows to return")
    include_fragment_svg: bool = Field(False, description="Attach a drawing of each substructure")


class BitExplanation(BaseModel):
    index:           int
    fragment_smiles: str   = Field(..., description="Substructure SMILES ('' for radius-0 bits)")
    atom_symbol:     str   = Field(..., description="Element at the environment centre")
    radius:          int
    raw_confidence:  float = Field(..., ge=0.0, le=1.0, description="Uncalibrated sigmoid output")
    confidence:      float = Field(..., ge=0.0, le=1.0, description="Calibrated presence probability")
    band:            str   = Field(..., description="Very likely / Likely / Possible / Unlikely")
    present:         bool  = Field(..., description="Whether the candidate actually contains it")
    group:           str   = Field(..., description="missing / match / unexpected / uncertain")
    atoms:           List[int] = Field(default_factory=list, description="Atom indices to highlight")
    bonds:           List[int] = Field(default_factory=list, description="Bond indices to highlight")
    fragment_svg:    Optional[str] = Field(None, description="Drawing of the substructure, when requested")


class BitHighlightRequest(BaseModel):
    smiles: str       = Field(..., min_length=1, max_length=MAX_SMILES_LENGTH)
    atoms:  List[int] = Field(default_factory=list, max_length=1000,
                              description="Atom indices to highlight")
    bonds:  List[int] = Field(default_factory=list, max_length=1000,
                              description="Bond indices to highlight")


class BitHighlightResponse(BaseModel):
    smiles: str
    svg:    Optional[str] = Field(None, description="SVG markup, or null if rendering is unavailable")


class BitExplainResponse(BaseModel):
    smiles:          str
    calibrated:      bool = Field(..., description="False when the model ships no calibration curve")
    bits:            List[BitExplanation]
    totals:          Dict[str, int] = Field(..., description="Row count per group before truncation")
    total_shown:     int
    total_available: int


# ── Result card ───────────────────────────────────────────────────────────────

class DatabaseLinks(BaseModel):
    coconut: Optional[str] = None
    lotus:   Optional[str] = None
    npmrd:   Optional[str] = None


class NPClassification(BaseModel):
    """NPClassifier annotation. Each tier can hold several labels, or none."""
    pathway:     List[str] = Field(default_factory=list)
    superclass:  List[str] = Field(default_factory=list)
    npclass:     List[str] = Field(default_factory=list,
                                   description="Class tier ('class' is reserved in JS)")
    isglycoside: bool      = False


class ResultCard(BaseModel):
    index:                        int
    smiles:                       str
    similarity:                   float = Field(ge=0.0, le=1.0)
    cosine_similarity:            Optional[float] = Field(None, ge=0.0, le=1.0)
    tanimoto_similarity:          Optional[float] = Field(None, ge=0.0, le=1.0)
    svg:                          Optional[str]   = None
    plain_svg:                    Optional[str]   = None
    name:                         Optional[str]   = None
    primary_link:                 Optional[str]   = None
    database_links:               DatabaseLinks   = Field(default_factory=DatabaseLinks)
    retrieved_molecule_fp_indices: List[int]      = Field(default_factory=list)
    exact_mass:                   Optional[float] = None
    npclassifier:                 Optional[NPClassification] = Field(
        None, description="Null when the model directory ships no annotations")


# ── Predict response ──────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    results:     List[ResultCard]
    total_count: int = Field(ge=0)
    offset:      int = Field(default=0, ge=0)
    limit:       int = Field(ge=0)
    pred_fp:     Optional[List[float]] = None


# ── SMILES search response ────────────────────────────────────────────────────

class SmilesSearchResponse(BaseModel):
    results:      List[ResultCard]
    total_count:  int = Field(ge=0)
    offset:       int = Field(default=0, ge=0)
    limit:        int = Field(ge=0)
    query_smiles: str
    query_fp:     Optional[List[float]] = None


# ── Custom SMILES card ────────────────────────────────────────────────────────

class CustomSmilesCardRequest(BaseModel):
    smiles:       str              = Field(..., min_length=1, max_length=MAX_SMILES_LENGTH,
                                            description="Target SMILES")
    reference_fp: List[float]      = Field(..., min_length=1, max_length=MAX_FP_LENGTH,
                                            description="Reference fingerprint (predicted or query FP)")
    model_id:     Optional[str]    = Field(None)


class CustomSmilesCardResponse(BaseModel):
    result: ResultCard
