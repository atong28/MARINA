"""
Pydantic request and response models for all API endpoints.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ── Spectral data input ───────────────────────────────────────────────────────

class SpectralInput(BaseModel):
    """Combined spectral data payload for /predict."""
    hsqc:      Optional[List[float]] = Field(None, description="HSQC triplets [H1,C1,I1, ...]")
    h_nmr:     Optional[List[float]] = Field(None, description="¹H NMR shifts (ppm)")
    c_nmr:     Optional[List[float]] = Field(None, description="¹³C NMR shifts (ppm)")
    mass_spec: Optional[List[float]] = Field(None, description="Mass spec pairs [m/z,I, ...]")
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
    k:         int                 = Field(10, ge=1, le=50)
    model_id:  Optional[str]       = Field(None)
    mw_min:    Optional[float]     = Field(None, gt=0, description="Min MW filter (Da)")
    mw_max:    Optional[float]     = Field(None, gt=0, description="Max MW filter (Da)")


# ── SMILES search ─────────────────────────────────────────────────────────────

class SmilesSearchRequest(BaseModel):
    smiles:   str              = Field(..., min_length=1, description="Query SMILES")
    k:        int              = Field(10, ge=1, le=50)
    model_id: Optional[str]   = Field(None)
    mw_min:   Optional[float] = Field(None, gt=0)
    mw_max:   Optional[float] = Field(None, gt=0)


# ── Fingerprints ──────────────────────────────────────────────────────────────

class FingerprintIndicesRequest(BaseModel):
    smiles:   str            = Field(..., min_length=1)
    model_id: Optional[str] = Field(None)


class FingerprintIndicesResponse(BaseModel):
    smiles:     str              = Field(..., description="Input SMILES")
    fp_indices: Optional[List[int]] = Field(None, description="Active bit indices (sorted)")


# ── Result card ───────────────────────────────────────────────────────────────

class DatabaseLinks(BaseModel):
    coconut: Optional[str] = None
    lotus:   Optional[str] = None
    npmrd:   Optional[str] = None


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
    smiles:       str              = Field(..., min_length=1, description="Target SMILES")
    reference_fp: List[float]      = Field(..., min_length=1, description="Reference fingerprint (predicted or query FP)")
    model_id:     Optional[str]    = Field(None)


class CustomSmilesCardResponse(BaseModel):
    result: ResultCard
