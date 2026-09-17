from typing import List, Dict
from pydantic.dataclasses import dataclass
from dataclasses import field
from ..core import SMARTArgs
from ..core.const import INPUT_TYPES


@dataclass
class MARINAArgs(SMARTArgs):
    experiment_name: str = 'marina-development'
    project_name: str = 'MARINA'

    input_types: List[INPUT_TYPES] = field(
        default_factory=lambda: ['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mass_spec_neg', 'mw', 'formula']
    )

    additional_test_types: list[list[str]] = field(default_factory=lambda: [
        ['hsqc'], ['h_nmr'], ['c_nmr'], ['mass_spec'], ['mass_spec_neg'], ['hmbc'], ['cosy'], ['hsqc', 'hmbc', 'cosy']
    ])

    dim_model: int = 784
    nmr_dim_coords: List[int] = field(default_factory=lambda: [391, 391, 2])
    nmr_is_sign_encoding: List[bool] = field(default_factory=lambda: [False, False, True])
    c_nmr_dim_coords: List[int] = field(default_factory=lambda: [784])
    c_nmr_is_sign_encoding: List[bool] = field(default_factory=lambda: [False])
    h_nmr_dim_coords: List[int] = field(default_factory=lambda: [784])
    h_nmr_is_sign_encoding: List[bool] = field(default_factory=lambda: [False])
    # MARINA2.0 2D modalities: HMBC (dC, dH) and COSY (dH, dH) as two-coordinate peaks
    hmbc_dim_coords: List[int] = field(default_factory=lambda: [392, 392])
    hmbc_is_sign_encoding: List[bool] = field(default_factory=lambda: [False, False])
    cosy_dim_coords: List[int] = field(default_factory=lambda: [392, 392])
    cosy_is_sign_encoding: List[bool] = field(default_factory=lambda: [False, False])
    # training-time per-peak dropout of the ceiling 2D lists (calibrated on curated deposits;
    # analysis/hmbc-cosy-calibration/results/marina2_dropout_params.json is the default table)
    dropout_2d: bool = True
    dropout_2d_params: str = ''
    hmbc_max_peaks: int = 200
    cosy_max_peaks: int = 120
    ms_dim_coords: List[int] = field(default_factory=lambda: [392, 392])
    ms_is_sign_encoding: List[bool] = field(default_factory=lambda: [False, False])
    heads: int = 8
    layers: int = 8
    self_attn_layers: Dict[str, int] = field(default_factory=
        lambda: {'hsqc': 2, 'h_nmr': 1, 'c_nmr': 2, 'mass_spec': 1, 'mass_spec_neg': 1, 'hmbc': 3, 'cosy': 2}
    )
    ff_dim: int = 3072
    out_dim: int = 16384

    formula_tokens: int = 1

    c_wavelength_bounds: List[float] = field(
        default_factory=lambda: [0.01, 400.0])
    h_wavelength_bounds: List[float] = field(
        default_factory=lambda: [0.01, 20.0])
    mz_wavelength_bounds: List[float] = field(
        default_factory=lambda: [0.01, 3100.0])
    intensity_wavelength_bounds: List[float] = field(
        default_factory=lambda: [0.001, 2.0])
