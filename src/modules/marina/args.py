from typing import List, Dict
from pydantic.dataclasses import dataclass
from dataclasses import field
from ..core import SMARTArgs


@dataclass
class MARINAArgs(SMARTArgs):
    experiment_name: str = 'marina-development'
    project_name: str = 'MARINA'

    dim_model: int = 784
    nmr_dim_coords: List[int] = field(default_factory=lambda: [391, 391, 2])
    nmr_is_sign_encoding: List[bool] = field(default_factory=lambda: [False, False, True])
    c_nmr_dim_coords: List[int] = field(default_factory=lambda: [784])
    c_nmr_is_sign_encoding: List[bool] = field(default_factory=lambda: [False])
    h_nmr_dim_coords: List[int] = field(default_factory=lambda: [784])
    h_nmr_is_sign_encoding: List[bool] = field(default_factory=lambda: [False])
    ms_dim_coords: List[int] = field(default_factory=lambda: [392, 392])
    ms_is_sign_encoding: List[bool] = field(default_factory=lambda: [False, False])
    mw_dim_coords: List[int] = field(default_factory=lambda: [784])
    mw_is_sign_encoding: List[bool] = field(default_factory=lambda: [False])
    heads: int = 8
    # Depth of the shared cross-attention stack. Was 16; the layers sweep found 8 matches it
    # at 3/3 seeds (rank@1 0.7565 +/- 0.0015 vs 0.7546 +/- 0.0011) for ~33% fewer FLOPs, which
    # is consistent with the attribution result that blocks 0-9 read nothing reaching the output.
    layers: int = 8
    self_attn_layers: Dict[str, int] = field(default_factory=
        lambda: {'hsqc': 2, 'h_nmr': 1, 'c_nmr': 2, 'mass_spec': 1, 'mw': 1}
    )
    ff_dim: int = 3072
    out_dim: int = 16384

    c_wavelength_bounds: List[float] = field(
        default_factory=lambda: [0.01, 400.0])
    h_wavelength_bounds: List[float] = field(
        default_factory=lambda: [0.01, 20.0])
    mz_wavelength_bounds: List[float] = field(
        default_factory=lambda: [0.01, 5000.0])
    intensity_wavelength_bounds: List[float] = field(
        default_factory=lambda: [0.001, 200.0])
    mw_wavelength_bounds: List[float] = field(
        default_factory=lambda: [0.01, 7000.0])
