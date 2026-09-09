import os
from typing import Tuple

from ..core.const import DATASET_ROOT, PVC_ROOT
from ..marina.args import MARINAArgs
from ..spectre.args import SPECTREArgs


def get_data_paths(args: MARINAArgs | SPECTREArgs, today: str) -> Tuple[str, str]:
    results_path = os.path.join(
        DATASET_ROOT,
        "results",
        args.experiment_name,
        today
    )

    final_path = os.path.join(
        PVC_ROOT,
        "results",
        args.experiment_name,
        today
    )

    return results_path, final_path
