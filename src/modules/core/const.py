from typing import Literal, Dict, List, Set
from pathlib import Path
import os

from ..log import get_logger

logger = get_logger(__file__)

INPUT_TYPES = Literal['hsqc', 'h_nmr', 'c_nmr', 'mass_spec', 'mass_spec_neg', 'mw', 'formula', 'hmbc', 'cosy']
INPUTS_CANONICAL_ORDER: List[INPUT_TYPES] = ['hsqc', 'hmbc', 'cosy', 'c_nmr', 'h_nmr', 'mass_spec', 'mass_spec_neg', 'mw', 'formula']

DEBUG_LEN: int = 3000

NON_SPECTRAL_INPUTS: Set[INPUT_TYPES] = {'mw', 'formula'}
SELF_ATTN_INPUTS: Set[INPUT_TYPES] = {'hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mass_spec_neg', 'hmbc', 'cosy'}


if 'src/marina/src/modules' in __file__:
    logger.info('Detected website setup')
    CODE_ROOT = None
    DATASET_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / 'data'
    WANDB_API_KEY_FILE = None
    PVC_ROOT = None
    BENCHMARK_ROOT = None
elif 'nas-gpu' in __file__:
    logger.info('Detected yuzu setup')
    CODE_ROOT = '/data/nas-gpu/wang/atong/MARINA'
    DATASET_ROOT = '/data/nas-gpu/wang/atong/MARINA/data/dataset'
    WANDB_API_KEY_FILE = '/data/nas-gpu/wang/atong/MARINA/wandb_api_key.json'
    PVC_ROOT = CODE_ROOT
    BENCHMARK_ROOT = DATASET_ROOT
elif '/code' in __file__:
    logger.info('Detected nautilus setup')
    CODE_ROOT = '/code'
    DATASET_ROOT = '/workspace'
    WANDB_API_KEY_FILE = '/root/gurusmart/Moonshot/wandb_api_key.json'
    PVC_ROOT = '/root/gurusmart/Moonshot'
    BENCHMARK_ROOT = '/root/gurusmart/Benchmark'
elif os.environ.get('DATASET_ROOT'):
    logger.info('Detected env-var setup (DATASET_ROOT=%s)', os.environ['DATASET_ROOT'])
    CODE_ROOT = os.environ.get('CODE_ROOT')
    DATASET_ROOT = os.environ['DATASET_ROOT']
    WANDB_API_KEY_FILE = None
    PVC_ROOT = None
    BENCHMARK_ROOT = os.environ.get('BENCHMARK_ROOT')
else:
    raise ValueError('Unknown setup – set the DATASET_ROOT environment variable')

CODE_ROOT = os.environ.get('CODE_ROOT', CODE_ROOT)
DATASET_ROOT = os.environ.get('DATASET_ROOT', DATASET_ROOT)
BENCHMARK_ROOT = os.environ.get('BENCHMARK_ROOT', BENCHMARK_ROOT)
PVC_ROOT = os.environ.get('PVC_ROOT', PVC_ROOT)
WANDB_API_KEY_FILE = os.environ.get('WANDB_API_KEY_FILE', WANDB_API_KEY_FILE)
for _name, _value in (('CODE_ROOT', CODE_ROOT), ('DATASET_ROOT', DATASET_ROOT),
                      ('BENCHMARK_ROOT', BENCHMARK_ROOT), ('PVC_ROOT', PVC_ROOT),
                      ('WANDB_API_KEY_FILE', WANDB_API_KEY_FILE)):
    if os.environ.get(_name):
        logger.info('%s overridden from environment: %s', _name, _value)

DO_NOT_OVERRIDE = [
    'train', 'test', 'visualize', 'load_from_checkpoint', 'input_types', 'requires',
    'benchmark', 'restrictions', 'benchmark_split', 'resume',
    'experiment_name', 'project_name', 'seed', 'lr', 'epochs', 'patience',
    'early_stopping_metric', 'modality_drop_override', 'modality_dropout_scheme',
]

HSQC_TYPE = 0
C_NMR_TYPE = 1
H_NMR_TYPE = 2
MW_TYPE = 3
MS_TYPE = 4
MS_NEG_TYPE = 5
FORMULA_TYPE = 6
HMBC_TYPE = 7
COSY_TYPE = 8

INPUT_MAP = {
    'hsqc': HSQC_TYPE,
    'c_nmr': C_NMR_TYPE,
    'h_nmr': H_NMR_TYPE,
    'mw': MW_TYPE,
    'mass_spec': MS_TYPE,
    'mass_spec_neg': MS_NEG_TYPE,
    'formula': FORMULA_TYPE,
    'hmbc': HMBC_TYPE,
    'cosy': COSY_TYPE,
}

FORMULA_ELEMENTS: List[str] = [
    'H', 'C', 'O', 'N', 'S', 'Cl', 'Br', 'P', 'F', 'I', 'Si', 'B', 'Se', 'As', 'Fe',
    'Mg', 'Co', 'Na', 'Te', 'V', 'Al', 'Zn', 'Sn', 'Au', 'Mo', 'Cu', 'W', 'Ni', 'Bi',
    'Ga', 'K', 'Pb', 'Ca', 'Gd', 'Sb', 'Sr', 'Ru', 'Ho', 'Cr', '*',
]
