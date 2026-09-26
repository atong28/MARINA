import os
import pickle
import torch
import traceback
import sys
from itertools import islice
from typing import Any, List, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

from .args import MARINAArgs

from ..core.const import DEBUG_LEN, INPUTS_CANONICAL_ORDER, DATASET_ROOT, NON_SPECTRAL_INPUTS

from ..data.fp_loader import FPLoader
from ..data.inputs import MARINAInputLoader, MFInputLoader
from ..data.augment import PeakAugmenter
from ..log import get_logger

logger = get_logger(__file__)

def collate(batch):
    dicts, fps = zip(*batch)
    batch_inputs = {}
    for mod in INPUTS_CANONICAL_ORDER:
        seqs = [d.get(mod) for d in dicts]
        if all(x is None for x in seqs):
            continue
        D = next(x.shape[1] for x in seqs if isinstance(
            x, torch.Tensor) and x.ndim == 2)
        seqs = [
            x if (isinstance(x, torch.Tensor) and x.ndim ==
                    2) else torch.zeros((0, D), dtype=torch.float)
            for x in seqs
        ]
        batch_inputs[mod] = pad_sequence(seqs, batch_first=True)

    batch_fps = torch.stack(fps, dim=0)
    return batch_inputs, batch_fps

def format_inference_data(data: dict[int, Any]) -> dict[str, Any]:
    '''
    Return collated data for inference.
    
    data: stores same thing as input_loader would give:
    {
        'hsqc': ..., # shape: (N, 3)
        'c_nmr': ..., # shape: (N, 1)
        'h_nmr': ..., # shape: (N, 1)
        'mw': ... # shape: (1,)
    }
    
    Usage: 
    >>> inputs = data_module.format_inference_data(data)
    >>> output = model(**inputs)
    '''
    if 'mw' in data:
        data['mw'] = torch.tensor(data['mw']).view(1, 1)
    batch_inputs, _ = collate([(data, torch.tensor([0.0]))])
    return {'batch': batch_inputs}

class MARINADataset(Dataset):
    def __init__(self, args: MARINAArgs, fp_loader: FPLoader, split: str = 'train', override_input_types: Optional[list[str]] = None):
        try:
            self.args = args
            self.split = split
            if split != 'train':
                args.requires = args.input_types
            logger.debug(
                f'[MARINADataset] Initializing {split} dataset with input types {args.input_types} and required inputs {args.requires}')
            self.input_types = args.input_types if override_input_types is None else override_input_types
            self.requires = args.requires if override_input_types is None else override_input_types

            with open(os.path.join(DATASET_ROOT, 'index.pkl'), 'rb') as f:
                data: dict[int, Any] = pickle.load(f)
            data = {
                idx: entry for idx, entry in data.items()
                if entry['split'] == split and
                any(
                    entry[f'has_{input_type}']
                    for input_type in self.input_types
                    if input_type not in NON_SPECTRAL_INPUTS
                )
            }
            data_len = len(data)
            logger.debug(
                f'[MARINADataset] Requiring the following items to be present: {self.requires}')
            data = {
                idx: entry for idx, entry in data.items()
                if all(entry[f'has_{dtype}'] for dtype in self.requires)
            }
            logger.debug(
                f'[MARINADataset] Purged {data_len - len(data)}/{data_len} items. {len(data)} items remain')
            logger.debug(f'[MARINADataset] Dataset size: {len(data)}')
            if args.debug and len(data) > DEBUG_LEN:
                logger.debug(
                    f'[MARINADataset] Debug mode activated. Data length set to {DEBUG_LEN}')
                data = dict(islice(data.items(), DEBUG_LEN))

            if len(data) == 0:
                raise RuntimeError(
                    f'[MARINADataset] Dataset split {split} is empty!')

            self.jittering = args.jittering if split == 'train' else 0.0
            self.spectral_loader = MARINAInputLoader(
                DATASET_ROOT, data, split=split)
            self.mfp_loader = MFInputLoader(fp_loader)

            if split == 'train' and (args.aug_add_prob > 0 or args.aug_remove_prob > 0):
                dist_path = os.path.join(DATASET_ROOT, 'peak_distributions.npz')
                self.augmenter = PeakAugmenter(
                    dist_path,
                    p_add=args.aug_add_prob,
                    alpha_add=args.aug_alpha_add,
                    p_remove=args.aug_remove_prob,
                    alpha_remove=args.aug_alpha_remove,
                )
            else:
                self.augmenter = None
            
            self.drop_percentage = self.compute_drop_percentage(data)
            self.dropout_scheme = getattr(args, 'modality_dropout_scheme', 'bernoulli')
            if split == 'train':
                logger.debug(
                    f'[MARINADataset] Modality dropout scheme: {self.dropout_scheme}')

            self.data = list(data.items())

            logger.debug('[MARINADataset] Setup complete!')

        except Exception:
            logger.error(traceback.format_exc())
            logger.error(
                '[MARINADataset] While instantiating the dataset, ran into the above error.')
            sys.exit(1)
            
    def compute_drop_percentage(self, data: dict[int, Any]):
        drop_percentage = {}
        override = getattr(self.args, 'modality_drop_override', None)
        logger.debug(f'[MARINADataset] Computing drop percentage for input types: {self.input_types}')
        for input_type in self.input_types:
            if override is not None and input_type not in NON_SPECTRAL_INPUTS:
                # mw is deliberately excluded: it is always present in every dataset,
                # so leaving it at the hardcoded 0.5 keeps it identical across runs.
                drop_percentage[input_type] = override
            elif input_type not in NON_SPECTRAL_INPUTS:
                percent_present = sum(1 for entry in data.values() if entry[f'has_{input_type}']) / len(data)
                drop_percentage[input_type] = 1 - (0.5 / percent_present) if percent_present > 0.5 else 0.0
            else:
                drop_percentage[input_type] = 0.5 # hard coded for now for mw (always present)
        if override is not None:
            logger.info(f'[MARINADataset] modality_drop_override={override}; using it for all input types')
        return drop_percentage

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_idx, data_obj = self.data[idx]
        if self.split != 'train':
            input_types = set(self.input_types)
            return self.spectral_loader.load(data_idx, input_types), self.mfp_loader.load(data_idx)
        if self.dropout_scheme == 'uniform_cardinality':
            input_types = self._sample_uniform_cardinality(data_obj)
        else:
            input_types = self._sample_bernoulli(data_obj)
        return self.spectral_loader.load(data_idx, input_types, jittering=self.jittering, augmenter=self.augmenter), self.mfp_loader.load(data_idx)

    def _sample_bernoulli(self, data_obj):
        '''Legacy scheme: force-keep one random present spectral modality, then drop every
        other present modality independently at drop_percentage.'''
        available_types = {
            'hsqc': data_obj['has_hsqc'],
            'c_nmr': data_obj['has_c_nmr'],
            'h_nmr': data_obj['has_h_nmr'],
            'mass_spec': data_obj['has_mass_spec'],
            'mass_spec_neg': data_obj.get('has_mass_spec_neg', False)
        }
        drop_candidates = [
            k for k, v in available_types.items() if k in self.input_types and v]
        assert len(drop_candidates) > 0, 'Found an empty entry!'

        always_keep = drop_candidates[torch.randint(len(drop_candidates), (1,)).item()]
        input_types = set(self.input_types)
        for input_type in self.input_types:
            if not data_obj[f'has_{input_type}']:
                input_types.remove(input_type)
            elif (input_type != always_keep and
                  input_type not in self.requires and
                  torch.rand(1).item() < self.drop_percentage[input_type]):
                input_types.remove(input_type)
        return input_types

    def _sample_uniform_cardinality(self, data_obj):
        '''Uniform-over-cardinality scheme: pick the number of modalities uniformly, then a
        uniform subset of that size among the present modalities, requiring >=1 spectral
        modality. Unlike the bernoulli scheme, `requires` is not pinned here (it still
        filters which compounds enter the split); every present modality is a free
        candidate. Rejection-samples the (rare) all-non-spectral subset, which can only
        occur at cardinality <= len(NON_SPECTRAL_INPUTS).'''
        available = [m for m in self.input_types if data_obj[f'has_{m}']]
        assert available, 'Found an empty entry!'
        has_spectral = any(m not in NON_SPECTRAL_INPUTS for m in available)
        k = torch.randint(1, len(available) + 1, (1,)).item()
        while True:
            perm = torch.randperm(len(available))[:k].tolist()
            chosen = {available[i] for i in perm}
            if not has_spectral or any(m not in NON_SPECTRAL_INPUTS for m in chosen):
                return chosen


class MARINADataModule(pl.LightningDataModule):
    def __init__(self, args: MARINAArgs, fp_loader: FPLoader):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.val_num_workers = getattr(args, 'val_num_workers', args.num_workers)
        self.persistent_workers = bool(
            args.persistent_workers and self.num_workers > 0)
        self.val_persistent_workers = bool(
            args.persistent_workers and self.val_num_workers > 0)
        self.fp_loader = fp_loader
        self.test_types = [args.input_types] + args.additional_test_types
        self.test_types = [types for types in self.test_types if all(t in args.input_types for t in types)]
        self._fit_is_setup = False
        self._test_is_setup = False

    def setup(self, stage: Optional[str]):
        if (stage == "fit" or stage == "validate" or stage is None) and not self._fit_is_setup:
            self.train = MARINADataset(
                self.args,
                self.fp_loader,
                split='train'
            )

            self.val = [
                MARINADataset(
                    self.args,
                    self.fp_loader,
                    split='val',
                    override_input_types=input_type
                ) for input_type in self.test_types
            ]

            self._fit_is_setup = True

        if (stage == "test") and not self._test_is_setup:
            self.test = [
                MARINADataset(
                    self.args,
                    self.fp_loader,
                    split='test',
                    override_input_types=input_type
                ) for input_type in self.test_types
            ]

            self._test_is_setup = True

        if stage == "predict":
            raise NotImplementedError("Predict setup not implemented")

    def __getitem__(self, idx):
        if not self._fit_is_setup:
            self.setup(stage='fit')
        return self.train[idx]

    def train_dataloader(self) -> DataLoader:
        if not self._fit_is_setup:
            self.setup(stage='fit')

        return DataLoader(
            self.train,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
            multiprocessing_context="fork",
        )

    def val_dataloader(self) -> List[DataLoader]:
        if not self._fit_is_setup:
            self.setup(stage='fit')

        return [
            DataLoader(
                val_dl,
                batch_size=self.batch_size,
                collate_fn=self._collate_fn,
                num_workers=self.val_num_workers,
                pin_memory=True,
                persistent_workers=self.val_persistent_workers,
                multiprocessing_context="fork",
            )
            for val_dl in self.val
        ]

    def test_dataloader(self) -> List[DataLoader]:
        if not self._test_is_setup:
            self.setup(stage='test')

        return [
            DataLoader(
                test_dl,
                batch_size=self.batch_size,
                collate_fn=self._collate_fn,
                num_workers=self.val_num_workers,
                pin_memory=True,
                persistent_workers=self.val_persistent_workers,
                multiprocessing_context="fork",
            )
            for test_dl in self.test
        ]

    def _collate_fn(self, batch):
        """
        batch: list of (data_inputs: dict, mfp: Tensor)
        returns: (batch_inputs: dict[str→Tensor], batch_fps: Tensor)
        """
        return collate(batch)

    def format_inference_data(self, data: dict[int, Any]) -> dict[str, Any]:
        '''
        Return collated data for inference.
        
        data: stores same thing as input_loader would give:
        {
            'hsqc': ..., # shape: (N, 3)
            'c_nmr': ..., # shape: (N, 1)
            'h_nmr': ..., # shape: (N, 1)
            'mw': ... # shape: (1,)
        }
        
        Usage: 
        >>> inputs = data_module.format_inference_data(data)
        >>> output = model(**inputs)
        '''
        if 'mw' in data:
            data['mw'] = torch.tensor(data['mw']).view(1, 1)
        batch_inputs, _ = self._collate_fn([(data, torch.tensor([0.0]))])
        return {'batch': batch_inputs}