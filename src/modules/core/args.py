from typing import Literal, List, Optional
from pydantic.dataclasses import dataclass
from dataclasses import field


@dataclass
class SMARTArgs:
    experiment_name: str = 'smart-development'
    project_name: str = 'SMART'
    # random seed
    seed: int = 0
    # path to load checkpoint from
    load_from_checkpoint: str | None = None
    # Resume a run in place rather than warm-starting from it. load_from_checkpoint only
    # restores weights, so a chained job restarts the LR warmup and early-stopping patience
    # from scratch; this restores optimizer, schedule, epoch counter and callback state too.
    # Also writes checkpoints somewhere persistent and keeps a rolling `last.ckpt`, since on
    # Nautilus the default results dir is an emptyDir that dies with the pod.
    # Requires a stable SMART_RUN_ID across chunks so they share one results directory.
    resume: bool = False
    # How often --resume writes its rolling checkpoint. This is the only knob on resume's
    # I/O cost: a 2.17GB checkpoint takes ~4.6s to the PVC (measured 470 MB/s) versus ~1.4s
    # to ephemeral, so every 5 epochs is well under 1% of a 600s epoch. Raise it if the
    # filesystem is loaded, at the cost of redoing up to this many epochs after a restart.
    checkpoint_every_n_epochs: int = 5
    # whether to do training
    train: bool = True
    # whether to do testing
    test: bool = True
    # whether to do benchmarking
    benchmark: bool = True
    # restrictions on the input types to be used for benchmarking
    restrictions: Optional[List[Literal['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mass_spec_neg', 'mw']]] = None

    input_types: List[Literal['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mass_spec_neg', 'mw']] = field(
        default_factory=lambda: ['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw']
    )

    requires: List[Literal['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mass_spec_neg', 'mw']] = field(
        default_factory=lambda: []
    )

    # training args
    debug: bool = False
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 750
    patience: int = 30
    # metric monitored for early stopping and checkpointing (maximized)
    early_stopping_metric: str = 'val/mean_cos'
    persistent_workers: bool = True
    lr: float = 2e-4
    eta_min: float = 1e-5
    weight_decay: float = 0.0
    scheduler: Literal['cosine', 'none'] = 'cosine'
    freeze_weights: bool = False
    use_jaccard: bool = False
    warmup: bool = False
    accumulate_grad_batches_num: int = 4
    precision: Literal['bf16-mixed', '16-mixed', '32-true'] = 'bf16-mixed'
    dropout: float = 0.1
    
    # jittering default value to wobble the spectra
    jittering: float = 0.5

    # Fixed per-modality drop probability. None keeps the computed default
    # (1 - 0.5/availability). Set 0.0 to disable dropping entirely, which is what a
    # mostly single-modality dataset wants since always_keep already protects those.
    modality_drop_override: Optional[float] = None

    # Modality-dropout sampling scheme for TRAINING masks.
    #   'bernoulli'          : legacy — one random present spectral modality is force-kept,
    #                          every other present modality is dropped independently at
    #                          drop_percentage (see MARINADataset.compute_drop_percentage).
    #                          The number of surviving modalities is binomial → concentrated
    #                          in the middle, starving the single- and full-modality regimes.
    #   'uniform_cardinality': draw a modality COMBINATION with uniform-over-cardinality
    #                          weighting — pick the number of modalities uniformly, then a
    #                          uniform subset of that size among the present modalities,
    #                          requiring >=1 spectral modality. Equalizes the low- and
    #                          full-modality regimes. `requires` still filters which compounds
    #                          enter the split, but does NOT pin modalities under this scheme.
    modality_dropout_scheme: Literal['bernoulli', 'uniform_cardinality'] = 'bernoulli'

    # peak augmentation (injection + dropout); applies to hsqc/c_nmr/h_nmr during training only
    aug_add_prob: float = 0.0
    aug_remove_prob: float = 0.0
    aug_alpha_add: float = 0.1
    aug_alpha_remove: float = 0.1

    # BCE and cosine similarity loss lambda. 0 for full cosine similarity loss, 1 for full BCE loss.
    lambda_hybrid: float = 0.5
    
    # fp type for prediction and evaluation. fingerprint details should be stored in 
    #   DATASET_ROOT/RankingEntropy/
    # with the proper formatting.
    fp_type: Literal['RankingEntropy', 'RankingEntropySubstructure',
                     'RankingEntropyMultiplicity',
                     'RankingEntropyMultiplicityUncapped'] = 'RankingEntropy'
    
    # additional test types to be used for testing, always will test on all inputs
    additional_test_types: list[list[str]] = field(default_factory=lambda: [
        ['hsqc'], ['h_nmr'], ['c_nmr'], ['mass_spec']
    ])

    # split to use for benchmarking
    benchmark_split: Literal['val', 'test', 'all'] = 'val'