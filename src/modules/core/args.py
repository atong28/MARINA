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
    restrictions: Optional[List[Literal['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw']]] = None

    input_types: List[Literal['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw']] = field(
        default_factory=lambda: ['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw']
    )

    requires: List[Literal['hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mw']] = field(
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
    # Trainer precision. bf16-mixed is the default every run to date has used, but bf16 has
    # no hardware support below sm_80 -- on SDSC's V100s (sm_70) it runs without tensor-core
    # acceleration, while 16-mixed does engage them. fp16 has a much narrower dynamic range,
    # so Lightning applies automatic loss scaling for it; switching changes numerics and
    # makes runs non-comparable with existing bf16 ones.
    precision: Literal['bf16-mixed', '16-mixed', '32-true'] = 'bf16-mixed'
    dropout: float = 0.1
    
    # jittering default value to wobble the spectra
    jittering: float = 0.5

    # Fixed per-modality drop probability. None keeps the computed default
    # (1 - 0.5/availability). Set 0.0 to disable dropping entirely, which is what a
    # mostly single-modality dataset wants since always_keep already protects those.
    modality_drop_override: Optional[float] = None

    # peak augmentation (injection + dropout); applies to hsqc/c_nmr/h_nmr during training only
    aug_add_prob: float = 0.0
    aug_remove_prob: float = 0.0
    aug_alpha_add: float = 0.1
    aug_alpha_remove: float = 0.1

    # BCE and cosine similarity loss lambda. 0 for full cosine similarity loss, 1 for full BCE loss.
    lambda_hybrid: float = 0.0
    
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