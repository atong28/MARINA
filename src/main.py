import os
from datetime import datetime

import torch

from .modules import (
    MARINA,
    MARINAArgs,
    MARINADataModule,
    SPECTRE,
    SPECTREArgs,
    SPECTREDataModule,
    parse_args,
    train_marina,
    test_marina,
    benchmark_marina
)
from .modules.util import (
    configure_system,
    set_global_seed,
    get_data_paths,
    configure_wandb,
    write_results
)
from .modules.log import get_logger
from .modules.data.fp_loader import make_fp_loader
from .modules.core.const import DATASET_ROOT


MARINA_MODEL_CLASSES = {
    "MARINA": MARINA,
    "SPECTRE": SPECTRE,
}

MARINA_DATAMODULE_CLASSES = {
    "MARINA": MARINADataModule,
    "SPECTRE": SPECTREDataModule,
}

def _resume_ckpt_dir(args, final_path: str, logger):
    """
    Where checkpoints go when --resume is set: somewhere that outlives the pod.

    results_path lives under DATASET_ROOT, which on Nautilus is an emptyDir -- a preempted
    or walltime-killed run loses its checkpoints entirely and has nothing to resume from.
    final_path is on the PVC. Returns None to leave the default alone when not resuming.
    """
    if not args.resume:
        return None
    if not final_path:
        logger.warning(
            '[Main] --resume set but no persistent results path is configured (PVC_ROOT '
            'unset); checkpoints stay on ephemeral storage and will not survive the pod.'
        )
        return None
    os.makedirs(final_path, exist_ok=True)
    return final_path


def launch_marina(args: MARINAArgs | SPECTREArgs, today: str):
    fp_loader = make_fp_loader(
        args.fp_type,
        entropy_out_dim=args.out_dim,
        retrieval_path=os.path.join(
            DATASET_ROOT,
            'retrieval.pkl'
        )
    )
    # create a mapping
    model_class = MARINA_MODEL_CLASSES[args.project_name]
    data_module_class = MARINA_DATAMODULE_CLASSES[args.project_name]
    # define the model and data module
    model: MARINA | SPECTRE = model_class(args, fp_loader)
    data_module: MARINADataModule | SPECTREDataModule = data_module_class(
        args,
        fp_loader
    )
    # paths to output data
    results_path, final_path = get_data_paths(args, today)

    logger = get_logger(__file__)

    # Resolved before the wandb/logging setup so logs.txt and params.json land alongside
    # the checkpoints on persistent storage; otherwise a preempted run takes its log with it.
    ckpt_dir = _resume_ckpt_dir(args, final_path, logger)

    # create a wandb run
    wandb_run = configure_wandb(args, results_path, today, log_dir=ckpt_dir)

    # Warm-start from a checkpoint before training. `train_marina` builds its own
    # Trainer and never sees load_from_checkpoint, so without this the flag silently
    # does nothing on a training run and the model trains from scratch.
    if args.train and args.load_from_checkpoint:
        state = torch.load(args.load_from_checkpoint, map_location='cpu')
        state_dict = state.get('state_dict', state)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(
            f'[Main] Warm-started from {args.load_from_checkpoint} '
            f'(missing={len(missing)}, unexpected={len(unexpected)})'
        )
        if missing or unexpected:
            logger.warning(f'[Main] missing={missing[:8]} unexpected={unexpected[:8]}')

    # train a model using the args as input
    if args.train:
        train_marina(
            args,
            data_module,
            model,
            results_path,
            wandb_run=wandb_run,
            fp_loader=fp_loader,
            ckpt_dir=ckpt_dir
        )
    elif args.test:
        test_marina(
            args,
            data_module,
            model,
            results_path,
            ckpt_path=args.load_from_checkpoint,
            wandb_run=wandb_run,
            fp_loader=fp_loader
        )
    elif args.benchmark:
        benchmark_marina(
            args,
            data_module,
            model,
            fp_loader,
            wandb_run=wandb_run,
            load_from_checkpoint=args.load_from_checkpoint
        )
    else:
        raise ValueError("[Main] Nothing to do!")

    # if it was a training run write out the results to a path and end the wandb run
    write_results(args, final_path, results_path, logger, wandb_run)

def main():
    configure_system()
    # create timestamp for run (allow overriding so launchers can pre-create
    # the results directory and capture stdout/stderr into it)
    today = os.environ.get("SMART_RUN_ID") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # parse the args
    args: MARINAArgs | SPECTREArgs = parse_args()
    set_global_seed(args.seed)
    launch_marina(args, today)

if __name__ == "__main__":
    main()
