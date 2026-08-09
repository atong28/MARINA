import os
import time

import torch
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .marina import MARINA, MARINAArgs, MARINADataModule
from .spectre import SPECTRE, SPECTREArgs
from .log import get_logger, ErrorLoggingCallback
from .test import test_marina
from .benchmark_callback import BenchmarkCosineCallback
from .data.fp_loader import EntropyFPLoader

logger = get_logger(__file__)

def train_marina(
    args: MARINAArgs | SPECTREArgs,
    data_module: MARINADataModule,
    model: MARINA | SPECTRE,
    results_path: str,
    wandb_run=None,
    fp_loader: EntropyFPLoader | None = None,
    ckpt_dir: str | None = None
) -> None:
    torch.set_float32_matmul_precision('high')
    logger.info(f'[Main] Results Path: {results_path}')
    # ckpt_dir is where checkpoints are written. It differs from results_path only when
    # resuming, where they must outlive the pod.
    ckpt_dir = ckpt_dir or results_path
    if ckpt_dir != results_path:
        logger.info(f'[Main] Checkpoint Path: {ckpt_dir}')
    try:
        logger.info(f'[Main] Using GPU : {torch.cuda.get_device_name()}')
    except:
        logger.info(f'[Main] Using GPU: unknown type')
    wandb_logger = WandbLogger(experiment=wandb_run)
    metric = args.early_stopping_metric
    ckpt_callback = cb.ModelCheckpoint(
        monitor=metric,
        mode='max',
        save_last=False,
        save_top_k=1,
        dirpath=ckpt_dir,
        filename='epoch_{epoch:d}'
    )
    early_stopping = EarlyStopping(
        monitor=metric,
        mode='max',
        patience=args.patience
    )
    lr_monitor = cb.LearningRateMonitor(logging_interval="step")
    error_callback = ErrorLoggingCallback()
    benchmark_cos = BenchmarkCosineCallback(args, fp_loader)
    # Lightning >=2.6 defaults to RichProgressBar whenever `rich` is importable
    # (it is, via tyro). Rich renders nothing to a non-TTY, so pod logs show no
    # epoch output at all. Pin tqdm, which still writes progress when piped.
    progress_bar = cb.TQDMProgressBar()

    callbacks = [early_stopping, lr_monitor, ckpt_callback, error_callback,
                 benchmark_cos, progress_bar]

    if args.resume:
        # A rolling checkpoint on its own schedule. save_last=True on the monitored
        # callback would NOT do this: it only fires when the metric improves, so once
        # val/mean_cos plateaus the "last" checkpoint goes stale for hundreds of epochs
        # and resuming from it silently redoes all of them.
        # monitor=None + save_top_k=1 overwrites one file every_n_epochs, so the write
        # cost is bounded regardless of run length.
        callbacks.append(cb.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='last',
            save_top_k=1,
            every_n_epochs=args.checkpoint_every_n_epochs,
        ))
    early_stopping = EarlyStopping(
        monitor=metric,
        mode='max',
        patience=args.patience
    )
    lr_monitor = cb.LearningRateMonitor(logging_interval="step")
    error_callback = ErrorLoggingCallback()
    benchmark_cos = BenchmarkCosineCallback(args, fp_loader)
    # Lightning >=2.6 defaults to RichProgressBar whenever `rich` is importable
    # (it is, via tyro). Rich renders nothing to a non-TTY, so pod logs show no
    # epoch output at all. Pin tqdm, which still writes progress when piped.
    progress_bar = cb.TQDMProgressBar()

    callbacks = [early_stopping, lr_monitor, ckpt_callback, error_callback,
                 benchmark_cos, progress_bar]

    if args.resume:
        # A rolling checkpoint on its own schedule. save_last=True on the monitored
        # callback would NOT do this: it only fires when the metric improves, so once
        # val/mean_cos plateaus the "last" checkpoint goes stale for hundreds of epochs
        # and resuming from it silently redoes all of them.
        # monitor=None + save_top_k=1 overwrites one file every_n_epochs, so the write
        # cost is bounded regardless of run length.
        callbacks.append(cb.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='last',
            save_top_k=1,
            every_n_epochs=args.checkpoint_every_n_epochs,
        ))

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        precision="bf16-mixed",
        logger=wandb_logger,
        callbacks=callbacks,
        accumulate_grad_batches=args.accumulate_grad_batches_num,
        # MARINA3 batches routinely contain no 1H / 13C / MS-MS at all (no modality
        # exceeds 53.7% coverage), so those per-modality encoders receive no gradient
        # and plain DDP aborts at reducer._rebuild_buckets(). Numerically identical to
        # find_unused_parameters=False; only the reduction-set discovery changes.
        strategy='ddp_find_unused_parameters_true',
        gradient_clip_val=1.0
    )

    # Lightning restores optimizer, LR schedule, epoch counter and callback state (including
    # EarlyStopping's patience counter) only via ckpt_path. Passing the checkpoint to
    # load_from_checkpoint instead would restore weights alone and restart the schedule.
    resume_ckpt = None
    if args.resume:
        candidate = os.path.join(ckpt_dir, 'last.ckpt')
        if os.path.exists(candidate):
            resume_ckpt = candidate
            logger.info(f'[Main] Resuming from {resume_ckpt}')
        else:
            logger.info(f'[Main] No {candidate} yet; starting this run from scratch.')

    logger.info("[Main] Begin Training!")
    fit_start = time.time()
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt)
    logger.info(f'[Main] fit() finished in {time.time() - fit_start:.1f}s')
    trainer.strategy.barrier()

    if args.test and trainer.local_rank == 0:
        # Everything from here on holds the GPUs while doing little with them. Timing the
        # test phase separately from the results move is what distinguishes "the benchmark
        # is slow" from "the copy is slow" -- the two are otherwise one opaque gap between
        # the last training log line and the job exiting.
        test_start = time.time()
        test_marina(args, data_module, model, results_path, None, wandb_run=wandb_run, fp_loader=fp_loader)
        logger.info(f'[Main] test/benchmark phase finished in {time.time() - test_start:.1f}s')
