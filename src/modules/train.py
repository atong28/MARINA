import os
import time
import glob
import zipfile

import torch
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO

from .marina import MARINA, MARINAArgs, MARINADataModule
from .spectre import SPECTRE, SPECTREArgs
from .log import get_logger, ErrorLoggingCallback
from .test import test_marina
from .benchmark_callback import BenchmarkCosineCallback
from .data.fp_loader import EntropyFPLoader

logger = get_logger(__file__)


class AtomicCheckpointIO(TorchCheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        tmp = f'{path}.partial'
        super().save_checkpoint(checkpoint, tmp, storage_options=storage_options)
        os.replace(tmp, path)


def _is_loadable(path: str) -> bool:
    try:
        with zipfile.ZipFile(path):
            return True
    except Exception:
        return False


def _resolve_resume_checkpoint(ckpt_dir: str) -> str | None:
    candidates = sorted(
        (p for p in glob.glob(os.path.join(ckpt_dir, '*.ckpt')) if os.path.isfile(p)),
        key=os.path.getmtime, reverse=True
    )
    for path in candidates:
        if _is_loadable(path):
            return path
        logger.error(f'[Main] Ignoring corrupt checkpoint {path} '
                     f'({os.path.getsize(path)} bytes, unreadable as a zip archive).')
    return None


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
    progress_bar = cb.TQDMProgressBar()

    callbacks = [early_stopping, lr_monitor, ckpt_callback, error_callback,
                 benchmark_cos, progress_bar]

    if args.resume:
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
        strategy='ddp_find_unused_parameters_true',
        gradient_clip_val=1.0,
        plugins=[AtomicCheckpointIO()]
    )

    resume_ckpt = None
    if args.resume:
        resume_ckpt = _resolve_resume_checkpoint(ckpt_dir)
        if resume_ckpt:
            logger.info(f'[Main] Resuming from {resume_ckpt}')
        else:
            logger.info(f'[Main] No intact checkpoint in {ckpt_dir}; '
                        f'starting this run from scratch.')

    logger.info("[Main] Begin Training!")
    fit_start = time.time()
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt)
    logger.info(f'[Main] fit() finished in {time.time() - fit_start:.1f}s')
    trainer.strategy.barrier()

    if args.test and trainer.local_rank == 0:
        test_start = time.time()
        test_marina(args, data_module, model, results_path, None, wandb_run=wandb_run, fp_loader=fp_loader)
        logger.info(f'[Main] test/benchmark phase finished in {time.time() - test_start:.1f}s')
