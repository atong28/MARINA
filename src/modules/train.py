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
    fp_loader: EntropyFPLoader | None = None
) -> None:
    torch.set_float32_matmul_precision('high')
    logger.info(f'[Main] Results Path: {results_path}')
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
        dirpath=results_path,
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
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        precision="bf16-mixed",
        logger=wandb_logger,
        callbacks=[early_stopping, lr_monitor, ckpt_callback, error_callback, benchmark_cos, progress_bar],
        accumulate_grad_batches=args.accumulate_grad_batches_num,
        # MARINA3 batches routinely contain no 1H / 13C / MS-MS at all (no modality
        # exceeds 53.7% coverage), so those per-modality encoders receive no gradient
        # and plain DDP aborts at reducer._rebuild_buckets(). Numerically identical to
        # find_unused_parameters=False; only the reduction-set discovery changes.
        strategy='ddp_find_unused_parameters_true',
        gradient_clip_val=1.0
    )

    logger.info("[Main] Begin Training!")
    trainer.fit(model, datamodule=data_module)
    trainer.strategy.barrier()

    if args.test and trainer.local_rank == 0:
        test_marina(args, data_module, model, results_path, None, wandb_run=wandb_run, fp_loader=fp_loader)
