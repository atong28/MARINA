import os
import time
import wandb
import shutil
import logging


def _tree_bytes(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _timed_move(src: str, dst: str, logger: logging.Logger = None) -> None:
    """
    Move one entry and record how long it took.

    results_path is node-local and final_path is CephFS, so this is a cross-device copy.
    It normally runs at a few hundred MB/s, but it happens *after* trainer.fit() while the
    job still holds its GPUs -- so when it is slow, the GPUs sit idle for the duration.
    Logging bytes and throughput makes a slow copy diagnosable after the fact instead of
    something to be reconstructed from wall-clock gaps.
    """
    nbytes = _tree_bytes(src)
    start = time.time()
    shutil.move(src, dst)
    elapsed = time.time() - start
    if logger:
        rate = (nbytes / elapsed / 1e6) if elapsed > 0 else float('inf')
        logger.info(
            "[Main] moved %s (%.2f GB) in %.1fs (%.0f MB/s)",
            os.path.basename(src), nbytes / 1e9, elapsed, rate
        )

from ..log import is_main_process
from ..marina.args import MARINAArgs
from ..spectre.args import SPECTREArgs


def write_results(
    args: MARINAArgs | SPECTREArgs,
    final_path: str,
    results_path: str,
    logger: logging.Logger = None,
    wandb_run=None
) -> None:
    """_summary_

    Args:
        args (MARINAArgs | SPECTREArgs): _description_
        final_path (str): _description_
        result_path (str): _description_
        logger (logging.Logger, optional): _description_. Defaults to None.
        wandb_run (_type_, optional): _description_. Defaults to None.
    """
    if is_main_process() and args.train:
        logger and logger.info("[Main] Moving results to final destination")
        overall_start = time.time()
        overall_bytes = _tree_bytes(results_path)

        os.makedirs(os.path.dirname(final_path), exist_ok=True)

        # If a launcher pre-created final_path (e.g., to tee stdout/stderr there),
        # avoid nesting results under final_path/today by merging contents instead.
        if os.path.exists(final_path):
            if not os.path.isdir(final_path):
                raise RuntimeError(f"[Main] final_path exists but is not a directory: {final_path}")

            os.makedirs(final_path, exist_ok=True)
            for name in os.listdir(results_path):
                src = os.path.join(results_path, name)
                dst = os.path.join(final_path, name)
                if os.path.exists(dst):
                    shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
                _timed_move(src, dst, logger)
            os.rmdir(results_path)
        else:
            _timed_move(results_path, final_path, logger)

        overall_elapsed = time.time() - overall_start
        logger and logger.info(
            "[Main] Results move complete: %.2f GB in %.1fs (%.0f MB/s) -> %s",
            overall_bytes / 1e9, overall_elapsed,
            (overall_bytes / overall_elapsed / 1e6) if overall_elapsed > 0 else float('inf'),
            final_path
        )

        if wandb_run is not None:
            wandb.finish()

    return
