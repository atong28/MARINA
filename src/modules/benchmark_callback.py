import os
import pickle

import torch
import torch.distributed as dist
import pytorch_lightning as pl

from .benchmark import cos_sim, filter_data
from .core.const import BENCHMARK_ROOT
from .log import get_logger

logger = get_logger(__file__)


class BenchmarkCosineCallback(pl.Callback):
    """
    Every validation epoch, run a forward pass over the annotated benchmark and
    log the mean cosine similarity between the predicted and ground-truth
    structural fingerprints. This is the same `mean_cos` as the post-training
    benchmark, but *without* the expensive ranking/dereplication retrieval, so
    it is cheap enough to track live.

    Work is sharded across DDP ranks and reduced, so the per-epoch cost is one
    forward pass over (benchmark size / world_size) entries.
    """

    def __init__(self, args, fp_loader):
        super().__init__()
        self.args = args
        self.fp_loader = fp_loader
        self.restrictions = (
            args.input_types if args.restrictions is None else args.restrictions
        )
        self._prepared = False
        # name -> list of (raw_input_dict, normalized_target_fp)
        self._entries: dict[str, list] = {}

    def _prepare(self) -> None:
        if self._prepared:
            return
        self._prepared = True
        if BENCHMARK_ROOT is None:
            logger.warning("[BenchmarkCosine] BENCHMARK_ROOT not set; skipping.")
            return
        sources = [
            ("benchmark", "benchmark.pkl"),
            ("benchmark_journal", "benchmark-journal.pkl"),
        ]
        for name, fname in sources:
            path = os.path.join(BENCHMARK_ROOT, fname)
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                data = pickle.load(f)
            if self.args.benchmark_split != "all":
                data = {
                    k: v for k, v in data.items()
                    if v["split"] == self.args.benchmark_split
                }
            prepared = []
            for entry in data.values():
                sfp = self.fp_loader.build_mfp_for_smiles(entry["smiles"])
                sfp = sfp / torch.norm(sfp)
                prepared.append((entry["input"], sfp))
            self._entries[name] = prepared
            logger.info(
                f"[BenchmarkCosine] Prepared {len(prepared)} '{name}' entries "
                f"(split={self.args.benchmark_split})."
            )

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        self._prepare()
        if not self._entries:
            return

        dm = trainer.datamodule
        device = pl_module.device
        world = trainer.world_size
        rank = trainer.global_rank

        for name, prepared in self._entries.items():
            shard = prepared[rank::world] if world > 1 else prepared
            local_sum = 0.0
            for raw_input, sfp in shard:
                inputs = dm.format_inference_data(
                    filter_data(raw_input, self.restrictions)
                )
                batch = {k: v.to(device) for k, v in inputs["batch"].items()}
                pred = torch.sigmoid(pl_module(batch)[0])
                local_sum += cos_sim(pred, sfp.to(device)).item()

            stats = torch.tensor([local_sum, float(len(shard))], device=device)
            if world > 1 and dist.is_available() and dist.is_initialized():
                dist.all_reduce(stats)
            mean_cos = (stats[0] / stats[1]).item() if stats[1] > 0 else 0.0

            # All ranks hold the same reduced value -> log without further sync.
            pl_module.log(
                f"val/{name}_cos", mean_cos,
                on_epoch=True, on_step=False, sync_dist=False,
            )
