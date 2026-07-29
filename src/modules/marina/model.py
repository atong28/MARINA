import math
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanMetric

from .args import MARINAArgs

from ..core.const import SELF_ATTN_INPUTS
from ..core.metrics import cm
from ..core.ranker import RankingSet

from ..data.fp_loader import FPLoader
from ..data.encoder import build_encoder

from ..loss import BCECosineHybridLoss
from ..log import get_logger

logger = get_logger(__file__)
logger_should_sync_dist = torch.cuda.device_count() > 1


def _ln_component(ln: nn.LayerNorm, c: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Apply the linear part of `ln` to one additive component of its input.

    LayerNorm(z) = w * (z - mean(z)) / sigma(z) + b. Centering and the division
    by sigma both distribute over a sum z = sum_k c_k, so mapping every component
    through this and adding `b` once reproduces LayerNorm(z) exactly. `sigma`
    must be computed from the full z, not from `c`.
    """
    return ln.weight * (c - c.mean(-1, keepdim=True)) / sigma


def _ln_sigma(z: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(z.var(-1, keepdim=True, unbiased=False) + eps)


class CrossAttentionBlock(nn.Module):
    """
    Query (global CLS) attends to Key/Value (the spectral peaks + other tokens).
    """

    def __init__(self, dim_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_model,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim_model)
        self.ff = nn.Sequential(
            nn.Linear(dim_model, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim_model),
        )
        self.norm2 = nn.LayerNorm(dim_model)

    def forward(self, query, key, value, key_padding_mask=None):
        attn_out, _ = self.attn(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        q1 = self.norm1(query + attn_out)
        ff_out = self.ff(q1)
        out = self.norm2(q1 + ff_out)
        return out

    def attn_contributions(self, query, kv, key_padding_mask, spans):
        """
        Split this block's attention output into one additive term per span.

        Attention output is a sum over source positions, so partitioning that sum
        by modality is exact rather than approximate:

            attn_out = sum_j sum_h a[h,j] * W_O^h v[h,j] + b_O

        `spans` maps a bucket name to a boolean mask over the source positions.
        Returns (parts, attn_out), where sum(parts.values()) + out_proj.bias
        reproduces attn_out.
        """
        B, L, E = kv.shape
        H = self.attn.num_heads
        hd = E // H

        attn_out, alpha = self.attn(
            query,
            kv,
            kv,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        alpha = alpha[:, :, 0, :]                                # (B, H, L)

        v = F.linear(kv, self.attn.in_proj_weight[2 * E:], self.attn.in_proj_bias[2 * E:])
        v = v.view(B, L, H, hd).permute(0, 2, 1, 3)              # (B, H, L, hd)

        parts = {}
        for name, sel in spans.items():
            o = torch.einsum('bhl,bhld->bhd', alpha[:, :, sel], v[:, :, sel])
            parts[name] = F.linear(o.reshape(B, E), self.attn.out_proj.weight)
        return parts, attn_out

class MARINA(pl.LightningModule):
    def __init__(self, args: MARINAArgs, fp_loader: FPLoader):
        super().__init__()

        self.args = args
        self.fp_loader = fp_loader
        if self.global_rank == 0:
            logger.info("[MARINA] Started Initializing")
        self.fp_length = args.out_dim
        self.out_dim = args.out_dim
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.heads = args.heads
        self.layers = args.layers
        self.ff_dim = args.ff_dim
        self.dropout = args.dropout
        self.scheduler = args.scheduler
        self.dim_model = args.dim_model
        self.use_jaccard = args.use_jaccard
        self.freeze_weights = args.freeze_weights

        self.enc_nmr = build_encoder(
            args.dim_model,
            args.nmr_dim_coords,
            [args.c_wavelength_bounds, args.h_wavelength_bounds],
            args.nmr_is_sign_encoding
        )
        self.enc_c_nmr = build_encoder(
            args.dim_model,
            args.c_nmr_dim_coords,
            [args.c_wavelength_bounds],
            args.c_nmr_is_sign_encoding
        )
        self.enc_h_nmr = build_encoder(
            args.dim_model,
            args.h_nmr_dim_coords,
            [args.h_wavelength_bounds],
            args.h_nmr_is_sign_encoding
        )
        self.enc_ms = build_encoder(
            args.dim_model,
            args.ms_dim_coords,
            [args.mz_wavelength_bounds, args.intensity_wavelength_bounds],
            args.ms_is_sign_encoding
        )
        self.enc_mw = build_encoder(
            args.dim_model,
            args.mw_dim_coords,
            [args.mw_wavelength_bounds],
            args.mw_is_sign_encoding
        )
        self.encoders = {
            "hsqc": self.enc_nmr,
            "h_nmr": self.enc_h_nmr,
            "c_nmr": self.enc_c_nmr,
            "mass_spec": self.enc_ms,
            "mw": self.enc_mw
        }
        self.encoders = nn.ModuleDict(
            {k: v for k, v in self.encoders.items() if k in self.args.input_types})
        self.self_attn = nn.ModuleDict({
            modality: nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.dim_model, nhead=self.heads,
                    dim_feedforward=self.ff_dim,
                    batch_first=True, dropout=self.dropout
                ),
                num_layers=args.self_attn_layers[modality],
                enable_nested_tensor=False
            )
            for modality in self.encoders
        })
        self.mod_tokens = nn.ParameterDict({
            modality: nn.Parameter(torch.randn(1, 1, self.dim_model))
            for modality in self.encoders
        })
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(
                dim_model=self.dim_model,
                num_heads=self.heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout
            )
            for _ in range(self.layers)
        ])
        self.global_cls = nn.Parameter(torch.randn(1, 1, self.dim_model))
        self.fc = nn.Linear(self.dim_model, self.out_dim)
        self.loss = BCECosineHybridLoss(lambda_bce=args.lambda_hybrid)
        self._val_mm = torch.nn.ModuleDict()
        self._test_mm = torch.nn.ModuleDict()
        if self.freeze_weights:
            for parameter in self.parameters():
                parameter.requires_grad = False
        self.ranker = None
        self.spectral_types = ['all_inputs'] + ['_'.join(types) for types in self.args.additional_test_types if all(t in self.args.input_types for t in types)]
        if self.global_rank == 0:
            logger.info("[MARINA] Initialized")

    def _get_metric_mm(self, store: nn.ModuleDict, feat: str, input_type: str, sync_on_compute: bool = True) -> MeanMetric:
        key = f"{feat}__{input_type}"
        if key not in store:
            store[key] = MeanMetric(
                sync_on_compute=sync_on_compute).to(self.device)
        return store[key]

    def _encode(self, batch):
        """
        Run the per-modality encoders and self-attention stacks, then concatenate
        them into the joint memory the global CLS token reads from.

        Returns (joint_seq, joint_mask, segments), where `segments` lists
        (modality, length) in concatenation order. The first position of each
        segment is that modality's learned mod_token.
        """
        all_points = []
        all_masks = []
        segments = []
        for m, x in batch.items():
            if m not in SELF_ATTN_INPUTS:
                continue
            B, L, D_in = x.shape
            mask = (x.abs().sum(-1) == 0)
            enc_seq = self.encoders[m](x.view(B * L, D_in)).view(B, L, self.dim_model)
            mod_token = self.mod_tokens[m].to(enc_seq.device).expand(B, 1, -1)
            enc_seq = torch.cat([mod_token, enc_seq], dim=1)
            mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=enc_seq.device), mask], dim=1)
            attended = self.self_attn[m](enc_seq, src_key_padding_mask=mask)
            all_points.append(attended)
            all_masks.append(mask)
            segments.append((m, L + 1))
        return torch.cat(all_points, dim=1), torch.cat(all_masks, dim=1), segments

    def forward(self, batch, batch_idx=None, return_representations=False):
        B = next(iter(batch.values())).size(0)
        joint_seq, joint_mask, _ = self._encode(batch)
        global_token = self.global_cls.expand(B, 1, -1)
        for block in self.cross_blocks:
            global_token = block(
                query=global_token,
                key=joint_seq,
                value=joint_seq,
                key_padding_mask=joint_mask
            )
        out = self.fc(global_token.squeeze(1))
        if return_representations:
            return global_token.squeeze(1).detach().cpu().numpy()
        return out

    @torch.no_grad()
    def forward_contributions(self, batch, per_layer=False):
        """
        Decompose the final CLS token into additive per-modality contributions.

        The cross-attention stack reads from a memory that never updates across
        blocks, so each block's attention output splits exactly by source position
        and the surrounding residuals and LayerNorms carry those parts through
        linearly. The decomposition is exact, to float precision:

            sum(contributions.values()) == cls_final

        Buckets are one per modality, '<modality>:token' for that modality's
        learned mod_token, plus 'ffn' (the non-linear feedforward output, which
        cannot be attributed to a source), 'cls_init' (the learned query) and
        'bias'. A modality absent from the batch has all of its real positions
        masked, so its modality bucket goes to zero and only ':token' survives --
        that split separates reliance on actual peaks from a learned prior.

        Contributions are tracked per (bucket, block) internally and summed over
        blocks for the return value. With `per_layer=True` the breakdown is also
        returned, as (injection, survival) dicts keyed by (bucket, block):

            injection  ||c|| when the block wrote it into the residual stream
            survival   <c, cls_hat> / ||cls|| after every later LayerNorm

        Summing `survival` over blocks reproduces the aggregate share exactly, so
        the split is a refinement of the same quantity rather than a new measure.
        Their ratio is the attenuation an early write suffers before reaching the
        output -- note it folds in alignment with the final direction, not just
        loss of magnitude.

        Returns (cls_final, contributions[, injection, survival]).
        """
        B = next(iter(batch.values())).size(0)
        joint_seq, joint_mask, segments = self._encode(batch)
        L = joint_seq.size(1)

        spans = {}
        offset = 0
        for m, length in segments:
            token_sel = torch.zeros(L, dtype=torch.bool, device=joint_seq.device)
            token_sel[offset] = True
            spans[f'{m}:token'] = token_sel
            peak_sel = torch.zeros(L, dtype=torch.bool, device=joint_seq.device)
            peak_sel[offset + 1:offset + length] = True
            spans[m] = peak_sel
            offset += length

        global_token = self.global_cls.expand(B, 1, -1)

        # One slot per (bucket, block), preallocated so the whole set can be
        # pushed through each LayerNorm as a single broadcast op. Slots for
        # blocks not yet reached hold zeros, which _ln_component maps to zeros.
        keys = [('cls_init', -1), ('bias', -1)]
        for li in range(len(self.cross_blocks)):
            keys.extend((name, li) for name in spans)
            keys.append(('ffn', li))
        index = {k: i for i, k in enumerate(keys)}
        stack = torch.zeros(len(keys), B, 1, self.dim_model,
                            device=joint_seq.device, dtype=joint_seq.dtype)
        stack[index[('cls_init', -1)]] = global_token
        bias_i = index[('bias', -1)]
        injection = {}

        for li, block in enumerate(self.cross_blocks):
            parts, attn_out = block.attn_contributions(
                global_token, joint_seq, joint_mask, spans)
            for name, c in parts.items():
                stack[index[(name, li)]] = c.unsqueeze(1)
                injection[(name, li)] = c.norm(dim=-1)
            stack[bias_i] += block.attn.out_proj.bias

            z1 = global_token + attn_out
            sigma1 = _ln_sigma(z1, block.norm1.eps)
            stack = _ln_component(block.norm1, stack, sigma1)
            stack[bias_i] += block.norm1.bias
            q1 = block.norm1(z1)

            ff_out = block.ff(q1)
            stack[index[('ffn', li)]] = ff_out
            injection[('ffn', li)] = ff_out.squeeze(1).norm(dim=-1)

            z2 = q1 + ff_out
            sigma2 = _ln_sigma(z2, block.norm2.eps)
            stack = _ln_component(block.norm2, stack, sigma2)
            stack[bias_i] += block.norm2.bias
            global_token = block.norm2(z2)

        cls_final = global_token.squeeze(1)
        contribs = {}
        for (name, _), i in index.items():
            c = stack[i].squeeze(1)
            contribs[name] = contribs[name] + c if name in contribs else c
        if not per_layer:
            return cls_final, contribs

        norm = cls_final.norm(dim=-1)
        u = cls_final / norm.unsqueeze(-1)
        survival = {
            k: (stack[index[k]].squeeze(1) * u).sum(-1) / norm
            for k in injection
        }
        return cls_final, contribs, injection, survival

    def training_step(self, batch, batch_idx):
        batch_inputs, fps = batch
        logits = self.forward(batch_inputs)
        loss = self.loss(logits, fps)
        self.log("tr/loss", loss, prog_bar=True,
                 on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        batch_inputs, fps = batch
        logits = self.forward(batch_inputs)
        loss = self.loss(logits, fps)
        metrics, _ = cm(
            logits, fps, self.ranker, loss, self.loss,
            no_ranking=True
        )
        input_type_key = self.spectral_types[dataloader_idx]
        for feat, val in metrics.items():
            mm = self._get_metric_mm(self._val_mm, feat, input_type_key)
            mm.update(torch.tensor(
                val, device=self.device, dtype=torch.float32))

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        batch_inputs, fps = batch
        logits = self.forward(batch_inputs)
        loss = self.loss(logits, fps)
        metrics, _ = cm(
            logits, fps, self.ranker, loss, self.loss,
            no_ranking=False
        )
        input_type_key = self.spectral_types[dataloader_idx]
        for feat, val in metrics.items():
            mm = self._get_metric_mm(
                self._test_mm, feat, input_type_key, sync_on_compute=False)
            mm.update(torch.tensor(
                val, device=self.device, dtype=torch.float32))  

    def predict_step(self, batch, batch_idx, return_representations=False):
        raise NotImplementedError()

    def on_validation_epoch_end(self):
        keys = list(self._val_mm.keys())
        if not keys:
            return
        input_types = sorted({k.split("__", 1)[1] for k in keys})
        feats = sorted({k.split("__", 1)[0] for k in keys})
        di = {}
        for feat in feats:
            vals_for_avg = []
            for input_type in input_types:
                mm = self._get_metric_mm(
                    self._val_mm, feat, input_type, sync_on_compute=True)
                v = mm.compute().item()
                di[f"val/mean_{feat}/{input_type}"] = v
                if input_type == "all_inputs":
                    di[f"val/mean_{feat}"] = v
                vals_for_avg.append(v)
        for k, v in di.items():
            self.log(k, v, on_epoch=True, on_step=False, sync_dist=True)
        for mm in self._val_mm.values():
            mm.reset()

    def on_test_epoch_end(self):
        keys = list(self._test_mm.keys())
        if not keys:
            return
        input_types = sorted({k.split("__", 1)[1] for k in keys})
        feats = sorted({k.split("__", 1)[0] for k in keys})
        di = {}
        for feat in feats:
            vals_for_avg = []
            for input_type in input_types:
                mm = self._get_metric_mm(
                    self._test_mm, feat, input_type, sync_on_compute=False)
                v = mm.compute().item()
                di[f"test/mean_{feat}/{input_type}"] = v
                if input_type == "all_inputs":
                    di[f"test/mean_{feat}"] = v
                vals_for_avg.append(v)
        for k, v in di.items():
            self.log(k, v, on_epoch=True, on_step=False)
        for mm in self._test_mm.values():
            mm.reset()

    def configure_optimizers(self):
        if not self.scheduler:
            return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.scheduler == "cosine":
            opt = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay, betas=(0.9, 0.95))
            total_steps = self.trainer.estimated_stepping_batches
            steps_per_epoch = max(1, total_steps // self.trainer.max_epochs)
            warmup_steps = int((self.args.epochs // 10) *
                               steps_per_epoch) if self.args.warmup else 0

            # final LR as a fraction of base LR
            min_factor = self.args.eta_min / self.args.lr

            def lr_lambda(step: int):
                if step < warmup_steps:
                    return max(1e-6, step / max(1, warmup_steps))
                t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * t))
                return min_factor + (1 - min_factor) * cosine

            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sched,
                    "interval": "step",  # step every optimizer step
                    "name": "lr",
                },
            }

    def setup_ranker(self):
        store = self.fp_loader.load_rankingset(self.args.fp_type)
        self.ranker = RankingSet(store=store, metric="cosine")
