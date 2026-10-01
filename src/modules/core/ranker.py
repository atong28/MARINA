from typing import Tuple

import torch
import torch.nn.functional as F

from .utils import set_float32_highest_precision
from ..log import get_logger

logger = get_logger(__file__)

# A predicted bit is on when sigmoid(logit) >= 0.5 (logit >= 0), as in core.metrics.cm.
JACCARD_QUERY_THRESH = 0.5


def query_bits(q: torch.Tensor) -> torch.Tensor:
    """Binarize predicted probabilities (sigmoid outputs) into on-bits."""
    return (q >= JACCARD_QUERY_THRESH).to(torch.float32)


@set_float32_highest_precision
class RankingSet(torch.nn.Module):
    """
    Minimal fast similarity-ranking over a bank of fingerprints.

    Stores:
      - `data`: (N, D) float32 fingerprint matrix.
        May be dense or torch.sparse_csr_tensor.

    Typical usage (backward compatible):
      >>> ranker = RankingSet(store=fps)                     # cosine (default)
      >>> idxs = ranker.retrieve_idx(query_fp, n=50)
      >>> counts = ranker.batched_rank(queries, truths)

    New usage (for nonnegative count vectors, e.g., 39-D log1p counts):
      >>> ranker = RankingSet(store=fps, metric="tanimoto")  # real-valued Jaccard
    """

    def __init__(self, store: torch.Tensor, metric: str = "cosine", eps: float = 1e-12, debug: bool = False):
        """
        Args:
            store : (N, D) float32 tensor (dense or CSR). For cosine, rows should
                    already be L2-normalized if you want true cosine w.r.t. queries.
                    For tanimoto, store should be raw nonnegative weights (e.g., log1p(counts)).
                    For jaccard, any nonnegative store; nonzero entries are the on-bits.
            metric: "cosine" (default, backward-compatible), "tanimoto", or "jaccard"
                    (binary Tanimoto: queries are sigmoid probabilities, binarized at
                    JACCARD_QUERY_THRESH; truths are binarized at > 0).
            eps   : numerical floor to stabilize divisions.
            debug : extra logs.
        """
        super().__init__()

        self.metric = metric.lower()
        self.eps = float(eps)
        self.debug = debug

        if store.dtype != torch.float32:
            store = store.to(torch.float32)

        if self.metric == "jaccard":
            # binary Tanimoto: bank rows are on-bit sets (stored values, e.g. 1/sqrt(nnz), -> 1)
            if store.layout == torch.sparse_csr:
                store = torch.sparse_csr_tensor(store.crow_indices(), store.col_indices(),
                                                torch.ones_like(store.values()), size=store.size())
                row_nnz = store.crow_indices().diff().to(torch.float32)
            else:
                store = (store > 0).to(torch.float32)
                row_nnz = store.sum(dim=1)
            self.register_buffer("row_nnz", row_nnz, persistent=False)

        # Keep as buffer so it moves with .to(device) but isn't a parameter
        self.register_buffer("data", store, persistent=False)
        logger.info(
            f"[RankingSet] Initialized with {self.data.size(0)} sample(s); metric={self.metric}")

    @property
    def device(self) -> torch.device:
        return self.data.device

    # -------- Utilities --------

    @staticmethod
    def round(fp: torch.Tensor) -> torch.Tensor:
        """
        Convert a fingerprint vector to a binary vector with 1s at indices equal to the max value.
        """
        fp = fp.flatten()
        hi = torch.max(fp)
        out = torch.zeros_like(fp)
        out[fp == hi] = 1
        return out

    @staticmethod
    def normalized_to_nonzero(fp: torch.Tensor) -> Tuple[int, ...]:
        """
        Indices whose values equal (within tolerance) the max of `fp`.
        """
        hi = torch.max(fp)
        nonzero = torch.nonzero(torch.isclose(fp, hi), as_tuple=False)
        return tuple(nonzero[:, 0].tolist())

    def pair_sim(self, queries: torch.Tensor, truths: torch.Tensor) -> torch.Tensor:
        """(Q,) similarity of each query to its own truth under the ranking metric, i.e. the
        threshold its gold row sits at in the bank. Truths may be binary or L2-normalized."""
        if queries.dim() == 1:
            queries, truths = queries.unsqueeze(0), truths.unsqueeze(0)
        if self.metric == "cosine":
            return torch.sum(F.normalize(queries, dim=1, p=2.0) * F.normalize(truths, dim=1, p=2.0), dim=1)
        if self.metric == "jaccard":
            qb, tb = query_bits(queries), (truths > 0).to(torch.float32)
            inter = torch.sum(qb * tb, dim=1)
            union = qb.sum(dim=1) + tb.sum(dim=1) - inter
            return inter / union.clamp_min(1.0)
        raise ValueError(f"pair_sim not implemented for metric {self.metric}")

    # -------- Internal helpers --------
    def _sims(self, queries: torch.Tensor) -> torch.Tensor:
        q = queries.to(self.device)

        if self.metric == "cosine":
            qn = F.normalize(q, dim=1, p=2.0)
            return self.data @ qn.T

        elif self.metric == "jaccard":
            qb = query_bits(q)                                       # (Q, D) 0/1
            inter = self.data @ qb.T                                 # (N, Q) |row & q|
            union = self.row_nnz.unsqueeze(1) + qb.sum(dim=1).unsqueeze(0) - inter
            return inter / union.clamp_min(1.0)

        elif self.metric == "tanimoto":
            q = torch.clamp(q, min=0)              # <- ensure nonnegative
            dot = self.data @ q.T                  # (N, Q)
            q_sq = (q * q).sum(dim=1, keepdim=True)  # (Q,1)
            denom = (self.row_sq.unsqueeze(1) +
                     q_sq.T - dot).clamp_min(self.eps)
            return dot / denom

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    # -------- Retrieval & ranking --------

    def retrieve_idx(self, query: torch.Tensor, n: int = 50) -> torch.Tensor:
        """
        Top-N nearest neighbors by the configured metric.

        Args:
            query: (D,) or (B, D)
            n: number of neighbors.

        Returns:
            (n,) if single query; else (n, B) indices.
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        sims = self._sims(query)                                 # (N, B)
        _, idxs = torch.topk(sims, k=min(n, sims.size(0)), dim=0)
        return idxs.squeeze(1) if idxs.size(1) == 1 else idxs

    def jaccard_rank(
        self,
        data: torch.Tensor,
        queries: torch.Tensor,
        truths: torch.Tensor,
        thresh: torch.Tensor,
    ) -> torch.Tensor:
        """
        Count, per query, how many entries in `data` meet/exceed the Jaccard threshold.
        This path is unchanged and expects *binary* queries/truths.
        """
        assert queries.size() == truths.size(), "queries and truths must share shape"
        counts = []
        with torch.no_grad():
            for i, q in enumerate(queries):
                inter = torch.sum((data * q) > 0, dim=1)
                union = torch.sum((data + q) > 0, dim=1).clamp_min(1)
                jacc = inter / union
                counts.append(torch.sum(jacc >= thresh[i], dtype=torch.int32))
        out = torch.stack(counts).to(torch.int32)
        return out - 1  # discount the gold row if present

    def dot_prod_rank(
        self,
        data: torch.Tensor,
        queries: torch.Tensor,
        truths: torch.Tensor,
        thresh: torch.Tensor,
        tie_aware: bool = False,
    ) -> torch.Tensor:
        """
        Count, per query, how many entries in `data` meet/exceed the *cosine* threshold.

        Expects `queries` and `truths` already L2-normalized. Returns (Q,) int32 counts.

        tie_aware=False (default): strict/pessimistic -- rows tied with the truth
            threshold count against it; the self-row is dropped (minus 1). This is the
            training test-loop convention (matches test_result.pkl rank@k).
        tie_aware=True: optimistic (decision D6) -- only rows STRICTLY better than the
            truth threshold count; fingerprint-identical twins tied with the truth (and
            the self-row, which sits at thresh) are excluded, so no -1.
        """
        assert queries.size() == truths.size(), "queries and truths must share shape"
        with torch.no_grad():
            sims = data @ queries.T  # (N, Q)
            close = torch.isclose(sims, thresh)
            if tie_aware:
                ct = torch.sum(
                    torch.logical_and(sims > thresh, torch.logical_not(close)),
                    dim=0,
                    keepdim=True,
                    dtype=torch.int32,
                )
            else:
                ct = torch.sum(
                    torch.logical_or(sims >= thresh, close),
                    dim=0,
                    keepdim=True,
                    dtype=torch.int32,
                )
                ct = ct - 1
            if self.debug:
                truth_sims = data @ truths.T
                logger.debug(f"truth_sims shape: {truth_sims.shape}")
                logger.debug(f"ct_greater:\n{ct}")
            return ct.squeeze(0)

    def batched_rank(
        self,
        queries: torch.Tensor,
        truths: torch.Tensor,
        use_jaccard: bool = False,
        tie_aware: bool = False,
    ) -> torch.Tensor:
        """
        Rank each query against `self.data` and count how many entries beat its query-specific threshold.

        If `use_jaccard`:
            - Threshold = Jaccard(query, truth) per query (binary).
        Else if metric == "cosine":
            - Threshold = cosine(query_i, truth_i) per query (after L2-normalization).
        Else if metric == "tanimoto":
            - Threshold = Tanimoto(query_i, truth_i) per query (no normalization).
        Else if metric == "jaccard":
            - Threshold = binary Tanimoto(query_i >= 0.5, truth_i > 0); same strict / tie-aware
              conventions as the cosine path.
        """
        with torch.no_grad():
            if use_jaccard:
                intersection = torch.sum(queries * truths, dim=1)
                union = torch.sum((queries + truths) > 0, dim=1).clamp_min(1)
                thresh = intersection / union  # (Q,)
                return self.jaccard_rank(self.data, queries, truths, thresh)
            if self.metric == "cosine":
                qn = F.normalize(queries, dim=1, p=2.0)
                tn = F.normalize(truths, dim=1, p=2.0)
                thresh = torch.sum((qn * tn), dim=1, keepdim=True).T  # (1, Q)
                return self.dot_prod_rank(self.data, qn, tn, thresh, tie_aware=tie_aware)
            if self.metric == "jaccard":
                sims = self._sims(queries)                                    # (N, Q)
                thresh = self.pair_sim(queries.to(self.device), truths.to(self.device)).unsqueeze(0)
                close = torch.isclose(sims, thresh)
                if tie_aware:
                    ct = torch.sum(torch.logical_and(sims > thresh, torch.logical_not(close)),
                                   dim=0, dtype=torch.int32)
                else:
                    ct = torch.sum(torch.logical_or(sims >= thresh, close), dim=0, dtype=torch.int32) - 1
                return ct
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
