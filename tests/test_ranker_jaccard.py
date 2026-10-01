"""RankingSet metric='jaccard' (binary Tanimoto) against a brute-force reference.

Run: DATASET_ROOT=/tmp PYTHONPATH=. pixi run python tests/test_ranker_jaccard.py   (pytest-compatible too;
DATASET_ROOT only satisfies the src.modules import guard, no data is read)
"""
import torch

from src.modules.core.ranker import RankingSet


def _bank(n=400, d=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    bits = (torch.rand(n, d, generator=g) < 0.15).float()
    bits[0] = 0.0                      # an empty row
    bits[1] = bits[2]                  # fingerprint-identical twins
    return bits


def _csr_normalized(bits):
    """Rankingset as stored on disk: CSR rows with values 1/sqrt(nnz)."""
    norm = bits / bits.sum(1, keepdim=True).clamp_min(1).sqrt()
    return norm.to_sparse_csr()


def _jaccard(a, b):
    inter = (a * b).sum(-1)
    union = a.sum(-1) + b.sum(-1) - inter
    return inter / union.clamp_min(1)


def _brute_rank(bits, prob, gold, tie_aware):
    q = (prob >= 0.5).float()
    sims = _jaccard(bits, q.unsqueeze(0))
    thr = _jaccard(q, gold)
    if tie_aware:
        return int((sims > thr + 1e-7).sum())
    return int((sims >= thr - 1e-7).sum()) - 1


def test_jaccard_matches_brute_force():
    bits = _bank()
    ranker = RankingSet(_csr_normalized(bits), metric="jaccard")
    g = torch.Generator().manual_seed(1)
    gold_idx = torch.arange(3, 43)
    gold = bits[gold_idx]
    # noisy predictions around the gold (flip ~10% of bits), as probabilities
    flip = (torch.rand(gold.shape, generator=g) < 0.1).float()
    prob = (gold + flip) % 2 * 0.8 + 0.1
    for tie_aware in (False, True):
        got = ranker.batched_rank(prob, gold, tie_aware=tie_aware)
        want = torch.tensor([_brute_rank(bits, p, t, tie_aware) for p, t in zip(prob, gold)])
        assert torch.equal(got.cpu().long(), want), (tie_aware, got, want)
    # L2-normalized truths (as the journal loop passes them) give the same thresholds
    gold_n = gold / gold.norm(dim=1, keepdim=True)
    assert torch.equal(ranker.batched_rank(prob, gold_n), ranker.batched_rank(prob, gold))


def test_jaccard_sims_and_retrieval():
    bits = _bank()
    ranker = RankingSet(_csr_normalized(bits), metric="jaccard")
    prob = bits[5:9] * 0.9 + 0.05                         # exact on-bits of rows 5..8
    sims = ranker._sims(prob)                            # (N, Q)
    ref = _jaccard(bits.unsqueeze(1), (prob >= 0.5).float().unsqueeze(0))
    assert torch.allclose(sims, ref)
    top = ranker.retrieve_idx(prob, 1)
    assert top.flatten().tolist() == [5, 6, 7, 8]
    assert torch.allclose(ranker.pair_sim(prob, bits[5:9]), torch.ones(4))


def test_cosine_path_unchanged():
    bits = _bank()
    store = _csr_normalized(bits)
    ranker = RankingSet(store, metric="cosine")
    prob = torch.rand(10, bits.size(1), generator=torch.Generator().manual_seed(2))
    gold = bits[10:20]
    got = ranker.batched_rank(prob, gold)
    qn = torch.nn.functional.normalize(prob, dim=1)
    gn = torch.nn.functional.normalize(gold, dim=1)
    sims = store.to_dense() @ qn.T
    thr = (qn * gn).sum(1)
    want = ((sims >= thr) | torch.isclose(sims, thr.expand_as(sims))).sum(0) - 1
    assert torch.equal(got.cpu().long(), want.long())
    assert torch.allclose(ranker.pair_sim(prob, gold), thr)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
