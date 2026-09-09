from typing import Any, List
import numpy as np
import torch
import os
import pickle
from collections import OrderedDict
from tqdm import tqdm
from rdkit import Chem

from wandb.sdk.wandb_run import Run
import wandb

from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs import ConvertToNumpyArray
from .marina import MARINAArgs,MARINADataModule, MARINA
from .spectre import SPECTREArgs, SPECTREDataModule, SPECTRE
from .log import get_logger
from .core.const import BENCHMARK_ROOT, INPUT_TYPES
from .data.fp_loader import EntropyFPLoader
from .data.fp_utils import load_smiles_index
from .data.formula import formula_to_vector

logger = get_logger(__file__)

_gen = GetMorganGenerator(radius=2, fpSize=2048)


def formula_vec_from_smiles(smiles: str) -> torch.Tensor:
    """Formula element-count vector (1, n_elements) for a benchmark molecule, using the same
    RDKit formula string + fixed vocabulary as the index.pkl precompute so it matches training."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    formula = rdMolDescriptors.CalcMolFormula(mol)
    return torch.tensor(formula_to_vector(formula), dtype=torch.float32).view(1, -1)

def cos_sim(pred, target):
    return torch.dot(pred, target) / (torch.norm(pred) * torch.norm(target))

def tanimoto_sim(pred, target):
    pred_bin = (pred > 0).int()
    target_bin = (target > 0).int()
    intersection = (pred_bin & target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection
    return intersection.float() / union.float()

def get_mfp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = _gen.GetFingerprint(mol)
    arr = np.zeros((2048,), dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return torch.tensor(arr, dtype=torch.float32)

def _to_device(obj, device):
    """Recursively move tensors inside dict/list/tuple to `device` (for GPU inference)."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def load_model(args: MARINAArgs | SPECTREArgs, model: MARINA | SPECTRE) -> None:
    if args.project_name == 'MARINA':
        model.load_state_dict(torch.load(args.load_from_checkpoint)['state_dict'])
    elif args.project_name == 'SPECTRE':
        state_dict = torch.load(args.load_from_checkpoint)['state_dict']
        encoders = []
        for k, v in model.state_dict().items():
            if 'sin_term' in k or 'cos_term' in k:
                encoders.append((k, v))
        state = [(k, v) for k, v in state_dict.items()]
        state = [state[0]] + encoders + state[1:]
        state_dict = OrderedDict(state)
        model.load_state_dict(state_dict)
    model.setup_ranker()
    # Move the whole model (incl. the ranker's rankingset buffer) to GPU when available —
    # otherwise the 531k-row cosine ranking + annotation retrieval run on CPU (0% GPU, slow).
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

def filter_data(data: dict[int, Any], restrictions: List[INPUT_TYPES]) -> dict[int, Any]:
    return {k: v for k, v in data.items() if k in restrictions}

def _rank_conventions(pred: torch.Tensor, sfp: torch.Tensor, ranker) -> tuple[int, int]:
    """0-based rank of the gold structure against the full bank, both tie conventions.

    strict : bank rows with sim >= cos(pred, gold) count against the gold (ties count),
             minus the self-row -- the training test-loop convention.
    tie    : only rows STRICTLY better than the gold count; fingerprint-identical twins
             tied with the gold (and the self-row, which sits at the threshold) are
             excluded (decision D6).
    """
    q = pred.reshape(1, -1)
    sims = ranker._sims(q).squeeze(1).float()                      # (N,)
    qn = torch.nn.functional.normalize(q, dim=1, p=2.0)[0]
    gn = torch.nn.functional.normalize(sfp.reshape(1, -1).to(qn.device), dim=1, p=2.0)[0]
    thr = torch.dot(qn, gn).float()
    close = torch.isclose(sims, thr.expand_as(sims))
    rank_strict = int((((sims >= thr) | close).sum() - 1).item())
    rank_tie = int(((sims > thr) & ~close).sum().item())
    return rank_strict, rank_tie


# Annotation success rate (SPECTRE-paper definition): a query is a "hit" at k if any of its
# top-k retrieved structures has ECFP4 (Morgan r=2, 2048-bit; get_mfp) cosine >= 0.8 to the
# true structure. Distinct from dereplication (exact-structure rank).
ANN_TOPK = 10       # deepest retrieval rank the @1/5/10 columns need
ANN_THRESH = 0.8    # ECFP4 cosine threshold for an annotation hit


def _retrieval_smiles(fp_loader, ranker):
    """idx->SMILES list aligned with the ranker's bank rows, for the annotation metric.
    Returns None (annotation disabled) if the retrieval index is unavailable or its length
    does not match the bank (e.g. an augmented SPECTRE bank)."""
    path = getattr(fp_loader, 'retrieval_path', None)
    if not path or not os.path.exists(path):
        logger.warning('[annotation] no retrieval_path on fp_loader; annotation disabled')
        return None
    try:
        idx2smi = load_smiles_index(path)
    except Exception as e:
        logger.warning(f'[annotation] could not load retrieval index: {e!r}; disabled')
        return None
    smiles = [idx2smi.get(i) for i in range(len(idx2smi))]
    bank = ranker.data.size(0)
    if len(smiles) != bank:
        logger.warning(f'[annotation] retrieval size {len(smiles)} != bank {bank}; '
                       'annotation disabled (augmented/mismatched bank)')
        return None
    return smiles


def _annotation_rank(gold_smiles: str, pred: torch.Tensor, ranker, retr_smiles, ecfp_cache) -> int:
    """0-based rank of the first top-ANN_TOPK retrieval whose ECFP4 cosine to the gold
    structure >= ANN_THRESH; returns ANN_TOPK if none (so it fails @1/5/10). ECFP4 of
    retrieved rows is cached by bank index across queries."""
    try:
        gold = get_mfp(gold_smiles)
    except Exception:
        return ANN_TOPK
    idxs = ranker.retrieve_idx(pred, ANN_TOPK).tolist()
    for r, idx in enumerate(idxs):
        if idx not in ecfp_cache:
            smi = retr_smiles[idx] if 0 <= idx < len(retr_smiles) else None
            try:
                ecfp_cache[idx] = get_mfp(smi) if smi else None
            except Exception:
                ecfp_cache[idx] = None
        fp = ecfp_cache[idx]
        if fp is not None and cos_sim(gold, fp).item() >= ANN_THRESH:
            return r
    return ANN_TOPK


def _run_benchmark_loop(
    benchmark_data: dict,
    data_module: MARINADataModule | SPECTREDataModule,
    model: MARINA | SPECTRE,
    fp_loader: EntropyFPLoader,
    restrictions: List,
    desc: str = 'Benchmarking',
) -> list[dict]:
    """One forward pass per entry under `restrictions`; return per-entry records
    [{cos, rank_strict, rank_tie[, ann_rank]}]. `ann_rank` (annotation success) is added
    when a retrieval-index SMILES map is available (disabled for augmented banks). Entries
    with no spectral modality left after the restriction are skipped."""
    recs = []
    ann_smiles = _retrieval_smiles(fp_loader, model.ranker)
    ecfp_cache: dict = {}
    dev = next(model.parameters()).device
    for entry in tqdm(benchmark_data.values(), desc=desc):
        raw_input = entry['input']
        if 'formula' in restrictions:
            raw_input = {**raw_input, 'formula': formula_vec_from_smiles(entry['smiles'])}
        clean = filter_data(raw_input, restrictions)
        if not any(k not in ('mw', 'formula') for k in clean):
            continue  # nothing spectral to feed for this subset
        inputs = _to_device(data_module.format_inference_data(clean), dev)
        with torch.no_grad():
            output = model(**inputs)
        pred = torch.sigmoid(output[0])
        sfp = fp_loader.build_mfp_for_smiles(entry['smiles'])
        sfp = (sfp / torch.norm(sfp)).to(pred.device)
        rs, rt = _rank_conventions(pred, sfp, model.ranker)
        rec = {'cos': cos_sim(pred, sfp).item(), 'rank_strict': rs, 'rank_tie': rt}
        if ann_smiles is not None:
            rec['ann_rank'] = _annotation_rank(
                entry['smiles'], pred, model.ranker, ann_smiles, ecfp_cache)
        recs.append(rec)
    return recs


def _summarise(recs: list[dict], prefix: str, wandb_metrics: dict) -> None:
    """Aggregate records into mean_cos + rank_strict/tie @1/5/10, log to the console,
    and accumulate into `wandb_metrics` (the caller flushes once)."""
    n = len(recs)
    if n == 0:
        logger.info(f'[{prefix}] no entries; skipped')
        return
    mean_cos = sum(r['cos'] for r in recs) / n
    m = {f"{prefix}/mean_cos": mean_cos, f"{prefix}/n": n}
    for k in (1, 5, 10):
        m[f"{prefix}/rank_strict_top{k}_pct"] = 100.0 * sum(r['rank_strict'] < k for r in recs) / n
        m[f"{prefix}/rank_tie_top{k}_pct"] = 100.0 * sum(r['rank_tie'] < k for r in recs) / n
    ann = [r['ann_rank'] for r in recs if 'ann_rank' in r]
    if ann:
        na = len(ann)
        for k in (1, 5, 10):
            m[f"{prefix}/ann_top{k}_pct"] = 100.0 * sum(a < k for a in ann) / na
    logger.info(
        f"[{prefix}] n={n} mean_cos={mean_cos:.4f} | strict @1/5/10 = "
        f"{m[f'{prefix}/rank_strict_top1_pct']:.2f}/{m[f'{prefix}/rank_strict_top5_pct']:.2f}/"
        f"{m[f'{prefix}/rank_strict_top10_pct']:.2f} | tie @1/5/10 = "
        f"{m[f'{prefix}/rank_tie_top1_pct']:.2f}/{m[f'{prefix}/rank_tie_top5_pct']:.2f}/"
        f"{m[f'{prefix}/rank_tie_top10_pct']:.2f}"
        + (f" | ann @1/5/10 = {m[f'{prefix}/ann_top1_pct']:.2f}/"
           f"{m[f'{prefix}/ann_top5_pct']:.2f}/{m[f'{prefix}/ann_top10_pct']:.2f}" if ann else "")
    )
    wandb_metrics.update(m)


def _journal_subsets(base: List) -> dict:
    """Journal modality subsets fed to the model. 'all' = the model's own inputs; 'nmr' =
    the three NMR modalities; 'msms' = positive + negative MS/MS; plus each spectral modality
    on its own and the pairwise NMR combinations (for the per-combo results tables). Every
    subset is intersected with `base` so a model only gets modalities it was trained with."""
    subs = {'all': list(base)}
    nmr = [m for m in ('hsqc', 'c_nmr', 'h_nmr') if m in base]
    msms = [m for m in ('mass_spec', 'mass_spec_neg') if m in base]
    if nmr:
        subs['nmr'] = nmr
    if msms:
        subs['msms'] = msms
    # single-modality ablations (Tables 1/3/S1 combos: HSQC / 13C / 1H / MS+ / MS-)
    for m in ('hsqc', 'c_nmr', 'h_nmr', 'mass_spec', 'mass_spec_neg'):
        if m in base:
            subs[m] = [m]
    # pairwise NMR combinations (Table S1 combos 6-8)
    for a, b in (('hsqc', 'c_nmr'), ('hsqc', 'h_nmr'), ('c_nmr', 'h_nmr')):
        if a in base and b in base:
            subs[f'{a}_{b}'] = [a, b]
    # NMR + MS/MS spectra (Table 1/S1 "NMR+MS/MS*" base) — the only combo that needs the
    # sim journal (real NMR + simulated MS/MS); present only if the model has MS/MS encoders.
    if nmr and msms:
        subs['nmr_msms'] = nmr + msms
    # "full NMR + formula" (Table 2 combo): NMR + MW + formula, MS/MS explicitly WITHHELD so
    # it is well-defined even when run on the sim journal (on the plain journal it equals
    # 'all'). `mw`/`formula` included only if the model was trained with them.
    if nmr:
        subs['nmr_mw_formula'] = nmr + [m for m in ('mw', 'formula') if m in base]
    return subs


def benchmark_marina(
    args: MARINAArgs | SPECTREArgs,
    data_module: MARINADataModule | SPECTREDataModule,
    model: MARINA | SPECTRE,
    fp_loader: EntropyFPLoader,
    wandb_run: Run | None = None,
    load_from_checkpoint: str | None = None,
) -> None:
    """Benchmark a MARINA/SPECTRE model on the journal set.

    Runs the journal (benchmark-journal.pkl) for BOTH the val and test splits, each under
    three modality subsets (all / nmr / msms), reporting mean_cos and the strict and
    tie-aware rank@1/5/10 for every combination. The NP-MRD benchmark and the >0.99
    dereplication metric are retired.
    """
    base = args.input_types if args.restrictions is None else args.restrictions
    if load_from_checkpoint is not None:
        load_model(args, model)
    if BENCHMARK_ROOT is None:
        raise ValueError('Benchmarking is not supported on this setup')

    journal_path = os.path.join(BENCHMARK_ROOT, "benchmark-journal.pkl")
    if not os.path.exists(journal_path):
        logger.warning(f'[Benchmark] benchmark-journal.pkl not found at {journal_path}; skipping.')
        return

    logger.info(f'[Benchmark] Benchmarking {model.__class__.__name__} (journal, val+test)')
    journal: dict[str, Any] = pickle.load(open(journal_path, 'rb'))
    subsets = _journal_subsets(base)
    os.makedirs(os.path.join(BENCHMARK_ROOT, 'benchmarks'), exist_ok=True)

    wandb_metrics: dict = {}
    saved: dict = {}
    for split in ('val', 'test'):
        split_data = {k: v for k, v in journal.items() if v.get('split') == split}
        for sub_name, sub_mods in subsets.items():
            recs = _run_benchmark_loop(
                split_data, data_module, model, fp_loader, sub_mods,
                desc=f'journal/{split}/{sub_name}')
            _summarise(recs, f'benchmark_journal/{split}/{sub_name}', wandb_metrics)
            saved[f'{split}/{sub_name}'] = recs

    out = os.path.join(BENCHMARK_ROOT, 'benchmarks', f"{args.experiment_name}_benchmark_journal_results.pkl")
    with open(out, 'wb') as f:
        pickle.dump(saved, f)
    logger.info(f'[Benchmark] Saved journal results to {out} ({os.path.getsize(out)} bytes)')
    if wandb_run is not None and wandb_metrics:
        wandb.log(wandb_metrics)