#!/usr/bin/env python3
"""Build the main-paper result tables from the raw evaluation outputs.

Reads paper/specs/<table>.json and paper/results/raw-<metric>/<bench>/ (written by `paper/run_eval.sh collect`), and
writes paper/results/tables[-jaccard]/<table>.{tex,json} for the split the paper reports plus <table>_<other split>.
--metric cosine (paper, default) or jaccard picks the raw dir, the output dir and the mean-similarity column.
Missing inputs become \\tbd cells and are listed under "missing" in the json, so the tables can be rebuilt as runs land.

Run (from the MARINA repo root):  DATASET_ROOT=/tmp pixi run python paper/make_tables.py
(DATASET_ROOT only satisfies the src import guard used by unpickling; nothing else reads it.)
"""
import argparse
import json
import os
import pickle
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
KS = (1, 5, 10)
METRIC = "cosine"                       # ranking metric of the raw results; set from --metric
SIM = {"cosine": "cos", "jaccard": "tani"}  # per-record / sim-json key of the mean similarity column
LABEL = {"cosine": ("exp-mean-cos", "cos", "cosine"), "jaccard": ("exp-mean-Tanimoto", "Tani", "binary Tanimoto")}

PRE = r"""\begingroup
\color{cInk}
%s\providecommand{\pmm}[1]{\ensuremath{{\scriptscriptstyle\,\pm#1}}}
\providecommand{\tbd}{--}
\setlength{\tabcolsep}{%s}
\renewcommand{\arraystretch}{%s}
"""
POST = "  \\bottomrule\n\\end{tabular}\n\\endgroup\n"


class Raw:
    """Loads raw results from paper/results/raw/<bench>/ and records what is missing."""

    def __init__(self, root):
        self.root, self.missing, self._cache = root, [], {}

    def journal(self, bench, name):
        path = os.path.join(self.root, bench, f"{name}_benchmark_journal_results.pkl")
        if path not in self._cache:
            self._cache[path] = pickle.load(open(path, "rb")) if os.path.exists(path) else None
            if self._cache[path] is None:
                self.missing.append(os.path.relpath(path, HERE))
        return self._cache[path]

    def sim(self, bench, name):
        path = os.path.join(self.root, bench, f"{name}_sim_results.json")
        if not os.path.exists(path):
            self.missing.append(os.path.relpath(path, HERE))
            return None
        return json.load(open(path))["metrics"]


def jmetrics(saved, key):
    """rank@k / ann@k (%) and mean cosine / Tanimoto for one split/subset of a journal result pkl."""
    if saved is None or key not in saved:
        return None
    recs = saved[key]
    n = len(recs)
    m = {"n": n}
    for k in KS:
        m[f"rank@{k}"] = 100.0 * sum(r["rank_strict"] < k for r in recs) / n
        if all("ann_rank" in r for r in recs):
            m[f"ann@{k}"] = 100.0 * sum(r["ann_rank"] < k for r in recs) / n
    if all("tani" in r for r in recs):
        m["tani"] = sum(r["tani"] for r in recs) / n
    m["cos"] = sum(r["cos"] for r in recs) / n
    m["sim"] = m.get(SIM[METRIC])
    return m


def agg(vals):
    """mean ± sample std over seeds; None if any seed is missing."""
    if not vals or any(v is None for v in vals):
        return None
    return {"mean": st.mean(vals), "std": st.stdev(vals) if len(vals) > 1 else 0.0, "seeds": vals}


def seed_stat(per_seed, metric):
    return agg([None if m is None else m.get(metric) for m in per_seed])


def pick_flagship(raw, spec_models, bench):
    """The flagship experiments named in the spec."""
    return spec_models["experiments"], "flagship"


def cell(s, pct=True, digits=2, sign=False):
    if s is None:
        return r"\tbd"
    m = f"{s['mean']:+.{digits}f}" if sign else f"{s['mean']:.{digits}f}"
    if sign:
        m = ("$+$" if s["mean"] >= 0 else "$-$") + m[1:]
    pc = "\\%" if pct and not sign else ""
    return f"{m}{pc}\\pmm{{{s['std']:.{digits}f}}}"


OUT_DIR = os.path.join(HERE, "results", "tables")  # tables-jaccard for --metric jaccard


def out(name, tex, data):
    d = OUT_DIR
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, f"{name}.tex"), "w").write(tex)
    json.dump(data, open(os.path.join(d, f"{name}.json"), "w"), indent=2)


# ---------------------------------------------------------------------------------------------------------- tables
def results_main(raw, spec, split):
    bench = spec["benchmark"]["bench"]
    exps, which = pick_flagship(raw, spec, bench)
    seeds = [raw.journal(bench, e) for e in exps]
    data = {"experiments": exps, "flagship": which, "split": split, "rows": {}}
    lines = []
    for part, (title, unit, metric) in enumerate([("(a) Structure dereplication", "exp-rank@$k$", "rank"),
                                                   ("(b) Structure annotation", "exp-ann@$k$", "ann")]):
        if part:
            lines.append("  \\midrule")
        lines.append(f"  \\multicolumn{{7}}{{@{{}}l}}{{\\hd{{{title}}}~({unit})}} \\\\")
        lines.append("  \\addlinespace[2pt]")
        for label, sub in spec["rows"]:
            base = [jmetrics(s, f"{split}/{sub}") if s else None for s in seeds]
            withf = [jmetrics(s, f"{split}/{sub}_formula") if s else None for s in seeds]
            cells, row = [], {}
            for k in KS:
                b = seed_stat(base, f"{metric}@{k}")
                f = [None if (x is None or y is None) else y[f"{metric}@{k}"] - x[f"{metric}@{k}"]
                     for x, y in zip(base, withf)]
                d = agg(f)
                cells += [cell(b), cell(d, sign=True)]
                row[f"{metric}@{k}"], row[f"{metric}@{k}_plusF_delta"] = b, d
            data["rows"].setdefault(label, {}).update(row)
            lines.append(f"  {label:<13}& " + " & ".join(cells) + " \\\\")
    head = (PRE % ("", "6pt", "1.2") + "\\begin{tabular}{@{}l cc cc cc@{}}\n  \\toprule\n"
            "  \\multirow{2}{*}{\\hd{Input}}\n"
            "      & \\multicolumn{2}{c}{\\hd{@1}}\n      & \\multicolumn{2}{c}{\\hd{@5}}\n"
            "      & \\multicolumn{2}{c}{\\hd{@10}} \\\\\n"
            "  \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "      & \\hd{Base} & \\hd{$+$F} & \\hd{Base} & \\hd{$+$F} & \\hd{Base} & \\hd{$+$F} \\\\\n  \\midrule\n")
    note = f"% {spec['table']} — journal {split}, flagship: {', '.join(exps)}; ranking = {LABEL[METRIC][2]}.\n"
    return note + head + "\n".join(lines) + "\n" + POST, data


def spectre_comparison(raw, spec, split):
    bench = spec["benchmark"]["bench"][split]
    mar = spec["models"]["MARINA"]
    exps, which = pick_flagship(raw, mar, bench)
    seeds = [raw.journal(bench, e) for e in exps]
    sp = raw.journal(bench, "spectre-deployed")
    data = {"experiments": exps, "flagship": which, "split": split, "rows": {}}
    lines = []
    for row in spec["rows"]:
        label, sub = row[0], row[1]
        marina_only = len(row) > 2
        cells, drow = [], {}
        for k in KS:
            m = seed_stat([jmetrics(s, f"{split}/{sub}") if s else None for s in seeds], f"rank@{k}")
            s = None if (marina_only or sp is None) else jmetrics(sp, f"{split}/{sub}")
            sv = None if s is None else s[f"rank@{k}"]
            mc = cell(m, pct=False)
            if m is not None and (marina_only or (sv is not None and m["mean"] > sv)):
                mc = r"\best{" + mc.replace(r"\pmm", r"}\pmm", 1) if r"\pmm" in mc else mc
            cells += [mc, r"\na" if marina_only else (r"\tbd" if sv is None else f"{sv:.2f}")]
            drow[f"rank@{k}"] = {"MARINA": m, "SPECTRE": sv}
        data["rows"][label] = drow
        lines.append(f"  {label:<26}& " + " & ".join(cells) + " \\\\")
    head = (PRE % ("", "6pt", "1.2") + "\\begin{tabular}{@{}l cc cc cc@{}}\n  \\toprule\n"
            "  \\multirow{2}{*}{\\hd{Input}}\n"
            "      & \\multicolumn{2}{c}{\\hd{exp-rank@1}}\n      & \\multicolumn{2}{c}{\\hd{exp-rank@5}}\n"
            "      & \\multicolumn{2}{c}{\\hd{exp-rank@10}} \\\\\n"
            "  \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "      & \\hd{MARINA} & \\hd{SPECTRE}\n      & \\hd{MARINA} & \\hd{SPECTRE}\n"
            "      & \\hd{MARINA} & \\hd{SPECTRE} \\\\\n  \\midrule\n")
    note = (f"% {spec['table']} — SPECTRE-clean journal {split} (n={spec['benchmark']['n'][split]}), "
            f"flagship: {', '.join(exps)}; ranking = {LABEL[METRIC][2]} for both models.\n")
    return note + head + "\n".join(lines) + "\n" + POST, data


def fp_comparison(raw, spec, split, main_only):
    bench, sub = spec["benchmark"]["bench"], spec["input"]["subset"]
    arms = {lab: a for lab, a in spec["arms"].items() if a["main"] or not main_only}
    stats = {}
    for lab, a in arms.items():
        per = [jmetrics(raw.journal(bench, e), f"{split}/{sub}") for e in a["experiments"]]
        stats[lab] = {**{f"rank@{k}": seed_stat(per, f"rank@{k}") for k in KS}, "sim": seed_stat(per, "sim")}
    cols = [f"rank@{k}" for k in KS] + ["sim"]
    best = {c: max((s[c]["mean"] for s in stats.values() if s[c]), default=None) for c in cols}
    lines = []
    for lab, s in stats.items():
        cells = []
        for c in cols:
            txt = cell(s[c], pct=False, digits=3 if c == "sim" else 2)
            if s[c] and best[c] is not None and s[c]["mean"] == best[c] and len(stats) > 1:
                txt = r"\best{" + txt.replace(r"\pmm", r"}\pmm", 1)
            cells.append(txt)
        lines.append(f"  {lab:<24}& " + " & ".join(cells) + " \\\\")
    head = (PRE % ("", "7pt", "1.2") + "\\begin{tabular}{@{}l cccc@{}}\n  \\toprule\n  \\hd{Fingerprint}\n"
            "      & \\hd{exp-rank@1} & \\hd{exp-rank@5} & \\hd{exp-rank@10} & \\hd{" + LABEL[METRIC][0] + "} \\\\\n  \\midrule\n")
    note = (f"% {spec['table']} — MARINA-DB-PRIVATE arms, journal {split}, input {sub} (NMR + formula; MW and MS/MS "
            f"withheld); ranking = {LABEL[METRIC][2]}; {'main rows' if main_only else 'all five fingerprints (appendix)'}.\n")
    return note + head + "\n".join(lines) + "\n" + POST, {"split": split, "subset": sub, "rows": stats}


def results_training_regime(raw, spec, split):
    bench = spec["experimental"]["bench"]
    data = {"split": split, "rows": {}}
    stats = {}
    for regime, exps in spec["arms"].items():
        js = [raw.journal(bench, e) for e in exps]
        ss = [raw.sim(bench, e) for e in exps]
        for label, jsub, scombo in spec["eval_inputs"]:
            per = [jmetrics(j, f"{split}/{jsub}") for j in js]
            st_ = {f"exp_r@{k}": seed_stat(per, f"rank@{k}") for k in KS}
            st_["exp_sim"] = seed_stat(per, "sim")
            for k in KS:
                st_[f"sim_r@{k}"] = agg([None if s is None or f"test/mean_rank_{k}/{scombo}" not in s
                                         else 100.0 * s[f"test/mean_rank_{k}/{scombo}"] for s in ss])
            key = f"test/mean_{SIM[METRIC]}/{scombo}"
            st_["sim_sim"] = agg([None if s is None or key not in s else s[key] for s in ss])
            stats[(regime, label)] = st_
    cols = [f"exp_r@{k}" for k in KS] + ["exp_sim"] + [f"sim_r@{k}" for k in KS] + ["sim_sim"]
    regimes = list(spec["arms"])
    lines = []
    for ri, regime in enumerate(regimes):
        if ri:
            lines.append("  \\midrule")
        for li, (label, _, _) in enumerate(spec["eval_inputs"]):
            cells = []
            for c in cols:
                s = stats[(regime, label)][c]
                txt = cell(s, pct=False, digits=3 if c.endswith("_sim") else 2)
                others = [stats[(r, label)][c] for r in regimes if r != regime]
                if s and all(o for o in others) and all(s["mean"] - s["std"] > o["mean"] + o["std"] for o in others):
                    txt = r"\best{" + txt.replace(r"\pmm", r"}\pmm", 1)
                cells.append(txt)
            first = (f"  \\multirow{{{len(spec['eval_inputs'])}}}{{*}}{{\\shortstack[l]{{{regime}}}}}" if li == 0
                     else "     ")
            lines.append(f"{first}\n      & {label:<14} & " + " & ".join(cells) + " \\\\")
            data["rows"][f"{regime} | {label}"] = stats[(regime, label)]
    head = (PRE % ("\\small\n", "5pt", "1.15") + "\\begin{tabular}{@{}l l cccc cccc@{}}\n  \\toprule\n"
            "  \\multirow{2}{*}{\\hd{Training}} & \\multirow{2}{*}{\\hd{Eval input}}\n"
            f"      & \\multicolumn{{4}}{{c}}{{\\hd{{Experimental (journal {split}, $n{{=}}{232 if split == 'val' else 234}$)}}}}\n"
            "      & \\multicolumn{4}{c}{\\hd{Simulated (MARINA-DB-PRIVATE test)}} \\\\\n"
            "  \\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\n"
            "      & & \\hd{r@1} & \\hd{r@5} & \\hd{r@10} & \\hd{" + LABEL[METRIC][1] + "}\n"
            "        & \\hd{r@1} & \\hd{r@5} & \\hd{r@10} & \\hd{" + LABEL[METRIC][1] + "} \\\\\n  \\midrule\n")
    note = (f"% {spec['table']} — uncapped-multiplicity MARINA-DB-PRIVATE arms; experimental = journal {split}, "
            f"simulated = MARINA-DB-PRIVATE test; ranking = {LABEL[METRIC][2]}.\n")
    return note + head + "\n".join(lines) + "\n" + POST, data


def sim_exp_gap(raw, spec, split="test"):
    """Simulated-vs-experimental gap for the flagship (Draft-2 layout): MARINA-DB test set (general simulated),
    MARINA-Bench with Mnova-simulated NMR, MARINA-Bench with experimental NMR. Cells are 3-seed means, 1 decimal."""
    exps = spec["experiments"]
    sims = [raw.sim("full", e) for e in exps]
    jsim = [raw.journal("simnmr", e) for e in exps]
    jexp = [raw.journal("full", e) for e in exps]

    def from_sim(m, combo):
        if m is None or f"test/mean_rank_1/{combo}" not in m:
            return None
        out = {f"rank@{k}": 100.0 * m[f"test/mean_rank_{k}/{combo}"] for k in KS}
        out["cos"] = m[f"test/mean_cos/{combo}"]
        return out

    lines, data = [], {"experiments": exps, "rows": {}}
    for ii, (label, combo, sub) in enumerate(spec["inputs"]):
        per_row = [[from_sim(m, combo) for m in sims],
                   [jmetrics(j, f"{split}/{sub}") for j in jsim],
                   [jmetrics(j, f"{split}/{sub}") for j in jexp]]
        if ii:
            lines.append("  \\midrule")
        for ri, (row, per) in enumerate(zip(spec["rows"], per_row)):
            st_ = {c: seed_stat(per, c) for c in ["rank@1", "rank@5", "rank@10", "cos"]}
            data["rows"][f"{label} | {row['label']}"] = st_
            cells = [r"\tbd" if st_[c] is None else (f"{st_[c]['mean']:.3f}" if c == "cos" else f"{st_[c]['mean']:.1f}")
                     for c in ["rank@1", "rank@5", "rank@10", "cos"]]
            first = f"\\multirow{{3}}{{*}}{{{label}}}" if ri == 0 else ""
            lines.append(f"  {first:<28} & {row['label']:<17} & " + " & ".join(cells) + " \\\\")
    head = (PRE % ("", "6pt", "1.18") +
            "\\begin{tabular}{@{}l l S[table-format=2.1] S[table-format=2.1] S[table-format=2.1] S[table-format=1.3]@{}}\n"
            "  \\toprule\n  & & \\multicolumn{3}{c}{\\hd{Retrieval accuracy (\\%)}} & \\\\\n  \\cmidrule(lr){3-5}\n"
            "  \\hd{Spectra} & \\hd{Input} & {\\hd{Top-1}} & {\\hd{Top-5}} & {\\hd{Top-10}} & {\\hd{Mean cos}} \\\\\n  \\midrule\n")
    note = (f"% {spec['table']} — flagship: {', '.join(exps)}; test set = MARINA-DB test split; benchmark = MARINA-Bench "
            f"{split} with Mnova-simulated / experimental NMR; ranking = {LABEL[METRIC][2]}.\n")
    return note + head + "\n".join(lines) + "\n" + POST, data

def main():
    global OUT_DIR, METRIC
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="cosine", choices=["cosine", "jaccard"])
    ap.add_argument("--raw", default=None, help="default paper/results/raw-<metric>")
    ap.add_argument("--out", default=None, help="default paper/results/tables (cosine) or tables-jaccard")
    a = ap.parse_args()
    METRIC = a.metric
    a.raw = a.raw or os.path.join(HERE, "results", f"raw-{METRIC}")
    OUT_DIR = a.out or os.path.join(HERE, "results", "tables" if METRIC == "cosine" else "tables-jaccard")
    specs = {k: json.load(open(os.path.join(HERE, "specs", f"{k}.json")))
             for k in ("results_main", "spectre_comparison", "fp_comparison", "results_training_regime")}
    specs_extra = {"sim_exp_gap": json.load(open(os.path.join(HERE, "specs", "sim_exp_gap.json")))}
    summary = {}
    for table, spec in specs.items():
        paper_split = (spec.get("benchmark") or spec["experimental"])["paper_split"]
        for split in (paper_split, "test" if paper_split == "val" else "val"):
            raw = Raw(a.raw)
            suffix = "" if split == paper_split else f"_{split}"
            if table == "results_main":
                tex, data = results_main(raw, spec, split)
            elif table == "spectre_comparison":
                tex, data = spectre_comparison(raw, spec, split)
            elif table == "fp_comparison":
                tex, data = fp_comparison(raw, spec, split, main_only=True)
                atex, adata = fp_comparison(raw, spec, split, main_only=False)
                adata["missing"] = sorted(set(raw.missing))
                out(f"fp_comparison_all{suffix}", atex, adata)
            else:
                tex, data = results_training_regime(raw, spec, split)
            data["missing"] = sorted(set(raw.missing))
            out(f"{table}{suffix}", tex, data)
            summary[f"{table}{suffix}"] = "complete" if not data["missing"] else f"{len(data['missing'])} inputs missing"
    for table, spec in specs_extra.items():   # appendix tables (one version)
        raw = Raw(a.raw)
        tex, data = sim_exp_gap(raw, spec)
        data["missing"] = sorted(set(raw.missing))
        out(table, tex, data)
        summary[table] = "complete" if not data["missing"] else f"{len(data['missing'])} inputs missing"
    for k, v in summary.items():
        print(f"{k:<34} {v}")


if __name__ == "__main__":
    main()
