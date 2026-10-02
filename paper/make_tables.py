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
    tbd = sum(line.count(r"\tbd") for line in tex.split("\n") if "providecommand" not in line)
    if tbd:  # placeholder cells: inputs missing inside files that exist (e.g. sim combos not run yet)
        data.setdefault("missing", [])
        data["tbd_cells"] = tbd
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
            f"      & \\multicolumn{{4}}{{c}}{{\\hd{{Experimental (MARINA-Bench {split}, $n{{=}}{232 if split == 'val' else 234}$)}}}}\n"
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

S1_ROWS = [("NMR$+$MS/MS*", "nmr_msms", "hsqc_c_nmr_h_nmr_mass_spec_mass_spec_neg"), ("NMR", "nmr", "hsqc_c_nmr_h_nmr"),
           ("ME-HSQC", "hsqc", "hsqc"), ("$^{13}$C", "c_nmr", "c_nmr"), ("$^{1}$H", "h_nmr", "h_nmr"),
           ("ME-HSQC $+$ $^{13}$C", "hsqc_c_nmr", "hsqc_c_nmr"), ("ME-HSQC $+$ $^{1}$H", "hsqc_h_nmr", "hsqc_h_nmr"),
           ("$^{13}$C $+$ $^{1}$H", "c_nmr_h_nmr", "c_nmr_h_nmr")]
S1_TAGS = [("spectra", ""), ("$+$Formula", "_formula"), ("$+$MW", "_mw")]


def results_full(raw, spec, kind):
    """Appendix S1 (kind='derep': rank@k + cos) / S1b (kind='ann': ann@k), exp = journal val, sim = MARINA-DB test."""
    exps, split = spec["experiments"], spec["experimental"]["split"]
    js = [raw.journal("full", e) for e in exps]
    ss = [raw.sim("full", e) for e in exps]
    cols = ([f"rank@{k}" for k in KS] + ["cos"]) if kind == "derep" else [f"ann@{k}" for k in KS]
    skey = {**{f"rank@{k}": f"mean_rank_{k}" for k in KS}, **{f"ann@{k}": f"mean_ann_{k}" for k in KS}, "cos": "mean_cos"}
    lines, data = [], {"experiments": exps, "rows": {}}
    for i, (lab, sub, scombo) in enumerate(S1_ROWS):
        for j, (tag, suf) in enumerate(S1_TAGS):
            per = [jmetrics(x, f"{split}/{sub}{suf}") for x in js]
            exp = {c: seed_stat(per, c) for c in cols}
            sim = {c: agg([None if m is None or f"test/{skey[c]}/{scombo}{suf}" not in m
                           else (1.0 if c == "cos" else 100.0) * m[f"test/{skey[c]}/{scombo}{suf}"] for m in ss])
                   for c in cols}
            data["rows"][f"{lab} | {tag}"] = {"exp": exp, "sim": sim}
            cells = [cell(d[c], pct=False, digits=3 if c == "cos" else 2) for d in (exp, sim) for c in cols]
            first = f"  \\multirow{{3}}{{*}}{{{lab}}} & {tag}" if j == 0 else f"      & {tag:<10}"
            lines.append(first + " & " + " & ".join(cells) + " \\\\")
        if i != len(S1_ROWS) - 1:
            lines.append("  \\cmidrule(l){2-10}" if kind == "derep" else "  \\cmidrule(l){2-8}")
    n = len(cols)
    sub_hd = " & ".join(f"\\hd{{{'r@' + c.split('@')[1] if c.startswith('rank') else c}}}" for c in cols)
    head = (PRE % ("\\small\n", "5pt" if kind == "derep" else "6pt", "1.15") +
            f"\\begin{{tabular}}{{@{{}}l l {'c' * n} {'c' * n}@{{}}}}\n  \\toprule\n"
            "  \\multirow{2}{*}{\\hd{Input}} & \\multirow{2}{*}{\\hd{}}\n"
            f"      & \\multicolumn{{{n}}}{{c}}{{\\hd{{Experimental (MARINA-Bench {split}, $n{{=}}{232 if split == 'val' else 234}$)}}}}\n"
            f"      & \\multicolumn{{{n}}}{{c}}{{\\hd{{Simulated (MARINA-DB test)}}}} \\\\\n"
            f"  \\cmidrule(lr){{3-{2 + n}}}\\cmidrule(lr){{{3 + n}-{2 + 2 * n}}}\n"
            f"      & & {sub_hd}\n        & {sub_hd} \\\\\n  \\midrule\n")
    what = "dereplication" if kind == "derep" else "annotation (a top-k retrieval has ECFP4 cos >= 0.8 to the true structure)"
    note = (f"% Table S1{'' if kind == 'derep' else 'b'} — full flagship {what}; flagship: {', '.join(exps)}; "
            f"exp = journal {split}, sim = MARINA-DB test; ranking = {LABEL[METRIC][2]}.\n")
    return note + head + "\n".join(lines) + "\n" + POST, data


def single_atom(raw, spec):
    """Appendix single-atom-bit tables (overall + per element), val+test pooled per seed, mean of seeds."""
    exps = spec["experiments"]
    paths = [os.path.join(raw.root, "singleatom", f"{e}.json") for e in exps]
    missing = [os.path.relpath(p, HERE) for p in paths if not os.path.exists(p)]
    raw.missing += missing
    if missing:
        return None, None, {"missing": missing}
    pooled = []
    for p in paths:
        res = json.load(open(p))["results"]
        out = {}
        for combo in ("nmr", "nmr_formula"):
            sp = [res[s][combo] for s in res]
            N = sum(c["n"] for c in sp)
            wrong = sum(c["bit"]["mean_wrong_per_mol"] * c["n"] for c in sp) / N
            ncols = sp[0]["bit"]["n_single_cols"]
            s_cor = sum(c["element_total_strict"]["correct"] for c in sp)
            s_tot = sum(c["element_total_strict"]["total"] for c in sp)
            s_mol = sum(c["element_total_strict"]["mol_all"] for c in sp)
            el = {}
            for c in sp:
                for k, v in c["element_total"]["per_key"].items():
                    a = el.setdefault(k, [0.0, 0]); a[0] += v["acc_pct"] / 100 * v["n"]; a[1] += v["n"]
            out[combo] = {"n": N, "wrong": wrong, "bit_acc": 100 * (1 - wrong / ncols), "el_acc": 100 * s_cor / s_tot,
                          "mol_all": 100 * s_mol / N, "per_el": {k: (100 * c / n, n) for k, (c, n) in el.items()}}
        pooled.append(out)
    m = {}
    for combo in ("nmr", "nmr_formula"):
        cs = [p[combo] for p in pooled]
        m[combo] = {k: sum(c[k] for c in cs) / len(cs) for k in ("wrong", "bit_acc", "el_acc", "mol_all")}
        m[combo]["n"] = cs[0]["n"]
        m[combo]["per_el"] = {k: (sum(c["per_el"][k][0] for c in cs) / len(cs), cs[0]["per_el"][k][1])
                              for k in cs[0]["per_el"]}
    nmr, frm = m["nmr"], m["nmr_formula"]

    def b(val, other, fmt, lower=False):
        txt = format(val, fmt)
        better = (round(val, 2) < round(other, 2)) if lower else (round(val, 2) > round(other, 2))
        return r"\best{" + txt + "}" if better else txt

    def sgn(v, fmt):
        return ("+" if v >= 0 else "") + format(v, fmt)

    note = (f"% single_atom_formula_overall — single-atom multiplicity bits, NMR vs NMR+Formula; flagship: {', '.join(exps)};\n"
            f"% MARINA-Bench val+test pooled (n={nmr['n']}), mean of {len(exps)} seeds. Element-count acc. and All counts\n"
            "% correct are phantom-penalising (a predicted element absent from the molecule counts as an error).\n")
    rows = [("NMR", nmr, frm), ("NMR $+$ Formula", frm, nmr)]
    body = []
    for lab, x, y in rows:
        body.append(f"  {lab:<18} & {b(x['wrong'], y['wrong'], '.2f', lower=True)} & {b(x['bit_acc'], y['bit_acc'], '.2f')} & "
                    f"{b(x['el_acc'], y['el_acc'], '.1f')} & {b(x['mol_all'], y['mol_all'], '.1f')} \\\\")
    d = {k: frm[k] - nmr[k] for k in ("wrong", "bit_acc", "el_acc", "mol_all")}
    overall = (note + "\\begingroup\n\\color{cInk}\n\\sisetup{retain-explicit-plus=true}\n\\setlength{\\tabcolsep}{6pt}\n"
               "\\renewcommand{\\arraystretch}{1.18}\n"
               "\\begin{tabular}{@{}l S[table-format=+1.2] S[table-format=+2.2] S[table-format=+2.1] S[table-format=+2.1]@{}}\n"
               "  \\toprule\n  \\hd{Input} & {\\hd{Wrong bits/mol $\\downarrow$}} & {\\hd{Bit acc.\\ \\%}}\n"
               "    & {\\hd{Element-count acc.\\ \\%}} & {\\hd{All counts correct \\%}} \\\\\n  \\midrule\n"
               + "\n".join(body) + "\n  \\addlinespace[2pt]\n"
               f"  $\\Delta$ (Formula) & {sgn(d['wrong'], '.2f')} & {sgn(d['bit_acc'], '.2f')} & {sgn(d['el_acc'], '.1f')} & "
               f"{sgn(d['mol_all'], '.1f')} \\\\\n" + POST)
    order = ["C", "N", "O", "Cl", "Br", "S", "Si", "I", "P", "F"]
    els = [e for e in order if e in nmr["per_el"]] + [e for e in nmr["per_el"] if e not in order]
    na = [round(nmr["per_el"][e][0]) for e in els]; fa = [round(frm["per_el"][e][0]) for e in els]
    rowc = lambda xs, ys: " & ".join(r"\best{" + str(x) + "}" if x > y else str(x) for x, y in zip(xs, ys))
    per = (f"% single_atom_formula_perelement — per-element exact-count accuracy (%), NMR vs NMR+Formula; flagship: "
           f"{', '.join(exps)};\n% MARINA-Bench val+test pooled, mean of {len(exps)} seeds; n(mol) = benchmark molecules "
           "containing the element.\n\\begingroup\n\\color{cInk}\n\\setlength{\\tabcolsep}{5pt}\n"
           "\\renewcommand{\\arraystretch}{1.18}\n"
           f"\\begin{{tabular}}{{@{{}}l *{{{len(els)}}}{{S[table-format=3.0]}}@{{}}}}\n  \\toprule\n"
           "  \\hd{Input} & " + " & ".join(f"{{\\hd{{{e}}}}}" for e in els) + " \\\\\n"
           "  {\\itshape n\\ (mol)} & " + " & ".join(str(nmr["per_el"][e][1]) for e in els) + " \\\\\n  \\midrule\n"
           f"  NMR             & {rowc(na, fa)} \\\\\n  NMR $+$ Formula & {rowc(fa, na)} \\\\\n" + POST)
    return overall, per, {"experiments": exps, "nmr": nmr, "nmr_formula": frm}


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
    sp = json.load(open(os.path.join(HERE, "specs", "results_full.json")))
    for kind, name in (("derep", "results_full_derep"), ("ann", "results_full_ann")):
        raw = Raw(a.raw)
        tex, data = results_full(raw, sp, kind)
        data["missing"] = sorted(set(raw.missing))
        out(name, tex, data)
        summary[name] = ("complete" if not data["missing"] and not data.get("tbd_cells")
                         else f"{len(data['missing'])} files missing, {data.get('tbd_cells', 0)} cells pending")
    raw = Raw(a.raw)
    ov, pe, data = single_atom(raw, json.load(open(os.path.join(HERE, "specs", "single_atom_formula.json"))))
    if ov:
        out("single_atom_formula_overall", ov, data)
        out("single_atom_formula_perelement", pe, data)
    summary["single_atom_formula"] = "complete" if ov else f"{len(data['missing'])} inputs missing"
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
