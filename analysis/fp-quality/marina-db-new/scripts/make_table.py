"""Write fp_quality_marinadb.tex (same layout/macros as fp_quality.tex) from the new-set results.

\\best{} = winning value per directional column among the UNCOMMENTED rows, on displayed
(rounded) values, ties all marked — the rule the original table follows.

Usage: python make_table.py --results <marina-db-new/results> --out <fp_quality_marinadb.tex>
"""
import argparse, datetime, glob, json, os

# (key, label, commented-out?) — mirrors fp_quality.tex row-for-row
MARINA = [("uniqmult", r"Multiplicity", False),
          ("uncapped", r"Multiplicity (uncapped)", True),
          ("cap5", r"Multiplicity (capped $k{=}5$)", True),
          ("sherlock", r"Sherlock \cite{SPECTRE}", False),
          ("substructure", r"Substructure", True)]
REF = [("ECFP4_2048", r"ECFP (r${=}2$) \cite{ECFP}"),
       ("ECFP4_16384", r"ECFP (r${=}2$) \cite{ECFP}"),
       ("Morgan_r10_16384", r"ECFP (r${=}10$) \cite{ECFP}"),
       ("FCFP9_2048", r"FCFP9 (feature, r${=}9$)"),
       ("FCFP9_16384", r"FCFP9 (feature, r${=}9$)"),
       ("MAP4_2048", r"MAP4 \cite{MAP4}"),
       ("MAP4_16384", r"MAP4 \cite{MAP4}"),
       ("AtomPair_2048", r"Atom Pair \cite{AtomPair}"),
       ("AtomPair_16384", r"Atom Pair \cite{AtomPair}"),
       ("Biosynfoni", r"Biosynfoni \cite{Biosynfoni}")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ex2 = {}
    for f in glob.glob(os.path.join(a.results, "exp2_*.json")):
        ex2.update(json.load(open(f))["fingerprints"])
    rho = json.load(open(os.path.join(a.results, "exp1_spearman.json")))["spearman"]

    rows = [(k, lab, com) for k, lab, com in MARINA] + [(k, lab, False) for k, lab in REF]
    cells = {}
    for k, _, _ in rows:
        e = ex2[k]
        cells[k] = [f"{e['D']}", f"{e['onbits_mean']:.1f}", f"{e['collision_pct']:.2f}",
                    f"{e['bad_collision_pct']:.2f}", f"{e['largest_group']}", f"{rho[k]['rho']:.3f}"]
    active = [k for k, _, com in rows if not com]
    for col, better in ((2, min), (3, min), (4, min), (5, max)):
        win = better(float(cells[k][col]) for k in active)
        for k in active:
            if float(cells[k][col]) == win:
                cells[k][col] = rf"\best{{{cells[k][col]}}}"

    def line(k, lab, com):
        return ("  % " if com else "  ") + rf"\quad {lab:<34s}& " + " & ".join(cells[k]) + r" \\"

    today = datetime.date.today().isoformat()
    L = [rf"% tab:fp_quality over the 531,927-molecule MARINA-DB (CH-NMR-NP-first) retrieval set",
         r"% (Datasets/MARINA-DB-OPEN/retrieval.pkl = the old 531,087 rows, same order, + 840 CH-NMR-NP rows).",
         r"% Produced by MARINA/analysis/fp-quality/marina-db-new/ (run_refs.sh, run_mces.sh,",
         r"% scripts/{exp2_with_prefix,make_pairs,merge_mces,make_table}.py; repo builders + exp1_analyze.py);",
         r"% results: marina-db-new/results/{exp2_*.json, exp1_spearman.json}. MCES rho: fresh RASCAL pass over",
         rf"% 100k pairs resampled from the new set (seed 0, original recipe). Generated {today}.",
         r"% The old-set table (531,087 molecules) is fp_quality.tex, left untouched.",
         r"\begingroup", r"\color{cInk}", r"\setlength{\tabcolsep}{6pt}", r"\renewcommand{\arraystretch}{1.18}",
         r"\begin{tabular}{@{}l S[table-format=5.0] S[table-format=3.1] S[table-format=2.2] S[table-format=2.2] S[table-format=5.0] S[table-format=1.3]@{}}",
         r"  \toprule",
         r"   & & & \multicolumn{3}{c}{\hd{Specificity}} & \\",
         r"  \cmidrule(lr){4-6}",
         r"  \hd{Fingerprint} & {\hd{Bits}} & {\hd{On-bits/mol}} & {\hd{Collision \% $\downarrow$}}",
         r"    & {\hd{Bad ${>}200$\,Da \% $\downarrow$}} & {\hd{Largest tie $\downarrow$}}",
         r"    & {\hd{MCES $\rho$ $\uparrow$}} \\",
         r"  \midrule",
         r"  \multicolumn{7}{@{}l}{\itshape MARINA (entropy-selected, r${=}10$)} \\"]
    L += [line(k, lab, com) for k, lab, com in MARINA]
    L += [r"  \midrule", r"  \multicolumn{7}{@{}l}{\itshape Reference} \\"]
    L += [line(k, lab, False) for k, lab in REF]
    L += [r"  \bottomrule", r"\end{tabular}", r"\endgroup"]
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
