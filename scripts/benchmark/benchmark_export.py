import pickle
import os
import pandas as pd

BENCHMARK_ROOT = os.environ.get('BENCHMARK_ROOT', '/home/user/atong/Benchmark')
BENCHMARKS_DIR = os.path.join(BENCHMARK_ROOT, 'benchmarks')


def format_results(results):
    rows = []
    for k, v in results.items():
        topk = v['predictions']['dereplication_topk']
        rows.append({
            'npid': v.get('npid', k),
            'dereplication_rank': (
                -1 if not any(topk.values())
                else next(rank for rank, hit in topk.items() if hit)
            ),
            'cosine_sim': v['predictions']['cosine_sim'].item(),
            'h_nmr': "\n".join(
                f'{row[0]:.3f}' for row in v['input']['h_nmr'].tolist()
            ),
            'c_nmr': "\n".join(
                f'{row[0]:.3f}' for row in v['input']['c_nmr'].tolist()
            ),
            'hsqc': "\n".join(
                f'{row[1]:.3f}\t{row[0]:.3f}\t{int(row[2])}' for row in v['input']['hsqc'].tolist()
            ),
            'mw': round(float(v['input']['mw']), 5),
            'smiles': v['smiles'],
        })
    return pd.DataFrame(rows)


for seed in range(3):
    out_path = os.path.join(BENCHMARK_ROOT, f'marina-final-run-seed-{seed}_benchmark.xlsx')
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        nm_pkl = os.path.join(BENCHMARKS_DIR, f'marina-final-run-seed-{seed}_benchmark_results.pkl')
        format_results(pickle.load(open(nm_pkl, 'rb'))).to_excel(writer, sheet_name='NP-MRD', index=False)

        jn_pkl = os.path.join(BENCHMARKS_DIR, f'marina-final-run-seed-{seed}_benchmark_journal_results.pkl')
        format_results(pickle.load(open(jn_pkl, 'rb'))).to_excel(writer, sheet_name='Journal', index=False)

    print(f'Wrote {out_path}')
