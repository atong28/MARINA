"""
Prototype 2D HSQC peak-aware similarity, generalizing NMRPeak's 1D metric to
cross-peaks. Each HSQC cross-peak is a (13C, 1H) coordinate. The per-nucleus
exponential kernels are combined multiplicatively (separable 2D kernel), which
naturally handles the very different 13C vs 1H ppm scales. Same two-stage
(Hungarian + greedy) matching and 0.8*avg + 0.2*penalty structure as the paper.

Phase/multiplicity (col 2): only the SIGN matters (CH/CH3 = +, CH2 = -). Annotated
encodes it with small magnitudes (~0.1-0.3) but the sign is comparable to Journal/
Sim (+/-1). With use_phase=True a matched cross-peak pair must have the same phase
sign to be valid (opposite-sign pairs score 0), reflecting that a CH2 and a CH/CH3
cross-peak at the same position are chemically distinct.

Each cross-peak is (13C, 1H, sign) with sign in {+1, -1}.
"""
from typing import List, Tuple, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment


def phi(d: float, sigma: float) -> float:
    return float(np.exp(-d / sigma))


class HSQC2DSimilarityScorer:
    def __init__(self, scale_c: float = 5.0, scale_h: float = 1.0,
                 tol_c: float = 20.0, tol_h: float = 2.0, use_phase: bool = True):
        self.sc, self.sh = scale_c, scale_h
        self.tc, self.th = tol_c, tol_h
        self.use_phase = use_phase

    def _sim(self, p, q) -> float:
        if self.use_phase and p[2] != q[2]:
            return 0.0
        return phi(abs(p[0] - q[0]), self.sc) * phi(abs(p[1] - q[1]), self.sh)

    def _valid(self, p, q) -> bool:
        ok = abs(p[0] - q[0]) <= self.tc and abs(p[1] - q[1]) <= self.th
        if self.use_phase:
            ok = ok and (p[2] == q[2])
        return ok

    def _match(self, A: List[Tuple[float, float]], B: List[Tuple[float, float]]):
        """Two-stage matching. Returns list of (round, i_short, j_long, sim, valid)
        where short/long are chosen by size (matching is symmetric in A,B)."""
        if len(A) <= len(B):
            short, long_, swapped = A, B, False
        else:
            short, long_, swapped = B, A, True
        ns, nl = len(short), len(long_)

        # Stage I: Hungarian on cost = 1 - sim
        sim_mat = np.zeros((ns, nl))
        for i, p in enumerate(short):
            for j, q in enumerate(long_):
                sim_mat[i, j] = self._sim(p, q)
        cost = 1.0 - sim_mat
        rows, cols = linear_sum_assignment(cost)

        matches = []
        matched_long = set()
        for i, j in zip(rows, cols):
            matches.append((1, i, j, sim_mat[i, j], self._valid(short[i], long_[j])))
            matched_long.add(j)

        # Stage II: greedy for unmatched long peaks
        for j in set(range(nl)) - matched_long:
            sims = [self._sim(short[i], long_[j]) for i in range(ns)]
            i = int(np.argmax(sims))
            matches.append((2, i, j, sims[i], self._valid(short[i], long_[j])))
        return matches

    def calculate_similarity_score(self, A, B) -> Dict:
        if len(A) == 0 or len(B) == 0:
            return {"similarity": 0.0, "avg_peak_similarity": 0.0,
                    "penalty_coefficient": 0.0, "n_matches": 0, "n_delta_peak": abs(len(A) - len(B))}
        matches = self._match(A, B)
        n_match = len(matches)                 # == max(|A|,|B|)
        n_delta = abs(len(A) - len(B))
        n_max = max(len(A), len(B))

        peak_sims, valid_r1 = [], 0
        for rnd, i, j, sim, valid in matches:
            if valid:
                if rnd == 1:
                    valid_r1 += 1
                    peak_sims.append(sim)
                else:
                    peak_sims.append(0.8 * sim)
            else:
                peak_sims.append(0.0)
        avg = float(np.mean(peak_sims)) if peak_sims else 0.0
        penalty = n_max / (n_match + n_delta + (n_match - valid_r1) + 1e-8)
        final = 0.8 * avg + 0.2 * penalty
        return {"similarity": final, "avg_peak_similarity": avg,
                "penalty_coefficient": penalty, "n_matches": n_match, "n_delta_peak": n_delta}
