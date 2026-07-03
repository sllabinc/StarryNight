"""
Monte Carlo random-placement null test (Supplementary Note 4, pillar A).

Question: could a random arrangement of k points match a fixed constellation as
well as the painting's asterism does?

Procedure: draw k points uniformly at random in the unit square [0,1]^2 (these
play the role of the canvas), fit the FIXED constellation onto them, record
rho_CS. Repeat N times to build the null distribution. The one-sided p-value is
the fraction of random placements whose rho_CS is at most the observed value.

Because rho_CS is scale-invariant and rotation-free, the unit square is an
arbitrary but sufficient sampling domain: the null depends only on the random
SHAPE of k points, not on their absolute coordinates.

Reproducibility: the RNG seed and sample size are fixed below; the reported
p-values are exactly reproducible. Raw null draws are NOT stored (they are
regenerable from the seed); only summary statistics are reported.

NOTE on the floor: with N = 200,000 the smallest measurable non-zero p is
1/200,000 = 5e-6. Fits whose true tail probability is below this report 0 hits
("< 5e-6"); fits with a true p of order 1e-5 (e.g. the F612 Hyades fit at
8.17 %) return a small non-zero count and a p of order 1e-5, which is the honest
reproducible value.

Author: Elijah J. H. Kim. MIT License.
"""
from __future__ import annotations
import numpy as np
from procrustes import procrustes, center, build_fit, load_replica_F1540

SEED = 20250722          # fixed for reproducibility (date motif: 1889-07-22)
N_DEFAULT = 200_000


def mc_null(template, k: int, rho_obs: float, n: int = N_DEFAULT, seed: int = SEED):
    """Return (p_value, median, pct5, n_hits, n) for the random-placement null."""
    rng = np.random.default_rng(seed)
    Tc = center(template)
    rhos = np.empty(n, dtype=float)
    for i in range(n):
        rhos[i] = procrustes(rng.random((k, 2)), Tc)["rho_CS"]
    hits = int((rhos <= rho_obs).sum())
    p = hits / n
    return p, float(np.median(rhos)), float(np.percentile(rhos, 5)), hits, n


def _fmt_p(p, n):
    return f"< {1/n:.0e}" if p == 0 else f"{p:.2g}"


def run(n: int = N_DEFAULT):
    print(f"Monte Carlo random-placement null  (N = {n:,}, seed = {SEED})")
    print(f"{'fit':16s}{'k':>3s}{'rho_obs':>9s}{'p':>11s}{'median':>9s}{'5th pct':>9s}")
    # Order follows the manuscript (Table S7): Hyades, then the F1540 replica,
    # with the reference-only Aries fit last.
    cases = []
    Xh, Th = build_fit("F612_Hyades")
    cases.append(("F612 Hyades", Th, 5, procrustes(Xh, Th)["rho_CS"]))
    rep = load_replica_F1540()
    cases.append(("F1540 replica", Th, 5, procrustes(rep, Th)["rho_CS"]))
    Xa, Ta = build_fit("F612_Aries")
    cases.append(("F612 Aries", Ta, 3, procrustes(Xa, Ta)["rho_CS"]))
    out = {}
    for name, tmpl, k, obs in cases:
        p, med, p5, hits, _ = mc_null(tmpl, k, obs, n=n)
        out[name] = (p, med, p5, hits)
        print(f"{name:16s}{k:>3d}{obs:>8.2f}%{_fmt_p(p, n):>11s}{med:>8.1f}%{p5:>8.1f}%")
    return out


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT
    run(n)
