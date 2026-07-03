"""
Fixed-correspondence position-based fits: the two surviving 1888 Van Gogh
nocturnes (pipeline validation) and Millet's La Nuit etoilee (transferability
to a different artist).

Each fit is a full, fixed-correspondence fit (every measured point used, in the
listed order), so no point/label search is needed here -- that machinery applies
only to the F612 Hyades fit (see permutation_tests.py). We simply reproduce the
observed rho_CS and run the random-placement Monte Carlo null.

Listed in manuscript (chronological) order:
    Winter asterism, Orion & Canis Major (Millet, ~1850-65) : k = 8, rho_CS = 3.85 %  [transferability]
    Big Dipper  (The Starry Night over the Rhône, 1888)     : k = 7, rho_CS = 6.54 %  [validation]
    Summer Triangle + b/d Cyg (Cafe Terrace, 1888)          : k = 5, rho_CS = 5.46 %  [validation]

The Millet fit uses star positions only (no FOV reconstruction), showing the
shape-matching step transfers to another artist's work.

Author: Elijah J. H. Kim. MIT License.
"""
from __future__ import annotations
import os
import numpy as np
from procrustes import procrustes, center, DATA_DIR
from montecarlo_null import mc_null, N_DEFAULT, _fmt_p


def _load_benchmark(name):
    canvas, sky = [], []
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        for line in f:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            if line.startswith("star_canvas"):
                continue
            p = line.rstrip("\n").split(",")
            canvas.append((float(p[2]), float(p[3])))
            sky.append((float(p[4]), float(p[5])))
    return np.array(canvas), np.array(sky)


# Order follows the manuscript (Table 1 / Table S7): transferability first,
# then the two 1888 validation nocturnes.
BENCHMARKS = {
    "Millet (Winter ast.)": ("benchmark_millet.csv", 8, 3.85), # transferability
    "Rhone (Big Dipper)":  ("benchmark_rhone.csv", 7, 6.54),   # validation, VG 1888
    "Cafe (Summer Tri.)":  ("benchmark_cafe.csv", 5, 5.46),    # validation, VG 1888
}


def run(n_mc: int = N_DEFAULT):
    print(f"Benchmark fits and Monte Carlo null  (N = {n_mc:,})")
    print(f"{'fit':22s}{'k':>3s}{'rho_CS':>9s}{'target':>9s}{'MC p':>11s}")
    out = {}
    for name, (fname, k, target) in BENCHMARKS.items():
        canvas, sky = _load_benchmark(fname)
        rho = procrustes(canvas, sky)["rho_CS"]
        p, med, p5, hits, _ = mc_null(sky, k, rho, n=n_mc)
        out[name] = (rho, p, hits)
        print(f"{name:22s}{k:>3d}{rho:>8.2f}%{target:>8.2f}%{_fmt_p(p, n_mc):>11s}")
    return out


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT
    run(n)
