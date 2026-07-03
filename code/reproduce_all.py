"""
Reproduce every quantitative result in Supplementary Notes 3-4 of
"The actual sky of Van Gogh's Starry Night".

Run:  python reproduce_all.py
(optional)  python reproduce_all.py 1000000     # larger Monte Carlo N

Prints the Procrustes fits, the Monte Carlo null, and the permutation tests,
then checks the headline numbers against their reported values. Exits 0 if all
checks pass.

Author: Elijah J. H. Kim. MIT License.
"""
from __future__ import annotations
import sys
import numpy as np
from procrustes import procrustes, build_fit, load_replica_F1540
import montecarlo_null as mc
import permutation_tests as perm
import benchmark_fits as bench


def section(title):
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64)


def main(n_mc=mc.N_DEFAULT):
    checks = []

    section("1. Procrustes shape fits (correspondence-based: Hyades, F1540, Aries)")
    # Order: Hyades, then F1540 replica, with reference-only Aries last.
    fits = {}
    X, T = build_fit("F612_Hyades")
    fits["F612_Hyades"] = procrustes(X, T)
    print(f"  {'F612 Hyades (k=5)':20s} rho_CS={fits['F612_Hyades']['rho_CS']:6.2f}%  "
          f"rho_max={fits['F612_Hyades']['rho_max']:6.2f}%  "
          f"theta={fits['F612_Hyades']['theta_deg']:6.2f}  CS={fits['F612_Hyades']['CS']:.2f}")
    rep = procrustes(load_replica_F1540(), T)
    print(f"  {'F1540 replica (k=5)':20s} rho_CS={rep['rho_CS']:6.2f}%  "
          f"rho_max={rep['rho_max']:6.2f}%  CS={rep['CS']:.2f}")
    Xa, Ta = build_fit("F612_Aries")
    fits["F612_Aries"] = procrustes(Xa, Ta)
    print(f"  {'F612 Aries (k=3)':20s} rho_CS={fits['F612_Aries']['rho_CS']:6.2f}%  "
          f"rho_max={fits['F612_Aries']['rho_max']:6.2f}%  "
          f"theta={fits['F612_Aries']['theta_deg']:6.2f}  CS={fits['F612_Aries']['CS']:.2f}  (reference-only)")
    checks += [("Hyades rho_CS = 8.17%", abs(fits["F612_Hyades"]["rho_CS"] - 8.17) < 0.01),
               ("F1540  rho_CS = 6.62%", abs(rep["rho_CS"] - 6.62) < 0.01),
               ("Aries  rho_CS = 5.36%", abs(fits["F612_Aries"]["rho_CS"] - 5.36) < 0.01)]

    section(f"2. Benchmark fits: 1888 validation + Millet transferability (N={n_mc:,})")
    bres = bench.run(n_mc)
    checks += [("Millet (Winter ast.) rho_CS = 3.85%", abs(bres["Millet (Winter ast.)"][0] - 3.85) < 0.01),
               ("Rhone (Big Dipper)  rho_CS = 6.54%", abs(bres["Rhone (Big Dipper)"][0] - 6.54) < 0.01),
               ("Cafe  (Summer Tri.) rho_CS = 5.46%", abs(bres["Cafe (Summer Tri.)"][0] - 5.46) < 0.01),
               ("Millet MC p < 1e-4", bres["Millet (Winter ast.)"][1] < 1e-4),
               ("Rhone  MC p < 1e-4", bres["Rhone (Big Dipper)"][1] < 1e-4),
               ("Cafe   MC p < 1e-4", bres["Cafe (Summer Tri.)"][1] < 1e-4)]

    section(f"3. Monte Carlo random-placement null, target asterisms (N={n_mc:,})")
    mcres = mc.run(n_mc)
    checks += [("Hyades MC p < 1e-3", mcres["F612 Hyades"][0] < 1e-3),
               ("F1540  MC p < 1e-4", mcres["F1540 replica"][0] < 1e-4),
               ("Aries  MC p ~ 0.01 (0.005-0.02)", 0.005 < mcres["F612 Aries"][0] < 0.02),
               ("k=5 null median ~ 40%", abs(mcres["F612 Hyades"][1] - 40) < 3)]

    section("4. Permutation and direction tests")
    perm.run()
    b1_rank, b1_p, *_ = perm.b1_label_permutation()
    b2_rank, b2_best, b2_ids, b2_has8 = perm.b2_subset_enumeration()
    cdir = perm.c_direction_constraint()
    checks += [("B1 anatomical labelling rank 1/120", b1_rank == 1),
               ("B2 our subset rank 1/462", b2_rank == 1),
               ("B2 winner excludes Sn8 (Venus)", not b2_has8),
               ("C  Hyades first under direction gate", cdir[0][1] == ("Sn2", "Sn3", "Sn6", "Sn7", "Sn10"))]

    section("Validation summary")
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}]  {name}")
        ok = ok and passed
    print("\nALL CHECKS PASSED." if ok else "\nSOME CHECKS FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else mc.N_DEFAULT
    sys.exit(main(n))
