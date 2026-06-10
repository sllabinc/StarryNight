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

    section("1. Procrustes shape fits (Tables S1-S4)")
    fits = {}
    for label, fit in (("F612 Hyades (k=5)", "F612_Hyades"),
                       ("F612 Aries  (k=3)", "F612_Aries")):
        X, T = build_fit(fit)
        r = procrustes(X, T)
        fits[fit] = r
        print(f"  {label:20s} rho_CS={r['rho_CS']:6.2f}%  rho_max={r['rho_max']:6.2f}%  "
              f"theta={r['theta_deg']:6.2f}  CS={r['CS']:.2f}")
    rep = procrustes(load_replica_F1540(), build_fit("F612_Hyades")[1])
    print(f"  {'F1540 replica (k=5)':20s} rho_CS={rep['rho_CS']:6.2f}%  "
          f"rho_max={rep['rho_max']:6.2f}%  CS={rep['CS']:.2f}")
    checks += [("Hyades rho_CS = 8.17%", abs(fits["F612_Hyades"]["rho_CS"] - 8.17) < 0.01),
               ("Aries  rho_CS = 5.36%", abs(fits["F612_Aries"]["rho_CS"] - 5.36) < 0.01),
               ("F1540  rho_CS = 6.62%", abs(rep["rho_CS"] - 6.62) < 0.01)]

    section(f"2. Method-validation benchmark fits (N={n_mc:,})")
    bres = bench.run(n_mc)
    checks += [("Rhone (Big Dipper) rho_CS = 6.54%", abs(bres["Rhone (Big Dipper)"][0] - 6.54) < 0.01),
               ("Cafe  (Summer Tri.) rho_CS = 5.46%", abs(bres["Cafe (Summer Tri.)"][0] - 5.46) < 0.01),
               ("Rhone MC p < 1e-4", bres["Rhone (Big Dipper)"][1] < 1e-4),
               ("Cafe  MC p < 1e-4", bres["Cafe (Summer Tri.)"][1] < 1e-4)]

    section(f"3. Monte Carlo random-placement null, target asterisms (N={n_mc:,})")
    mcres = mc.run(n_mc)
    checks += [("Hyades MC p < 1e-3", mcres["F612 Hyades"][0] < 1e-3),
               ("Aries  MC p ~ 0.01 (0.005-0.02)", 0.005 < mcres["F612 Aries"][0] < 0.02),
               ("F1540  MC p < 1e-4", mcres["F1540 replica"][0] < 1e-4),
               ("k=5 null median ~ 40%", abs(mcres["F612 Hyades"][1] - 40) < 3)]

    section("4. Permutation and direction tests")
    perm.run()
    b1_rank, b1_p, *_ = perm.b1_label_permutation()
    b2_rank, b2_best, b2_ids, b2_has8 = perm.b2_subset_enumeration()
    cdir = perm.c_direction_constraint()
    checks += [("B1 anatomical labelling rank 1/120", b1_rank == 1),
               ("B2 our subset rank 1/462", b2_rank == 1),
               ("B2 winner excludes S8 (Venus)", not b2_has8),
               ("C  Hyades first under direction gate", cdir[0][1] == ("S2", "S3", "S6", "S7", "S10"))]

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
