"""
Permutation tests on point selection and labelling (Supplementary Note 4, pillar B)
plus the physical direction constraint (pillar C).

These are exhaustive and deterministic (no random seed needed).

B1  Label permutation. Fix the five selected points {Sn2,Sn3,Sn6,Sn7,Sn10} and try all
    5! = 120 ways of assigning the Hyades skeleton labels (eps, alpha, theta,
    delta, gamma Tau). The anatomically correct labelling should give the unique
    minimum rho_CS  ->  p ~ 1/120 ~ 0.008.

B2  Subset enumeration. Try all C(11,5) = 462 ways of choosing five of the eleven
    measured points, each fitted with its best-of-120 labelling. The pre-specified
    subset {Sn2,Sn3,Sn6,Sn7,Sn10} should rank first, with no better subset, and the
    winner should NOT contain Sn8 (Venus) -> the five-point choice is not post-hoc.

C   Direction constraint. A single instant of sky shares one tilt, so a physical
    match must align near the painting's orientation. Restricting to fits with
    |theta - 46.5 deg| <= 20 deg, the Hyades subset remains first; the runner-up is
    a Venus-contaminated subset well above it.

Author: Elijah J. H. Kim. MIT License.
"""
from __future__ import annotations
from itertools import permutations, combinations
import numpy as np
from procrustes import procrustes, build_fit, load_canvas_mean, load_sky

HYADES_ORDER = ["eps Tau", "alpha Tau", "theta Tau", "delta Tau", "gamma Tau"]
SELECTED = ["Sn2", "Sn3", "Sn6", "Sn7", "Sn10"]   # pre-specified five points (anatomical order)
THETA_REF = 46.5                              # painting orientation (deg)
GATE = 20.0                                   # allowed deviation (deg)


def _ang_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def b1_label_permutation():
    """Return (rank, p, rho_anat, rho_min) for the 120 labellings of the fixed 5 points."""
    cmean = load_canvas_mean()
    sky = load_sky("sky_hyades.csv")
    pts = np.array([cmean[i] for i in SELECTED])
    tmpl = np.array([sky[s] for s in HYADES_ORDER])
    rho_anat = procrustes(pts, tmpl)["rho_CS"]          # identity perm = anatomical
    rhos = [procrustes(pts[list(p)], tmpl)["rho_CS"] for p in permutations(range(5))]
    rho_min = min(rhos)
    rank = 1 + sum(1 for r in rhos if r < rho_anat - 1e-9)
    return rank, rank / 120, rho_anat, rho_min


def b2_subset_enumeration():
    """Return (rank, rho_best, best_ids, contains_S8) over all 462 five-point subsets."""
    cmean = load_canvas_mean()
    sky = load_sky("sky_hyades.csv")
    tmpl = np.array([sky[s] for s in HYADES_ORDER])
    ids = list(cmean)
    results = []
    for combo in combinations(range(11), 5):
        P = np.array([cmean[ids[i]] for i in combo])
        best = min(procrustes(P[list(pm)], tmpl)["rho_CS"] for pm in permutations(range(5)))
        results.append((best, combo))
    results.sort(key=lambda t: t[0])
    target = tuple(sorted(ids.index(x) for x in SELECTED))
    rank = [c for _, c in results].index(target) + 1
    best_rho, best_combo = results[0]
    best_ids = [ids[i] for i in best_combo]
    return rank, best_rho, best_ids, ("Sn8" in best_ids)


def c_direction_constraint():
    """Return [(rho, ids), ...] top two fits passing the |theta - 46.5| <= 20 gate."""
    cmean = load_canvas_mean()
    sky = load_sky("sky_hyades.csv")
    tmpl = np.array([sky[s] for s in HYADES_ORDER])
    ids = list(cmean)
    gated = []
    for combo in combinations(range(11), 5):
        P = np.array([cmean[ids[i]] for i in combo])
        for pm in permutations(range(5)):
            r = procrustes(P[list(pm)], tmpl)
            if _ang_diff(r["theta_deg"], THETA_REF) <= GATE:
                gated.append((r["rho_CS"], tuple(ids[i] for i in combo)))
    gated.sort(key=lambda t: t[0])
    return gated[:2]


def run():
    rank, p, anat, mn = b1_label_permutation()
    print("B1 label permutation (120):"
          f"  anatomical rho_CS={anat:.2f}%  rank={rank}/120  p={p:.3f}  min={mn:.2f}%")
    rank, best, ids, has8 = b2_subset_enumeration()
    print("B2 subset enumeration (462):"
          f"  best={best:.2f}% {ids}  contains Sn8(Venus)={has8}  our-subset rank={rank}/462")
    top = c_direction_constraint()
    print("C  direction gate |theta-46.5|<=20:"
          f"  1st {top[0][0]:.2f}% {list(top[0][1])} ;  2nd {top[1][0]:.2f}% {list(top[1][1])}")


if __name__ == "__main__":
    run()
