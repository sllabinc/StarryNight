# Procrustes shape-matching and null tests for *The Starry Night* (F612) and benchmark nocturnes

Code and data accompanying **"The actual sky of Van Gogh's *Starry Night*: a
multidisciplinary reconstruction"** (Supplementary Notes 3–4).

The analysis quantifies how well the central asterism of *The Starry Night*
(F612) matches the **Hyades** cluster, and tests whether that match could arise
by chance, by an arbitrary choice of points, or by an arbitrary labelling.

> **Identity, not date.** `rho_CS` is scale-invariant and rotation-free, so it
> measures pure *shape* and is therefore **date-invariant**. Every test here
> probes *identification* ("are these points the Hyades?"), never the calendar
> date. No date is claimed or tested by this code.

## Contents

```
data/
  F612_measurements.csv        11 central stars of F612, three CAD measurements (mm)
  F612_mean.csv                their means (analysis input); Sn8 = Venus
  sky_hyades.csv               Hyades V-asterism, J2000.0 gnomonic, arcmin, centred
  sky_aries.csv                Aries (non-diagnostic comparison)
  replica_F1540_centered.csv   replica drawing F1540, five central points, centred (mm)
  benchmark_rhone.csv          Big Dipper benchmark (Rhone, 1888): canvas + sky, k=7
  benchmark_cafe.csv           Summer Triangle benchmark (Cafe, 1888): canvas + sky, k=5
  benchmark_millet.csv         Winter-asterism transferability fit (Millet, ~1850-65): canvas + sky, k=8
  correspondences.csv          anatomical point-to-star correspondences (F612)
code/
  procrustes.py                closed-form 2D Procrustes engine + CSV loaders
  benchmark_fits.py            1888 validation (Rhone, Cafe) + Millet transferability fits + their MC null
  montecarlo_null.py           random-placement null test (pillar A)
  permutation_tests.py         label (B1), subset (B2), direction (C) tests
  reproduce_all.py             runs everything and checks the headline numbers
```

## Method

The optimal rotation, isotropic scale and translation that map a fixed sky
constellation onto a canvas configuration are obtained in closed form from the
cross-product matrix `M = Xc^T Tc`:

```
theta = atan2(M21 - M12, M11 + M22)      scale = (sigma0 + sigma1) / Tr(T^T T)
```

This is mathematically identical to the SVD solution `R = U V^T`, and being built
from cos/sin it is always a proper rotation (`det = +1`), so reflections are
excluded. The shape-mismatch index is the centroid-size-normalised RMS residual,
`rho_CS = 100 * RMS(r_i) / CS`, with `CS = sqrt(sum ||Xc_i||^2)`.

**Coordinate registration.** The canvas is measured in image coordinates (the y
axis runs downward); the sky is a gnomonic projection (declination runs upward).
Because the fit allows proper rotation only, the two coordinate systems must
share the same handedness, so one axis is negated to register them. In the
stored data this is folded into `sky_eta` (declination axis already negated),
so every fit is simply `procrustes(canvas, sky)` with no per-fit flip. This is a
fixed coordinate-system registration, not a fitted reflection.

## Reproduce

```
pip install -r requirements.txt
cd code
python reproduce_all.py            # ~ seconds; prints results and PASS/FAIL checks
python reproduce_all.py 1000000    # optional: larger Monte Carlo sample
```

Expected headline values:

Fits are listed in manuscript order (transferability, then 1888 validation,
then the Hyades application and its replica, with the reference-only Aries last):

| quantity | value |
|---|---|
| Millet winter-asterism transferability fit (k = 8) | rho_CS = 3.85 %, rho_max = 7.22 % (Rigel), CS = 497.18 mm |
| Rhone Big Dipper benchmark (k = 7) | rho_CS = 6.54 % |
| Cafe Summer Triangle benchmark (k = 5) | rho_CS = 5.46 % |
| F612 Hyades fit (k = 5) | rho_CS = 8.17 %, rho_max = 10.59 %, theta = 46.5°, CS = 472.93 mm |
| F1540 replica fit (k = 5) | rho_CS = 6.62 %, rho_max = 7.85 %, CS = 317.34 mm |
| F612 Aries fit (k = 3) | rho_CS = 5.36 % (non-diagnostic / reference-only, see below) |
| Monte Carlo, Millet / Rhone / Cafe | p < 5 × 10⁻⁶ (0 of 200,000) |
| Monte Carlo, F612 Hyades | p ≈ 3 × 10⁻⁵ (null median ≈ 40 % at k = 5) |
| Monte Carlo, F1540 replica | p ≈ 5 × 10⁻⁶ |
| Monte Carlo, Aries (k = 3) | p ≈ 0.008 (k = 3 null is permissive: 5th percentile 13.2 % vs 26.8 % at k = 5) |
| Label permutation (B1) | anatomical labelling ranks 1 / 120, p ≈ 1/120 ≈ 0.008 |
| Subset enumeration (B2) | {Sn2,Sn3,Sn6,Sn7,Sn10} ranks 1 / 462; winner excludes Sn8 (Venus) |
| Direction gate (C) | Hyades first; runner-up 8.85 % is Venus-contaminated |

**Reproducibility of the Monte Carlo p-values.** The random-placement null draws
k points uniformly in the unit square `[0,1]^2`; the RNG seed (`20250722`) and
sample size (`200,000`) are fixed, so the p-values are exactly reproducible. With
N = 200,000 the smallest measurable non-zero p is `5 × 10⁻⁶`. A fit at rho_CS =
8.17 % (the F612 Hyades fit) has a true tail probability of order `10⁻⁵`, so it
returns a small non-zero count (p ≈ 3 × 10⁻⁵) rather than the sampling floor;
lower-residual fits (≤ 6.6 %) reach the floor (p < 5 × 10⁻⁶). Raw null draws are
not stored because they are regenerable from the seed.

**Why Aries is non-diagnostic.** Its raw rho_CS (5.36 %) is low only because k = 3
has two degrees of freedom; the permutation ceiling for k = 3 is `1/3! ≈ 0.167`,
so no k = 3 fit can be statistically distinguished from chance regardless of its
residual.

## Data sources

Celestial coordinates are from the Hipparcos catalogue (J2000.0). The painting
and drawing images analysed here are held by the Van Gogh Museum (F612, F1540)
and are not redistributed in this repository.

## License

Code and data are released under the MIT License (see `LICENSE`).
