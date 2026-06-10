"""
Closed-form 2D Procrustes shape-matching engine.

Used to align a canvas star configuration (X) to a fixed sky constellation (T)
and to quantify the residual shape mismatch as rho_CS (centroid-size-normalised
RMS residual, in percent).

The rotation is obtained directly from the cross-product matrix M = Xc^T Tc via
the closed-form angle theta = atan2(M21 - M12, M11 + M22). This is mathematically
identical to the SVD solution R = U V^T, and because it is built from cos/sin it
is always a proper rotation (det = +1), so reflections are excluded automatically.

rho_CS is scale-invariant and rotation-free: it measures pure shape, and is
therefore DATE-INVARIANT. All tests here probe IDENTITY (is this the Hyades?),
not the calendar date.

Author: Elijah J. H. Kim. Released under the MIT License.
"""
from __future__ import annotations
import csv
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def center(P) -> np.ndarray:
    """Subtract the column mean (translate the centroid to the origin)."""
    P = np.asarray(P, dtype=float)
    return P - P.mean(axis=0)


def procrustes(X, T) -> dict:
    """Fit fixed template T onto target X by optimal translation, rotation and
    isotropic scale (full ordinary Procrustes), and return the fit metrics.

    Parameters
    ----------
    X : (k, 2) array_like   target points (e.g. centred canvas asterism)
    T : (k, 2) array_like   template points (e.g. sky constellation)

    Returns
    -------
    dict with keys: theta_deg, scale, fit (k,2), r_i (k,), rho_i (k,),
                    CS, rho_CS, rho_max   (rho_* in percent)
    """
    Xc = center(X)
    Tc = center(T)
    a = float((Xc[:, 0] * Tc[:, 0] + Xc[:, 1] * Tc[:, 1]).sum())  # M11 + M22
    b = float((Xc[:, 1] * Tc[:, 0] - Xc[:, 0] * Tc[:, 1]).sum())  # M21 - M12
    mag = np.hypot(a, b)
    c, s = a / mag, b / mag
    rot = np.column_stack([c * Tc[:, 0] - s * Tc[:, 1],
                           s * Tc[:, 0] + c * Tc[:, 1]])
    scale = mag / (Tc ** 2).sum()                       # (sigma0 + sigma1) / Tr(T^T T)
    fit = scale * rot
    r_i = np.hypot(*(Xc - fit).T)
    CS = np.sqrt((Xc ** 2).sum())                        # centroid size of the target
    rho_CS = 100.0 * np.sqrt((r_i ** 2).mean()) / CS     # RMS divisor = COUNT(points)
    rho_max = 100.0 * r_i.max() / CS
    theta_deg = np.degrees(np.arctan2(b, a))
    return dict(theta_deg=theta_deg, scale=scale, fit=fit, r_i=r_i,
                rho_i=100.0 * r_i / CS, CS=CS, rho_CS=rho_CS, rho_max=rho_max)


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------
def _read_csv(name: str) -> list[dict]:
    path = os.path.join(DATA_DIR, name)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for line in f:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            header = line.rstrip("\n").split(",")
            break
        reader = csv.DictReader((ln for ln in f if not ln.lstrip().startswith("#")),
                                fieldnames=header)
        for r in reader:
            rows.append(r)
    return rows


def load_canvas_mean() -> dict:
    """Return {star_id: (x_mm, y_mm)} for the F612 11-star means."""
    return {r["star_id"]: (float(r["x_mm"]), float(r["y_mm"]))
            for r in _read_csv("F612_mean.csv")}


def load_sky(name: str) -> dict:
    """Return {star: (x, y)} for a sky-coordinate file."""
    rows = _read_csv(name)
    return {r["star"]: (float(r["x_arcmin"]), float(r["y_arcmin"])) for r in rows}


def load_replica_F1540() -> np.ndarray:
    """Return the 5 centred replica points in Hyades order (eps, alpha, theta, delta, gamma)."""
    rows = _read_csv("replica_F1540_centered.csv")
    return np.array([(float(r["Xc_mm"]), float(r["Yc_mm"])) for r in rows], dtype=float)


def load_correspondence(fit: str):
    """Return (canvas_ids, sky_file, sky_stars) for a named fit in correspondences.csv."""
    rows = [r for r in _read_csv("correspondences.csv") if r["fit"] == fit]
    canvas_ids = [r["canvas_id"] for r in rows]
    sky_file = rows[0]["sky_file"] + ".csv"
    sky_stars = [r["sky_star"] for r in rows]
    return canvas_ids, sky_file, sky_stars


def build_fit(fit: str):
    """Assemble (X canvas array, T sky array) for a named fit using the means."""
    canvas_ids, sky_file, sky_stars = load_correspondence(fit)
    cmean = load_canvas_mean()
    sky = load_sky(sky_file)
    X = np.array([cmean[i] for i in canvas_ids], dtype=float)
    T = np.array([sky[s] for s in sky_stars], dtype=float)
    return X, T


if __name__ == "__main__":
    for fit in ("F612_Hyades", "F612_Aries"):
        X, T = build_fit(fit)
        r = procrustes(X, T)
        print(f"{fit:14s} k={len(X)}  rho_CS={r['rho_CS']:.2f}%  "
              f"rho_max={r['rho_max']:.2f}%  theta={r['theta_deg']:.2f}  CS={r['CS']:.2f}")
    X = load_replica_F1540()
    _, T = build_fit("F612_Hyades")
    r = procrustes(X, T)
    print(f"{'F1540_Hyades':14s} k={len(X)}  rho_CS={r['rho_CS']:.2f}%  "
          f"rho_max={r['rho_max']:.2f}%  CS={r['CS']:.2f}")
