#!/usr/bin/env python
"""Verify the matrix assembly: packing, index order, additivity, and the split.

The vendored full-space engine is the same bytes as the frozen oracle, so
comparing their *numbers* proves only that a copy is a copy. What is genuinely
untested until now is everything ``mhs`` wraps around them:

  [a] PACKING AND INDEX ORDER. A matrix is only useful if contracting it
      reproduces the field. ``einsum("oisk,sk->oi", G, slip)`` must equal the
      oracle's directly-computed displacement. A transposed index pair, an
      obs/source swap, or a slip axis in the wrong place all survive a
      shape check and die here.

  [b] THE SAME FOR STRESS, against the oracle's TOTAL stress (eigenstress
      included), since ``H`` is the total.

  [c] THE EIGENSTRESS, against the oracle's own closed form -- and this is the
      clause that matters most for the half-space work, because the reuse of the
      full-space eigenstress is the premise the whole decomposition rests on.

  [d] THE THREE-WAY SPLIT as an ALGEBRAIC identity: total - eigenstress ==
      elastic, to a few ulp rather than to a tolerance. If that ever needs a
      tolerance, the three entry points are not three views of one computation.

  [e] ADDITIVITY over sources. A multi-source matrix contracted with per-source
      slip must equal the sum of the single-source fields. Catches a loop that
      overwrites instead of filling its own column, which is invisible at
      n_src = 1 -- the case every other clause uses.

  [f] eps IS PER SOURCE. A graded eps array must give the same answer as
      assembling each source separately at its own eps. Catches an eps that is
      read once and reused, which no single-eps test can see.

Run from anywhere:  python tests/gates/mhs/verify_fullspace_assembly.py
"""
from __future__ import annotations

import sys

import numpy as np

CHECKS: list[bool] = []

MU, LAM = 30.0, 45.0        # nu = 0.30, NOT 1/4: a lam/mu swap is invisible there
EPS = 0.1
SLIP = np.array([0.013, -0.007, 0.004])     # all three components live

#: Observers spanning the regimes the primitives switch between: well off the
#: element, close to it, in its plane, and beyond an edge.
OBS = np.array([
    [0.30, 0.20, -1.00],      # off the element
    [0.25, 0.25, -2.00],      # in the element's plane, inside
    [0.10, 0.10, -2.05],      # just below it
    [3.00, -2.00, -4.00],     # far field
    [-1.00, 0.50, -2.00],     # in plane, outside the footprint
])

TRI = np.array([[0.0, 0.0, -2.0], [1.0, 0.0, -2.0], [0.3, 0.9, -2.1]])


def check(label: str, value: float, tol: float, fmt: str = "{:.3e}") -> bool:
    ok = bool(np.isfinite(value)) and value < tol
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {label:56s} "
          f"{fmt.format(value)} < {fmt.format(tol)}")
    return ok


def check_true(label: str, ok: bool, note: str = "") -> bool:
    ok = bool(ok)
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {label:56s} {note}")
    return ok


def relerr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    den = np.abs(b).max()
    return float(np.abs(a - b).max() / den) if den > 0 else float(np.abs(a).max())


def _nu(mu, lam):
    return 0.5 * lam / (lam + mu)


def main() -> bool:
    from mhs.kernels import assemble as asm
    from mhs_oracle import clq

    print("=" * 76)
    print("Full-space matrix assembly: packing, the split, additivity, graded eps")
    print("=" * 76)
    nu = _nu(MU, LAM)
    print(f"  mu = {MU:g}, lam = {LAM:g} (nu = {nu:.3f}, not 1/4), eps = {EPS:g}")
    print(f"  {OBS.shape[0]} observers spanning off-element / in-plane / far field")

    n_obs = OBS.shape[0]
    tris = TRI[None, :, :]
    eps1 = np.full(1, EPS)

    # ---------------------------------------------------------------- [a] ---
    print("\n[a] PACKING: contracting the matrix reproduces the field")
    G = np.zeros((n_obs, 3, 1, 3))
    asm.assemble_fullspace_disp(OBS, tris, MU, LAM, eps1, G)
    u_mat = np.einsum("oisk,sk->oi", G, SLIP[None, :])
    u_ref = np.asarray(clq.displacement(OBS, TRI, SLIP, MU, nu, EPS))
    check("a disp matrix contracted == clq.displacement", relerr(u_mat, u_ref),
          1e-13)
    # and the shape contract the API promises
    check_true("a disp matrix shape is (n_obs, 3, n_src, 3)",
               G.shape == (n_obs, 3, 1, 3), str(G.shape))

    # ---------------------------------------------------------------- [b] ---
    print("\n[b] PACKING: the same for TOTAL stress")
    S = np.zeros((n_obs, 3, 3, 1, 3))
    asm.assemble_fullspace_total_stress(OBS, tris, MU, LAM, eps1, S)
    s_mat = np.einsum("omnsk,sk->omn", S, SLIP[None, :])
    s_tot = np.asarray(clq.stress(OBS, TRI, SLIP, MU, nu, EPS,
                                  subtract_eigenstress=False))
    check("b stress matrix contracted == clq total stress",
          relerr(s_mat, s_tot), 1e-13)
    check_true("b stress is SYMMETRIC",
               float(np.abs(s_mat - np.swapaxes(s_mat, 1, 2)).max())
               / max(float(np.abs(s_mat).max()), 1e-300) < 1e-14,
               "sigma_mn == sigma_nm")

    # ---------------------------------------------------------------- [c] ---
    print("\n[c] THE EIGENSTRESS, the premise the half-space reuse rests on")
    E = np.zeros((n_obs, 3, 3, 1, 3))
    asm.assemble_eigenstress(OBS, tris, MU, LAM, eps1, E)
    e_mat = np.einsum("omnsk,sk->omn", E, SLIP[None, :])
    e_ref = np.asarray(clq.eigenstress(OBS, TRI, SLIP, MU, nu, EPS))
    check("c eigenstress matrix contracted == clq.eigenstress",
          relerr(e_mat, e_ref), 1e-13)
    # it must be the C:sym(s (x) n) form, hence symmetric, and NONZERO here
    check_true("c eigenstress is symmetric",
               float(np.abs(e_mat - np.swapaxes(e_mat, 1, 2)).max())
               / max(float(np.abs(e_mat).max()), 1e-300) < 1e-14, "")
    check_true("c eigenstress is live at these observers (not a quiet zero)",
               float(np.abs(e_mat).max()) > 1e-9,
               f"max |C:eps*| = {np.abs(e_mat).max():.3e}")

    # ---------------------------------------------------------------- [d] ---
    print("\n[d] THE SPLIT as an identity, not a tolerance")
    elastic = s_mat - e_mat
    e_el = np.asarray(clq.stress(OBS, TRI, SLIP, MU, nu, EPS,
                                 subtract_eigenstress=True))
    ulp = np.spacing(np.abs(e_el).max())
    worst = float(np.abs(elastic - e_el).max())
    check_true("d total - eigenstress == elastic, to a few ulp",
               worst <= 8.0 * ulp,
               f"{worst:.3e} <= 8 ulp = {8.0 * ulp:.3e}")

    # ---------------------------------------------------------------- [e] ---
    print("\n[e] ADDITIVITY over sources (invisible at n_src = 1)")
    tri2 = np.array([[0.0, 0.0, -3.0], [0.8, 0.2, -3.0], [0.1, 0.7, -3.2]])
    many = np.stack([TRI, tri2])
    slips = np.array([SLIP, np.array([-0.002, 0.009, 0.001])])
    Gm = np.zeros((n_obs, 3, 2, 3))
    asm.assemble_fullspace_disp(OBS, many, MU, LAM, np.full(2, EPS), Gm)
    u_many = np.einsum("oisk,sk->oi", Gm, slips)
    u_sum = (np.asarray(clq.displacement(OBS, TRI, slips[0], MU, nu, EPS))
             + np.asarray(clq.displacement(OBS, tri2, slips[1], MU, nu, EPS)))
    check("e 2-source matrix == sum of the two fields", relerr(u_many, u_sum),
          1e-13)
    # each column must be its OWN source, so column 0 must match the 1-source run
    check("e column 0 is unchanged by adding a second source",
          relerr(Gm[:, :, 0, :], G[:, :, 0, :]), 1e-15)

    # ---------------------------------------------------------------- [f] ---
    print("\n[f] eps IS PER SOURCE")
    eps_graded = np.array([EPS, 3.0 * EPS])
    Gg = np.zeros((n_obs, 3, 2, 3))
    asm.assemble_fullspace_disp(OBS, many, MU, LAM, eps_graded, Gg)
    u_graded = np.einsum("oisk,sk->oi", Gg, slips)
    u_sep = (np.asarray(clq.displacement(OBS, TRI, slips[0], MU, nu, EPS))
             + np.asarray(clq.displacement(OBS, tri2, slips[1], MU, nu,
                                           3.0 * EPS)))
    check("f graded eps == each source at its own eps",
          relerr(u_graded, u_sep), 1e-13)
    # and the graded run must DIFFER from the uniform one, or the check is blind
    moved = relerr(Gg[:, :, 1, :], Gm[:, :, 1, :])
    check_true("f tripwire: a 3x eps actually changes that column",
               moved > 1e-3, f"relative change {moved:.3e}")

    print("-" * 76)
    if all(CHECKS):
        print(f"PASS: assembly packs, splits and grades correctly "
              f"({len(CHECKS)} checks)")
    else:
        print(f"FAIL: {sum(1 for c in CHECKS if not c)} of {len(CHECKS)} "
              f"checks failed")
    return all(CHECKS)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
