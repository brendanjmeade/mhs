#!/usr/bin/env python
"""The eps -> 0 limit against cutde: the only EXTERNAL check in the package.

Every other gate is an identity, a physics condition, or a comparison against a
vendored oracle. None of them can catch a SLIP-BASIS error -- a strike/dip
swap, a dip-sign flip, a reflected normal -- because the oracle parity compares
two copies that would share it, and the identities (``total - eigenstress ==
stress``, reciprocity, the free-surface condition) hold in ANY orthonormal slip
basis. ``cutde.halfspace`` is an independent implementation of the classical
limit, so it is the one thing that can.

THE BASIS IS IMPOSED, NOT FITTED. A least-squares fit of the 3x3 map between
cutde's (strike, dip, tensile) slip and mhs's Cartesian slip would ABSORB a
rotation error and pass. So the map is built from the element's vertices --
``strike = (-n2, n1, 0)`` normalised, ``dip = n x strike`` -- and imposed. (The
fit was useful for DISCOVERING the convention: it came out orthogonal with
det +0.9998 and equal to that geometric frame to 5e-5, which is how the
convention was established rather than guessed from documentation.)

  [a] THE CONVENTION, per slip component separately. Imposing the frame and
      comparing column by column means a rotation error shows as disagreement
      in a column rather than being absorbed. Run at nu != 1/4, because a
      lam/mu swap is invisible there.

  [b] DISPLACEMENT CONVERGES AT ORDER 2. The order is the real check, not
      agreement at one eps: mollified -> classical is O(eps^2), and a
      convention error converges to the WRONG field rather than slowly. An
      absolute tolerance at a single eps would pass for a rotated basis.

  [c] STRESS CONVERGES AT ORDER 2, but only where cutde's stress is FINITE.
      cutde is the unmollified kernel: its stress diverges on the element and
      carries a 1/r line singularity at the edges. Measured, with the
      eigenstress as a fraction of the signal in the third column:

        dist/L   |sig| cutde   eigen/sig     eps=0.1    eps=0.025   order
          3.00        0.1901    3.4e-08     1.5e-03     9.5e-05      2.00
          1.50        1.0127    1.2e-06     3.3e-03     2.1e-04      2.00
          0.80        5.9450    7.8e-05     2.3e-02     1.5e-03      1.98
          0.40       73.9668    3.3e-02     1.8e-01     1.5e-02      1.79
          0.20      106.4034    2.8e+00     3.3e-01     3.3e-02      1.64

      so the comparison is clean beyond about one element length and degrades
      inside half of one -- where the eigenstress stops being negligible and
      becomes LARGER than the elastic stress it is subtracted from. This gate
      compares at >= 1 L and nowhere closer.

  [d] A TRIPWIRE THAT THE CLOSE-IN COMPARISON REALLY IS INVALID, so [c]'s
      restriction is a measured boundary rather than a convenient one, and so
      nobody extends it inward believing it will hold.

Needs the ``[tde]`` extra. It FAILS rather than skips without cutde: a skipped
external anchor is indistinguishable from a passing one in a summary line, and
this is the only external check there is.

Run from anywhere:  python tests/gates/mhs/verify_cutde_limit.py
"""
from __future__ import annotations

import sys

import numpy as np

from _common import Report

NU = 0.27                 # NOT 1/4: a lam/mu swap is invisible there
MU = 30.0
LAM = 2.0 * MU * NU / (1.0 - 2.0 * NU)

#: Dipping and off every symmetry plane, so no component vanishes for free.
TRI = np.array([[[0.1, 0.0, -2.0], [1.2, 0.3, -2.4], [0.5, 1.1, -1.5]]])

EPSS = (0.2, 0.1, 0.05, 0.025)
TOL_ORDER_MIN = 1.85      # measured 1.98 .. 2.19
TOL_DISP = 5e-4           # measured 1.4e-4 at the finest eps
TOL_STRESS = 2e-3         # measured 2.1e-4 at 1.5 L, finest eps
TOL_FRAME = 1e-3          # geometric frame vs the published convention
TRIP_CLOSE_ORDER = 1.85   # close in, the order must FALL below this
TRIP_EIGEN_SHARE = 0.5    # and the eigenstress must be a large fraction there


def slip_frame(tri):
    """cutde's (strike, dip, tensile) basis as rows, FROM THE SHIPPED MODULE.

    This was a private copy built from the vertices here, on the reasoning that
    a reference should not import what it checks. That had it backwards. The
    frame is a CONVENTION whose only external anchor is cutde, and with a
    private copy this gate passed no matter what ``mhs.tdcs`` said -- so a wrong
    frame could ship to callers while the comparison stayed green. Reading it
    from ``mhs.tdcs`` makes the fitted-map clause below check the SHIPPED
    convention against cutde's actual behaviour, which is the thing worth
    knowing, and is rule 10's point: a convention written twice can disagree
    with itself.
    """
    from mhs import tdcs
    return tdcs.slip_frame(np.asarray(tri, float)[None])[0]


def _order(errs, epss):
    return [float(np.log(errs[i] / errs[i + 1])
                  / np.log(epss[i] / epss[i + 1]))
            for i in range(len(errs) - 1)]


def main() -> bool:
    import cutde.halfspace as hs

    from mhs import Material
    from mhs.matrices import disp_matrix, eigenstress_matrix, stress_matrix

    rep = Report("eps -> 0 against cutde.halfspace (the external anchor)")
    mat = Material(mu=MU, lam=LAM)
    tri = TRI[0]
    cen = tri.mean(axis=0)
    L = max(float(np.linalg.norm(tri[i] - tri[j]))
            for i in range(3) for j in range(3))
    M = slip_frame(tri)                      # (slip_sdt, xyz)

    th = np.linspace(0.0, 2.0 * np.pi, 9)[:-1]

    def ring(mult, lift):
        r = mult * L
        o = cen + np.stack([r * np.cos(th), r * np.sin(th),
                            np.zeros_like(th)], axis=1)
        o[:, 2] = np.minimum(cen[2] + lift * r, -0.02)
        return o

    # ------------------------------------------------------------------ [a] ---
    print("\n[a] THE SLIP BASIS, imposed from the vertices and compared per "
          "component")
    print(f"    strike {np.round(M[0], 6)}")
    print(f"    dip    {np.round(M[1], 6)}")
    print(f"    normal {np.round(M[2], 6)}")
    rep.check("a the imposed frame is orthonormal",
              float(np.abs(M @ M.T - np.eye(3)).max()), TOL_FRAME,
              "a non-orthonormal basis would make every comparison below "
              "meaningless")
    rep.check_bool("a and right-handed (det = +1, not a reflection)",
                   abs(float(np.linalg.det(M)) - 1.0) < TOL_FRAME,
                   f"(det {float(np.linalg.det(M)):+.6f}) -- a reflected "
                   f"normal passes an orthonormality check and fails here")

    obs = ring(1.5, 0.35)
    cd = np.asarray(hs.disp_matrix(obs.copy(), TRI.copy(), NU))[:, :, 0, :]
    want = cd @ M                            # cutde slip -> Cartesian slip
    print(f"\n    per-slip-component displacement error at eps = "
          f"{EPSS[-1]:g}, {obs.shape[0]} observers on a ring at 1.5 L")
    got = disp_matrix(obs, TRI, mat, EPSS[-1])[:, :, 0, :]
    scale = float(np.abs(want).max())
    for k, name in enumerate(("x", "y", "z")):
        e = float(np.abs(got[:, :, k] - want[:, :, k]).max() / scale)
        print(f"      slip {name}: {e:.3e}")
        rep.check(f"a slip component {name} matches cutde", e, TOL_DISP,
                  "compared component by component with the frame IMPOSED, so "
                  "a rotation error cannot be absorbed into a fitted map")

    # ------------------------------------------------------------------ [b] ---
    print("\n[b] DISPLACEMENT: convergence order as eps -> 0")
    print(f"    {'eps':>8} {'rel err':>12}")
    du = []
    for eps in EPSS:
        g = disp_matrix(obs, TRI, mat, eps)[:, :, 0, :]
        du.append(float(np.abs(g - want).max() / scale))
        print(f"    {eps:8.3f} {du[-1]:12.3e}")
    ou = _order(du, EPSS)
    print(f"    orders: " + ", ".join(f"{o:.2f}" for o in ou))
    rep.check_bool(f"b displacement converges at order >= {TOL_ORDER_MIN}",
                   min(ou) >= TOL_ORDER_MIN,
                   f"(min {min(ou):.2f}) -- the ORDER is the check: a "
                   f"convention error converges to the wrong field, not slowly")
    rep.check("b and reaches cutde at the finest eps", du[-1], TOL_DISP)

    # ------------------------------------------------------------------ [c] ---
    def cutde_stress(o):
        E = np.asarray(hs.strain_matrix(o.copy(), TRI.copy(), NU))[:, :, 0, :]
        idx = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
        S = np.zeros((o.shape[0], 3, 3, 3))
        for k in range(3):
            e = np.zeros((o.shape[0], 3, 3))
            for v, (a, b) in enumerate(idx):
                e[:, a, b] = E[:, v, k]
                e[:, b, a] = E[:, v, k]
            tr = e[:, 0, 0] + e[:, 1, 1] + e[:, 2, 2]
            S[:, :, :, k] = 2.0 * MU * e
            for a in range(3):
                S[:, a, a, k] += LAM * tr
        return np.einsum("nabk,kc->nabc", S, M)

    print("\n[c] STRESS, where cutde's stress is FINITE (>= 1 L)")
    print(f"    {'dist/L':>8} {'eigen/sig':>11} {'coarsest':>11} "
          f"{'finest':>11} {'order':>7}")
    worst_s, worst_o = 0.0, 9.9
    for mult in (3.0, 1.5):
        o = ring(mult, 0.35)
        cs = cutde_stress(o)
        sc = float(np.abs(cs).max())
        eg = eigenstress_matrix(o, TRI, mat, EPSS[0])[:, :, :, 0, :]
        es = []
        for eps in (EPSS[0], EPSS[-1]):
            ms = stress_matrix(o, TRI, mat, eps)[:, :, :, 0, :]
            es.append(float(np.abs(ms - cs).max() / sc))
        order = float(np.log(es[0] / es[1]) / np.log(EPSS[0] / EPSS[-1]))
        worst_s = max(worst_s, es[-1])
        worst_o = min(worst_o, order)
        print(f"    {mult:8.2f} {np.abs(eg).max() / sc:11.2e} "
              f"{es[0]:11.3e} {es[1]:11.3e} {order:7.2f}")
    rep.check("c stress reaches cutde at the finest eps", worst_s, TOL_STRESS,
              "with the eigenstress removed, which at these distances is "
              "below 1e-6 of the signal anyway")
    rep.check_bool(f"c stress converges at order >= {TOL_ORDER_MIN}",
                   worst_o >= TOL_ORDER_MIN, f"(min {worst_o:.2f})")

    # ------------------------------------------------------------------ [d] ---
    print("\n[d] TRIPWIRE: the close-in stress comparison really IS invalid,")
    print("    so [c]'s >= 1 L is a measured boundary and not a convenient one")
    o = ring(0.2, 0.35)
    cs = cutde_stress(o)
    sc = float(np.abs(cs).max())
    eg = eigenstress_matrix(o, TRI, mat, EPSS[0])[:, :, :, 0, :]
    share = float(np.abs(eg).max() / sc)
    es = []
    for eps in (EPSS[0], EPSS[-1]):
        ms = stress_matrix(o, TRI, mat, eps)[:, :, :, 0, :]
        es.append(float(np.abs(ms - cs).max() / sc))
    order = float(np.log(es[0] / es[1]) / np.log(EPSS[0] / EPSS[-1]))
    print(f"    at 0.2 L: |sig| cutde {sc:.2f}, eigenstress/sig {share:.2f}, "
          f"order {order:.2f}")
    rep.check_bool(f"d the order DEGRADES close in (< {TRIP_CLOSE_ORDER})",
                   order < TRIP_CLOSE_ORDER,
                   f"({order:.2f}) -- cutde's stress is diverging here, so "
                   f"this is a limitation of the REFERENCE, not of mhs")
    rep.check_bool(f"d and the eigenstress dominates there "
                   f"(> {TRIP_EIGEN_SHARE:g} of the signal)",
                   share > TRIP_EIGEN_SHARE,
                   f"({share:.2f}x) -- which is WHY: the elastic stress is a "
                   f"difference of two terms that barely resemble each other. "
                   f"The near field is anchored by the adaptive-quadrature "
                   f"reference and the oracle instead, both at 1e-15.")
    return rep.finish()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
