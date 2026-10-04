#!/usr/bin/env python
"""The shipped moment table against GAUSS QUADRATURE, plus the two-route identity.

Ported from upstream ``clq``'s ``verify_moments``, pointed at ``mhs.fullspace``,
and it is the first gate here that establishes correctness rather than
agreement. The parity gate compares the shipped engine with the frozen oracle,
which today is a copy against a copy. **This gate's reference is Gauss
quadrature of the same integrand, computed inside the gate**, sharing no code
with the closed form it checks -- so it can tell whether the mathematics is
right, not merely whether two files match.

What it covers, and why each case is here:

  * OFF-PLANE observers, near, far and on the other side, against a 200x200
    rule at 1e-10.
  * NEAR-PLANE, z = 0.2 h, where the closed form's edge expansions start to
    matter and the quadrature reference itself needs the finer rule.
  * ON-PLANE and mollified -- inside the triangle, outside its footprint, at a
    VERTEX and at an EDGE MIDPOINT. These are the four places the edge
    primitives switch regime, and the vertex and edge cases are where a naive
    difference loses digits. Quadrature-limited at 1e-7.
  * The TWO-ROUTE ``M^(1,1)`` identity at **1e-13**. ``M_n^(1,1)`` can be
    reached by recursing in either in-plane direction, and the two must agree.
    It is the only entry with no interior term, so it is the sharpest statement
    that the master recurrence is right -- and it needs no external reference at
    all.
  * eps = 0 OFF the plane: the singular kernel, whose integrals are still
    finite. Catches a closed form that only works because eps regularised it.
  * eps = 0 ON the plane must RAISE, not return a quiet number.
  * The FORCE (Kelvin single-layer) degrees, which reach rows the dislocation
    kernels never touch: ``n = 1`` at degree 2, which via the degree closure
    pulls in ``n = -1`` -- positive powers of R, seeded DOWNWARD instead of
    upward. Without this sweep those two rows are exercised only indirectly.

Tolerances are upstream's, unchanged. They were measured there and loosening one
in translation is the one way porting a gate can do harm.

Run from anywhere:  python tests/gates/mhs/verify_moments.py
"""
from __future__ import annotations

import sys

import numpy as np

from _common import TRI, Report, relmax, shipped


def quad_moments(fr, obs, eps, degrees, n):
    """``M_n^(a,b)`` by Gauss quadrature on the reference triangle.

    The independent reference. It evaluates the integrand directly -- in-plane
    monomials over a power of the mollified distance -- and shares nothing with
    the seed/recurrence/edge-primitive machinery under test.
    """
    gauss_triangle = shipped("moments").gauss_triangle
    z, X = fr.to_plane(obs)
    x1, x2, w = gauss_triangle(n)
    lam = np.stack([1 - x1 - x2, x1, x2], 1)
    eta = lam @ fr.p
    wq = w * 2.0 * fr.area
    xi = eta[None] - X[:, None]
    R2 = xi[..., 0] ** 2 + xi[..., 1] ** 2 + (z * z + eps * eps)[:, None]
    out = {}
    for nn, d in degrees.items():
        out[nn] = np.zeros((len(obs), d + 1, d + 1))
        for a in range(d + 1):
            for b in range(d + 1 - a):
                out[nn][:, a, b] = (wq * xi[..., 0] ** a * xi[..., 1] ** b
                                    * R2 ** (-0.5 * nn)).sum(1)
    return out


def worst(tab, ref) -> float:
    """The worst relative error over every entry of the table."""
    w = 0.0
    for n, d in tab.degrees.items():
        for a in range(d + 1):
            for b in range(d + 1 - a):
                w = max(w, relmax(tab.M[n][:, a, b], ref[n][:, a, b]))
    return w


def main() -> bool:
    moments = shipped("moments")
    frame = shipped("frame")
    MomentTable = moments.MomentTable
    kernel_degrees = moments.kernel_degrees

    rep = Report("Shipped 2-D moment table vs Gauss quadrature "
                 "(mhs.fullspace)")
    fr = frame.local_frame(TRI)
    v1, v2, v3 = TRI
    deg2 = kernel_degrees(2, ("U", "H", "E"))

    # --- off-plane ----------------------------------------------------------
    obs = np.array([[0.60, -0.10, 0.60], [2.0, 1.5, 3.0], [-0.4, 0.0, -0.8]])
    eps = 0.1
    tab = MomentTable(fr, obs, eps, deg2, check_identity=True)
    rep.check("P2 table, off-plane, vs 200x200 Gauss",
              worst(tab, quad_moments(fr, obs, eps, tab.degrees, 200)), 1e-10)

    # --- near-plane (z = 0.2 h on this tilted triangle) ---------------------
    obs_np = np.array([[0.9, 0.3, 0.15]])
    tab_np = MomentTable(fr, obs_np, eps, deg2)
    rep.check("P2 table, near-plane z=0.2 h, vs 200x200 Gauss",
              worst(tab_np, quad_moments(fr, obs_np, eps, tab_np.degrees, 200)),
              1e-10)
    rep.check("two-route M(1,1) identity (off-plane)",
              tab.identity_residual, 1e-13)

    # --- on-plane, mollified: inside, outside, vertex, edge midpoint --------
    eps = 0.15 * fr.L
    obs = np.array([v1 + 0.3 * (v2 - v1) + 0.3 * (v3 - v1),
                    v1 + 1.2 * (v2 - v1) - 0.4 * (v3 - v1),
                    v1,
                    0.5 * (v2 + v3)])
    tab = MomentTable(fr, obs, eps, deg2, check_identity=True)
    rep.check("P2 table, on-plane h=eps=0.15L, vs 160x160 Gauss",
              worst(tab, quad_moments(fr, obs, eps, tab.degrees, 160)), 1e-7)
    rep.check("two-route M(1,1) identity (on-plane)",
              tab.identity_residual, 1e-13)

    # --- eps = 0 off the plane: singular kernel, finite integrals -----------
    obs = np.array([[0.60, -0.10, 0.60], [-0.4, 0.0, -0.8]])
    tab = MomentTable(fr, obs, 0.0, deg2)
    rep.check("P2 table, eps=0 off-plane, vs 200x200 Gauss",
              worst(tab, quad_moments(fr, obs, 0.0, tab.degrees, 200)), 1e-10)

    # --- order-3 smoke ------------------------------------------------------
    deg3 = kernel_degrees(3, ("U", "H", "E"))
    obs = np.array([[0.60, -0.10, 0.60], [2.0, 1.5, 3.0]])
    tab = MomentTable(fr, obs, 0.1, deg3)
    # degree-7 entries lose a few digits in the binomial edge expansion
    rep.check("P3 table (smoke), off-plane, vs 200x200 Gauss",
              worst(tab, quad_moments(fr, obs, 0.1, tab.degrees, 200)), 1e-8)

    # --- n = 9, which the HALF-SPACE IMAGE kernel needs ---------------------
    # The seed ladder used to stop at I_7, because the full-space kernels stop
    # at R^-7. The image kernel needs I_9: the Papkovich-Neuber form already
    # takes one derivative and the slip-to-stress readout two more, so three
    # derivatives act on the potentials' R^-3 term. The three hardcoded cases
    # are now ONE LOOP over the same vertical identity, which also makes n = 11
    # reachable -- so this clause checks the LADDER, not just the rung needed.
    need9 = {9: 2, 7: 4, 5: 3, 3: 1}
    obs9 = np.array([[0.60, -0.10, 0.60], [2.0, 1.5, 3.0], [-0.4, 0.0, -0.8]])
    tab9 = MomentTable(fr, obs9, 0.1, need9)
    rep.check_bool("the seed ladder reaches n = 9", 9 in tab9.M,
                   f"(built n = {sorted(tab9.degrees)})")
    rep.check("M_9 table, off-plane, vs 200x200 Gauss",
              worst(tab9, quad_moments(fr, obs9, 0.1, tab9.degrees, 200)),
              1e-10)
    obs_on = np.array([v1 + 0.3 * (v2 - v1) + 0.3 * (v3 - v1), v1])
    e_on = 0.15 * fr.L
    tab9o = MomentTable(fr, obs_on, e_on, need9)
    rep.check("M_9 table, ON-plane h=eps=0.15L, vs 160x160 Gauss",
              worst(tab9o, quad_moments(fr, obs_on, e_on, tab9o.degrees, 160)),
              1e-7)
    # Generic, not special-cased to 9: one rung higher must build too, or the
    # loop has been turned back into a case list.
    tab11 = MomentTable(fr, obs9, 0.1, {11: 1, 9: 2, 7: 4, 5: 3, 3: 1})
    rep.check("M_11 too (the ladder is a loop, not a case list)",
              worst(tab11, quad_moments(fr, obs9, 0.1, tab11.degrees, 200)),
              1e-10)

    # --- eps = 0 on the plane must raise, not return a quiet number ---------
    try:
        MomentTable(fr, np.array([v1 + 0.3 * (v2 - v1) + 0.3 * (v3 - v1)]),
                    0.0, deg2)
        rep.check_bool("eps=0 on the plane raises ValueError", False)
    except ValueError:
        rep.check_bool("eps=0 on the plane raises ValueError", True)

    # --- force degrees: the n = 1 and n = -1 rows ---------------------------
    degF = kernel_degrees(2, ("G", "S"))
    rep.check_bool("kernel_degrees(2, ('G','S')) == {1: 2, 3: 4, 5: 5}",
                   degF == {1: 2, 3: 4, 5: 5}, f"(got {degF})")
    rep.check_bool("kernel_degrees(0, ('S',)) == kernel_degrees(0, ('U',)) "
                   "(S and U share G1)",
                   kernel_degrees(0, ("S",)) == kernel_degrees(0, ("U",)),
                   f"(S {kernel_degrees(0, ('S',))}, "
                   f"U {kernel_degrees(0, ('U',))})")
    obs_f = np.array([[0.60, -0.10, 0.60], [2.0, 1.5, 3.0], [-0.4, 0.0, -0.8]])
    tabF = MomentTable(fr, obs_f, 0.1, degF, check_identity=True)
    rep.check_bool("force degrees close to n = 1 (degree 2) and n = -1",
                   tabF.degrees.get(1, -1) == 2 and -1 in tabF.degrees,
                   f"(closed degrees {tabF.degrees})")
    rep.check("P2 force table (n = 1, -1, 3, 5), off-plane, vs 200x200 Gauss",
              worst(tabF, quad_moments(fr, obs_f, 0.1, tabF.degrees, 200)),
              1e-11)
    rep.check("two-route M(1,1) identity (force degrees)",
              tabF.identity_residual, 1e-13)
    eps_f = 0.15 * fr.L
    obs_fon = np.array([v1 + 0.3 * (v2 - v1) + 0.3 * (v3 - v1),
                        v1 + 1.2 * (v2 - v1) - 0.4 * (v3 - v1),
                        v1,
                        0.5 * (v2 + v3)])
    tabF_on = MomentTable(fr, obs_fon, eps_f, degF)
    rep.check("P2 force table, on-plane h=eps=0.15L, vs 160x160 Gauss",
              worst(tabF_on, quad_moments(fr, obs_fon, eps_f, tabF_on.degrees,
                                          160)), 1e-12)
    tabF_0 = MomentTable(fr, obs_f[[0, 2]], 0.0, degF)
    rep.check("P2 force table, eps=0 off-plane, vs 200x200 Gauss",
              worst(tabF_0, quad_moments(fr, obs_f[[0, 2]], 0.0,
                                         tabF_0.degrees, 200)), 1e-11)
    part_series(rep)
    return rep.finish()


def part_series(rep) -> None:
    """THE MEMOISED EDGE SERIES, against the per-call forms, BITWISE.

    ``edge_table`` shares the exponent-indexed work of both binomial series
    across the ``(k, m)`` pairs, because the power differences and the bases
    depend only on an exponent and those collide heavily -- 416 array
    evaluations covering ~69 distinct exponents on the image kernel's spec. The
    claim that goes with it is that nothing was reassociated: the same products
    in the same order, so every entry is bitwise what the per-call form gives.

    That is worth a clause rather than a measurement taken once. A tolerance
    here would be the wrong instrument -- if these ever merely AGREE rather
    than being identical, the sharing reassociated something, and the right
    response is to find out what, not to widen a bound.

    Both regimes, and every ``m`` the kernels reach including the force
    family's ``m = 1`` and the downward-seeded ``m = -1``.
    """
    from mhs.fullspace import defaults as fd
    from mhs.fullspace.primitives import (_LargeUSeries, _SmallUSeries,
                                          _series_dP, _small_u_series_dP)

    print("\n  [series] memoised contexts vs the per-call forms, bitwise")
    spec = {-1: 2, 1: 2, 3: 2, 5: 4, 7: 5, 9: 6}
    nT = fd.SERIES_TERMS
    rng = np.random.default_rng(17)
    n_arr = n_same = 0
    for _ in range(3):
        N = 256
        # large-|u|: same sign, |u| >> rho, which is where SERIES_U_OVER_RHO
        # sends a batch
        rho2 = np.abs(rng.normal(size=N)) * 0.05 + 1e-4
        ua = np.abs(rng.normal(size=N)) * 4.0 + 4.0 * np.sqrt(rho2)
        ub = ua + np.abs(rng.normal(size=N)) * 3.0 + 1e-3
        ctx = _LargeUSeries(ua, ub, rho2, nT)
        for m, kmax in spec.items():
            for k in range(kmax + 1):
                n_arr += 1
                n_same += int(np.array_equal(
                    ctx.dP(k, m), _series_dP(k, m, ua, ub, rho2, nT)))
        # small-|u|: |u| << rho, MIXED signs, which is the branchier one
        rho2s = np.abs(rng.normal(size=N)) * 3.0 + 1.0
        rhos = np.sqrt(rho2s)
        uas = rng.normal(size=N) * 0.1 * rhos
        ubs = uas + np.abs(rng.normal(size=N)) * 0.1 * rhos + 1e-9
        ctx2 = _SmallUSeries(uas, ubs, rho2s, nT)
        for m, kmax in spec.items():
            for k in range(kmax + 1):
                n_arr += 1
                n_same += int(np.array_equal(
                    ctx2.dP(k, m),
                    _small_u_series_dP(k, m, uas, ubs, rho2s, nT)))
    rep.check_bool(f"memoised edge series == per-call, BITWISE "
                   f"({n_same}/{n_arr} arrays)", n_same == n_arr,
                   "no tolerance: these are the same products in the same "
                   "association and the same accumulation order, so anything "
                   "short of identical means the sharing reassociated "
                   "something")

    # And the tripwire: the regime masks must PARTITION, because edge_table
    # fills an np.empty and a row claimed by no branch is uninitialised memory
    # in a moment table. edge_table asserts this at runtime; this checks that
    # the assertion is reachable rather than vacuous, by driving a batch that
    # lands in all four regimes at once.
    from mhs.fullspace.primitives import edge_table
    ua = np.array([2.0, 0.01, 1.0, -1.0, 3.0])
    ub = np.array([9.0, 0.02, 2.0, 1.0, 4.0])
    rho2 = np.array([1e-4, 4.0, 0.3, 0.3, 0.0])      # ser, small, closed, closed, zero
    tab = edge_table(ua, ub, rho2, {3: 1, 5: 2})
    rep.check_bool("a mixed batch hits every edge regime and stays finite",
                   all(np.all(np.isfinite(v)) for v in tab.values()),
                   "large-|u| series, small-|u| series, closed form (both "
                   "signs) and rho = 0 in one call -- an uninitialised row "
                   "would show up as garbage or NaN here")


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
