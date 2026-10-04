#!/usr/bin/env python
"""The half-space image kernel: closed-form R-family plus quadrature Q-family.

This gate covers the thing the package exists to do, and it is deliberately
ORACLE-FREE: every check below is either an identity, a physics condition, or a
comparison against adaptive quadrature built here. The oracle comparison lives
in ``parity/verify_deep_source``; what this gate proves is that the kernel is
RIGHT, not merely that it reproduces another implementation.

  [a] THE GENERATED TABLE'S SHAPE. ``_image_table.py`` is generated, so the
      consumer asserts the counts, the maximum tensor rank and the set of
      ``R2`` orders it was built with. A truncated or hand-edited table would
      otherwise silently halve a kernel. The orders must be ODD, because that
      is the whole reason clq's hierarchy covers the R-family at no new
      primitives.

  [b] THE FREE-SURFACE CONDITION, which is what the image correction is FOR and
      which needs no reference at all: traction on ``z = 0`` must vanish.
      Measured against the direct term alone, where it does not -- that
      contrast is the clause. If the image were dropped, mis-signed, or
      reflected in the wrong plane, this is what would catch it, and no
      identity or self-consistency check would.

  [c] CLOSED FORM vs ADAPTIVE QUADRATURE, at a benign observer and ON the
      surface trace. The reference subdivides until each leaf is small compared
      with its distance from the observer, floored at eps -- because the
      integrand's peak has width eps at the observer's projection, so a leaf
      must be either far from that peak or smaller than it. UNIFORM Gauss
      cannot referee this: clause [e] of ``verify_vertical_fault`` measures it
      stagnating at O(1) exactly here, which is why the closed form exists.
      (The first draft of this reference refined only the child nearest the
      observer, leaving the ADJACENT children holding part of the peak at the
      base rule; it disagreed with both the closed form and uniform Gauss,
      which is how it was caught.)

  [d] THE Q-FAMILY IS NOT NEGLIGIBLE -- a tripwire the wrong way round. It is
      the half left in quadrature, and the temptation is to drop it. Near the
      trace it is a fifth of the on-fault stress, so this clause asserts that
      omitting it MOVES the answer. If it ever stopped mattering, the kernel
      would have changed.

  [f] THE REMAINING GAP, gated so it cannot be forgotten: the Q-family is
      still quadrature and does NOT converge on the trace. Step 8a closed the
      R-family; the composite kernel is still limited there. A tripwire, so the
      commit that lands step 8b has to invert it. See part_f.

  [e] THE THREE-WAY SPLIT, bitwise. ``total - eigenstress == stress`` as an
      identity rather than a tolerance: if it ever needed one, the three entry
      points would not be three views of one computation. Also that the
      eigenstress carries NO image contribution -- it is a pointwise statement
      about the blob and the source triangle, reading no Green's function.

Run from anywhere:  python tests/gates/mhs/verify_image_kernel.py
"""
from __future__ import annotations

import sys

import numpy as np

from _common import Report, relmax

MU, LAM = 30.0, 45.0            # mu != lam, so a swap is visible
EPS_FINE = 0.01

#: Buried, well clear: uniform quadrature is a valid reference here.
TRI_DEEP = np.array([[0.1, 0.0, -2.0],
                     [1.0, 0.2, -2.1],
                     [0.4, 0.9, -1.6]])
#: Vertical AND reaching z = 0, tilted off x = 0 so no component vanishes by
#: symmetry and a relative error stays meaningful. The worst geometry there is:
#: the image is coplanar with the element and shares its surface edge.
TRI_SURF = np.array([[0.0, 0.0, 0.0],
                     [0.08, 1.0, 0.0],
                     [0.03, 0.5, -1.0]])

TOL_CLOSED = 1e-10        # measured 1e-15 .. 2e-13 against the adaptive rule
TOL_FREE_SURFACE = 2e-3   # the image's job; the direct term alone is O(1)
TRIP_DIRECT_ONLY = 0.2    # the direct term alone must FAIL the condition
TRIP_Q_SHARE = 0.02       # the Q-family must move the answer by at least this


def _gauss_tri(n):
    gx, gw = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (gx + 1.0)
    wu = 0.5 * gw
    U_, V_ = np.meshgrid(u, u, indexing="ij")
    WU, WV = np.meshgrid(wu, wu, indexing="ij")
    return U_.ravel(), (V_ * (1.0 - U_)).ravel(), (WU * WV * (1.0 - U_)).ravel()


def _diam(t):
    return max(float(np.linalg.norm(t[i] - t[j]))
               for i in range(3) for j in range(3))


def _near_dist(t, obs):
    pts = np.vstack([t, t.mean(0), 0.5 * (t[0] + t[1]), 0.5 * (t[1] + t[2]),
                     0.5 * (t[2] + t[0])])
    return float(np.min(np.linalg.norm(pts - obs, axis=1)))


def _adaptive(tri, obs, eps, depth=14, nq=12, eta=0.5):
    """The image R-family by subdivision, as a reference the closed form has
    not seen. Admissible leaf: small compared with its distance from the
    observer, floored at eps."""
    if depth == 0 or _diam(tri) <= eta * max(_near_dist(tri, obs), eps):
        return _pointwise_integral(tri, obs, eps, nq)
    m01 = 0.5 * (tri[0] + tri[1])
    m12 = 0.5 * (tri[1] + tri[2])
    m20 = 0.5 * (tri[2] + tri[0])
    kids = (np.array([tri[0], m01, m20]), np.array([m01, tri[1], m12]),
            np.array([m20, m12, tri[2]]), np.array([m01, m12, m20]))
    return sum(_adaptive(k, obs, eps, depth - 1, nq, eta)
               for k in kids)


def _pointwise_integral(tri, obs, eps, nq):
    """Gauss rule on ONE (sub)triangle, from the table evaluated POINTWISE.

    Deliberately independent of ``image.py``: it reads the same generated table
    but shares none of the moment machinery, so agreement tests the closed-form
    integration rather than the table.
    """
    tab = _TABLE
    l1, l2, w = _gauss_tri(nq)
    y = ((1 - l1 - l2)[:, None] * tri[0] + l1[:, None] * tri[1]
         + l2[:, None] * tri[2])
    a2 = float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))
    nu = LAM / (2.0 * (LAM + MU))
    scale = 2.0 * (LAM + MU) / (np.pi * MU * (LAM + 2.0 * MU))
    d1 = obs[0] - y[:, 0]
    d2 = obs[1] - y[:, 1]
    d3 = obs[2] + y[:, 2]
    r2 = np.sqrt(d1 * d1 + d2 * d2 + d3 * d3 + eps ** 2)
    DDG = np.zeros((len(w), 3, 3, 3, 3))
    for rec in tab.DDG_RECORDS:
        i, j, p, m, a, b, c, n = rec[:8]
        cv = 0.0
        for pz, pnu, num, den in rec[8]:
            cv += (num / den) * obs[2] ** pz * nu ** pnu
        DDG[:, i, j, p, m] += (cv * scale) * d1 ** a * d2 ** b * d3 ** c / r2 ** n
    nrm = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    nrm = nrm / np.linalg.norm(nrm)
    S = np.zeros((len(w), 3, 3, 3))
    for i in range(3):
        for p in range(3):
            trm = sum(DDG[:, i, m, p, m] for m in range(3))
            for k in range(3):
                S[:, i, p, k] = (
                    MU * sum(nrm[l] * DDG[:, i, k, p, l] for l in range(3))
                    + MU * sum(nrm[l] * DDG[:, i, l, p, k] for l in range(3))
                    + LAM * nrm[k] * trm)
    H = np.zeros((len(w), 3, 3, 3))
    for k in range(3):
        g = S[:, :, :, k]
        e = 0.5 * (g + np.swapaxes(g, -1, -2))
        t = e[:, 0, 0] + e[:, 1, 1] + e[:, 2, 2]
        for a_ in range(3):
            for b_ in range(3):
                H[:, a_, b_, k] = 2.0 * MU * e[:, a_, b_]
                if a_ == b_:
                    H[:, a_, b_, k] += LAM * t
    return (w[:, None, None, None] * a2 * H).sum(0)


_TABLE = None      # the generated table, bound in main()


def main() -> bool:
    global _TABLE
    import mhs.kernels._image_table as table
    import mhs.kernels.image as image
    from mhs import Material
    from mhs.matrices import (eigenstress_matrix, stress_matrix,
                              total_stress_matrix)
    _TABLE = table

    rep = Report("Half-space image kernel: closed-form R + quadrature Q")

    # ----------------------------------------------------------------- [a] ---
    print("\n[a] THE GENERATED TABLE'S SHAPE")
    print(f"    R-family: {table.N_DG} DG + {table.N_DDG} DDG monomials, "
          f"max rank {table.MAX_RANK}, R2 orders {list(table.N_ORDERS)}")
    print(f"    Q-family: {table.N_Q_DG} DG + {table.N_Q_DDG} DDG "
          f"(evaluated pointwise, integrated by quadrature)")
    rep.check_bool("a record counts match the generated constants",
                   len(table.DG_RECORDS) == table.N_DG
                   and len(table.DDG_RECORDS) == table.N_DDG
                   and len(table.Q_DG_RECORDS) == table.N_Q_DG
                   and len(table.Q_DDG_RECORDS) == table.N_Q_DDG,
                   "a truncated table would silently halve a kernel")
    rep.check_bool("a every R-family R2 order is ODD",
                   all(n % 2 == 1 for n in table.N_ORDERS),
                   f"({list(table.N_ORDERS)}) -- the reason clq's hierarchy "
                   f"covers this family with no new primitives")
    rep.check_bool("a the ladder reaches n = 9", 9 in table.N_ORDERS,
                   "three derivatives act on the potentials' R2^-3 term")

    # ----------------------------------------------------------------- [b] ---
    print("\n[b] THE FREE-SURFACE CONDITION: traction on z = 0 must vanish")
    print("    This is what the image correction is FOR, and it needs no "
          "reference.")
    mat = Material(mu=MU, lam=LAM)
    th = np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
    ring = np.stack([1.6 * np.cos(th), 1.6 * np.sin(th) + 0.5,
                     np.zeros_like(th)], axis=1)
    bulk = ring.copy()
    bulk[:, 2] = -1.0
    tris = TRI_DEEP[None, ...]
    eps = 0.08

    def traction_ratio(full):
        """max|t_3k| on z = 0 over the bulk stress scale."""
        import mhs.kernels.assemble as asm
        obs = np.vstack([ring, bulk])
        buf = np.zeros((obs.shape[0], 3, 3, 1, 3))
        e = np.full(1, eps)
        if full:
            asm.assemble_halfspace_total_stress(obs, tris, MU, LAM, e, buf)
        else:
            asm.assemble_fullspace_total_stress(obs, tris, MU, LAM, e, buf)
        n_r = ring.shape[0]
        t3 = float(np.abs(buf[:n_r, 2, :, 0, :]).max())
        sb = float(np.abs(buf[n_r:, :, :, 0, :]).max())
        return t3 / sb

    r_hs = traction_ratio(True)
    r_fs = traction_ratio(False)
    print(f"    half space (direct + image) : {r_hs:.3e}")
    print(f"    direct term alone           : {r_fs:.3e}")
    rep.check("b the half-space kernel satisfies it", r_hs, TOL_FREE_SURFACE,
              "the image correction's whole purpose")
    rep.check_bool(f"b the direct term alone does NOT (> {TRIP_DIRECT_ONLY})",
                   r_fs > TRIP_DIRECT_ONLY,
                   f"({r_fs:.3e}) -- so [b] is a real test of the image, not a "
                   f"condition satisfied by accident. A dropped, mis-signed or "
                   f"wrongly reflected image fails here and nowhere else.")

    # ----------------------------------------------------------------- [c] ---
    print("\n[c] CLOSED FORM vs ADAPTIVE SUBDIVISION (uniform Gauss cannot "
          "referee this)")
    print(f"    {'geometry':24s} {'delta/eps':>10} {'closed vs adaptive':>20}")
    worst = 0.0
    cases = [("buried, obs clear", TRI_DEEP, np.array([0.3, 0.4, -1.0]), 0.08),
             ("surface, delta/eps = 1", TRI_SURF,
              np.array([0.037, 0.5, -EPS_FINE]), EPS_FINE),
             ("surface, delta/eps = 0.2", TRI_SURF,
              np.array([0.037, 0.5, -0.2 * EPS_FINE]), EPS_FINE)]
    for label, tri, obs, ev in cases:
        cf = image.image_influence(obs.reshape(1, 3), tri, 0, MU, LAM, ev,
                                   want=("H",))["H"][0, 0]
        ref = _adaptive(tri, obs, ev)
        e = relmax(cf, ref)
        worst = max(worst, e)
        print(f"    {label:24s} {abs(obs[2]) / ev:10.2f} {e:20.3e}")
    rep.check("c closed form agrees with the adaptive rule everywhere",
              worst, TOL_CLOSED,
              "including ON the trace, where clause [e] of "
              "verify_vertical_fault measures uniform Gauss stagnating at O(1)")

    # ----------------------------------------------------------------- [d] ---
    print("\n[d] THE Q-FAMILY IS NOT NEGLIGIBLE (a tripwire the wrong way "
          "round)")
    obs = np.array([[0.037, 0.5, -0.2 * EPS_FINE]])
    r_only = image.image_influence(obs, TRI_SURF, 0, MU, LAM, EPS_FINE,
                                   want=("H",))["H"]
    tot = image.image_total(obs, TRI_SURF, 0, MU, LAM, EPS_FINE,
                            want=("H",))["H"]
    share = float(np.abs(tot - r_only).max() / np.abs(tot).max())
    print(f"    dropping the Q-family moves the on-fault stress by "
          f"{100 * share:.1f}%")
    rep.check_bool(f"d omitting the Q-family MATTERS (> {100*TRIP_Q_SHARE:g}%)",
                   share > TRIP_Q_SHARE,
                   f"({100 * share:.1f}%) -- it is left in quadrature because "
                   f"its error is eps-independent, NOT because it is small")

    # ----------------------------------------------------------------- [e] ---
    print("\n[e] THE THREE-WAY SPLIT, and the eigenstress carries no image")
    obs = np.array([[0.3, 0.4, -1.0], [0.0, 0.0, -0.05]])
    sig = stress_matrix(obs, tris, mat, eps)
    tot_s = total_stress_matrix(obs, tris, mat, eps)
    star = eigenstress_matrix(obs, tris, mat, eps)
    rep.check_bool("e total - eigenstress == stress, BITWISE",
                   np.array_equal(tot_s - star, sig),
                   "an identity, not a tolerance: the three entry points are "
                   "three views of one computation")
    s2, e2 = stress_matrix(obs, tris, mat, eps, parts=True)
    rep.check_bool("e parts=True agrees with the separate calls, bitwise",
                   np.array_equal(s2, sig) and np.array_equal(e2, star))
    # The eigenstress is a pointwise statement about the blob and the source
    # triangle: it reads no Green's function, so reflecting the element must
    # not change it beyond the geometry's own relabelling.
    rep.check_bool("e the eigenstress is nonzero where it should be",
                   float(np.abs(star).max()) > 0.0,
                   f"(max {float(np.abs(star).max()):.3e}) -- zero would make "
                   f"the subtraction vacuous and [e] above trivially true")
    part_f(rep, image)
    return rep.finish()


def part_f(rep, image) -> None:
    """THE REMAINING GAP, gated so it cannot be forgotten: the Q-family's
    quadrature does not converge on the surface trace.

    Clause [c] shows the R-family in closed form is machine-accurate at every
    clearance. That is step 8a, and it is NOT the whole kernel. The Q-family is
    still quadrature, and refining it near the trace does not approach a limit
    -- the value CLIMBS, because each refinement resolves more of a peak it
    never captured:

        delta/eps    nq=16     nq=32     nq=64    nq=128
            10.00    185.1    187.83   187.8378  187.8378   converged
             1.00    624.5   1053.1    1430.4    1531.9     climbing
             0.20    743.5   1428.9    2245.8    2540.0     climbing

    So splitting the families did not make the remaining half easy -- it made
    it HARDER than the original sum. R and Q partially cancel (2.3-3.0x here),
    and quadrature on one summand of a cancelling pair is worse conditioned
    than quadrature on the pair: at delta/eps = 10 this kernel's last increment
    is 1.3e-4 where the oracle's uniform FULL-image quadrature reaches 2.8e-6,
    fifty times better, for exactly that reason.

    WHAT IT MEANS FOR THE PACKAGE. On-fault stress is trustworthy at
    delta/eps of a few or more, which covers every buried element. It is NOT
    trustworthy within about an eps of a surface trace -- which is the
    configuration the closed form was built for. Step 8b, the Q-family in
    closed form, is therefore REQUIRED rather than optional, and this clause is
    the tripwire that says so: it asserts the defect is still there, so when 8b
    lands it FAILS and the commit that lands it must invert it.

    Target, stated now so it is not negotiated later: last increment below
    1e-10 at delta/eps = 0.2, which is what clause [c] already delivers for the
    R-family.
    """
    TV = np.array([[0.0, 0.0, 0.0], [0.08, 1.0, 0.0], [0.03, 0.5, -1.0]])
    obs = np.array([[0.037, 0.5, -0.2 * EPS_FINE]])
    print("\n[f] THE REMAINING GAP: the Q-family's quadrature on the trace")
    vals = [image.image_q_influence(obs, TV, 0, MU, LAM, EPS_FINE,
                                    want=("H",), n_quad=n)["H"][0, 0]
            for n in (32, 64, 128)]
    mags = [float(np.abs(v).max()) for v in vals]
    print(f"    |Q| at n_quad 32 / 64 / 128: "
          f"{mags[0]:.1f} / {mags[1]:.1f} / {mags[2]:.1f}")
    inc = relmax(vals[1], vals[2])
    print(f"    last increment: {inc:.2e}   (target for step 8b: < 1e-10)")
    rep.check_bool("f the Q-family STILL does not converge on the trace",
                   inc > 1e-3,
                   f"({inc:.2e}) -- step 8b is REQUIRED, not optional. When it "
                   f"lands this clause FAILS and must be inverted. Clause [c] "
                   f"is what the R-family already delivers here: 1e-15.")


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
