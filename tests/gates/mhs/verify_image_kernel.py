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

  [f] THE Q-FAMILY'S QUADRATURE BUDGET LAW, and that it is obeyed. The order
      needed is not constant -- it goes as L / sqrt(delta^2 + eps^2), the same
      law the direct term obeys with the same constant -- and the rule derives
      it per observer. Gated three ways: the law is obeyed, a starved order
      still fails, and the order actually varies. See part_f.

  [e] THE THREE-WAY SPLIT, bitwise. ``total - eigenstress == stress`` as an
      identity rather than a tolerance: if it ever needed one, the three entry
      points would not be three views of one computation. Also that the
      eigenstress carries NO image contribution -- it is a pointwise statement
      about the blob and the source triangle, reading no Green's function.

  [g] THE FLOOR, where the law is CLIPPED -- which is 99.7% of a real matrix's
      quadrature points, so it sets the cost of an assembly almost by itself.
      The law under-predicts in the far field (it asks for 1 at delta = 16 L),
      so out there the floor is the whole of the accuracy. Gated the same three
      ways as [f]: the floor holds at every patch order, a lower floor fails,
      and the `+ order` term is needed. See part_g.

  [h] THE CONTRACTED FORMS, traction and interaction, as identities against
      stress_matrix -- plus the one clause those identities cannot see: the
      self-interaction diagonal must be NEGATIVE, because slip relieves the
      shear that drives it. A frame flipped in both the kernel and the
      reference passes every identity and fails that. See part_h.

  [i] P1 AND P2 THROUGH THE PUBLIC API, by the partition of unity: uniform
      nodal slip must reproduce the P0 answer exactly, at every entry point,
      summed over each element's K columns. Also the check that catches a
      PARTIAL WRITE into the wider DOF buffer, with a tripwire that non-uniform
      nodal slip really does differ. See part_i.

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
TOL_BUDGET = 1e-7         # measured 2e-15 .. 1.1e-8 with the adaptive order
TRIP_STARVED = 1e-2       # a flat n_quad = 16 reaches O(1) near the trace
#: The floor is held to the SAME bar as the law it takes over from: clause [f]
#: passes at 1.05e-08 against TOL_BUDGET, so gating the floor tighter would
#: demand more of the far field than of the near one. Measured worst 3.8e-08,
#: at 2 L with P2 -- the crossover band on the worst geometry, improving to
#: 1.4e-10 by 3 L and 7e-15 by 16 L.
TOL_FLOOR = 1e-7
TRIP_FLOOR_LOW = 1e-5     # a floor of 4 must FAIL, or the floor is padding


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
    part_h(rep)
    part_i(rep)
    return rep.finish()


def part_i(rep) -> None:
    """P1 AND P2 THROUGH THE PUBLIC API, by the partition of unity.

    At order ``p`` each element carries ``K = 1, 3, 6`` nodal slip vectors and
    the source axis holds slip DEGREES OF FREEDOM, element-major. The shape
    functions sum to 1, so uniform nodal values represent exactly the constant
    slip field P0 represents -- which makes

        sum over an element's K columns  ==  that element's P0 column

    an identity, at every entry point, with no reference and no tolerance
    chosen by taste. It is also the check that catches a PARTIAL WRITE: an
    assembler that fills only the first ``n_src`` columns of an ``n_dof``-wide
    buffer leaves zeros behind, and nothing else here would notice. That is not
    hypothetical -- ``eigenstress_matrix`` did exactly that when the order
    argument was threaded through its buffer but not its assembler, and this
    clause is how it surfaced (0.57 against 1e-16 now).

    The tripwire is that NON-uniform nodal slip must give a different answer,
    because an implementation that ignored the nodal values entirely -- using
    the P0 kernel K times -- would satisfy the identity above perfectly.
    """
    from mhs import (Material, disp_matrix, eigenstress_matrix,
                     elastic_strain_matrix, interaction_matrix,
                     stress_matrix, tdcs, total_stress_matrix,
                     traction_matrix)
    from mhs.fullspace.shape import n_nodes
    from mhs.matrices import COLLOCATION_SHRINK

    mat = Material(mu=MU, lam=LAM)
    tris = np.array([TRI_SURF, TRI_DEEP])
    eps, n = 0.1, 2
    rng = np.random.default_rng(8)
    obs = np.column_stack([rng.normal(size=7) * 1.5, rng.normal(size=7) * 1.5,
                           -np.abs(rng.normal(size=7)) * 1.5 - 0.3])
    nrm = np.repeat(tdcs.slip_frame(tris)[:1, 2, :], len(obs), axis=0)

    print("\n[i] P1/P2 THROUGH THE API: the partition of unity")
    cases = [("disp_matrix", lambda p: disp_matrix(obs, tris, mat, eps,
                                                   order=p)),
             ("stress_matrix", lambda p: stress_matrix(obs, tris, mat, eps,
                                                       order=p)),
             ("total_stress_matrix",
              lambda p: total_stress_matrix(obs, tris, mat, eps, order=p)),
             ("eigenstress_matrix",
              lambda p: eigenstress_matrix(obs, tris, mat, eps, order=p)),
             ("elastic_strain_matrix",
              lambda p: elastic_strain_matrix(obs, tris, mat, eps, order=p)),
             ("traction_matrix",
              lambda p: traction_matrix(obs, nrm, tris, mat, eps, order=p))]
    worst = 0.0
    for name, fn in cases:
        ref = fn(0)
        for p in (1, 2):
            K = n_nodes(p)
            got = fn(p)
            if got.shape[-2] != n * K:
                rep.check_bool(f"i {name} has the right DOF count at P{p}",
                               False, f"(got {got.shape}, expected a source "
                                      f"axis of {n * K})")
                return
            summed = got.reshape(got.shape[:-2] + (n, K, 3)).sum(axis=-2)
            worst = max(worst, relmax(summed, ref))
    rep.check("i uniform nodal slip at P1/P2 reproduces P0, every entry point",
              worst, 1e-11,
              "summed over each element's K columns -- an identity, and the "
              "one check that catches an assembler filling only the first "
              "n_src columns of an n_dof buffer")

    g1 = disp_matrix(obs, tris, mat, eps, order=1)
    uniform = np.zeros((n * 3, 3))
    uniform[:, 0] = 1.0
    spiked = np.zeros((n * 3, 3))
    spiked[0::3, 0] = 3.0                    # same total, concentrated
    du = np.einsum("oisk,sk->oi", g1, uniform)
    ds = np.einsum("oisk,sk->oi", g1, spiked)
    rep.check_bool("i NON-uniform nodal slip gives a different field",
                   relmax(ds, du) > 1e-3,
                   f"({relmax(ds, du):.2e} relative) -- an implementation that "
                   f"ignored the nodal values and used the P0 kernel K times "
                   f"would satisfy the clause above exactly")

    for p in (0, 1, 2):
        K = interaction_matrix(tris, mat, eps, order=p, receiver="strike",
                               source="strike")
        d = np.diag(K[:, :, 0, 0])
        ok = K.shape == (n * n_nodes(p), n * n_nodes(p), 1, 1)
        rep.check_bool(f"i interaction_matrix at P{p}: DOF-shaped, diagonal "
                       f"still negative",
                       ok and bool(np.all(d < 0.0)),
                       f"(shape {K.shape}, diagonal {d.min():+.3e} .. "
                       f"{d.max():+.3e}) -- collocated at nodes shrunk "
                       f"{100 * COLLOCATION_SHRINK:g}% toward the centroid, "
                       f"because a raw P1 node is a VERTEX and reading stress "
                       f"on the element's own edge is the hardest case the "
                       f"kernel has")


def part_h(rep) -> None:
    """THE CONTRACTED FORMS, as identities -- traction and interaction.

    ``traction_matrix`` and ``interaction_matrix`` exist because the object they
    would otherwise be contracted out of cannot be allocated: a 10k x 10k stress
    matrix is 20.1 GiB and the ceiling refuses it, where a single-component
    interaction is 0.75 GiB. They contract INSIDE the source loop, so what has
    to be proved is that moving the contraction there changed no arithmetic.

    Every clause is an identity against ``stress_matrix``, so none needs a
    tolerance chosen by taste -- and the last two are tripwires, because "agrees
    with the stress matrix" would also be satisfied by a form that was not
    actually smaller, or by a component selection that ignored its argument.
    """
    from mhs import (Material, chunking, interaction_matrix, stress_matrix,
                     tdcs, traction_matrix)

    mat = Material(mu=MU, lam=LAM)
    tris = np.array([TRI_SURF, TRI_DEEP,
                     TRI_DEEP + np.array([0.6, -0.3, -0.4]),
                     np.array([[0.2, 0.1, -1.0], [1.1, 0.0, -1.0],
                               [0.7, 0.8, -1.0]])])          # one HORIZONTAL
    eps = 0.1
    frame = tdcs.slip_frame(tris)

    print("\n[h] THE CONTRACTED FORMS: traction and interaction")
    gram = np.einsum("sab,scb->sac", frame, frame)
    rep.check("h the slip frame is orthonormal",
              float(np.abs(gram - np.eye(3)).max()), 1e-14,
              "rows are [strike, dip, tensile], so the transpose is the inverse")
    dets = np.linalg.det(frame)
    rep.check_bool("h the slip frame is right-handed and finite everywhere",
                   bool(np.all(np.isfinite(frame)))
                   and float(dets.min()) > 0.999,
                   f"(det {dets.min():+.4f} .. {dets.max():+.4f}) -- the set "
                   f"includes a HORIZONTAL element, whose (-n_y, n_x, 0) strike "
                   f"is the zero vector and whose frame is NaN unguarded")
    sdt = np.array([[0.3, -0.7, 0.2], [1.0, 0.1, -0.4],
                    [-0.2, 0.5, 0.9], [0.6, 0.6, -0.1]])
    rep.check("h the frame round-trips slip",
              float(np.abs(tdcs.from_cartesian(
                  tris, tdcs.to_cartesian(tris, sdt)) - sdt).max()), 1e-14,
              "(strike, dip, tensile) -> Cartesian -> back")

    obs = tris.mean(axis=1) + np.array([0.25, 0.15, -0.35])
    nrm = frame[:, 2, :]
    worst = 0.0
    for sub in (True, False):
        S = stress_matrix(obs, tris, mat, eps, subtract_eigenstress=sub)
        ref = np.einsum("omnsk,on->omsk", S, nrm)
        worst = max(worst, relmax(
            traction_matrix(obs, nrm, tris, mat, eps,
                            subtract_eigenstress=sub), ref))
    rep.check("h traction_matrix == stress_matrix contracted with the normal",
              worst, 1e-13,
              "both eigenstress settings; contracting in the source loop must "
              "not change the arithmetic")

    cen = tris.mean(axis=1)
    t_ref = np.einsum("omnsk,on->omsk", stress_matrix(cen, tris, mat, eps), nrm)
    k_ref = np.einsum("oai,oisk,sbk->osab", frame, t_ref, frame)
    rep.check("h interaction_matrix == stress resolved in both frames",
              relmax(interaction_matrix(tris, mat, eps), k_ref), 1e-13,
              "receivers at the element centroids, where the mollified kernel "
              "is finite once the eigenstress is removed")

    sel = interaction_matrix(tris, mat, eps, receiver="strike",
                             source="strike")
    nsel = interaction_matrix(tris, mat, eps, receiver=("normal",),
                              source=("tensile",))
    rep.check_bool("h component selection picks the named slices",
                   sel.shape == (4, 4, 1, 1)
                   and relmax(sel[:, :, 0, 0], k_ref[:, :, 0, 0]) < 1e-13
                   and relmax(nsel[:, :, 0, 0], k_ref[:, :, 2, 2]) < 1e-13,
                   "strike/strike and normal/tensile against the full 3x3 -- a "
                   "selection that ignored its argument would return the wrong "
                   "slice, and one returning the full tensor would fail on "
                   "shape")

    # [h] THE SIGN, which none of the clauses above can see. They all compare
    # against stress_matrix contracted with the SAME frame, so a frame or
    # normal flipped in both places passes every one of them -- and would
    # silently invert every cycle model built on this. Slip must RELIEVE the
    # shear that drives it, so the self-interaction diagonal is negative. This
    # is a physics statement with no reference and no tolerance.
    dipping = np.array([
        [[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 0.6, -1.8]],
        [[1.0, 0.0, -1.0], [1.0, 0.6, -1.8], [0.0, 0.6, -1.8]],
        [[1.0, 0.0, -1.0], [2.0, 0.0, -1.0], [1.0, 0.6, -1.8]],
    ])
    for comp in ("strike", "dip"):
        diag = np.diag(interaction_matrix(dipping, mat, 0.1, receiver=comp,
                                          source=comp)[:, :, 0, 0])
        rep.check_bool(f"h self-{comp} interaction is NEGATIVE "
                       f"(slip relieves its own shear)",
                       bool(np.all(diag < 0.0)),
                       f"({diag.min():+.3e} .. {diag.max():+.3e} on a dipping "
                       f"patch) -- the one clause here that a frame flipped in "
                       f"BOTH the kernel and the reference cannot satisfy")

    big = chunking.pair_bytes("stress")
    rep.check_bool("h the contracted forms really are smaller per pair",
                   chunking.pair_bytes("traction") * 3 == big
                   and chunking.pair_bytes((1, 1)) * 27 == big,
                   f"(stress {big} B, traction "
                   f"{chunking.pair_bytes('traction')} B, 1x1 interaction "
                   f"{chunking.pair_bytes((1, 1))} B per obs/source pair) -- so "
                   f"20.1 GiB at 10k x 10k becomes 6.7 or 0.75, which is the "
                   f"entire reason these entry points exist")


def part_f(rep, image) -> None:
    """THE QUADRATURE BUDGET LAW for the Q-family, and that it is obeyed.

    The Q-family is the half left in quadrature. Unlike the R-family it does not
    stagnate -- it converges -- but its required order is NOT constant, which an
    earlier version of this gate asserted on the strength of a buried-element
    measurement. It obeys

        n_quad ~ C * L / sqrt(delta^2 + eps^2),      delta = dist to the IMAGE

    and that is the SAME law the direct term obeys (``n_quad ~ 8 L / eps``,
    clause [b] of oracle/verify_vertical_fault) with the same constant: the
    direct term's observer sits ON its own element, so delta = 0 and the scale
    is eps. One law, two cases -- which is why C did not have to be fitted
    separately. Measured 5.8 to 8.4 for 1e-9 relative, over delta/h in
    {0.02 .. 0.33} and eps/h in {0.01 .. 0.1}.

    Three clauses, and the middle one is the point:

      - the law is OBEYED: with the adaptive order the full image kernel is at
        machine precision at every on-fault point tested, including 0.03 h below
        a surface trace at eps/h = 0.003, where a flat 16 gave O(1).

      - a STARVED order still fails. Without this the clause above could pass
        because the configuration is easy rather than because the law works.

      - the order actually VARIES with geometry. A law that returned a constant
        would pass the first two clauses and be no law at all.

    What the law replaced, for the record, since the numbers are the reason it
    exists -- error of the full image stress ON the fault plane, flat n_quad=16
    against the adaptive rule:

        eps/h    P0 (h/3)        P1/P2 (h/6)      depth 0.03 h
        0.100    4.4e-08 -> 5.2e-15    2.5e-04 -> 4.3e-15   9.7e-03 -> 5.0e-15
        0.010    7.5e-08 -> 1.4e-15    6.9e-04 -> 3.4e-15   1.0e+00 -> 3.9e-09
        0.003    7.5e-08 -> 2.6e-15    6.9e-04 -> 2.2e-15   1.1e+00 -> 1.1e-08

    and it costs 4.1x the flat 16 while being 4.2x CHEAPER than a flat 64 at
    the same accuracy.
    """
    from mhs.fullspace.frame import local_frame
    TV = np.array([[0.0, 0.0, 0.0], [0.08, 1.0, 0.0], [0.03, 0.5, -1.0]])
    nrm = np.cross(TV[1] - TV[0], TV[2] - TV[0])
    nrm = nrm / np.linalg.norm(nrm)

    def on_plane(depth):
        q = np.array([0.0, 0.5, -depth])
        q = q - nrm * float((q - TV[0]) @ nrm)
        return q.reshape(1, 3)

    print("\n[f] THE Q-FAMILY'S QUADRATURE BUDGET LAW")
    print("    n_quad ~ C L / sqrt(delta^2 + eps^2), delta = dist to the IMAGE")
    print(f"    {'eps/h':>7} {'point':>14} {'n_quad':>8} {'adaptive err':>14} "
          f"{'flat 16 err':>13}")
    worst_ad = 0.0
    worst_flat = 0.0
    tri_img = TV * np.array([1.0, 1.0, -1.0])
    L = local_frame(tri_img).L
    for eh in (0.1, 0.003):
        for label, d in (("P0 h/3", 1.0 / 3.0), ("P1/P2 h/6", 1.0 / 6.0),
                         ("0.03 h", 0.03)):
            o = on_plane(d)
            nq = int(image.q_gauss_orders(o, tri_img, eh, L)[0])
            # Reference at FOUR TIMES the order the law asks for, not a flat
            # 256: a flat high order costs 65k points x 1134 records per case
            # and makes the gate unrunnable, while 4x the law is resolved
            # wherever the law itself is right -- which the starved clause
            # below independently confirms it is.
            ref = image.image_total(o, TV, 0, MU, LAM, eh, want=("H",),
                                    n_quad=min(4 * nq, 256))["H"][0, 0]
            ad = relmax(image.image_total(o, TV, 0, MU, LAM, eh,
                                          want=("H",))["H"][0, 0], ref)
            fl = relmax(image.image_total(o, TV, 0, MU, LAM, eh, want=("H",),
                                          n_quad=16)["H"][0, 0], ref)
            worst_ad = max(worst_ad, ad)
            worst_flat = max(worst_flat, fl)
            print(f"    {eh:7.3f} {label:>14} {nq:8d} {ad:14.2e} {fl:13.2e}")
    rep.check("f the law is OBEYED: adaptive order is accurate everywhere",
              worst_ad, TOL_BUDGET,
              "including 0.03 h below a surface trace at eps/h = 0.003, where "
              "a flat 16 gives O(1)")
    rep.check_bool(f"f a STARVED order still FAILS (> {TRIP_STARVED:g})",
                   worst_flat > TRIP_STARVED,
                   f"({worst_flat:.2e} at flat n_quad = 16) -- so the clause "
                   f"above passes because the law works, not because the "
                   f"configuration is easy")
    orders = image.q_gauss_orders(
        np.vstack([on_plane(d) for d in (1.0 / 3.0, 1.0 / 6.0, 0.01)]),
        tri_img, 0.01, L)
    rep.check_bool("f the order VARIES with geometry (it is a law, not a "
                   "constant)", len(set(orders.tolist())) == len(orders),
                   f"({orders.tolist()} for depths h/3, h/6, 0.01 h) -- a rule "
                   f"returning a constant would pass the two clauses above and "
                   f"be no law at all")
    part_g(rep, image)


def part_g(rep, image) -> None:
    """THE FLOOR, and the patch-order term -- the FAR field, not the near one.

    Clause [f] gates the law where the law governs. This one gates where it is
    CLIPPED, which is the overwhelming majority of a real matrix: on a 400
    element surface-breaking fault at eps/h = 0.1, 99.9% of the 160,000 pairs
    take the floor and they hold 99.7% of all quadrature points. So the floor
    sets the cost of an assembly almost by itself, and it had never been
    measured -- it was a round number with a one-line comment.

    Measured now, and it is NOT padding: the law is calibrated on the near-field
    scale and UNDER-predicts far away, asking for 1 at delta = 16 L where P0
    needs 4 for 1e-9. Beyond about 2 L the floor is the whole of the accuracy.

    Three clauses, matching [f]'s shape:

      - the floor HOLDS across the clipped region, at every patch order. The
        far-field requirement is eps-independent (it is set by delta/L), so
        this sweeps distance and order rather than eps.

      - a LOWER floor fails. Without this the clause above could pass because
        the far field is easy rather than because the floor is right, which is
        exactly how a flat 16 survived into a shipped default once already.

      - the ``+ order`` term is NEEDED. P1 and P2 add polynomial degree to the
        integrand, which the distance law cannot see; dropping the term leaves
        P2 an order short near the crossover.
    """
    tri = TRI_SURF
    tri_img = tri * np.array([1.0, 1.0, -1.0])
    L = float(np.max(np.linalg.norm(tri - tri.mean(0), axis=1)))
    cen = tri_img.mean(0)
    rng = np.random.default_rng(2)
    dirs = rng.normal(size=(24, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    def observers(dist):
        pts = cen[None, :] + dist * dirs
        pts[:, 2] = -np.abs(pts[:, 2]) - 1e-9       # the body is z <= 0
        return pts

    print("    dist/L  P  n_used   floor-as-shipped   floor 4   no +order")
    worst, worst_low, worst_noterm = 0.0, 0.0, 0.0
    for mult in (2.0, 3.0, 5.0, 8.0, 16.0):
        obs = observers(mult * L)
        for order in (0, 1, 2):
            # Converged reference: a far pair is smooth, so Gauss is geometric
            # here -- but checked rather than assumed, because comparing an
            # unconverged rule against itself is the trap clause [b] of
            # oracle/verify_vertical_fault records.
            ref = image.image_q_influence(obs, tri, order, MU, LAM, EPS_FINE,
                                          want=("H",), n_quad=44)["H"]
            chk = image.image_q_influence(obs, tri, order, MU, LAM, EPS_FINE,
                                          want=("H",), n_quad=52)["H"]
            if relmax(chk, ref) > 1e-13:
                rep.check_bool("g the far-field reference is converged",
                               False, f"(order 44 vs 52 differ by "
                                      f"{relmax(chk, ref):.1e} at "
                                      f"{mult:g} L, P{order})")
                return
            used = int(image.q_gauss_orders(obs, tri_img, EPS_FINE, L,
                                            order).min())

            def at(n):
                return relmax(image.image_q_influence(
                    obs, tri, order, MU, LAM, EPS_FINE, want=("H",),
                    n_quad=n)["H"], ref)

            e_used, e_low = at(used), at(4)
            e_noterm = at(max(used - order, 1))
            worst = max(worst, e_used)
            worst_low = max(worst_low, e_low)
            worst_noterm = max(worst_noterm, e_noterm)
            print(f"    {mult:6.1f} {order:2d} {used:7d} {e_used:18.2e} "
                  f"{e_low:9.2e} {e_noterm:11.2e}")

    rep.check("g the FLOOR holds across the clipped far field, every order",
              worst, TOL_FLOOR,
              "the law asks for 1 to 7 out here, so this is the floor's "
              "accuracy and not the law's")
    rep.check_bool(f"g a LOWER floor (4) still FAILS (> {TRIP_FLOOR_LOW:g})",
                   worst_low > TRIP_FLOOR_LOW,
                   f"({worst_low:.2e}) -- so the clause above passes because "
                   f"the floor is sized right, not because the far field is "
                   f"easy. Lowering the floor is worth 2.3x on 99.7% of a "
                   f"matrix's quadrature points, so this is the clause any "
                   f"such change has to move.")
    rep.check_bool("g WITHOUT the `+ order` term the clause above FAILS",
                   worst_noterm > TOL_FLOOR,
                   f"({worst_noterm:.2e} without it against {worst:.2e} with "
                   f"it, tol {TOL_FLOOR:g}) -- so the term is load-bearing "
                   f"rather than decorative: P1/P2 add polynomial degree that "
                   f"a distance law cannot see")


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
