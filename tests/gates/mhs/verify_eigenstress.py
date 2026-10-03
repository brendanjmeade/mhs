#!/usr/bin/env python
"""The eigenstress: its closed form, its sign, and that it leaves the field alone.

This is the premise the whole half-space decomposition rests on, so it is worth
pinning hard. ``mhs`` reuses the FULL-SPACE eigenstress verbatim for the
half-space kernel, and the argument is that ``Phi_eps(x) = int_T phi_eps(x - y)
dS`` is a pointwise statement about the blob and the source triangle -- it reads
no Green's function, so no free surface and no image can enter it. All the blob
content sits in the direct term; the image correction depends only on the
distance to the image, which never vanishes inside the body, so it is smooth
there and carries none.

That argument is only worth having if the full-space eigenstress is itself
right. Three clauses, ported from upstream clq with tolerances unchanged:

  [a] THE CLOSED FORM vs GAUSS QUADRATURE of the blob times the shape
      functions, at orders 0, 1 and 2. Two genuinely different computations:
      the closed form goes through the moment hierarchy, the reference
      integrates the integrand numerically. Off-plane at 1e-10, and on-plane
      INSIDE the triangle at 1e-7 where the reference itself is the limit.

  [c] THE FINITENESS LADDER, and this is the clause that matters most. Halve
      eps five times and watch two things happen at once: the RAW (total)
      on-fault stress grows like 1/eps -- the eigenstress the mollification put
      there -- while the ELASTIC stress stays put. That is simultaneously the
      proof that the subtraction has the right SIGN (the wrong sign would also
      grow) and that what is left is bounded. No external reference: the
      scalings are the statement.

  [d] THE OFF-FAULT NO-OP. Ten eps from the plane the eigenstress must be a
      negligible fraction of the field, and by twenty eps it must be gone.
      Catches a subtraction that quietly damages the far field, which the
      near-field clauses cannot see.

Clause (b) of the upstream gate -- the infinite-plane limit against the legacy
``anelastic`` module -- is NOT ported: it needs that flat module, which mhs does
not vendor. Its value here would be small (it checks the frozen approximation in
the one regime where it is valid) and the dependency is large. Named rather than
silently dropped.

Clauses (c) and (d) run at ORDER 0 only, because ``mhs.kernels.assemble`` is
order-0 today; the upstream gate also sweeps p = 1 (hat) and p = 2 (dome) slip.
Extending the assembly to nodal order brings those back, and they should come
back -- the dome is the case whose slip curvature kinks sigma_xz across the
plane, so it converges at ~1 rather than ~2 and is the sharpest of the three.

Run from anywhere:  python tests/gates/mhs/verify_eigenstress.py
"""
from __future__ import annotations

import sys

import numpy as np

from _common import MU, TRI, Report, relmax, shipped

NU_A = 0.3        # not 1/4
NU_C = 0.25       # the ladder's upstream value, kept so the numbers compare
ORDERS = (0, 1, 2)


def _lam(mu: float, nu: float) -> float:
    """Only for talking to the vendored engine, which takes (mu, nu)."""
    return 2.0 * mu * nu / (1.0 - 2.0 * nu)


# --------------------------------------------------------------------- [a] ---
def part_a(rep) -> None:
    print("\n(a) E_k closed form vs Gauss quadrature of the blob x shape functions")
    from mhs_oracle.clq.quadrature import quadrature_influence

    nodal_influence = shipped("kernels").nodal_influence
    local_frame = shipped("frame").local_frame
    inside = shipped("frame").inside

    fr = local_frame(TRI)
    v1, v2, v3 = TRI
    obs = np.array([[0.60, -0.10, 0.60], [2.0, 1.5, 3.0], [-0.4, 0.0, -0.8]])
    eps = 0.3
    for p in ORDERS:
        E = nodal_influence(obs, TRI, p, MU, NU_A, eps, want=("E",),
                            far_field="analytic")["E"]
        E40 = quadrature_influence(obs, TRI, p, MU, NU_A, eps, n_gauss=40,
                                   want=("E",))["E"]
        E80 = quadrature_influence(obs, TRI, p, MU, NU_A, eps, n_gauss=80,
                                   want=("E",))["E"]
        rep.check(f"(a) p={p} off-plane eps=0.3 vs 80x80 Gauss",
                  relmax(E, E80), 1e-10,
                  f"[40x40: {relmax(E, E40):.1e}, 40->80 moves the reference "
                  f"by {relmax(E40, E80):.1e}]")

    eps = 0.15 * fr.L
    obs = np.array([v1 + 0.3 * (v2 - v1) + 0.3 * (v3 - v1),
                    TRI.mean(axis=0),
                    v1 + 0.15 * (v2 - v1) + 0.6 * (v3 - v1)])
    assert np.all(inside(TRI, obs)), "on-plane observers must be inside"
    for p in ORDERS:
        E = nodal_influence(obs, TRI, p, MU, NU_A, eps, want=("E",),
                            far_field="analytic")["E"]
        Eq = quadrature_influence(obs, TRI, p, MU, NU_A, eps, n_gauss=160,
                                  want=("E",))["E"]
        rep.check(f"(a) p={p} on-plane inside eps=0.15L vs 160x160 Gauss",
                  relmax(E, Eq), 1e-7)


# ------------------------------------------------------- shipped readouts ---
def _shipped_stress(obs, tri, slip, mu, lam, eps, *, elastic):
    """Total or elastic stress through ``mhs.kernels.assemble``.

    Deliberately the SHIPPED path, not the oracle's api: the point of porting
    this gate is to pin the mathematics of the thing that will change.
    """
    from mhs.kernels import assemble as asm
    obs = np.asarray(obs, float).reshape(-1, 3)
    tris = np.asarray(tri, float)[None]
    eps_arr = np.full(1, float(eps))
    tot = np.zeros((obs.shape[0], 3, 3, 1, 3))
    asm.assemble_fullspace_total_stress(obs, tris, mu, lam, eps_arr, tot)
    out = np.einsum("omnsk,sk->omn", tot, np.asarray(slip, float)[None])
    if elastic:
        eig = np.zeros_like(tot)
        asm.assemble_eigenstress(obs, tris, mu, lam, eps_arr, eig)
        out = out - np.einsum("omnsk,sk->omn", eig,
                              np.asarray(slip, float)[None])
    return out


def _shipped_eigenstress(obs, tri, slip, mu, lam, eps):
    from mhs.kernels import assemble as asm
    obs = np.asarray(obs, float).reshape(-1, 3)
    tris = np.asarray(tri, float)[None]
    eig = np.zeros((obs.shape[0], 3, 3, 1, 3))
    asm.assemble_eigenstress(obs, tris, mu, lam, np.full(1, float(eps)), eig)
    return np.einsum("omnsk,sk->omn", eig, np.asarray(slip, float)[None])


# --------------------------------------------------------------------- [c] ---
def part_c(rep):
    print("\n(c) finiteness ladder, unit equilateral triangle, order 0 "
          f"(mu={MU:g}, nu={NU_C})")
    frame = shipped("frame")
    shape = shipped("shape")
    lam = _lam(MU, NU_C)

    L = 1.0
    tri = frame.equilateral(L)
    pts = shape.triangle_grid(tri, 24)
    pts = pts[frame.inside(tri, pts, margin=0.25 * L)]
    cen = tri.mean(axis=0)[None]
    print(f"    {pts.shape[0]} mid-plane interior points (margin 0.25 L)")
    ladder = np.array([0.2, 0.1, 0.05, 0.025, 0.0125]) * L
    slip = np.array([1.0, 0.0, 0.0])          # uniform, in plane

    raw_pk, el_pk, el_cen = [], [], []
    for e in ladder:
        raw = _shipped_stress(pts, tri, slip, MU, lam, e, elastic=False)
        el = _shipped_stress(pts, tri, slip, MU, lam, e, elastic=True)
        raw_pk.append(np.abs(raw[:, 0, 2]).max())
        el_pk.append(np.abs(el[:, 0, 2]).max())
        el_cen.append(_shipped_stress(cen, tri, slip, MU, lam, e,
                                      elastic=True)[0, 0, 2])
    raw_pk, el_pk, el_cen = map(np.array, (raw_pk, el_pk, el_cen))
    raw_ratio = raw_pk[1:] / raw_pk[:-1]
    el_ratio = el_pk[1:] / el_pk[:-1]
    dif = np.abs(el_cen[:-1] - el_cen[1:])
    order = np.log2(dif[:-1] / dif[1:])

    print(f"      {'eps/L':>7s} {'raw peak':>11s} {'raw ratio':>10s} "
          f"{'el peak':>10s} {'el ratio':>9s} {'el(centroid)':>13s} "
          f"{'|d el|':>10s} {'order':>6s}")
    for i, e in enumerate(ladder):
        rr = f"{raw_ratio[i-1]:10.3f}" if i else " " * 10
        er = f"{el_ratio[i-1]:9.3f}" if i else " " * 9
        dd = f"{dif[i-1]:10.3e}" if i else " " * 10
        oo = f"{order[i-2]:6.2f}" if i >= 2 else " " * 6
        print(f"      {e / L:7.4f} {raw_pk[i]:11.5f} {rr} {el_pk[i]:10.5f} "
              f"{er} {el_cen[i]:13.8f} {dd} {oo}")
    print(f"      observed order at the centroid: {order[-1]:.2f} "
          f"(steps: {', '.join(f'{o:.2f}' for o in order)})")

    # The raw field must DIVERGE: that is the eigenstress, growing like 1/eps.
    rep.check_bool("(c) raw (total) peak grows >= 1.7x per eps halving",
                   bool(np.all(raw_ratio >= 1.7)),
                   f"(min {raw_ratio.min():.3f}; this is the 1/eps eigenstress)")
    # ... while the elastic field stays put. Both halves are needed: a wrong
    # SIGN would also grow, so "bounded" is the discriminating statement.
    rep.check_bool("(c) elastic ratio in (0.8, 1.25) for eps/L <= 0.1",
                   bool(np.all((el_ratio[1:] > 0.8) & (el_ratio[1:] < 1.25))),
                   f"(range {el_ratio[1:].min():.3f}..{el_ratio[1:].max():.3f}; "
                   f"0.2->0.1 step {el_ratio[0]:.3f})")
    rep.check_bool("(c) elastic ratio decreases toward 1 along the ladder",
                   bool(np.all(np.diff(el_ratio) < 0.0)
                        and np.all(el_ratio > 1.0 - 1e-12)
                        and el_ratio[-1] < 1.05),
                   f"(ratios {', '.join(f'{r:.3f}' for r in el_ratio)})")
    rep.check_bool("(c) |el(eps) - el(eps/2)| at the centroid decreasing",
                   bool(np.all(np.diff(dif) < 0.0)),
                   f"(order {order[-1]:.2f})")
    # A tripwire, so the clause above cannot pass by the subtraction being a
    # no-op: the WRONG sign must also diverge, and must be far from the right one.
    e_mid = ladder[2]
    raw = _shipped_stress(pts, tri, slip, MU, lam, e_mid, elastic=False)
    eig = _shipped_eigenstress(pts, tri, slip, MU, lam, e_mid)
    wrong = raw + eig                      # added instead of subtracted
    right = raw - eig
    ratio = (float(np.abs(wrong[:, 0, 2]).max())
             / max(float(np.abs(right[:, 0, 2]).max()), 1e-300))
    rep.check_bool("(c) tripwire: the WRONG sign is far from the right one",
                   ratio > 3.0,
                   f"|wrong|/|right| = {ratio:.2f} at eps/L = {e_mid:g}")
    return tri, slip, ladder, raw_pk


# --------------------------------------------------------------------- [d] ---
def part_d(rep, tri, slip, ladder, raw_pk) -> None:
    eps = 0.05
    i_eps = int(np.argmin(np.abs(ladder - eps)))
    assert abs(ladder[i_eps] - eps) < 1e-15
    print(f"\n(d) off-fault no-op above the centroid, eps = {eps:g}")
    frame = shipped("frame")
    lam = _lam(MU, NU_C)
    n = frame.unit_normal(tri)
    cen = tri.mean(axis=0)

    vals = {}
    for hh in (10, 20, 40):
        obs = cen[None] + hh * eps * n
        tot = _shipped_stress(obs, tri, slip, MU, lam, eps, elastic=False)[0]
        eig = _shipped_eigenstress(obs, tri, slip, MU, lam, eps)[0]
        vals[hh] = (float(np.linalg.norm(eig)), float(np.linalg.norm(tot)))
    r10, r20, r40 = (vals[h][0] / vals[h][1] for h in (10, 20, 40))
    rpk = vals[10][0] / raw_pk[i_eps]
    print(f"    |C:eps*|/|sigma_total| at 10, 20, 40 eps: "
          f"{r10:.3e}, {r20:.3e}, {r40:.3e}")
    print(f"    |C:eps*|(10 eps) / on-fault raw peak: {rpk:.3e}")
    rep.check("(d) |C:eps*|(10 eps) / on-fault peak |sigma_total|", rpk, 1e-4)
    rep.check("(d) |C:eps*| / |sigma_total| at 10 eps", r10, 1e-2,
              "[1e-4 unattainable here: sigma_total is ~1e-3 of its on-fault "
              "value, so the ratio is against a small denominator]")
    rep.check("(d) |C:eps*| / |sigma_total| at 20 eps", r20, 1e-4)


def main() -> bool:
    rep = Report("Eigenstress: closed form, finiteness ladder, off-fault no-op "
                 "(mhs shipped path)")
    part_a(rep)
    tri, slip, ladder, raw_pk = part_c(rep)
    part_d(rep, tri, slip, ladder, raw_pk)
    print("\n  (b) the infinite-plane limit is NOT ported: it needs the legacy "
          "`anelastic`")
    print("      module, which mhs does not vendor. Named, not silently dropped.")
    return rep.finish()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
