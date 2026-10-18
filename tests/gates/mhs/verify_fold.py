#!/usr/bin/env python
"""The folded image of a source that reaches the free surface.

A source whose smeared slip reaches above ``z = 0`` cannot use the lifted
Mindlin image (right to O(eps^2) only when buried); its image in the band
below the surface is its mirror element plus the Boussinesq field of the
normal traction the symmetric pair leaves on ``z = 0``
(:mod:`mhs.kernels.fold`). Every check is oracle-free: an identity, a physics
condition, or the lift itself where the lift is known to be right.

  [a] THE POINT LOAD. The Boussinesq stress is ``C:grad u`` of the Boussinesq
      displacement by finite differences -- an independent statement of the
      same solution -- it is divergence-free, traction-free on the surface
      off the load, carries unit force through every depth, and integrated
      over a disk it is Love's closed form (stress and displacement), which
      is what restores the subtracted singular part.

  [b] THE PAIR LEAVES ONLY A NORMAL TRACTION on ``z = 0``, equal to twice the
      element's own: the symmetry the whole construction rests on.

  [c] THE SEAM. For a BURIED element the lift is right, so the folded image
      must reproduce it, and better the deeper: measured 1.9e-2 of the image
      at 12 eps and 1.6e-3 at 40 eps on a dipping element, in stress AND
      displacement. The mirror alone misses by O(1) -- the Boussinesq term is
      the one that matters for dip slip, and this is the tripwire that found
      a reversed offset in the point kernel (shears flipped, nothing else).

  [d] THE FREE-SURFACE CONDITION for a STRADDLING source, which the lift
      fails by O(1) within eps of the trace and the fold satisfies to the
      quadrature.

  [e] ONE FOLD IN EVERY ENTRY POINT. Traction is stress contracted with the
      normal, total minus eigenstress is stress bitwise, the interaction
      matrix is the frame-resolved traction, uniform nodal P2 slip at a trace
      node is uniform P0 slip, and ``fold=False`` is the lift: identical below
      the band, different above it.

Run from anywhere:  python tests/gates/mhs/verify_fold.py
"""
from __future__ import annotations

import sys

import numpy as np

from _common import Report, relmax

MU, LAM = 30.0, 20.0                 # nu = 0.2, not 1/4
EPS = 0.05
#: Dips ~33 deg toward +y with its top edge on the surface, off every axis.
TRI = np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.4, 0.9, -0.6]])
LATTICE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [0, .5, .5],
                    [.5, 0, .5], [1 / 3, 1 / 3, 1 / 3]])

TOL_KERNEL = 1e-7          # stress vs FD of displacement; measured 8e-10
TOL_SURFACE_KERNEL = 1e-13
TOL_FORCE = 2e-3           # the plane integral, a trapezoid in r
TOL_DISK = 5e-4            # Love's closed forms vs the integrated point kernel
TOL_PAIR = 1e-12
TOL_SEAM = (4e-2, 5e-3)    # at 12 and 40 eps; measured 1.9e-2, 1.6e-3
TRIP_MIRROR_ONLY = 0.5     # the mirror alone must miss by at least this
TOL_FREE_SURFACE = 1e-2    # fold; measured below
TRIP_LIFT_SURFACE = 0.2    # the lift must fail it by at least this
TOL_IDENTITY = 1e-12
TRIP_FOLD_MOVES = 0.1      # fold vs lift at a surface receiver


def _seam_receivers(tri):
    on = LATTICE @ tri
    above = on[:4].copy()
    above[:, 2] = 0.0
    body = on[[0, 1, 2, 6]] + np.array([0.0, 0.0, -0.2])
    body = np.vstack([body, on[6] + np.array([0.5, 0.0, 0.0])])
    return np.vstack([on, above, body])


def main() -> bool:
    import mhs
    from mhs import defaults as d
    from mhs.kernels import assemble as asm
    from mhs.kernels import fold
    from mhs.kernels.image import image_total

    rep = Report("the folded image of a surface-reaching source")
    mat = mhs.Material(mu=MU, lam=LAM)
    nu = 0.5 * LAM / (LAM + MU)

    # ----------------------------------------------------------------- [a] ---
    print("\n[a] THE POINT LOAD: stress vs FD of displacement, equilibrium, "
          "surface, force, Love's disk")
    rng = np.random.default_rng(1)
    P = rng.normal(size=(200, 3))
    P[:, 2] = -np.abs(P[:, 2]) - 0.05
    h = 1e-5
    G = np.zeros((200, 3, 3))
    div = np.zeros((200, 3))
    for i in range(3):
        dv = np.zeros(3)
        dv[i] = h
        up = fold.point_disp(P[:, 0] + dv[0], P[:, 1] + dv[1],
                             -(P[:, 2] + dv[2]), MU, nu)
        dn = fold.point_disp(P[:, 0] - dv[0], P[:, 1] - dv[1],
                             -(P[:, 2] - dv[2]), MU, nu)
        G[:, i, :] = (up - dn) / (2 * h)
        Sp = fold.point_stress(P[:, 0] + dv[0], P[:, 1] + dv[1],
                               -(P[:, 2] + dv[2]), nu)
        Sm = fold.point_stress(P[:, 0] - dv[0], P[:, 1] - dv[1],
                               -(P[:, 2] - dv[2]), nu)
        div += (Sp[:, i, :] - Sm[:, i, :]) / (2 * h)
    E = 0.5 * (G + np.swapaxes(G, 1, 2))
    S_fd = (LAM * np.trace(E, axis1=1, axis2=2)[:, None, None] * np.eye(3)
            + 2 * MU * E)
    S_cf = fold.point_stress(P[:, 0], P[:, 1], -P[:, 2], nu)
    rep.check("a stress == C:grad u of the Boussinesq displacement",
              relmax(S_fd, S_cf), TOL_KERNEL)
    rep.check("a div sigma == 0", float(np.abs(div).max() / np.abs(S_cf).max()),
              TOL_KERNEL)
    Ss = fold.point_stress(rng.normal(size=50), rng.normal(size=50),
                           np.zeros(50), nu)
    rep.check("a traction-free on z = 0 off the load",
              float(np.abs(Ss[:, :, 2]).max()), TOL_SURFACE_KERNEL)
    r = np.geomspace(1e-4, 1e4, 4001)
    th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    R, T = np.meshgrid(r, th, indexing="ij")
    Sz = fold.point_stress(R * np.cos(T), R * np.sin(T), np.ones_like(R), nu)
    Fz = float((Sz[..., 2, 2] * (np.gradient(r) * r)[:, None]).sum()
               * (2 * np.pi / th.size))
    rep.check("a unit force through the plane at depth 1", abs(Fz + 1.0),
              TOL_FORCE, f"({Fz:.6f})")
    worst_s = worst_u = 0.0
    for a_, z_ in ((1.0, 0.3), (2.0, 1.0), (0.5, 2.0)):
        rr = np.linspace(0, a_, 20001)[1:]
        wr = rr * (a_ / 20000)
        X = np.outer(rr, np.cos(th))
        Y = np.outer(rr, np.sin(th))
        Z = np.full(X.shape, z_)
        num_s = (fold.point_stress(X, Y, Z, nu)
                 * wr[:, None, None, None]).sum(axis=(0, 1)) * (2 * np.pi / th.size)
        num_u = (fold.point_disp(X, Y, Z, MU, nu)
                 * wr[:, None, None]).sum(axis=(0, 1)) * (2 * np.pi / th.size)
        worst_s = max(worst_s, float(np.abs(num_s - fold.disk_stress(a_, z_, nu)).max()))
        worst_u = max(worst_u, relmax(num_u, fold.disk_disp(a_, z_, MU, nu)))
    rep.check("a Love's disk stress == the integrated point kernel", worst_s,
              TOL_DISK)
    rep.check("a Love's disk displacement == the integrated point kernel",
              worst_u, TOL_DISK)

    # ----------------------------------------------------------------- [b] ---
    print("\n[b] THE PAIR'S SURFACE TRACTION IS NORMAL, twice the element's own")
    pts = TRI[:2, :2].mean(axis=0) + rng.normal(size=(40, 2)) * 0.3
    obs = np.column_stack([pts, np.zeros(40)])
    nz = np.tile([0.0, 0.0, 1.0], (40, 1))
    e1 = np.full(1, EPS)

    def traction(tri_):
        buf = np.zeros((40, 3, 1, 3))
        asm.assemble_halfspace_traction(obs, nz, tri_[None], MU, LAM, e1, buf,
                                        True, 0, None, fold=False)
        return buf[:, :, 0, :]
    # the pair, each with the lift switched off, is direct + mirror direct only
    # when its image is removed: build it from the full-space blocks instead
    dblk = asm.fullspace_blocks(obs, TRI, MU, LAM, EPS, want=("H", "E"))
    mblk = asm.fullspace_blocks(obs, fold.mirror(TRI), MU, LAM, EPS,
                                want=("H", "E"))
    t_d = (np.einsum("okijm,oj->okim", dblk["H"], nz)
           - asm._eigenstress_traction(
               asm._eigen_terms(obs, TRI, MU, LAM, EPS, 0, dblk["E"], False), nz))
    t_m = (np.einsum("okijm,oj->okim", mblk["H"], nz)
           - asm._eigenstress_traction(
               asm._eigen_terms(obs, fold.mirror(TRI), MU, LAM, EPS, 0,
                                mblk["E"], False), nz)) * fold.MIRROR_SLIP_SIGN
    pair = (t_d + t_m)[:, 0]                                 # (40, 3, slip)
    rep.check("b shear traction of the pair on z = 0 vanishes",
              float(np.abs(pair[:, :2, :]).max() / np.abs(pair[:, 2, :]).max()),
              TOL_PAIR)
    rep.check("b its normal traction is twice the element's own",
              relmax(fold.surface_load(pts, TRI, MU, LAM, EPS, 0)[:, 0, :],
                     pair[:, 2, :]), TOL_PAIR)
    del traction

    # ----------------------------------------------------------------- [c] ---
    print("\n[c] THE SEAM: a buried element's folded image is its lifted image")
    print(f"    {'depth of top edge':20s} {'stress':>10} {'disp':>10} "
          f"{'mirror only':>12}")
    for k, nd in enumerate((12.0, 40.0)):
        tri = TRI.copy()
        tri[:, 2] -= nd * EPS
        obs = _seam_receivers(tri)
        lift = image_total(obs, tri, 0, MU, LAM, EPS, want=("U", "H"))
        folded = fold.image(obs, tri, MU, LAM, EPS, 0, ("U", "H"))
        mir = (np.asarray(asm.fullspace_blocks(obs, fold.mirror(tri), MU, LAM,
                                               EPS, want=("H",))["H"])
               * fold.MIRROR_SLIP_SIGN)
        e_h = relmax(folded["H"], lift["H"])
        e_u = relmax(folded["U"], lift["U"])
        e_m = relmax(mir, lift["H"])
        print(f"    {nd:5.0f} eps            {e_h:10.2e} {e_u:10.2e} {e_m:12.2e}")
        rep.check(f"c stress seam at {nd:.0f} eps", e_h, TOL_SEAM[k])
        rep.check(f"c displacement seam at {nd:.0f} eps", e_u, TOL_SEAM[k])
        rep.check_bool(f"c the mirror alone misses (> {TRIP_MIRROR_ONLY})",
                       e_m > TRIP_MIRROR_ONLY,
                       f"({e_m:.2f}) -- the Boussinesq term is the fix for "
                       f"dip slip, not a refinement")

    # ----------------------------------------------------------------- [d] ---
    print("\n[d] THE FREE-SURFACE CONDITION for the straddling source")
    dist = np.array([0.5, 1.0, 2.0, 5.0, 10.0]) * EPS
    edge = TRI[1] - TRI[0]
    nrm_t = np.array([-edge[1], edge[0], 0.0]) / np.hypot(edge[0], edge[1])
    mid = 0.5 * (TRI[0] + TRI[1])
    surf = np.vstack([mid + s * dd * nrm_t for dd in dist for s in (-1, 1)])
    surf[:, 2] = 0.0
    nzs = np.array([0.0, 0.0, 1.0])
    body = np.array([[0.4, 0.5, -1.0], [0.6, 0.2, -0.8], [0.3, 0.6, -1.5]])
    scale = float(np.abs(mhs.stress_matrix(body, TRI[None], mat, EPS)).max())
    t_fold = mhs.traction_matrix(surf, nzs, TRI[None], mat, EPS)
    t_lift = mhs.traction_matrix(surf, nzs, TRI[None], mat, EPS, fold=False)
    r_fold = float(np.abs(t_fold).max()) / scale
    r_lift = float(np.abs(t_lift).max()) / scale
    print(f"    max |t(z=0)| / bulk stress: fold {r_fold:.2e}, lift {r_lift:.2e}")
    rep.check("d the folded kernel is traction-free on z = 0", r_fold,
              TOL_FREE_SURFACE)
    rep.check_bool(f"d the lift is NOT (> {TRIP_LIFT_SURFACE})",
                   r_lift > TRIP_LIFT_SURFACE,
                   f"({r_lift:.2e}) within 10 eps of the trace")

    # ----------------------------------------------------------------- [e] ---
    print("\n[e] ONE FOLD IN EVERY ENTRY POINT")
    obs = np.vstack([surf[:4], body[:1], [[0.5, 0.3, -0.2]]])
    nrm = np.tile([0.3, -0.5, 0.812], (obs.shape[0], 1))
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    tris = TRI[None]
    T = mhs.traction_matrix(obs, nrm, tris, mat, EPS)
    S = mhs.stress_matrix(obs, tris, mat, EPS)
    rep.check("e traction == stress . n",
              relmax(np.einsum("oabsk,ob->oask", S, nrm), T), TOL_IDENTITY)
    tot = mhs.total_stress_matrix(obs, tris, mat, EPS)
    eig = mhs.eigenstress_matrix(obs, tris, mat, EPS)
    rep.check_bool("e total - eigenstress == stress, BITWISE",
                   np.array_equal(tot - eig, S))
    from mhs import tdcs
    frame = tdcs.slip_frame(tris)
    K = mhs.interaction_matrix(tris, mat, EPS)                 # (1, 1, 3, 3)
    cen = mhs.collocation_points(tris)
    Tc = mhs.traction_matrix(cen, frame[:, 2, :], tris, mat, EPS)
    rep.check("e interaction == frame-resolved traction at the centroid",
              relmax(np.einsum("ai,oisk,bk->osab", frame[0], Tc, frame[0]), K),
              TOL_IDENTITY)
    # continuous P2 at the trace nodes of a two-element surface patch
    patch = np.array([TRI, [TRI[1], [1.2, 0.9, -0.6], TRI[2]]])
    sn = mhs.shared_nodes(patch)
    tr = np.flatnonzero(sn.trace)
    T2 = mhs.traction_matrix(sn.coords[tr], sn.normals[tr], patch, mat, EPS,
                             order=2, layout="continuous", shared=sn)
    T0 = mhs.traction_matrix(sn.coords[tr], sn.normals[tr], patch, mat, EPS)
    rep.check("e uniform nodal P2 slip at a trace node == uniform P0 slip",
              relmax(T2.sum(axis=2), T0.sum(axis=2)), 1e-10)
    deep = np.array([[0.4, 0.5, -(d.FOLD_BAND_DEPTH_EPS + 1.0) * EPS]])
    rep.check_bool("e below the band, fold == lift bitwise",
                   np.array_equal(mhs.stress_matrix(deep, tris, mat, EPS),
                                  mhs.stress_matrix(deep, tris, mat, EPS,
                                                    fold=False)))
    moved = relmax(mhs.disp_matrix(surf[:2], tris, mat, EPS, fold=False),
                   mhs.disp_matrix(surf[:2], tris, mat, EPS))
    rep.check_bool(f"e at the surface the fold moves the displacement "
                   f"(> {TRIP_FOLD_MOVES})", moved > TRIP_FOLD_MOVES,
                   f"({moved:.2f})")
    return rep.finish()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
