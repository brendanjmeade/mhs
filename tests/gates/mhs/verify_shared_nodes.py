#!/usr/bin/env python
"""The continuous layout: shared nodes, the scatter, and the penny crack.

``layout="continuous"`` is quadratic slip that is C0 across every element edge,
one Cartesian vector per shared node, collocated AT the nodes. Everything about
it is assembly-side -- the kernels are the P2 nodal ones every other gate
checks -- so what this gate proves is the TABLE, the SCATTER, and that the
layout buys on the half-space penny crack what it was measured to buy in the
sibling full-space package.

  [a] THE TABLE, on meshes small enough to count by hand: vertices welded,
      one node per edge, Euler's relation, the six ids of every element in the
      kernels' node order, the boundary mask, partition of unity, the
      three-midpoint integration rule exact for a quadratic, ``groups``
      keeping two faults apart along a shared line, and the refusals --
      non-manifold, inconsistent winding, a soup, P1, and the discontinuous
      arguments that do not apply.

  [b] THE SCATTER, as identities: a continuous-layout matrix equals the
      element-major P2 matrix with each element's six columns summed into the
      shared ones, for displacement, stress and traction; the interaction
      matrix equals the traction at the nodes resolved in the node frames; a
      row block equals the slice of the whole; ``by_rows`` is, to round-off, the
      serial result. At a surface trace the same partition of unity holds at
      a node ON z = 0, which is where the layout collocates on a
      surface-breaking fault.

  [c] THE PENNY CRACK, deep in the half space, dipping, on a polar disc whose
      rim lies exactly on the circle (the lattice disc's staircase rim is what
      both layouts see first, so it cannot referee this). Three things, each
      against the closed form and against P0 on the same mesh:
        * the READOUT of the prescribed Eshelby slip: the nodal error is
          ~1.7 eps/a, the same at vertices and midpoints and the same when
          eps is halved -- it is eps's -- where the P0 centroid readout does
          NOT fall with eps (the staircase floor);
        * the SOLVE: the stress-drop crack's slip at the free nodes, the rim
          locked, recovers Eshelby's profile 4x closer than P0 at the same eps
          and h, and its moment within 2 % where P0 is 10 % high;
        * the TENSILE penny (opening allowed, Sneddon): the same, 1.8x.
      The self-stiffness at every node is negative, the sign pin no identity
      can see. sym(-K) is reported: it loses definiteness at eps = 0.05 h
      (midpoint modes), which an LU does not care about and a complementarity
      solver must be told about.

Run from anywhere:  python tests/gates/mhs/verify_shared_nodes.py
"""
from __future__ import annotations

import functools
import math
import sys
import time

import numpy as np

CHECKS: list[bool] = []

MU, LAM = 30.0, 45.0            # nu = 0.30, not 1/4: a lam/mu swap is visible
NU = 0.5 * LAM / (LAM + MU)
DTAU = 0.01                     # GPa, the uniform stress drop
DEPTH = -20.0                   # km: 20 radii down, the free surface is weak
#: A row block against the whole matrix, and ``by_rows`` against the direct
#: call, agree to this fraction of the max entry rather than bitwise: the
#: batching differs, and OpenBLAS's small-problem kernels round differently
#: from its blocked ones (measured bitwise under Accelerate, last-bit under
#: the pip wheels on every CI platform).
ROWS_ROUNDOFF = 1e-13
DIP = 30.0                      # a horizontal disc is the frame trap

#: Measured on the polar disc at a/h = 6 (216 triangles, 469 nodes), the
#: numbers [c] pins. The readout error at the nodes in units of eps/a: 1.669
#: at eps = 0.1 h and 1.695 at 0.05 h (1.680 / 1.714 at a/h = 8), vertices and
#: midpoints within 0.21; the P0 centroid median 2.22 % at 0.1 h, which does
#: not halve with eps. Solve rms over the free nodes / d0: 1.60 % (0.1 h),
#: 1.13 % (0.05 h), P0 6.44 %; moments +0.8 % / -1.9 % against P0 +10.1 %;
#: tensile 2.36 % against P0 4.32 %.
READOUT_WINDOW = (1.2, 2.2)     # e / (eps/a), median over the interior
READOUT_EPS_RATIO = (1.6, 2.4)  # e(0.1 h) / e(0.05 h): proportional to eps
#: |median vertex error - median midpoint error| / (eps/a): both sit at the
#: readout's 1.7 eps/a and the kink at a vertex is not read differently from
#: the smooth midpoint. Measured 0.05 and 0.21 at eps/h = 0.1 and 0.05 here
#: and on macOS CI, but 0.52 at 0.05 h on the ubuntu runners: a difference
#: of two medians of a 1 % quantity, where the x86_64 wheels round the
#: closed forms differently. Pinned at 0.8, still under half the readout.
READOUT_ALIKE = 0.8
P0_READOUT_RATIO_MAX = 1.4      # the P0 median must NOT halve with eps
SOLVE_RMS_MAX = {0.1: 0.024, 0.05: 0.017}
SOLVE_MOMENT_TOL = {0.1: 0.02, 0.05: 0.03}
P0_OVER_P2_RMS_MIN = 2.5        # measured 4.0
P0_MOMENT_ERR_MIN = 0.05        # measured 0.10
TENSILE_RMS_MAX = 0.035         # measured 0.0236
P0_OVER_P2_TENSILE_MIN = 1.5    # measured 1.83


def check(label: str, value: float, tol: float, fmt: str = "{:.3e}") -> bool:
    ok = bool(np.isfinite(value)) and value < tol
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {label:60s} "
          f"{fmt.format(value)} < {fmt.format(tol)}")
    return ok


def check_true(label: str, ok: bool, note: str = "") -> bool:
    ok = bool(ok)
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {label:60s} {note}")
    return ok


def relmax(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    den = float(np.abs(b).max())
    return float(np.abs(a - b).max() / den) if den > 0 else float(np.abs(a).max())


def refuses(label, fn, exc=ValueError) -> bool:
    try:
        fn()
    except exc as e:
        return check_true(label, True, f"{exc.__name__}: {str(e)[:50]}")
    except Exception as e:  # noqa: BLE001
        return check_true(label, False, f"raised {type(e).__name__}: {e}")
    return check_true(label, False, "returned without raising")


# --------------------------------------------------------------- fixtures ---
def polar_disc(a: float, n_rings: int) -> np.ndarray:
    """A disc in z = 0: rings at k a / n_rings carrying 6 k vertices, the rim
    EXACTLY on the circle, 2k + 1 triangles per sector between rings, wound
    +z. Mean edge 1.12 a / n_rings. The rim is the point: a lattice disc's
    staircase rim moves the P2 moment by 4 % and lets P0 land by luck."""
    verts = [np.zeros(2)]
    ring = [[0]]
    for k in range(1, n_rings + 1):
        th = 2.0 * np.pi * np.arange(6 * k) / (6 * k)
        r = a * k / n_rings
        ring.append(list(range(len(verts), len(verts) + 6 * k)))
        verts += [np.array([r * np.cos(t), r * np.sin(t)]) for t in th]
    verts = np.asarray(verts)
    tris = [(0, ring[1][s], ring[1][(s + 1) % 6]) for s in range(6)]
    for k in range(1, n_rings):
        inner, outer = ring[k], ring[k + 1]
        ni, no = len(inner), len(outer)
        for s in range(6):
            def I(t):
                return inner[(s * k + t) % ni]

            def O(t):
                return outer[(s * (k + 1) + t) % no]
            tris += [(I(t), O(t), O(t + 1)) for t in range(k + 1)]
            tris += [(I(t), O(t + 1), I(t + 1)) for t in range(k)]
    xy = verts[np.asarray(tris)]
    area2 = ((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
             - (xy[:, 1, 1] - xy[:, 0, 1]) * (xy[:, 2, 0] - xy[:, 0, 0]))
    assert np.all(area2 > 0.0)
    out = np.zeros((xy.shape[0], 3, 3))
    out[:, :, :2] = xy
    return out


def dipping(flat: np.ndarray, dip_deg: float, depth: float):
    """Rotate a z = 0 mesh about the origin to dip toward +x, then sink it;
    returns the mesh and the in-plane radius function."""
    t = math.radians(dip_deg)
    R = np.array([[math.cos(t), 0.0, math.sin(t)], [0.0, 1.0, 0.0],
                  [-math.sin(t), 0.0, math.cos(t)]])
    out = flat @ R.T
    out[:, :, 2] += depth
    e1, e2 = R @ np.array([1.0, 0.0, 0.0]), R @ np.array([0.0, 1.0, 0.0])
    centre = np.array([0.0, 0.0, depth])

    def radius(p):
        d = np.asarray(p, float) - centre
        return np.hypot(d @ e1, d @ e2)
    return out, radius


def fan(n: int = 3, apex=(0.0, 0.0, -2.0)) -> np.ndarray:
    """``n`` triangles around one vertex, consistently wound (+z)."""
    apex = np.asarray(apex, float)
    th = np.linspace(0.0, 2.0 * np.pi, n + 1)[:-1] if n > 2 else \
        np.array([0.0, np.pi / 2.0])
    rim = apex + np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], axis=1)
    out = []
    for i in range(n):
        if n > 2 or i < n - 1:
            out.append([apex, rim[i], rim[(i + 1) % n]])
    return np.asarray(out)


def gauss_tri(tri, f, n: int = 6):
    """Gauss-Legendre on the triangle through the Duffy map, ``n x n`` points:
    exact to round-off for a quadratic, and built from numpy alone so the
    integration pin shares nothing with the three-midpoint rule it checks."""
    x, w = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (x + 1.0)
    wu = 0.5 * w
    U, V = np.meshgrid(u, u, indexing="ij")
    # x + y = u, y = u v: the Jacobian is u, and the unit triangle's area 1/2
    # is the 2 below
    W = np.outer(wu, wu) * U
    l1 = (1.0 - U).reshape(-1)
    l2 = (U * (1.0 - V)).reshape(-1)
    l3 = (U * V).reshape(-1)
    pts = np.stack([l1, l2, l3], axis=1) @ tri
    area = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
    return 2.0 * area * float(W.reshape(-1) @ f(pts))


# --------------------------------------------------------------------- [a] ---
def part_a() -> None:
    import mhs
    from mhs import tdcs
    from mhs.fullspace.shape import nodes as shape_nodes
    print("\n[a] THE TABLE: welding, edges, order, boundary, integration, groups")
    tris = fan(6)
    sn = mhs.shared_nodes(tris)
    check_true("a a 6-fan welds to 7 vertices, 12 edges, 19 nodes",
               (sn.n_vertices, sn.n_edges, sn.n) == (7, 12, 19),
               f"({sn.n_vertices}, {sn.n_edges}, {sn.n})")
    check_true("a Euler: V - E + F = 1 on a disc",
               sn.n_vertices - sn.n_edges + sn.n_tri == 1, "")
    worst = max(relmax(sn.coords[sn.ids[s]], shape_nodes(tris[s], 2))
                for s in range(sn.n_tri))
    check("a ids follow shape.nodes' order v1 v2 v3 m12 m23 m31", worst, 1e-14)
    check_true("a the boundary is the rim: 6 vertices and 6 midpoints, not the apex",
               int(sn.boundary.sum()) == 12 and not sn.boundary[sn.ids[0, 0]],
               f"{int(sn.boundary.sum())} boundary nodes")
    # the midpoint of a shared edge is ONE node in both elements
    shared_mid = [sn.ids[s, 5] == sn.ids[(s + 1) % 6, 3] for s in range(6)]
    check_true("a a shared edge's midpoint is one node in both elements",
               all(shared_mid), "m31 of element s is m12 of element s+1")
    ones = sn.expand(np.ones((sn.n, 3)))
    check_true("a partition of unity: expand(ones) is ones, element-major (36, 3)",
               ones.shape == (36, 3) and bool(np.all(ones == 1.0)), "")
    # integration: exact for a quadratic, against a 7-point rule
    q = lambda p: (p[:, 0] ** 2 - 0.3 * p[:, 0] * p[:, 1] + 0.7 * p[:, 1]
                   + 2.0)
    ref = sum(gauss_tri(tris[s], q) for s in range(sn.n_tri))
    check("a integrate() is exact for a quadratic (vs a Gauss rule)",
          abs(sn.integrate(q(sn.coords)) - ref) / abs(ref), 1e-13)
    cubic = lambda p: p[:, 0] ** 3
    ref3 = sum(gauss_tri(tris[s], cubic) for s in range(sn.n_tri))
    check_true("a TRIPWIRE: the rule is not exact for a cubic",
               abs(sn.integrate(cubic(sn.coords)) - ref3) / abs(ref3) > 1e-4,
               "so a quadratic's exactness is the rule's, not the fixture's")
    lin = lambda p: 0.5 * p[:, 0] + 1.0
    check("a element_mean() of a linear field is its centroid value",
          relmax(sn.element_mean(lin(sn.coords)), lin(tris.mean(axis=1))),
          1e-14)
    # normals and frames
    check("a node normals are unit and point up",
          float(np.abs(np.linalg.norm(sn.normals, axis=1) - 1.0).max())
          + float(np.clip(-sn.normals[:, 2], 0, None).max()), 1e-14)
    fr = sn.frames
    check("a node frames are orthonormal, right-handed",
          relmax(np.einsum("nab,ncb->nac", fr, fr), np.eye(3))
          + float(np.abs(np.linalg.det(fr) - 1.0).max()), 1e-13)
    check("a on a flat mesh the node frame is the element frame",
          relmax(fr, np.repeat(tdcs.slip_frame(tris[:1]), sn.n, axis=0)),
          1e-14)
    # groups: a second fault of two triangles attached along one rim edge of
    # the fan -- welded, the edge is shared; grouped, it is two separate edges
    r0, r1 = tris[0, 1], tris[0, 2]
    pq = r0 + r1 - tris[0, 0]
    qq = pq + (r1 - r0)
    strip = np.array([[r0, pq, r1], [pq, qq, r1]])
    two = np.concatenate([tris, strip])
    same = mhs.shared_nodes(two)
    apart = mhs.shared_nodes(two, groups=[0] * 6 + [1, 1])
    check_true("a groups keep a shared line apart (2 vertices and 1 midpoint more)",
               (same.n_vertices, same.n_edges) == (9, 16)
               and (apart.n_vertices, apart.n_edges) == (11, 17),
               f"welded ({same.n_vertices} V, {same.n_edges} E), grouped "
               f"({apart.n_vertices} V, {apart.n_edges} E)")
    check_true("a in the grouped table every node belongs to one group",
               bool(np.all(apart.group[apart.ids[:6]] == 0))
               and bool(np.all(apart.group[apart.ids[6:]] == 1))
               and int(apart.boundary.sum()) == int(same.boundary.sum()) + 4,
               "and the shared line is boundary on both sides (two more "
               "vertices, two midpoints where there was none)")
    # trace / front on a dipping square reaching z = 0
    sq = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, -1.0]],
                   [[1, 0, 0], [1, 1, -1.0], [0, 1, -1.0]]], float)
    st = mhs.shared_nodes(sq)
    check_true("a trace = boundary nodes at z = 0, front = the rest of the rim",
               int(st.trace.sum()) == 3 and int(st.front.sum()) == 5
               and not bool((st.trace & st.front).any()),
               f"trace {int(st.trace.sum())}, front {int(st.front.sum())}")
    # refusals
    e = fan(3)[0]
    three = np.stack([e, e + np.array([[0, 0, 0], [0, 0, 0], [0, 0, -1.0]]),
                      e + np.array([[0, 0, 0], [0, 0, 0], [0, 0, -2.0]])])
    refuses("a non-manifold (three triangles on one edge) refused",
            lambda: mhs.shared_nodes(three))
    flipped = fan(6).copy()
    flipped[2] = flipped[2][[0, 2, 1]]
    refuses("a inconsistent winding refused", lambda: mhs.shared_nodes(flipped))
    refuses("a a soup (nothing shared) refused",
            lambda: mhs.shared_nodes(np.stack([e, e + 5.0])))
    refuses("a a tolerance that welds a triangle's own vertices refused",
            lambda: mhs.shared_nodes(fan(6), tol_over_h=2.0))
    refuses("a a zero-area triangle refused",
            lambda: mhs.shared_nodes(np.array([[[0, 0, -1], [1, 0, -1],
                                                [2, 0, -1.0]]])))
    refuses("a frame_from_normals refuses a downward normal",
            lambda: tdcs.frame_from_normals(np.array([[0.0, 0.0, -1.0]])))
    mat = mhs.Material(mu=MU, lam=LAM)
    obs = tris.mean(axis=1)
    nrm = tdcs.unit_normals(tris)
    refuses("a P1 continuous refused",
            lambda: mhs.traction_matrix(obs, nrm, tris, mat, 0.1, order=1,
                                        layout="continuous"))
    refuses("a P0 continuous refused",
            lambda: mhs.interaction_matrix(tris, mat, 0.1, layout="continuous"))
    refuses("a shared= with the discontinuous layout refused",
            lambda: mhs.disp_matrix(obs, tris, mat, 0.1, order=2, shared=sn))
    refuses("a obs_pts with the continuous layout refused",
            lambda: mhs.interaction_matrix(tris, mat, 0.1, order=2,
                                           layout="continuous", obs_pts=obs))
    refuses("a shrink with the continuous layout refused",
            lambda: mhs.interaction_matrix(tris, mat, 0.1, order=2,
                                           layout="continuous", shrink=0.0))
    refuses("a a table of another mesh refused",
            lambda: mhs.disp_matrix(obs, tris[:5], mat, 0.1, order=2,
                                    layout="continuous", shared=sn))
    refuses("a an unknown layout refused",
            lambda: mhs.disp_matrix(obs, tris, mat, 0.1, layout="nodal"))


# --------------------------------------------------------------------- [b] ---
def part_b() -> None:
    import mhs
    from mhs import tdcs
    from mhs.parallel import by_rows
    print("\n[b] THE SCATTER: identities against the element-major P2 matrices")
    mat = mhs.Material(mu=MU, lam=LAM)
    tris, _ = dipping(polar_disc(1.0, 2), DIP, DEPTH)        # 24 triangles
    sn = mhs.shared_nodes(tris)
    eps = 0.1 / 2.0
    obs = sn.coords
    nrm = sn.normals
    cols = sn.ids.reshape(-1)

    def condensed(M, axis):
        out = np.zeros(M.shape[:axis] + (sn.n,) + M.shape[axis + 1:])
        idx = [slice(None)] * M.ndim
        idx[axis] = cols
        np.add.at(out, tuple(idx), M)
        return out

    worst = {}
    for name, cont, disc, axis in (
            ("disp_matrix",
             lambda: mhs.disp_matrix(obs, tris, mat, eps, order=2,
                                     layout="continuous", shared=sn),
             lambda: mhs.disp_matrix(obs, tris, mat, eps, order=2), 2),
            ("stress_matrix",
             lambda: mhs.stress_matrix(obs, tris, mat, eps, order=2,
                                       layout="continuous", shared=sn),
             lambda: mhs.stress_matrix(obs, tris, mat, eps, order=2), 3),
            ("traction_matrix",
             lambda: mhs.traction_matrix(obs, nrm, tris, mat, eps, order=2,
                                         layout="continuous", shared=sn),
             lambda: mhs.traction_matrix(obs, nrm, tris, mat, eps, order=2),
             2)):
        C = cont()
        D = disc()
        worst[name] = relmax(C, condensed(D, axis))
        check(f"b {name}: continuous == element columns summed into the "
              f"shared ones", worst[name], 1e-14)
        if name == "traction_matrix":
            T = C
    check_true("b the continuous source axis is V + E, the element-major one 6 F",
               T.shape[2] == sn.n == sn.n_vertices + sn.n_edges
               and D.shape[2] == 6 * sn.n_tri, f"{T.shape} vs {D.shape}")
    K = mhs.interaction_matrix(tris, mat, eps, order=2, layout="continuous",
                               shared=sn)
    fr = sn.frames
    check("b interaction_matrix == traction at the nodes in the node frames",
          relmax(K, np.einsum("oai,oigk,gbk->ogab", fr, T, fr)), 1e-14)
    Kr = mhs.interaction_matrix(tris, mat, eps, order=2, layout="continuous",
                                shared=sn, rows=(7, 30))
    # Round-off, not bitwise, wherever two calls batch the observers
    # differently: OpenBLAS runs a different kernel for a small problem than
    # for a large one, so the same row computed in a 23-row block and in the
    # whole matrix can differ in its last bit on the pip wheels (CI, all four
    # platforms) while agreeing bitwise under Accelerate. 1e-13 of the max.
    check("b rows=(lo, hi) is the slice of the whole (round-off)",
          relmax(Kr, K[7:30]), ROWS_ROUNDOFF)
    # the discontinuous layout takes rows too
    K0 = mhs.interaction_matrix(tris, mat, eps)
    check("b ... and so does the discontinuous layout (round-off)",
          relmax(mhs.interaction_matrix(tris, mat, eps, rows=(5, 20)),
                 K0[5:20]), ROWS_ROUNDOFF)
    call = functools.partial(mhs.interaction_matrix, tris, mat, eps, order=2,
                             layout="continuous", shared=sn,
                             receiver=("strike", "dip"), source=("strike", "dip"))
    direct = call()
    check("b by_rows(workers=1) == the direct call (round-off)",
          relmax(by_rows(call, sn.n, workers=1), direct), ROWS_ROUNDOFF)
    t0 = time.time()
    par = by_rows(call, sn.n, workers=2)
    check("b by_rows(workers=2) == the direct call (round-off)",
          relmax(par, direct), ROWS_ROUNDOFF)
    check_true("b collocation_points(layout='continuous') is the node table",
               np.array_equal(mhs.collocation_points(tris, 2,
                                                     layout="continuous",
                                                     shared=sn), sn.coords),
               "")
    # the three-way split survives the scatter
    tot = mhs.total_stress_matrix(obs, tris, mat, eps, order=2,
                                  layout="continuous", shared=sn)
    eig = mhs.eigenstress_matrix(obs, tris, mat, eps, order=2,
                                 layout="continuous", shared=sn)
    ela = mhs.stress_matrix(obs, tris, mat, eps, order=2, layout="continuous",
                            shared=sn)
    check("b total - eigenstress == elastic in the continuous layout",
          relmax(tot - eig, ela), 1e-13)
    # a caller-supplied out= starts at zero (the scatter accumulates)
    junk = np.full(T.shape, 7.0)
    mhs.traction_matrix(obs, nrm, tris, mat, eps, order=2,
                        layout="continuous", shared=sn, out=junk)
    check_true("b a scattered matrix into a dirty out= is the clean one",
               np.array_equal(junk, T), "")
    # a surface trace: partition of unity at a node ON z = 0
    surf = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.3, 0.9, -1.0]],
                     [[1.0, 0.0, 0.0], [1.2, 0.8, -1.0], [0.3, 0.9, -1.0]]])
    ss = mhs.shared_nodes(surf)
    tr = np.flatnonzero(ss.trace)
    pts = ss.coords[tr]
    nn = ss.normals[tr]
    Tc = mhs.traction_matrix(pts, nn, surf, mat, 0.1, order=2,
                             layout="continuous", shared=ss)
    T0 = mhs.traction_matrix(pts, nn, surf, mat, 0.1)
    check_true("b at a trace node ON z = 0 the kernels are finite",
               bool(np.isfinite(Tc).all()), f"{tr.size} trace nodes")
    # a shared column carries both elements, so the partition of unity is a
    # statement about the whole mesh: uniform slip on every node == uniform
    # slip on every element
    check("b ... and uniform nodal slip there is uniform P0 slip",
          relmax(Tc.sum(axis=2), T0.sum(axis=2)), 1e-11)


# --------------------------------------------------------------------- [c] ---
def part_c() -> None:
    import mhs
    from mhs import tdcs
    print("\n[c] THE PENNY CRACK in the half space: readout, solve, tensile")
    mat = mhs.Material(mu=MU, lam=LAM)
    a, a_over_h = 1.0, 6
    h = a / a_over_h
    tris, radius = dipping(polar_disc(a, a_over_h), DIP, DEPTH)
    n = tris.shape[0]
    sn = mhs.shared_nodes(tris)
    r_n = radius(sn.coords)
    r_c = radius(tris.mean(axis=1))
    d0 = 8.0 * (1.0 - NU) / (math.pi * (2.0 - NU)) * DTAU * a / MU
    w0 = 4.0 * (1.0 - NU) * DTAU * a / (math.pi * MU)
    prof = lambda r: np.sqrt(np.maximum(0.0, 1.0 - (r / a) ** 2))
    free = ~sn.boundary
    idx = np.flatnonzero(free)
    inner_n = r_n < a - h
    inner_c = r_c < a - h
    is_vertex = np.arange(sn.n) < sn.n_vertices
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0],
                                         tris[:, 2] - tris[:, 0]), axis=1)
    moment_exact = 2.0 * math.pi * d0 * a * a / 3.0
    print(f"       {n} triangles, {sn.n} nodes ({int(sn.boundary.sum())} on the "
          f"rim, locked), a/h {a_over_h}, dip {DIP:g}, depth {-DEPTH:g} a, "
          f"nu {NU:.2f}; rim radius "
          f"{r_n[sn.boundary].min():.4f}..{r_n[sn.boundary].max():.4f}")
    rim_v = sn.boundary & is_vertex
    check("c the rim vertices lie on the circle",
          float(np.abs(r_n[rim_v] - a).max()), 1e-12,
          "{:.1e}")
    print(f"       (rim midpoints at the chord sag, r >= "
          f"{r_n[sn.boundary & ~is_vertex].min():.4f}; a lattice disc's "
          f"staircase rim spans 0.94..1.02)")

    def blocks(K, rows, cols, comps):
        out = np.zeros((len(comps) * rows.size, len(comps) * cols.size))
        sub = K[np.ix_(rows, cols)]
        for i, a_ in enumerate(comps):
            for j, b_ in enumerate(comps):
                out[i::len(comps), j::len(comps)] = sub[:, :, a_, b_]
        return out

    got = {}
    for eoh in (0.1, 0.05):
        eps = eoh * h
        t0 = time.time()
        K = mhs.interaction_matrix(tris, mat, eps, order=2,
                                   layout="continuous", shared=sn,
                                   receiver=("strike", "dip", "normal"),
                                   source=("strike", "dip", "tensile"))
        t_k = time.time() - t0
        K0 = mhs.interaction_matrix(tris, mat, eps, order=0,
                                    receiver=("strike", "dip", "normal"),
                                    source=("strike", "dip", "tensile"))
        g = {}
        # the readout: traction along dip at the nodes from Eshelby's slip
        e = np.abs(K[:, :, 1, 1] @ (d0 * prof(r_n)) + DTAU) / DTAU
        e0 = np.abs(K0[:, :, 1, 1] @ (d0 * prof(r_c)) + DTAU) / DTAU
        g["read"] = float(np.median(e[inner_n])) / (eps / a)
        g["read_v"] = float(np.median(e[inner_n & is_vertex])) / (eps / a)
        g["read_m"] = float(np.median(e[inner_n & ~is_vertex])) / (eps / a)
        g["read0"] = float(np.median(e0[inner_c]))
        # the solve: shear, dip drop, rim locked
        A = blocks(K, idx, idx, (0, 1))
        rhs = np.zeros(A.shape[0])
        rhs[1::2] = -DTAU
        x = np.linalg.solve(A, rhs)
        d_dip = np.zeros(sn.n)
        d_dip[idx] = x[1::2]
        g["rms"] = float(np.sqrt(np.mean(((d_dip - d0 * prof(r_n))[free] / d0) ** 2)))
        g["moment"] = float(sn.integrate(d_dip) / moment_exact)
        g["centre"] = float(d_dip[np.argmin(r_n)] / d0)
        g["strike"] = float(np.sqrt(np.mean((x[0::2] / d0) ** 2)))
        g["mineig"] = float(np.linalg.eigvalsh(-0.5 * (A + A.T)).min())
        A0 = blocks(K0, np.arange(n), np.arange(n), (0, 1))
        rhs0 = np.zeros(2 * n)
        rhs0[1::2] = -DTAU
        x0 = np.linalg.solve(A0, rhs0)[1::2]
        g["rms0"] = float(np.sqrt(np.mean(((x0 - d0 * prof(r_c)) / d0) ** 2)))
        g["moment0"] = float((area * x0).sum() / moment_exact)
        # the tensile penny: opening allowed, normal drop
        An = K[np.ix_(idx, idx)][:, :, 2, 2]
        wn = np.zeros(sn.n)
        wn[idx] = np.linalg.solve(An, np.full(idx.size, -DTAU))
        g["tens"] = float(np.sqrt(np.mean(((wn - w0 * prof(r_n))[free] / w0) ** 2)))
        g["tens_centre"] = float(wn[np.argmin(r_n)] / w0)
        w0_ = np.linalg.solve(K0[:, :, 2, 2], np.full(n, -DTAU))
        g["tens0"] = float(np.sqrt(np.mean(((w0_ - w0 * prof(r_c)) / w0) ** 2)))
        g["diag"] = (float(np.diag(K[:, :, 0, 0]).max()),
                     float(np.diag(K[:, :, 1, 1]).max()))
        got[eoh] = g
        print(f"       eps/h {eoh:5.2f} ({t_k:.0f} s): readout e/(eps/a) median "
              f"{g['read']:.3f} (vertices {g['read_v']:.3f}, midpoints "
              f"{g['read_m']:.3f}); P0 centroids {100 * g['read0']:.2f} %")
        print(f"                   solve rms/d0 {100 * g['rms']:.2f} % (P0 "
              f"{100 * g['rms0']:.2f} %), moment {g['moment']:.4f} (P0 "
              f"{g['moment0']:.4f}), centre {g['centre']:.4f} d0, |strike| "
              f"{g['strike']:.1e} d0, min eig sym(-A) {g['mineig']:+.2f}")
        print(f"                   tensile rms/w0 {100 * g['tens']:.2f} % (P0 "
              f"{100 * g['tens0']:.2f} %), centre {g['tens_centre']:.4f} w0")

    for eoh, g in got.items():
        check_true(f"c eps/h {eoh:g}: the self-stiffness is negative at every node, "
                   f"strike and dip", g["diag"][0] < 0.0 and g["diag"][1] < 0.0,
                   f"max {g['diag'][0]:+.2e}, {g['diag'][1]:+.2e}")
        check_true(f"c eps/h {eoh:g}: the nodal readout error is "
                   f"{READOUT_WINDOW[0]}-{READOUT_WINDOW[1]} eps/a",
                   READOUT_WINDOW[0] < g["read"] < READOUT_WINDOW[1],
                   f"{g['read']:.3f}")
        check(f"c eps/h {eoh:g}: vertices and midpoints read alike (eps/a)",
              abs(g["read_v"] - g["read_m"]), READOUT_ALIKE, "{:.2f}")
        check(f"c eps/h {eoh:g}: solve rms over the free nodes / d0",
              g["rms"], SOLVE_RMS_MAX[eoh], "{:.4f}")
        check(f"c eps/h {eoh:g}: moment against 2 pi d0 a^2 / 3",
              abs(g["moment"] - 1.0), SOLVE_MOMENT_TOL[eoh], "{:.4f}")
        check_true(f"c eps/h {eoh:g}: the centre slips the right way, near d0",
                   0.95 < g["centre"] < 1.05, f"{g['centre']:.4f} d0")
        check(f"c eps/h {eoh:g}: no strike slip from a dip drop (rms / d0)",
              g["strike"], 1e-2, "{:.1e}")
        check(f"c eps/h {eoh:g}: tensile rms over the free nodes / w0",
              g["tens"], TENSILE_RMS_MAX, "{:.4f}")
        check_true(f"c eps/h {eoh:g}: the crack opens, near Sneddon's w0",
                   0.95 < g["tens_centre"] < 1.05, f"{g['tens_centre']:.4f} w0")
    r = (got[0.1]["read"] * 0.1) / (got[0.05]["read"] * 0.05)
    check_true("c the readout error is eps's: it halves when eps halves",
               READOUT_EPS_RATIO[0] < r < READOUT_EPS_RATIO[1],
               f"e(0.1 h) / e(0.05 h) = {r:.2f}")
    r0 = got[0.1]["read0"] / got[0.05]["read0"]
    check_true("c TRIPWIRE: the P0 centroid readout does NOT halve with eps",
               r0 < P0_READOUT_RATIO_MAX,
               f"{100 * got[0.1]['read0']:.2f} % -> {100 * got[0.05]['read0']:.2f} "
               f"% ({r0:.2f}x): the staircase floor")
    for eoh in (0.1, 0.05):
        check_true(f"c TRIPWIRE: P0's solve error is the larger, by "
                   f"{P0_OVER_P2_RMS_MIN}x at eps = {eoh:g} h",
                   got[eoh]["rms0"] > P0_OVER_P2_RMS_MIN * got[eoh]["rms"],
                   f"{100 * got[eoh]['rms0']:.2f} % against "
                   f"{100 * got[eoh]['rms']:.2f} %: "
                   f"{got[eoh]['rms0'] / got[eoh]['rms']:.1f}x")
    check_true("c TRIPWIRE: P0's moment is off by more than "
               f"{100 * P0_MOMENT_ERR_MIN:.0f} %",
               abs(got[0.1]["moment0"] - 1.0) > P0_MOMENT_ERR_MIN,
               f"{got[0.1]['moment0']:.4f}")
    check_true("c TRIPWIRE: P0's tensile error is the larger, by "
               f"{P0_OVER_P2_TENSILE_MIN}x",
               got[0.1]["tens0"] > P0_OVER_P2_TENSILE_MIN * got[0.1]["tens"],
               f"{got[0.1]['tens0'] / got[0.1]['tens']:.2f}x")
    print(f"       sym(-A) min eig {got[0.1]['mineig']:+.1f} at eps = 0.1 h, "
          f"{got[0.05]['mineig']:+.1f} at 0.05 h: reported, not gated -- an LU "
          f"does not care, a complementarity solver must check it")


def main() -> bool:
    print("=" * 76)
    print("The continuous layout: shared nodes, the scatter, the penny crack")
    print("=" * 76)
    print(f"  mu = {MU:g}, lam = {LAM:g} (nu = {NU:.3f}, not 1/4)")
    t0 = time.time()
    part_a()
    part_b()
    part_c()
    print("-" * 76)
    n_bad = sum(1 for c in CHECKS if not c)
    if n_bad == 0:
        print(f"PASS: the continuous layout holds ({len(CHECKS)} checks, "
              f"{time.time() - t0:.0f} s)")
    else:
        print(f"FAIL: {n_bad} of {len(CHECKS)} checks failed")
    return n_bad == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
