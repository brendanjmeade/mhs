#!/usr/bin/env python
"""Algebraic identities of the nodal tensors: no external reference at all.

Every clause here is a statement the kernels must satisfy *by algebra*, so none
of them needs an oracle, a quadrature rule or a tolerance chosen by taste. That
makes this the most convention-proof gate in the suite, and the reason it is
worth porting in full: it catches transposed indices, a wrong normal
contraction, and a mis-signed orientation -- the failures that survive a
comparison against a reference computed with the same mistake.

Eight families, each with a tripwire so no clause can pass by being vacuous:

  1. PARTITION OF UNITY -- the P1 and P2 tensors summed over nodes equal the
     order-0 tensor, and the shape functions sum to 1 at random points.
  2. P1 EMBEDDED IN P2, and P0 embedded in both: interpolating a lower-order
     density onto higher-order nodes must change nothing.
  3. CYCLIC VERTEX RELABEL with correspondingly permuted nodal values leaves
     everything unchanged -- and the expected permutation is pinned, so a
     relabel that happened to be the identity could not pass.
  4. REVERSED ORIENTATION. U and H are ODD in the normal, E is EVEN, and the
     FORCE kernels carry no normal dependence at all, so G and S are unchanged.
     With a tripwire that U really does flip, so "unchanged" is not vacuous.
  5. RIGID MOTION: ``U' = Q U Q^T``, ``H'`` with three rotations, ``E`` scalar,
     and the same for the force family. **Note for the half-space work:** this
     full covariance is exactly what a free surface at z = 0 breaks. The
     half-space kernel will satisfy only the in-plane subgroup plus translations
     parallel to the surface, and this clause is the full-space baseline against
     which that restriction has to be stated rather than discovered.
  6. SCALING of (triangle, observers, eps) at fixed mu. U invariant, H and E as
     1/alpha -- and the force kernels the OTHER WAY, because a force per unit
     area picks up the extra length from dS: G gains alpha, S is invariant. Two
     tripwires assert the families really do differ, since a bug that collapsed
     them would otherwise satisfy both laws' union.
  7. INTERPOLATION reproduces polynomials of degree <= p, with a tripwire that
     P1 does NOT reproduce a generic quadratic.
  8. FORCE/SLIP RECIPROCITY, the exact algebraic tie between the two families:

         U[i,j] == -n_m S[j,m,i]

     because the slip kernel is the traction operator applied to the same
     integrated gradient the force kernel contracts with C. No quadrature, no
     limit, just algebra -- measured ~3e-16. It is the sharpest single statement
     that the two families share one moment table, plus a sign tripwire and the
     tensor symmetries G[i,j] = G[j,i], S[i,j,c] = S[j,i,c].

Tolerances are upstream's, unchanged.

Run from anywhere:  python tests/gates/mhs/verify_identities.py
"""
from __future__ import annotations

import sys
import types

import numpy as np

from _common import MU, TRI, Report, random_rotation, relmax, shipped

NU = 0.3
EPS = 0.12
FF = "analytic"
ORDERS = (0, 1, 2)
KEYS = ("U", "H", "E")          # slip (displacement-discontinuity) source
KEYS_F = ("G", "S")             # force (Kelvin single-layer) source
ALL_KEYS = KEYS + KEYS_F


def _namespace() -> types.SimpleNamespace:
    """Bind the shipped engine's public names into one object.

    So the clause bodies below stay close to the upstream gate they were ported
    from. Porting's one real hazard is a transcription slip in an index or a
    sign, and the way to avoid it is to change as little as possible.
    """
    api, frame, shape, kernels = (shipped("api"), shipped("frame"),
                                  shipped("shape"), shipped("kernels"))
    return types.SimpleNamespace(
        displacement=api.displacement, stress=api.stress,
        eigenstress=api.eigenstress,
        force_displacement=api.force_displacement,
        force_stress=api.force_stress,
        nodes=shape.nodes, n_nodes=shape.n_nodes,
        interpolate=shape.interpolate, nodal_values=shape.nodal_values,
        shape_functions=shape.shape_functions,
        unit_normal=frame.unit_normal, local_frame=frame.local_frame,
        nodal_influence=kernels.nodal_influence)


C = _namespace()


def observers(tri):
    v1, v2, v3 = tri
    return np.array([[0.60, -0.10, 0.60],
                     [2.00, 1.50, 3.00],
                     [-0.40, 0.00, -0.80],
                     v1 + 0.3 * (v2 - v1) + 0.3 * (v3 - v1)])   # on-plane


def node_permutation(tri_a, tri_b, p):
    """``perm`` such that ``nodes(tri_b, p) == nodes(tri_a, p)[perm]``."""
    na, nb = C.nodes(tri_a, p), C.nodes(tri_b, p)
    perm = []
    for x in nb:
        d = np.linalg.norm(na - x, axis=1)
        k = int(np.argmin(d))
        if d[k] > 1e-12:
            raise RuntimeError("node sets of the relabelled triangles differ")
        perm.append(k)
    perm = np.array(perm)
    if len(set(perm.tolist())) != len(perm):
        raise RuntimeError("node permutation is not a bijection")
    return perm


def fields(obs, tri, slip):
    return {
        "u": C.displacement(obs, tri, slip, MU, NU, EPS, far_field=FF),
        "sigma_total": C.stress(obs, tri, slip, MU, NU, EPS,
                                subtract_eigenstress=False, far_field=FF),
        "C:eps*": C.eigenstress(obs, tri, slip, MU, NU, EPS, far_field=FF),
        "sigma_elastic": C.stress(obs, tri, slip, MU, NU, EPS, far_field=FF),
    }


def force_fields(obs, tri, force):
    """No eigenstress row: a mollified body force is a genuine body force."""
    return {
        "u_force": C.force_displacement(obs, tri, force, MU, NU, EPS,
                                        far_field=FF),
        "sigma_force": C.force_stress(obs, tri, force, MU, NU, EPS,
                                      far_field=FF),
    }


def influence_all(obs, tri):
    return {p: C.nodal_influence(obs, tri, p, MU, NU, EPS, want=ALL_KEYS,
                                 far_field=FF) for p in ORDERS}


def main() -> bool:
    rep = Report("Algebraic identities, no external reference "
                 "(nu = 0.3, eps = 0.12, tilted triangle)")
    rng = np.random.default_rng(7)
    obs = observers(TRI)
    inf = influence_all(obs, TRI)
    slips = {p: rng.standard_normal((C.n_nodes(p), 3)) for p in ORDERS}
    fld = {p: fields(obs, TRI, slips[p]) for p in ORDERS}
    forces = {p: rng.standard_normal((C.n_nodes(p), 3)) for p in ORDERS}
    ffld = {p: force_fields(obs, TRI, forces[p]) for p in ORDERS}

    # ---------------------------------------------------------------- (1) ---
    for p in (1, 2):
        for key in ALL_KEYS:
            rep.check(f"(1) P{p} partition of unity, {key}",
                      relmax(inf[p][key].sum(axis=1), inf[0][key][:, 0]), 1e-13)
    fr = C.local_frame(TRI)
    Xr = rng.uniform(-1.5 * fr.L, 1.5 * fr.L, size=(40, 2))
    pts = fr.from_plane(Xr, rng.uniform(-1.0, 1.0, size=40))
    for p in ORDERS:
        Nk = C.shape_functions(TRI, p, pts)
        rep.check(f"(1) P{p} sum_k N_k = 1 at random points",
                  float(np.max(np.abs(Nk.sum(axis=1) - 1.0))), 1e-13)

    # ---------------------------------------------------------------- (2) ---
    s1 = slips[1]
    s2 = C.interpolate(TRI, s1, C.nodes(TRI, 2))
    f2 = fields(obs, TRI, s2)
    for name in fld[1]:
        rep.check(f"(2) P1 embedded in P2: {name}",
                  relmax(f2[name], fld[1][name]), 1e-13)
    fq2 = C.interpolate(TRI, forces[1], C.nodes(TRI, 2))
    ff2 = force_fields(obs, TRI, fq2)
    for name in ffld[1]:
        rep.check(f"(2) P1 embedded in P2: {name}",
                  relmax(ff2[name], ffld[1][name]), 1e-13)
    for p in (1, 2):
        sp = np.repeat(slips[0], C.n_nodes(p), axis=0)
        fp = fields(obs, TRI, sp)
        rep.check(f"(2) P0 embedded in P{p}: worst over u/sigma/C:eps*",
                  max(relmax(fp[k], fld[0][k]) for k in fp), 1e-13)
        fqp = np.repeat(forces[0], C.n_nodes(p), axis=0)
        ffp = force_fields(obs, TRI, fqp)
        rep.check(f"(2) P0 force embedded in P{p}: worst over force fields",
                  max(relmax(ffp[k], ffld[0][k]) for k in ffp), 1e-13)

    # ---------------------------------------------------------------- (3) ---
    tri_c = TRI[[1, 2, 0]]
    rep.check("(3) cyclic relabel keeps nhat",
              relmax(C.unit_normal(tri_c), C.unit_normal(TRI)), 1e-15)
    inf_c = influence_all(obs, tri_c)
    expected_perm = {0: [0], 1: [1, 2, 0], 2: [1, 2, 0, 4, 5, 3]}
    for p in ORDERS:
        perm = node_permutation(TRI, tri_c, p)
        rep.check_bool(f"(3) P{p} cyclic node permutation is {expected_perm[p]}",
                       perm.tolist() == expected_perm[p],
                       f"(got {perm.tolist()})")
        for key in ALL_KEYS:
            rep.check(f"(3) P{p} cyclic relabel, nodal {key}",
                      relmax(inf_c[p][key], inf[p][key][:, perm]), 1e-13)
        fc = fields(obs, tri_c, slips[p][perm])
        rep.check(f"(3) P{p} cyclic relabel, fields (worst of 4)",
                  max(relmax(fc[k], fld[p][k]) for k in fc), 1e-13)
        ffc = force_fields(obs, tri_c, forces[p][perm])
        rep.check(f"(3) P{p} cyclic relabel, force fields (worst of 2)",
                  max(relmax(ffc[k], ffld[p][k]) for k in ffc), 1e-13)

    # ---------------------------------------------------------------- (4) ---
    tri_r = TRI[[0, 2, 1]]
    rep.check("(4) reversed orientation flips nhat",
              relmax(C.unit_normal(tri_r), -C.unit_normal(TRI)), 1e-15)
    inf_r = influence_all(obs, tri_r)
    expected_perm = {0: [0], 1: [0, 2, 1], 2: [0, 2, 1, 5, 4, 3]}
    for p in ORDERS:
        perm = node_permutation(TRI, tri_r, p)
        rep.check_bool(f"(4) P{p} reversed node permutation is "
                       f"{expected_perm[p]}",
                       perm.tolist() == expected_perm[p],
                       f"(got {perm.tolist()})")
        rep.check(f"(4) P{p} reversed, nodal U -> -U",
                  relmax(inf_r[p]["U"], -inf[p]["U"][:, perm]), 1e-13)
        rep.check(f"(4) P{p} reversed, nodal H -> -H",
                  relmax(inf_r[p]["H"], -inf[p]["H"][:, perm]), 1e-13)
        rep.check(f"(4) P{p} reversed, nodal E -> +E",
                  relmax(inf_r[p]["E"], inf[p]["E"][:, perm]), 1e-13)
        for key in KEYS_F:
            rep.check(f"(4) P{p} reversed, nodal {key} unchanged "
                      f"(no nhat dependence)",
                      relmax(inf_r[p][key], inf[p][key][:, perm]), 1e-13)
        d_flip = relmax(inf_r[p]["U"], inf[p]["U"][:, perm])
        rep.check_bool(f"(4) P{p} tripwire: nodal U does flip (> 1e-2)",
                       d_flip > 1e-2, f"(rel diff {d_flip:.2e})")
        ffr = force_fields(obs, tri_r, forces[p][perm])
        for name in ffr:
            rep.check(f"(4) P{p} reversed, force density kept: {name} unchanged",
                      relmax(ffr[name], ffld[p][name]), 1e-13)
        fneg = fields(obs, tri_r, -slips[p][perm])
        for name in fneg:
            rep.check(f"(4) P{p} reversed + negated slip: {name} unchanged",
                      relmax(fneg[name], fld[p][name]), 1e-13)
        fpos = fields(obs, tri_r, slips[p][perm])
        for name in ("u", "sigma_total", "C:eps*"):
            rep.check(f"(4) P{p} reversed, slip kept: {name} flips sign",
                      relmax(fpos[name], -fld[p][name]), 1e-13)

    # ---------------------------------------------------------------- (5) ---
    Q = random_rotation(3)
    t = np.array([0.7, -1.3, 0.4])
    rep.check("(5) Q is a proper rotation",
              max(relmax(Q @ Q.T, np.eye(3)), abs(np.linalg.det(Q) - 1.0)),
              1e-14)
    tri_q = TRI @ Q.T + t
    obs_q = obs @ Q.T + t
    inf_q = influence_all(obs_q, tri_q)
    for p in ORDERS:
        rep.check(f"(5) P{p} rigid motion, U' = Q U Q^T",
                  relmax(inf_q[p]["U"],
                         np.einsum("ia,jb,nkab->nkij", Q, Q, inf[p]["U"])),
                  1e-12)
        rep.check(f"(5) P{p} rigid motion, H' = Q Q Q H",
                  relmax(inf_q[p]["H"],
                         np.einsum("ma,lb,jc,nkabc->nkmlj", Q, Q, Q,
                                   inf[p]["H"])), 1e-12)
        rep.check(f"(5) P{p} rigid motion, E' = E",
                  relmax(inf_q[p]["E"], inf[p]["E"]), 1e-12)
        rep.check(f"(5) P{p} rigid motion, G' = Q G Q^T",
                  relmax(inf_q[p]["G"],
                         np.einsum("ia,jb,nkab->nkij", Q, Q, inf[p]["G"])),
                  1e-12)
        rep.check(f"(5) P{p} rigid motion, S' = Q Q Q S",
                  relmax(inf_q[p]["S"],
                         np.einsum("ia,jb,cd,nkabd->nkijc", Q, Q, Q,
                                   inf[p]["S"])), 1e-12)
        rep.check(f"(5) P{p} rigid motion, nodes' = Q nodes + t",
                  relmax(inf_q[p]["nodes"], inf[p]["nodes"] @ Q.T + t), 1e-14)
    fq = fields(obs_q, tri_q, slips[2] @ Q.T)
    rep.check("(5) P2 rigid motion, u' = Q u",
              relmax(fq["u"], fld[2]["u"] @ Q.T), 1e-12)
    for name in ("sigma_total", "C:eps*", "sigma_elastic"):
        rep.check(f"(5) P2 rigid motion, {name}' = Q sigma Q^T",
                  relmax(fq[name],
                         np.einsum("ia,jb,nab->nij", Q, Q, fld[2][name])),
                  1e-12)
    ffq = force_fields(obs_q, tri_q, forces[2] @ Q.T)
    rep.check("(5) P2 rigid motion, u_force' = Q u_force",
              relmax(ffq["u_force"], ffld[2]["u_force"] @ Q.T), 1e-12)
    rep.check("(5) P2 rigid motion, sigma_force' = Q sigma_force Q^T",
              relmax(ffq["sigma_force"],
                     np.einsum("ia,jb,nab->nij", Q, Q,
                               ffld[2]["sigma_force"])), 1e-12)

    # ---------------------------------------------------------------- (6) ---
    alpha = 3.7
    inf_s = {p: C.nodal_influence(alpha * obs, alpha * TRI, p, MU, NU,
                                  alpha * EPS, want=ALL_KEYS, far_field=FF)
             for p in ORDERS}
    for p in ORDERS:
        rep.check(f"(6) P{p} scaling, U invariant",
                  relmax(inf_s[p]["U"], inf[p]["U"]), 1e-12)
        rep.check(f"(6) P{p} scaling, H -> H/alpha",
                  relmax(inf_s[p]["H"], inf[p]["H"] / alpha), 1e-12)
        rep.check(f"(6) P{p} scaling, E -> E/alpha",
                  relmax(inf_s[p]["E"], inf[p]["E"] / alpha), 1e-12)
        rep.check(f"(6) P{p} scaling, G -> alpha G",
                  relmax(inf_s[p]["G"], alpha * inf[p]["G"]), 1e-12)
        rep.check(f"(6) P{p} scaling, S invariant",
                  relmax(inf_s[p]["S"], inf[p]["S"]), 1e-12)
        d_g = relmax(inf_s[p]["G"], inf[p]["G"])
        d_s = relmax(inf_s[p]["S"], inf[p]["S"] / alpha)
        rep.check_bool(f"(6) P{p} tripwire: G is NOT invariant (like U)",
                       d_g > 1e-2, f"(rel diff {d_g:.2e})")
        rep.check_bool(f"(6) P{p} tripwire: S does NOT scale as 1/alpha "
                       f"(like H)", d_s > 1e-2, f"(rel diff {d_s:.2e})")

    # ---------------------------------------------------------------- (7) ---
    def basis(P, p):
        _, X = fr.to_plane(P)
        cols = [np.ones(X.shape[0])]
        if p >= 1:
            cols += [X[:, 0], X[:, 1]]
        if p >= 2:
            cols += [X[:, 0] ** 2, X[:, 0] * X[:, 1], X[:, 1] ** 2]
        return np.stack(cols, axis=1)

    for p in ORDERS:
        coef = rng.standard_normal((basis(pts[:1], p).shape[1], 3))

        def f(P, p=p, coef=coef):
            return basis(P, p) @ coef

        vals = C.nodal_values(TRI, p, f)
        rep.check(f"(7) P{p} interpolate(nodal_values(f)) reproduces "
                  f"degree-{p} f",
                  relmax(C.interpolate(TRI, vals, pts), f(pts)), 1e-13)
    coef = rng.standard_normal((6, 3))

    def fquad(P):
        return basis(P, 2) @ coef

    d = relmax(C.interpolate(TRI, C.nodal_values(TRI, 1, fquad), pts),
               fquad(pts))
    rep.check_bool("(7) tripwire: P1 interpolation of a quadratic differs "
                   "by > 1e-2", d > 1e-2, f"(rel diff {d:.2e})")

    # ---------------------------------------------------------------- (8) ---
    nhat = C.unit_normal(TRI)
    for p in ORDERS:
        rhs = -np.einsum("m,nkjmi->nkij", nhat, inf[p]["S"])
        rep.check(f"(8) P{p} reciprocity: U[i,j] == -n_m S[j,m,i]",
                  relmax(inf[p]["U"], rhs), 1e-13)
        d_sign = relmax(inf[p]["U"], -rhs)
        rep.check_bool(f"(8) P{p} tripwire: the sign matters (> 1e-2)",
                       d_sign > 1e-2, f"(rel diff {d_sign:.2e})")
        rep.check(f"(8) P{p} G symmetric in its two indices",
                  relmax(inf[p]["G"], np.swapaxes(inf[p]["G"], 2, 3)), 1e-14)
        rep.check(f"(8) P{p} S symmetric in its first two indices",
                  relmax(inf[p]["S"], np.swapaxes(inf[p]["S"], 2, 3)), 1e-14)
        sig_f = ffld[p]["sigma_force"]
        rep.check(f"(8) P{p} force_stress tensor symmetric",
                  relmax(sig_f, np.swapaxes(sig_f, 1, 2)), 1e-14)

    return rep.finish()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
