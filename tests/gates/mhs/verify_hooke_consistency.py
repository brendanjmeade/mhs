#!/usr/bin/env python
"""FD-Hooke consistency, and the lam/mu-swap demonstration.

**This is the lam/mu gate.** A swap between Lame's first parameter and the shear
modulus in the displacement contraction is *exactly invisible* at nu = 1/4,
where lam == mu, and upstream records four files carrying that error for months
because every gate ran there. So everything here runs at nu = 0.25 AND nu = 0.30,
and both must pass.

The check is internal consistency through finite differences, which is what
makes it independent of any oracle: take the closed-form DISPLACEMENT kernel,
differentiate it numerically (central differences at h and h/2, Richardson
extrapolated), apply Hooke's law to the symmetric part, and require the result
to equal the closed-form STRESS kernel. Two different closed forms, related by a
derivative the gate computes itself -- a swap in one but not the other cannot
hide.

  * grad u by Richardson-extrapolated central differences, strain = sym(grad u),
    sigma = lam tr(strain) I + 2 mu strain, against the TOTAL stress kernel, at
    p = 0/1/2, nu in {0.25, 0.30}, three observers (two off-plane at |z| = 0.3 L
    and one ON the plane inside the triangle). 1e-7 relative.
  * A TRIPWIRE that the comparison discriminates total from elastic: on the
    plane, FD-Hooke must differ from the ELASTIC stress by more than 1e-2.
    Otherwise the clause above could pass against either and prove neither.
  * THE DEMONSTRATION. Rebuild the displacement by Gauss quadrature of the point
    kernel, once with the correct traction-operator pairing and once with lam and
    mu swapped on the first two contraction terms. Correct: reproduces the stress
    kernel to 1e-6 (quadrature-limited). Swapped: invisible at nu = 0.25 (passes
    the same 1e-6!) and caught above 1e-2 at nu = 0.30. That pair of outcomes is
    the whole argument for running kernel gates off nu = 1/4.

The FORCE (Kelvin single-layer) element gets the same battery, and note the
ASYMMETRY. For slip, FD-Hooke must be compared against the TOTAL stress, because
a mollified dislocation carries an eigenstrain and ``C:sym(grad u)`` includes it.
A mollified body force carries none -- it is a genuine body force -- so
``force_stress`` IS ``C:sym(grad u_force)`` and no eigenstress appears anywhere
in the force battery. ``force_stress`` therefore takes no
``subtract_eigenstress`` keyword at all, and a clause here asserts that passing
one raises, so the asymmetry is enforced rather than merely documented.

Tolerances are upstream's, unchanged.

Run from anywhere:  python tests/gates/mhs/verify_hooke_consistency.py
"""
from __future__ import annotations

import sys

import numpy as np

from _common import MU, TRI, Report, relmax, shipped

FD_REL_STEP = 1e-3        # h0 = FD_REL_STEP * eps
EPS_OVER_L = 0.2
TOL_FD = 1e-7
TOL_FD_FORCE = 1e-9       # measured worst ~3.4e-11 over the 18 force cases
TOL_QUAD = 1e-6
TRIPWIRE = 1e-2
N_GAUSS = 40


def lame_lambda(mu, nu):
    """The one place this gate forms the quotient, to talk to the engine."""
    return 2.0 * mu * nu / (1.0 - 2.0 * nu)


def hooke(grad_u, mu, nu):
    """``sigma = lam tr(e) I + 2 mu e`` with ``e = sym(grad u)``."""
    e = 0.5 * (grad_u + np.swapaxes(grad_u, 1, 2))
    tr = np.einsum("nii->n", e)
    return (lame_lambda(mu, nu) * tr[:, None, None] * np.eye(3)[None]
            + 2.0 * mu * e)


def fd_gradient(disp, obs, h0):
    """Richardson-extrapolated central-difference gradient ``(N, 3, 3)``.

    ``(4 D(h/2) - D(h)) / 3`` cancels the leading truncation term, which is what
    buys the 1e-7 tolerance: a plain central difference at this step size would
    leave far more than the thing being measured.
    """
    obs = np.asarray(obs, float).reshape(-1, 3)
    N = obs.shape[0]
    eye = np.eye(3)
    pts = []
    for h in (h0, 0.5 * h0):
        for m in range(3):
            for sgn in (+1.0, -1.0):
                pts.append(obs + sgn * h * eye[m])
    pts = np.concatenate(pts, axis=0)                      # (12 N, 3)
    u = np.asarray(disp(pts), float).reshape(2, 3, 2, N, 3)
    D = np.empty((2, N, 3, 3))
    for s, h in enumerate((h0, 0.5 * h0)):
        for m in range(3):
            D[s, :, :, m] = (u[s, m, 0] - u[s, m, 1]) / (2.0 * h)
    return (4.0 * D[1] - D[0]) / 3.0


def quad_displacement(obs, tri, slip, order, mu, nu, eps, swapped=False,
                      n_gauss=N_GAUSS):
    """Displacement by Gauss quadrature of the point kernel.

    ``swapped=True`` puts lam on the normal-derivative term and mu on the
    divergence term -- the historical pre-fix contraction. The two differ only
    through ``lam != mu``, which is the point.
    """
    local_frame = shipped("frame").local_frame
    gauss_triangle = shipped("moments").gauss_triangle
    shape_functions = shipped("shape").shape_functions
    pw = shipped("pointwise")

    fr = local_frame(tri)
    obs = np.asarray(obs, float).reshape(-1, 3)
    x1, x2, w = gauss_triangle(n_gauss)
    y = ((1 - x1 - x2)[:, None] * fr.v[0] + x1[:, None] * fr.v[1]
         + x2[:, None] * fr.v[2])
    wq = w * 2.0 * fr.area
    Nq = shape_functions(tri, order, y)                    # (Q, K)
    sq = Nq @ slip                                         # (Q, 3)
    lam = lame_lambda(mu, nu)
    n = fr.nhat
    M, Q = obs.shape[0], y.shape[0]
    d = (obs[:, None, :] - y[None, :, :]).reshape(-1, 3)
    DG = pw.kelvin_dG(d, mu, nu, eps).reshape(M, Q, 3, 3, 3)
    c1, c2 = (lam, mu) if swapped else (mu, lam)
    t1 = c1 * np.einsum("k,oqijk->oqij", n, DG)            # n_m dG_ij/dx_m
    t2 = c2 * np.einsum("j,oqikk->oqij", n, DG)            # n_j dG_im/dx_m
    t3 = mu * np.einsum("k,oqikj->oqij", n, DG)            # n_m dG_im/dx_j
    U = -(t1 + t2 + t3)
    return np.einsum("oqij,q,qj->oi", U, wq, sq)


def main() -> bool:
    api = shipped("api")
    local_frame = shipped("frame").local_frame
    n_nodes = shipped("shape").n_nodes

    rep = Report("FD-Hooke consistency: C:sym(grad u) vs the stress kernel "
                 "(the lam/mu gate)")
    fr = local_frame(TRI)
    L = fr.L
    eps = EPS_OVER_L * L
    h0 = FD_REL_STEP * eps
    v1, v2, v3 = TRI
    obs = np.array([
        fr.centroid + 0.3 * L * fr.nhat + 0.15 * L * fr.e1 - 0.10 * L * fr.e2,
        fr.centroid - 0.3 * L * fr.nhat - 0.20 * L * fr.e1 + 0.25 * L * fr.e2,
        v1 + 0.3 * (v2 - v1) + 0.3 * (v3 - v1),              # on-plane, inside
    ])
    z, _ = fr.to_plane(obs)
    print(f"  L = {L:.4f}, eps = {eps:.4f}, h0 = {h0:.2e}, obs z/L = "
          + ", ".join(f"{zz / L:+.3f}" for zz in z))
    rng = np.random.default_rng(20260904)
    slips = {p: rng.standard_normal((n_nodes(p), 3)) for p in (0, 1, 2)}

    worst = 0.0
    for nu in (0.25, 0.30):
        for p in (0, 1, 2):
            slip = slips[p]
            grad = fd_gradient(
                lambda pts: api.displacement(pts, TRI, slip, MU, nu, eps),
                obs, h0)
            sig_fd = hooke(grad, MU, nu)
            sig_tot = api.stress(obs, TRI, slip, MU, nu, eps,
                                 subtract_eigenstress=False)
            for i, label in enumerate(("off-plane +z", "off-plane -z",
                                       "on-plane")):
                d = relmax(sig_fd[i], sig_tot[i])
                worst = max(worst, d)
                rep.check(f"nu={nu:.2f} p={p}: FD-Hooke vs TOTAL stress, "
                          f"{label}", d, TOL_FD)
            if nu == 0.30:
                sig_el = api.stress(obs, TRI, slip, MU, nu, eps)
                d_el = relmax(sig_fd[2], sig_el[2])
                rep.check_bool(
                    f"nu={nu:.2f} p={p}: tripwire, FD-Hooke vs ELASTIC "
                    f"on-plane > {TRIPWIRE:.0e}", d_el > TRIPWIRE,
                    f"(rel diff {d_el:.2e})")
    print(f"  worst FD-Hooke vs total stress: {worst:.3e}")

    # --- force element: no eigenstress anywhere -----------------------------
    forces = {p: rng.standard_normal((n_nodes(p), 3)) for p in (0, 1, 2)}
    worst_f = 0.0
    for nu in (0.25, 0.30):
        for p in (0, 1, 2):
            force = forces[p]
            sig_fd = hooke(fd_gradient(
                lambda pts: api.force_displacement(pts, TRI, force, MU, nu,
                                                   eps), obs, h0), MU, nu)
            sig_f = api.force_stress(obs, TRI, force, MU, nu, eps)
            for i, label in enumerate(("off-plane +z", "off-plane -z",
                                       "on-plane")):
                d = relmax(sig_fd[i], sig_f[i])
                worst_f = max(worst_f, d)
                rep.check(f"nu={nu:.2f} p={p}: FD-Hooke of force_displacement "
                          f"vs force_stress, {label}", d, TOL_FD_FORCE)
    print(f"  worst FD-Hooke vs force stress: {worst_f:.3e}")
    try:
        api.force_stress(obs, TRI, forces[0], MU, 0.30, eps,
                         subtract_eigenstress=False)
        rep.check_bool("force_stress rejects subtract_eigenstress "
                       "(a body force is not an eigenstrain)", False)
    except TypeError:
        rep.check_bool("force_stress rejects subtract_eigenstress "
                       "(a body force is not an eigenstrain)", True)

    # --- the lam/mu pairing demonstration -----------------------------------
    slip0 = slips[0]
    for nu in (0.25, 0.30):
        sig_tot = api.stress(obs, TRI, slip0, MU, nu, eps,
                             subtract_eigenstress=False)
        for swapped in (False, True):
            sig_fd = hooke(fd_gradient(
                lambda pts: quad_displacement(pts, TRI, slip0, 0, MU, nu, eps,
                                              swapped=swapped), obs, h0),
                MU, nu)
            d = relmax(sig_fd, sig_tot)
            if not swapped:
                rep.check(f"nu={nu:.2f} p=0: point-kernel quadrature, CORRECT "
                          f"pairing", d, TOL_QUAD)
            elif nu == 0.25:
                rep.check(f"nu={nu:.2f} p=0: lam/mu swap is INVISIBLE "
                          f"(lam == mu)", d, TOL_QUAD)
            else:
                rep.check_bool(f"nu={nu:.2f} p=0: lam/mu swap is CAUGHT "
                               f"(> {TRIPWIRE:.0e})", d > TRIPWIRE,
                               f"(rel diff {d:.2e})")
    return rep.finish()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
