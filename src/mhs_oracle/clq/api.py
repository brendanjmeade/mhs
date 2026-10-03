"""Public API: one call signature for constant, linear and quadratic density,
for a dislocation (slip) source and for a force (Kelvin single-layer) source.

    displacement(obs, tri, slip, mu, nu, eps)       -> (N, 3)
    stress(obs, tri, slip, mu, nu, eps)             -> (N, 3, 3)   elastic (default)
    eigenstress(obs, tri, slip, mu, nu, eps)        -> (N, 3, 3)   C:eps*
    force_displacement(obs, tri, force, mu, nu, eps)-> (N, 3)
    force_stress(obs, tri, force, mu, nu, eps)      -> (N, 3, 3)   elastic
    influence(obs, tri, mu, nu, eps, order=p)       -> nodal tensors (BEM building block)

Conventions
-----------
* ``tri`` : (3, 3) vertices; the unit normal is ``(v2-v1) x (v3-v1)`` normalised.
* ``slip``: (K, 3) nodal Cartesian slip vectors with K = 1, 3, 6 (constant,
  linear, quadratic; nodes as in :func:`clq.shape.nodes`); a (3,) vector is
  a constant slip.  The order is inferred from K unless ``order=`` is given
  (then it must agree).
* ``force``: the same nodal layout, but a force per unit AREA on the triangle.
  Equilibrium is ``div sigma + f phi_eps = 0``, so a closed surface around the
  element carries ``int sigma . nhat dS = -int f dS``.  The force element has
  no ``nhat`` dependence and no eigenstress (see below).
* Slip sign: ``Delta u = u(+nhat side) - u(-nhat side)``.
* Units: any consistent set; ``mu`` sets the stress unit.  ``eps`` is a scalar
  >= 0.  ``eps = 0`` requires observation points off the plane, EXCEPT for
  ``force_displacement`` / ``want=("G",)``: the single layer is weakly singular,
  so it is evaluated on the element too.
* Stress: the kernel returns the TOTAL stress ``C:(eps_el + eps*)`` of the
  mollified dislocation; ``stress`` subtracts the exact eigenstress ``C:eps*``
  of the smeared slip by default (``subtract_eigenstress=True``), returning the
  elastic stress.  ``subtract_eigenstress=False`` gives the raw total (kernel
  diagnostics only).  ``force_stress`` has NO such argument: a mollified body
  force is a genuine body force, not an eigenstrain, so it is already the
  elastic stress.
* ``far_field``: "hybrid" (default and the safe choice: closed form within
  D_STAR * L of the centroid, Gauss quadrature beyond), "analytic" (closed
  form everywhere; loses digits with distance, ~2e-4 relative at 100 L and
  O(1) by 500 L for quadratic slip, so do not use it beyond ~20 L), or
  "quadrature" -- see :mod:`clq.moments` and the README "Far field" note.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .frame import as_triangle, unit_normal
from .kernels import nodal_influence
from .shape import order_from_count, n_nodes, nodes


class Influence(NamedTuple):
    U: np.ndarray | None      # (N, K, 3, 3)     slip -> displacement
    H: np.ndarray | None      # (N, K, 3, 3, 3)  slip -> TOTAL stress
    E: np.ndarray | None      # (N, K)           eigenstress weight
    nodes: np.ndarray         # (K, 3)
    order: int
    G: np.ndarray | None = None   # (N, K, 3, 3)     force density -> displacement
    S: np.ndarray | None = None   # (N, K, 3, 3, 3)  force density -> stress


def _check_material(mu, nu, eps):
    if not (mu > 0.0):
        raise ValueError("mu must be > 0")
    if not (-1.0 < nu < 0.5):
        raise ValueError("nu must be in (-1, 1/2)")
    eps = float(np.asarray(eps))
    if eps < 0.0:
        raise ValueError("eps must be >= 0")
    return eps


def _nodal_and_order(values, order, name="slip"):
    values = np.asarray(values, float)
    if values.ndim == 1:
        if values.shape != (3,):
            raise ValueError(f"a 1-D {name} must have 3 components")
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"{name} must have shape (K, 3), got {values.shape}")
    try:
        p = order_from_count(values.shape[0])
    except ValueError as exc:                       # name the density in the message
        raise ValueError(f"{name}: {exc}") from None
    if order is not None and int(order) != p:
        raise ValueError(f"order={order} disagrees with {values.shape[0]} nodal "
                         f"{name} values (order {p})")
    return values, p


def influence(obs, tri, mu, nu, eps, *, order=0, want=("U", "H", "E"),
              far_field="hybrid") -> Influence:
    """Nodal influence tensors of one triangle at the observation points.

    ``U[n, k, i, j]``: displacement component i at ``obs[n]`` per unit slip
    component j at node k; ``H[n, k, m, l, j]``: TOTAL stress ml per unit slip
    component j at node k; ``E[n, k]``: eigenstress weight, with
    ``C:eps* = sum_k E[n, k] [lam (s_k . nhat) I + mu (s_k nhat^T + nhat s_k^T)]``.
    A BEM block for the displacement is ``U.transpose(0, 2, 1, 3).reshape(3N, 3K)``.
    """
    eps = _check_material(mu, nu, eps)
    tri = as_triangle(tri)
    res = nodal_influence(obs, tri, int(order), mu, nu, eps, want=want, far_field=far_field)
    return Influence(U=res.get("U"), H=res.get("H"), E=res.get("E"),
                     nodes=res["nodes"], order=int(order),
                     G=res.get("G"), S=res.get("S"))


def _squeeze(out, obs):
    return out[0] if np.asarray(obs).ndim == 1 else out


def displacement(obs, tri, slip, mu, nu, eps, *, order=None, far_field="hybrid"):
    """Displacement (N, 3) at ``obs`` from the nodal slip on ``tri``."""
    slip, p = _nodal_and_order(slip, order)
    inf = influence(obs, tri, mu, nu, eps, order=p, want=("U",), far_field=far_field)
    u = np.einsum("nkij,kj->ni", inf.U, slip)
    return _squeeze(u, obs)


def eigenstress(obs, tri, slip, mu, nu, eps, *, order=None, far_field="hybrid"):
    """Eigenstress ``C:eps*`` (N, 3, 3) of the smeared slip (the anelastic term)."""
    slip, p = _nodal_and_order(slip, order)
    inf = influence(obs, tri, mu, nu, eps, order=p, want=("E",), far_field=far_field)
    return _squeeze(_eigenstress_from_weights(inf.E, slip, tri, mu, nu), obs)


def _eigenstress_from_weights(E, slip, tri, mu, nu):
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    n = unit_normal(tri)
    sym = 0.5 * (np.einsum("ki,j->kij", slip, n) + np.einsum("kj,i->kij", slip, n))   # (K,3,3)
    sig_k = lam * np.einsum("kii->k", sym)[:, None, None] * np.eye(3)[None] + 2.0 * mu * sym
    return np.einsum("nk,kij->nij", E, sig_k)


def stress(obs, tri, slip, mu, nu, eps, *, order=None, subtract_eigenstress=True,
           far_field="hybrid"):
    """Stress (N, 3, 3) at ``obs``: elastic (default) or total."""
    slip, p = _nodal_and_order(slip, order)
    want = ("H", "E") if subtract_eigenstress else ("H",)
    inf = influence(obs, tri, mu, nu, eps, order=p, want=want, far_field=far_field)
    sig = np.einsum("nkmlj,kj->nml", inf.H, slip)
    if subtract_eigenstress:
        sig = sig - _eigenstress_from_weights(inf.E, slip, tri, mu, nu)
    return _squeeze(sig, obs)


def force_displacement(obs, tri, force, mu, nu, eps, *, order=None,
                       far_field="hybrid"):
    """Displacement (N, 3) at ``obs`` from a nodal FORCE density on ``tri``.

    ``force`` is (K, 3) nodal force per unit AREA (or (3,) for a uniform
    density), interpolated by the same Lagrange shape functions as slip:
    ``u_i = sum_k G[n,k,i,j] f[k,j]``.  The sign convention is the Kelvin one,
    ``div sigma + f phi_eps = 0``, so a closed surface around the element
    carries ``int sigma . nhat dS = -int f dS``.
    """
    force, p = _nodal_and_order(force, order, name="force")
    inf = influence(obs, tri, mu, nu, eps, order=p, want=("G",), far_field=far_field)
    u = np.einsum("nkij,kj->ni", inf.G, force)
    return _squeeze(u, obs)


def force_stress(obs, tri, force, mu, nu, eps, *, order=None, far_field="hybrid"):
    """Stress (N, 3, 3) at ``obs`` from a nodal FORCE density on ``tri``.

    There is deliberately no ``subtract_eigenstress`` option: a mollified body
    force is a genuine body force, not an eigenstrain, so this is already the
    elastic stress and there is nothing to subtract (unlike :func:`stress`).
    """
    force, p = _nodal_and_order(force, order, name="force")
    inf = influence(obs, tri, mu, nu, eps, order=p, want=("S",), far_field=far_field)
    sig = np.einsum("nkijc,kc->nij", inf.S, force)
    return _squeeze(sig, obs)


def traction(sigma, tri):
    """Traction vector(s) ``sigma . nhat`` on the triangle's plane."""
    n = unit_normal(tri)
    sigma = np.asarray(sigma, float)
    return sigma @ n
