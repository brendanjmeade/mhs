"""Numerical oracle: Gauss product-rule integration of the point kernels times
the shape functions over the triangle (slow, for verification only)."""
from __future__ import annotations

import numpy as np

from .frame import local_frame
from .moments import gauss_triangle
from .shape import shape_functions, nodes
from . import pointwise as pw


def quadrature_influence(obs, tri, order, mu, nu, eps, n_gauss=40, want=("U", "H", "E")):
    """Same output as :func:`clq.kernels.nodal_influence`, by quadrature."""
    fr = local_frame(tri)
    obs = np.asarray(obs, float).reshape(-1, 3)
    x1, x2, w = gauss_triangle(n_gauss)
    y = (1 - x1 - x2)[:, None] * fr.v[0] + x1[:, None] * fr.v[1] + x2[:, None] * fr.v[2]
    wq = w * 2.0 * fr.area
    Nq = shape_functions(tri, order, y)               # (Q, K)
    K = Nq.shape[1]
    out = {"nodes": nodes(tri, order)}
    N = obs.shape[0]
    if "U" in want:
        U = np.zeros((N, K, 3, 3))
    if "H" in want:
        H = np.zeros((N, K, 3, 3, 3))
    if "E" in want:
        E = np.zeros((N, K))
    if "G" in want:
        G = np.zeros((N, K, 3, 3))
    if "S" in want:
        S = np.zeros((N, K, 3, 3, 3))
    for q in range(y.shape[0]):
        d = obs - y[q]
        wk = wq[q] * Nq[q]                           # (K,)
        if "U" in want:
            U += np.einsum("k,nij->nkij", wk, pw.dd_displacement_point(d, fr.nhat, mu, nu, eps))
        if "H" in want:
            H += np.einsum("k,nmlj->nkmlj", wk, pw.dd_stress_point(d, fr.nhat, mu, nu, eps))
        if "E" in want:
            E += np.einsum("k,n->nk", wk, pw.blob(d, eps))
        if "G" in want:
            G += np.einsum("k,nij->nkij", wk, pw.kelvin_G(d, mu, nu, eps))
        if "S" in want:
            S += np.einsum("k,nijc->nkijc", wk, pw.force_stress_point(d, mu, nu, eps))
    if "U" in want:
        out["U"] = U
    if "H" in want:
        out["H"] = H
    if "E" in want:
        out["E"] = E
    if "G" in want:
        out["G"] = G
    if "S" in want:
        out["S"] = S
    return out
