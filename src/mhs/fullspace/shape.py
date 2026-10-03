"""Lagrange shape functions of order ``p`` on a triangle.

Nodes (``nodes(tri, p)``), in this fixed order:

    p = 0 : [centroid]                                    (K = 1)
    p = 1 : [v1, v2, v3]                                  (K = 3)
    p = 2 : [v1, v2, v3, m12, m23, m31]                   (K = 6)
    p >= 3: vertices, then the p-1 interior nodes of edges 12, 23, 31
            (from the first to the second vertex), then interior nodes.

A node with lattice multi-index ``alpha = (a1, a2, a3)``, ``|alpha| = p``, sits
at barycentric ``alpha / p`` and carries the Lagrange polynomial

    N_alpha(lam) = prod_i prod_{j=0}^{a_i - 1} (p lam_i - j) / (j + 1),

which is ``lam_i`` for p = 1, ``lam_i (2 lam_i - 1)`` and ``4 lam_i lam_j`` for
p = 2.  The slip on the triangle is ``s(y) = sum_k N_k(y) s_k`` with nodal
Cartesian slip vectors ``s_k``.

In the triangle's centroid frame (see :mod:`clq.frame`) the barycentric
coordinates are affine, ``lam_k = A_k + B_k eta_1 + C_k eta_2``; for an
observation point with in-plane coordinates ``X`` the kernel integrals use the
obs-centred coordinate ``xi = eta - X``, in which
``lam_k = (A_k + B_k X_1 + C_k X_2) + B_k xi_1 + C_k xi_2``.  Only the constant
term depends on the observation point.  :func:`shape_coefficients` returns the
monomial coefficients ``c_k[a, b]`` of ``N_k`` in ``xi`` for every observation
point, which is what the weighted moment tables consume.
"""
from __future__ import annotations

import numpy as np

from .frame import Frame, local_frame, as_triangle


# ---------------------------------------------------------------------------
# Node lattice
# ---------------------------------------------------------------------------

def n_nodes(order: int) -> int:
    return (order + 1) * (order + 2) // 2


def order_from_count(K: int) -> int:
    """Inverse of :func:`n_nodes`; raises if ``K`` is not triangular."""
    p = 0
    while n_nodes(p) < K:
        p += 1
    if n_nodes(p) != K:
        raise ValueError(f"{K} nodal values do not correspond to a Lagrange "
                         "order on a triangle (need 1, 3, 6, 10, ...)")
    return p


def lattice(order: int) -> np.ndarray:
    """Multi-indices ``alpha`` (K, 3) in the fixed node order."""
    p = order
    if p == 0:
        return np.array([[0, 0, 0]])
    out = []
    # vertices
    out += [[p, 0, 0], [0, p, 0], [0, 0, p]]
    # edge interior nodes, edges 12, 23, 31, from first to second vertex
    for (i, j) in ((0, 1), (1, 2), (2, 0)):
        for s in range(1, p):
            a = [0, 0, 0]
            a[i] = p - s
            a[j] = s
            out.append(a)
    # interior nodes (lexicographic in (a1, a2) with a3 = p - a1 - a2, all >= 1)
    for a1 in range(1, p):
        for a2 in range(1, p - a1):
            a3 = p - a1 - a2
            if a3 >= 1:
                out.append([a1, a2, a3])
    arr = np.array(out, dtype=int)
    assert arr.shape[0] == n_nodes(p)
    return arr


def nodes(tri, order: int) -> np.ndarray:
    """Node coordinates (K, 3)."""
    tri = as_triangle(tri)
    if order == 0:
        return tri.mean(axis=0, keepdims=True)
    lam = lattice(order) / order
    return lam @ tri


# ---------------------------------------------------------------------------
# Shape functions as polynomials
# ---------------------------------------------------------------------------

def _poly_mul(P, Q, deg):
    """Product of two coefficient arrays (..., deg+1, deg+1) truncated at ``deg``."""
    out = np.zeros_like(P)
    for a in range(deg + 1):
        for b in range(deg + 1 - a):
            if not np.any(P[..., a, b]):
                continue
            for c in range(deg + 1 - a):
                for d in range(deg + 1 - a - b - c):
                    out[..., a + c, b + d] += P[..., a, b] * Q[..., c, d]
    return out


def barycentric_affine(frame: Frame) -> np.ndarray:
    """Affine barycentric coefficients in the centroid frame: rows
    ``(A_k, B_k, C_k)`` with ``lam_k(eta) = A_k + B_k eta_1 + C_k eta_2``."""
    A = np.vstack([np.ones(3), frame.p.T])   # [1; p_x; p_y] per vertex column
    return np.linalg.inv(A)                   # (3, 3): row k = (A_k, B_k, C_k)


def shape_coefficients(frame: Frame, order: int, X: np.ndarray) -> np.ndarray:
    """Monomial coefficients of the shape functions in the obs-centred frame.

    Returns ``c`` of shape (N, K, order+1, order+1) with
    ``N_k(xi) = sum_{a+b<=order} c[n, k, a, b] xi_1^a xi_2^b`` for observation
    point ``n`` whose in-plane centroid-frame coordinates are ``X[n]``.
    """
    X = np.asarray(X, float).reshape(-1, 2)
    N = X.shape[0]
    p = order
    K = n_nodes(p)
    D = p + 1
    if p == 0:
        c = np.zeros((N, 1, 1, 1))
        c[:, 0, 0, 0] = 1.0
        return c
    abc = barycentric_affine(frame)                 # (3, 3)
    # affine lam_i(xi) for every obs: (N, 3, D, D) coefficient arrays
    lam = np.zeros((N, 3, D, D))
    lam[:, :, 0, 0] = abc[:, 0][None, :] + X[:, 0:1] * abc[:, 1][None, :] + X[:, 1:2] * abc[:, 2][None, :]
    lam[:, :, 1, 0] = abc[:, 1][None, :]
    lam[:, :, 0, 1] = abc[:, 2][None, :]
    c = np.zeros((N, K, D, D))
    for k, alpha in enumerate(lattice(p)):
        poly = np.zeros((N, D, D))
        poly[:, 0, 0] = 1.0
        for i in range(3):
            for j in range(int(alpha[i])):
                # factor (p lam_i - j) / (j + 1)
                fac = p * lam[:, i] / (j + 1)
                fac[:, 0, 0] -= j / (j + 1)
                poly = _poly_mul(poly, fac, p)
        c[:, k] = poly
    return c


def shape_functions(tri, order: int, points) -> np.ndarray:
    """Evaluate all shape functions at the in-plane projections of ``points``:
    returns (M, K)."""
    fr = local_frame(tri)
    _, X = fr.to_plane(points)
    c = shape_coefficients(fr, order, X)      # coefficients in xi about each point
    return c[:, :, 0, 0]                        # xi = 0 at the point itself


def interpolate(tri, slip_nodes, points) -> np.ndarray:
    """Interpolated slip (M, 3) at ``points`` from nodal values (K, 3)."""
    slip_nodes = np.asarray(slip_nodes, float)
    if slip_nodes.ndim == 1:
        slip_nodes = slip_nodes[None, :]
    p = order_from_count(slip_nodes.shape[0])
    return shape_functions(tri, p, points) @ slip_nodes


def nodal_values(tri, order: int, func) -> np.ndarray:
    """Sample a callable ``func(points (K,3)) -> (K,3)`` at the nodes."""
    pts = nodes(tri, order)
    vals = np.asarray(func(pts), float)
    if vals.shape != (pts.shape[0], 3):
        raise ValueError(f"func must return shape {(pts.shape[0], 3)}, got {vals.shape}")
    return vals


# ---------------------------------------------------------------------------
# Observation grids on the triangle
# ---------------------------------------------------------------------------

def barycentric_grid(n: int) -> np.ndarray:
    """Barycentric lattice points (M, 3) with ``n`` subdivisions per edge
    (includes the edges and vertices; M = (n+1)(n+2)/2)."""
    out = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            out.append([i, j, n - i - j])
    return np.array(out, float) / n


def triangle_grid(tri, n: int, offset: float = 0.0) -> np.ndarray:
    """Points (M, 3) on the triangle at the barycentric lattice with ``n``
    subdivisions per edge, displaced by ``offset`` along the unit normal."""
    tri = as_triangle(tri)
    lam = barycentric_grid(n)
    pts = lam @ tri
    if offset:
        fr = local_frame(tri)
        pts = pts + offset * fr.nhat
    return pts


def grid_triangles(n: int) -> np.ndarray:
    """Connectivity (T, 3) of the :func:`barycentric_grid` lattice for plotting."""
    idx = {}
    k = 0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            idx[(i, j)] = k
            k += 1
    tris = []
    for i in range(n):
        for j in range(n - i):
            tris.append([idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]])
            if j < n - i - 1:
                tris.append([idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]])
    return np.array(tris, int)
