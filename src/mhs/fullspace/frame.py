"""Triangle geometry: local frame, projection, helpers.

Conventions (shared by every module in ``clq``):

* a triangle is a ``(3, 3)`` array ``tri`` whose rows are the vertices
  ``v1, v2, v3`` (any consistent length unit);
* its unit normal is ``nhat = (v2 - v1) x (v3 - v1) / |...|`` -- the vertex
  ORDER defines the orientation, there is no separate normal argument;
* the in-plane basis is ``e1 = (v2 - v1)/|v2 - v1|``, ``e2 = nhat x e1``, so
  the vertices are counter-clockwise in the ``(e1, e2)`` plane;
* the frame origin is the centroid; ``project`` returns the signed height
  ``z`` of an observation point and its in-plane coordinates ``X`` relative
  to the centroid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import defaults


@dataclass(frozen=True)
class Frame:
    """Local orthonormal frame of one flat triangle."""
    v: np.ndarray          # (3, 3) vertices
    e1: np.ndarray         # (3,)
    e2: np.ndarray         # (3,)
    nhat: np.ndarray       # (3,)
    centroid: np.ndarray   # (3,)
    p: np.ndarray          # (3, 2) in-plane vertex coordinates relative to the centroid
    area: float
    L: float               # longest edge length

    def to_plane(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Signed height ``z`` (N,) and in-plane coords ``X`` (N, 2) of points."""
        d = np.asarray(points, float).reshape(-1, 3) - self.centroid
        z = d @ self.nhat
        X = np.stack([d @ self.e1, d @ self.e2], axis=1)
        return z, X

    def from_plane(self, X: np.ndarray, z=0.0) -> np.ndarray:
        """Inverse of :meth:`to_plane`: (N, 2) in-plane coords -> (N, 3) points."""
        X = np.asarray(X, float).reshape(-1, 2)
        z = np.broadcast_to(np.asarray(z, float), (X.shape[0],))
        return (self.centroid + X[:, :1] * self.e1 + X[:, 1:] * self.e2
                + z[:, None] * self.nhat)


def as_triangle(tri) -> np.ndarray:
    tri = np.asarray(tri, dtype=float)
    if tri.shape != (3, 3):
        raise ValueError(f"tri must have shape (3, 3), got {tri.shape}")
    return tri


def unit_normal(tri) -> np.ndarray:
    tri = as_triangle(tri)
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    nn = np.linalg.norm(n)
    if nn == 0.0:
        raise ValueError("degenerate triangle (zero area)")
    return n / nn


def local_frame(tri) -> Frame:
    """Build the :class:`Frame` of ``tri``; raises on a degenerate triangle."""
    tri = as_triangle(tri)
    e1v = tri[1] - tri[0]
    e2v = tri[2] - tri[0]
    n = np.cross(e1v, e2v)
    area2 = np.linalg.norm(n)
    edges = np.array([np.linalg.norm(tri[1] - tri[0]),
                      np.linalg.norm(tri[2] - tri[1]),
                      np.linalg.norm(tri[0] - tri[2])])
    L = float(edges.max())
    if L == 0.0 or 0.5 * area2 < defaults.DEGENERATE_AREA_REL * L * L:
        raise ValueError("degenerate triangle (zero or near-zero area)")
    nhat = n / area2
    e1 = e1v / np.linalg.norm(e1v)
    e2 = np.cross(nhat, e1)
    centroid = tri.mean(axis=0)
    d = tri - centroid
    p = np.stack([d @ e1, d @ e2], axis=1)
    return Frame(v=tri, e1=e1, e2=e2, nhat=nhat, centroid=centroid, p=p,
                 area=0.5 * area2, L=L)


def equilateral(L: float = 1.0, center=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Equilateral triangle of edge ``L`` in the z = 0 plane, centroid at
    ``center``, counter-clockwise seen from +z (so ``nhat = +z``).  Vertex 1 is
    at the top (+y), vertices 2 and 3 at lower-left and lower-right."""
    h = L * np.sqrt(3.0) / 2.0
    cy = h / 3.0
    c = np.asarray(center, float)
    return np.array([[0.0, h - cy, 0.0],
                     [-L / 2.0, -cy, 0.0],
                     [L / 2.0, -cy, 0.0]]) + c


def barycentric(tri, points) -> np.ndarray:
    """Barycentric coordinates (N, 3) of the in-plane projections of points."""
    fr = local_frame(tri)
    _, X = fr.to_plane(points)
    A = np.vstack([np.ones(3), fr.p.T])          # (3, 3): [1; p_x; p_y] columns per vertex
    rhs = np.vstack([np.ones(X.shape[0]), X.T])  # (3, N)
    return np.linalg.solve(A, rhs).T


def inside(tri, points, margin: float = 0.0) -> np.ndarray:
    """Boolean mask (N,): in-plane projection lies inside ``tri`` with every
    edge at least ``margin`` away (signed distance to each edge line >= margin)."""
    fr = local_frame(tri)
    _, X = fr.to_plane(points)
    ok = np.ones(X.shape[0], dtype=bool)
    for k in range(3):
        a, b = fr.p[k], fr.p[(k + 1) % 3]
        t = b - a
        t = t / np.linalg.norm(t)
        n_in = np.array([-t[1], t[0]])            # inward normal for CCW vertices
        dist = (X - a) @ n_in
        ok &= dist >= margin
    return ok
