"""The shared-node table for CONTINUOUS quadratic slip on a triangulated fault.

WHY A SECOND LAYOUT. The nodal layouts of :mod:`mhs.matrices` are
discontinuous: every element carries its own ``K`` nodal values, a vertex
shared by six elements holds six independent slips, and the collocation points
are pulled toward the centroids because a raw node sits on the element's own
boundary. That layout is right for a density that may jump between elements.
Slip on a fault does not jump between elements, and letting it costs the
on-fault stress its convergence: a P0 slip is a staircase whose risers each
carry a line dislocation, so the elastic traction read between the centroids
has a floor set by the staircase and not by ``eps`` (the on-fault page quotes
it: 8 % of the stress drop in the facet cores at ``h = a/8``, both kernels,
and refinement does not move it).

The continuous layout welds the vertices and adds one node per edge, so a
fault with ``V`` vertices and ``E`` edges carries ``V + E`` quadratic Lagrange
nodes and ONE Cartesian slip vector at each -- the slip is C0 across every
edge, and what is left at an edge is a kink in its gradient, which is what
mollification bounds. The collocation points are the nodes themselves,
unshrunk: a kink is admissible at a mollified collocation point where a jump
is not. Measured in the sibling full-space package on the penny crack, the
nodal readout of a prescribed Eshelby slip converges at 1.6-1.8 eps/a on every
mesh where the P0 centroid readout is floored, and the stress-drop solve
recovers the slip to 1.6 % of its peak at ``a/h = 8`` against 4.8 % for P0.

THE TABLE IS GEOMETRY ONLY. It is built from the vertex coordinates, welded
by rounding to a tolerance -- a tolerance is required, not optional, because a
tiling reaches the same point by different arithmetic paths and the last bit
differs -- and carries the node coordinates, the per-element node ids in the
order the kernels expect (``v1 v2 v3 m12 m23 m31``, :mod:`mhs.fullspace.shape`),
the node normals, the boundary mask and the integration weights. Nothing about
the kernels, the material or ``eps`` enters it, so one table serves every
matrix built on the mesh.

WHAT IS REFUSED, because each would be silently wrong rather than loudly:

* a non-manifold tiling (an edge in three triangles): a node there has no
  single tangent plane;
* inconsistent winding (an interior edge traversed the same way by both
  triangles): a Cartesian slip at the shared node would then mean opposite
  jumps on the two sides, since each element reads the jump against its own
  vertex-order normal;
* a soup (several triangles sharing nothing): the discontinuous layout is the
  one for that.

``groups`` keeps faults apart. A network passes its ``fault_id`` so that two
faults meeting along a line -- a splay rooted on its host, whose junction
vertices the two tilings share bit for bit -- get SEPARATE nodes there: the
host's slip is continuous across the junction and the splay's is its own, and
welding them into one vector would tie the two together.

THE NODE NORMAL is the area-weighted mean of the incident elements' upward
normals (:func:`mhs.tdcs.unit_normals`), renormalised, so that on a faceted
fault the slip vector at a node is a VECTOR rather than a magnitude times a
per-facet direction. That distinction is measured, not stylistic: a per-facet
direction with a continuous magnitude leaves a line dislocation at every
crease, with a ``1/eps`` traction core at its irregular vertices; the nodal
vector removes that term. A log-type crease term remains, and on a rough
fault the normal traction within a facet of a crease is set by how the kink
is regularised -- read it at scales of two elements or more.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from . import defaults, tdcs
from .fullspace.shape import n_nodes

#: The one order the continuous layout exists for. Continuous P1 was measured
#: 15-22 % too stiff on the penny crack (its kink sits at the collocation
#: vertex itself, with no interior node to carry the curvature) and is refused;
#: P0 has no shared nodes to be continuous at.
CONTINUOUS_ORDER = 2

LAYOUTS = ("discontinuous", "continuous")


@dataclasses.dataclass(frozen=True)
class SharedNodes:
    """The nodes of a continuous P2 slip on one triangulated surface.

    ``coords`` (n, 3) are the vertices first (``n_vertices`` of them) then
    the edge midpoints; ``ids`` (n_tri, 6) names each element's six nodes in
    the kernels' order; ``normals`` (n, 3) point UP, as every frame in this
    package does; ``boundary`` marks the rim (the nodes of edges in one
    triangle only); ``weights`` (n,) integrate a nodal field exactly for a
    quadratic (the three-midpoint rule, A/3 at each midpoint, nothing at a
    vertex); ``group`` (n,) is the group each node was welded within.
    """

    coords: np.ndarray
    ids: np.ndarray
    normals: np.ndarray
    boundary: np.ndarray
    weights: np.ndarray
    group: np.ndarray
    n_vertices: int
    tol: float

    @property
    def n(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_tri(self) -> int:
        return int(self.ids.shape[0])

    @property
    def n_edges(self) -> int:
        return self.n - self.n_vertices

    @property
    def frames(self) -> np.ndarray:
        """``(n, 3, 3)`` with rows ``[strike, dip, tensile]`` at every node:
        cutde's frame built on the node normal, from :mod:`mhs.tdcs`."""
        return tdcs.frame_from_normals(self.normals)

    @property
    def trace(self) -> np.ndarray:
        """Boundary nodes ON the free surface ``z = 0``: a surface trace.

        These are the boundary nodes a solve may leave free -- the slip need
        not vanish where a fault breaks the surface -- where :attr:`front`
        names the ones it would normally lock."""
        return self.boundary & (self.coords[:, 2] >= -self.tol)

    @property
    def front(self) -> np.ndarray:
        """Boundary nodes inside the body: crack fronts, and a splay's rooted
        edge. The slip of a crack vanishes at its front, so a solve locks these
        by default; a caller with a reason overrides it."""
        return self.boundary & ~self.trace

    def expand(self, values: np.ndarray) -> np.ndarray:
        """Nodal values ``(n, ...)`` -> the element-major density
        ``(n_tri * 6, ...)`` the matrix functions' source axis is indexed by,
        node index fastest. This is the whole bridge between the two
        layouts: a continuous field is a discontinuous one whose shared nodes
        happen to agree, so every kernel and every ``order=2`` entry point
        consumes the expanded form unchanged."""
        v = np.asarray(values)
        if v.shape[0] != self.n:
            raise ValueError(f"expected {self.n} nodal values, got {v.shape[0]}")
        return v[self.ids].reshape((self.n_tri * n_nodes(CONTINUOUS_ORDER),)
                                   + v.shape[1:])

    def integrate(self, values: np.ndarray) -> np.ndarray:
        """``int values dS`` over the surface, exact for a quadratic field:
        the moment of a nodal slip is ``integrate(slip)``."""
        v = np.asarray(values, float)
        if v.shape[0] != self.n:
            raise ValueError(f"expected {self.n} nodal values, got {v.shape[0]}")
        return np.tensordot(self.weights, v, axes=(0, 0))

    def element_mean(self, values: np.ndarray) -> np.ndarray:
        """``(n_tri, ...)`` the mean of a nodal field over each element, by
        the same three-midpoint rule -- one value per element from a
        continuous field, for a per-element readout."""
        v = np.asarray(values, float)
        return v[self.ids[:, 3:]].mean(axis=1)


def _edges(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(edges (E, 2) sorted, edge_of (n_tri, 3), counts (E,))`` for the
    local edges 12, 23, 31 of every triangle."""
    local = np.stack([idx[:, [0, 1]], idx[:, [1, 2]], idx[:, [2, 0]]], axis=1)
    flat = np.sort(local.reshape(-1, 2), axis=1)
    edges, inv, counts = np.unique(flat, axis=0, return_inverse=True,
                                   return_counts=True)
    return edges, inv.reshape(-1, 3), counts


def shared_nodes(tris, *, groups=None,
                 tol_over_h: float = defaults.WELD_TOL_OVER_H) -> SharedNodes:
    """Build the :class:`SharedNodes` table of ``tris`` ``(n_tri, 3, 3)``.

    Vertices are welded when they round to the same point at
    ``tol_over_h`` times the shortest edge, within a group; ``groups``
    ``(n_tri,)`` keeps nodes of different faults apart (a network passes its
    ``fault_id``). See the module docstring for what is refused and why.
    """
    tri = np.ascontiguousarray(np.asarray(tris, dtype=np.float64))
    if tri.ndim != 3 or tri.shape[1:] != (3, 3) or tri.shape[0] == 0:
        raise ValueError(f"tris must have shape (n_tri, 3, 3), got {tri.shape}")
    if not np.all(np.isfinite(tri)):
        raise ValueError("tris must be finite")
    n_tri = tri.shape[0]
    grp = (np.zeros(n_tri, dtype=np.int64) if groups is None
           else np.asarray(groups).reshape(-1).astype(np.int64))
    if grp.shape[0] != n_tri:
        raise ValueError(f"groups has {grp.shape[0]} entries for {n_tri} "
                         f"triangles")
    edge_len = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2)
    h_min = float(edge_len.min())
    if not h_min > 0.0:
        raise ValueError("a triangle has a zero-length edge")
    tol = float(tol_over_h) * h_min

    # weld: round within a group, the group id as a fourth key so that two
    # faults sharing a line keep separate nodes on it
    flat = tri.reshape(-1, 3)
    keys = np.concatenate([np.round(flat / tol).astype(np.int64),
                           np.repeat(grp, 3)[:, None]], axis=1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n_v = uniq.shape[0]
    verts = np.zeros((n_v, 3))
    np.add.at(verts, inv, flat)
    verts /= np.bincount(inv, minlength=n_v)[:, None]
    idx = inv.reshape(-1, 3)
    if np.any(idx[:, 0] == idx[:, 1]) or np.any(idx[:, 1] == idx[:, 2]) \
            or np.any(idx[:, 2] == idx[:, 0]):
        raise ValueError(f"a triangle has two vertices within the weld "
                         f"tolerance {tol:.3e}: degenerate, or tol_over_h too "
                         f"loose for this mesh")

    edges, edge_of, counts = _edges(idx)
    if np.any(counts > 2):
        bad = int(np.argmax(counts))
        raise ValueError(
            f"non-manifold: edge {edges[bad].tolist()} is shared by "
            f"{int(counts[bad])} triangles, so a node on it has no single "
            f"tangent plane. The continuous layout needs a manifold tiling.")
    if n_tri > 1 and not np.any(counts == 2):
        raise ValueError(
            f"nothing is shared: {n_tri} triangles and no common edge. A "
            f"soup of separate elements wants the discontinuous layout; the "
            f"continuous one welds vertices at {tol:.3e} (tol_over_h "
            f"{tol_over_h:g} x the shortest edge {h_min:.3e}).")
    # orientation: an interior edge must be traversed once each way
    local = np.stack([idx[:, [0, 1]], idx[:, [1, 2]], idx[:, [2, 0]]],
                     axis=1).reshape(-1, 2)
    forward = local[:, 0] < local[:, 1]
    e_flat = edge_of.reshape(-1)
    n_forward = np.bincount(e_flat, weights=forward, minlength=edges.shape[0])
    inconsistent = (counts == 2) & (n_forward != 1)
    if np.any(inconsistent):
        bad = int(np.argmax(inconsistent))
        raise ValueError(
            f"inconsistent winding: the two triangles on edge "
            f"{edges[bad].tolist()} traverse it the same way, so their "
            f"vertex-order normals disagree and one Cartesian slip at a shared "
            f"node would mean opposite jumps on the two sides. Wind the mesh "
            f"consistently (every normal up is the usual choice).")

    n_e = edges.shape[0]
    mids = 0.5 * (verts[edges[:, 0]] + verts[edges[:, 1]])
    coords = np.concatenate([verts, mids])
    ids = np.concatenate([idx, n_v + edge_of], axis=1)       # v1 v2 v3 m12 m23 m31

    boundary = np.zeros(n_v + n_e, bool)
    rim = counts == 1
    boundary[n_v + np.flatnonzero(rim)] = True
    boundary[edges[rim].reshape(-1)] = True

    nrm_el = tdcs.unit_normals(tri)                          # upward
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                         tri[:, 2] - tri[:, 0]), axis=1)
    normals = np.zeros((n_v + n_e, 3))
    np.add.at(normals, ids.reshape(-1),
              np.repeat(area[:, None] * nrm_el, 6, axis=0))
    mag = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= mag
    weights = np.zeros(n_v + n_e)
    np.add.at(weights, ids[:, 3:].reshape(-1), np.repeat(area / 3.0, 3))
    group = np.zeros(n_v + n_e, dtype=np.int64)
    group[ids.reshape(-1)] = np.repeat(grp, 6)

    return SharedNodes(coords=coords, ids=ids, normals=normals,
                       boundary=boundary, weights=weights, group=group,
                       n_vertices=int(n_v), tol=tol)


def check_layout(layout: str) -> str:
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    return layout
