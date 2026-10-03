"""clq -- closed-form mollified dislocation and force kernels for constant,
linear and quadratic nodal density on an arbitrary flat triangle (full space,
Cortez regularisation R = sqrt(r^2 + eps^2)).

    import clq
    u     = clq.displacement(obs, tri, slip, mu, nu, eps)   # (N, 3)
    sigma = clq.stress(obs, tri, slip, mu, nu, eps)         # (N, 3, 3), elastic
    inf   = clq.influence(obs, tri, mu, nu, eps, order=2)   # nodal tensors

A slip (displacement-discontinuity) source gives ``U``/``H``/``E``; a force
(Kelvin single-layer) source -- a force per unit AREA on the triangle -- gives
``G``/``S``:

    u     = clq.force_displacement(obs, tri, force, mu, nu, eps)   # (N, 3)
    sigma = clq.force_stress(obs, tri, force, mu, nu, eps)         # (N, 3, 3)

``slip``/``force`` has shape (1, 3), (3, 3) or (6, 3) for constant / linear /
quadratic density (nodal values at ``clq.nodes(tri, order)``); a (3,) vector is
constant, and higher Lagrange orders (10, 15, ... rows) run as well.
"""
from .api import (Influence, influence, displacement, stress, eigenstress,
                  traction, force_displacement, force_stress)
from .frame import equilateral, inside, local_frame, unit_normal, barycentric
from .shape import (nodes, nodal_values, shape_functions, interpolate,
                    barycentric_grid, triangle_grid, grid_triangles, n_nodes,
                    order_from_count)

__version__ = "0.1.0"
__all__ = [
    "Influence", "influence", "displacement", "stress", "eigenstress", "traction",
    "force_displacement", "force_stress",
    "equilateral", "inside", "local_frame", "unit_normal", "barycentric",
    "nodes", "nodal_values", "shape_functions", "interpolate",
    "barycentric_grid", "triangle_grid", "grid_triangles", "n_nodes", "order_from_count",
]
