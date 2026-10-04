"""mhs -- mollified Mindlin half-space triangular-dislocation Green's functions.

A forward Green's function library for a **homogeneous half space with no
topography**. In that configuration nothing is solved: the free surface lives in
the kernel, the half space is laterally and vertically infinite so there are no
sides or base, there is no material interface, and fault slip is prescribed data.
No linear system, no collocation, no free term, no boundary density. The error
budget is the kernel's own mollification residual and the per-triangle
integration, and nothing else.

What it gives you that a classical triangular dislocation element does not: a
fault of finite width ``eps``, and an **interpretable stress on and near the
fault**, because the eigenstress ``C:eps*`` that every mollified double layer
carries is subtracted in the readout. ``stress_matrix`` returns the elastic
stress by default for that reason.

Import is deliberately cheap -- numpy and numba only, no sympy -- and a gate
enforces it.

    import mhs
    mat = mhs.Material(mu=30.0, lam=30.0)          # (mu, lam), never nu
    G = mhs.disp_matrix(obs, tris, mat, eps=0.1)   # (n_obs, 3, n_src, 3)
    u = np.einsum("oisk,sk->oi", G, slip)          # slip is CARTESIAN

and for on-fault stress interaction, the object cycle models actually want,
assembled directly because the stress matrix it would be contracted out of
cannot be allocated at production size:

    K = mhs.interaction_matrix(tris, mat, eps=0.1,
                               receiver="strike", source="strike")   # (n, n)
"""
from . import chunking, defaults, tdcs
from .materials import Material
from .matrices import (
    disp_matrix,
    eigenstress_matrix,
    elastic_strain_matrix,
    interaction_matrix,
    stress_matrix,
    total_stress_matrix,
    traction_matrix,
)

__all__ = [
    "Material",
    "disp_matrix",
    "stress_matrix",
    "traction_matrix",
    "interaction_matrix",
    "total_stress_matrix",
    "eigenstress_matrix",
    "elastic_strain_matrix",
    "chunking",
    "defaults",
    "tdcs",
]

__version__ = "0.1.0"
