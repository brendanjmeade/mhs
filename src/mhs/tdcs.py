"""The per-triangle (strike, dip, tensile) frame -- cutde's convention, once.

WHY THIS IS A MODULE AND NOT A DOCSTRING. The matrix functions take CARTESIAN
slip, deliberately: cutde's dip direction points *up*, and a convention
restated in each caller is a convention restated wrongly in one of them. So the
rotation lives here, in one function, and ``verify_cutde_limit`` reads it from
here rather than carrying its own copy -- which is the only way the external
anchor actually anchors the shipped convention.

THE FRAME, as cutde builds it:

    tensile = unit normal, oriented UPWARD (``n_z >= 0``)
    strike  = (-n_y, n_x, 0), normalised -- horizontal, along the surface trace
    dip     = tensile x strike

so ``dip`` has a positive vertical component: for a dipping fault it points up
the dip, not down it. That is cutde's choice and this package matches it rather
than correcting it, because the value of matching an external reference is that
the comparison is meaningful.

THE HORIZONTAL DEGENERACY is real and handled rather than left to produce NaN.
For a horizontal element ``n = (0, 0, +-1)``, so ``(-n_y, n_x, 0)`` is the zero
vector and there is no surface trace to take a strike from. The frame is then
fixed to ``strike = (1, 0, 0)``, ``dip = (0, 1, 0)`` -- an arbitrary but STATED
choice, because horizontal elements occur (a detachment, a sill) and silently
returning NaN for them would be worse than picking an axis.

NOTE ON THE NORMAL'S SIGN. This frame's ``tensile`` is the UPWARD normal, which
is not necessarily the normal implied by a triangle's stored vertex order --
that one sets which side of the surface the slip jump refers to and ``mhs`` does
not flip it (see ``assemble._unit_normal``). The two agree up to a sign, and
that sign is exactly the convention cutde fixes by orienting upward. Use
:func:`slip_frame` to talk to cutde or to read a fault database; use the mesh's
own normal when the question is which way the jump goes.
"""
from __future__ import annotations

import numpy as np

#: Order of the frame's rows, and of the slip components that pair with them.
COMPONENTS = ("strike", "dip", "tensile")

#: Below this the in-plane part of the normal is treated as zero and the
#: horizontal-element frame is used. The quantity is |(-n_y, n_x, 0)| for a
#: UNIT normal, so this is a direction tolerance and not a length.
FLAT_TOL = 1e-12


def unit_normals(tris: np.ndarray) -> np.ndarray:
    """Upward-oriented unit normals, ``(n, 3)``, from the vertices alone.

    Independent of anything else in the package: built from the cross product
    and then flipped to ``n_z >= 0``, which is what makes it cutde's normal
    rather than the mesh's.
    """
    tris = np.asarray(tris, float).reshape(-1, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    if not np.all(norm > 0.0):
        bad = int(np.argmin(norm))
        raise ValueError(
            f"tris[{bad}] is degenerate: its vertices are collinear, so it has "
            f"no normal and no frame. Zero-area triangles have to be removed "
            f"by the mesh, not absorbed here.")
    n = n / norm
    return np.where(n[:, 2:3] < 0.0, -n, n)


def slip_frame(tris: np.ndarray) -> np.ndarray:
    """``(n, 3, 3)`` with ROWS ``[strike, dip, tensile]`` in Cartesian xyz.

    So ``frame[s] @ v_xyz`` gives the (strike, dip, tensile) components of a
    Cartesian vector, and ``frame[s].T @ v_sdt`` goes the other way -- the rows
    are orthonormal, so the transpose IS the inverse and no solve is needed.
    """
    n = unit_normals(tris)
    strike = np.stack([-n[:, 1], n[:, 0], np.zeros(len(n))], axis=1)
    mag = np.linalg.norm(strike, axis=1, keepdims=True)
    flat = mag[:, 0] <= FLAT_TOL
    strike = np.where(flat[:, None], np.array([1.0, 0.0, 0.0]),
                      strike / np.where(mag > 0.0, mag, 1.0))
    dip = np.cross(n, strike)
    return np.stack([strike, dip, n], axis=1)


def to_cartesian(tris: np.ndarray, slip: np.ndarray) -> np.ndarray:
    """(strike, dip, tensile) slip ``(n, 3)`` -> Cartesian ``(n, 3)``."""
    frame = slip_frame(tris)
    slip = np.asarray(slip, float).reshape(-1, 3)
    if slip.shape[0] != frame.shape[0]:
        raise ValueError(f"slip has {slip.shape[0]} rows for "
                         f"{frame.shape[0]} triangles")
    return np.einsum("sab,sa->sb", frame, slip)


def from_cartesian(tris: np.ndarray, slip: np.ndarray) -> np.ndarray:
    """Cartesian slip ``(n, 3)`` -> (strike, dip, tensile) ``(n, 3)``."""
    frame = slip_frame(tris)
    slip = np.asarray(slip, float).reshape(-1, 3)
    if slip.shape[0] != frame.shape[0]:
        raise ValueError(f"slip has {slip.shape[0]} rows for "
                         f"{frame.shape[0]} triangles")
    return np.einsum("sab,sb->sa", frame, slip)


def component_index(names, allowed=COMPONENTS) -> np.ndarray:
    """Resolve component names to frame row indices, with a usable error.

    ``normal`` is accepted as a synonym for ``tensile`` on the RECEIVER side,
    where the natural word for the frame's third direction is the one normal to
    the plane the traction acts on.
    """
    if isinstance(names, str):
        names = (names,)
    out = []
    for nm in names:
        key = "tensile" if nm == "normal" else nm
        if key not in allowed:
            raise ValueError(
                f"unknown component {nm!r}; expected some of "
                f"{list(allowed)} (or 'normal' for 'tensile')")
        out.append(allowed.index(key))
    if not out:
        raise ValueError("at least one component is required")
    return np.array(out, dtype=int)
