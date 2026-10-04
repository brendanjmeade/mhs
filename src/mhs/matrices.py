"""The public surface: Green's function MATRICES for a mollified half space.

Every function here is a thin Python wrapper. All coercion, shape and convention
checking, the byte budget and the ``out=`` handling happen in this module, so the
numba kernels below ``mhs.kernels`` only ever see contiguous float64 arrays and
plain floats -- never ``None``, never a dataclass, never an array of unknown
layout or dtype.

**Index order** is ``(n_obs, <output>, n_dof, 3)``, matching cutde's
``(n_obs, vec_dim, n_src, 3)`` so an existing call site recognises it, except
that a tensor output keeps its full ``(3, 3)`` rather than collapsing to
Voigt-6. That is deliberate: Voigt forces a component ordering and a
factor-of-two question on the shear entries, and keeping the tensor means **no
Voigt convention is stated anywhere in mhs**. A caller who wants Voigt packs it
in one line.

**The source axis is slip DEGREES OF FREEDOM.** At ``order=p`` each element
carries ``K = 1, 3, 6`` nodal slip vectors (P0, P1, P2) and element ``s``
occupies columns ``s*K .. (s+1)*K`` -- element-major, node index fastest, so a
caller reshapes to ``(n_src, K, 3)``. At the default ``order=0``, ``K == 1`` and
``n_dof == n_src``, so nothing written for elements changes; higher order only
makes the axis longer. The node axis is flattened rather than added because a
return rank that depended on an argument would be worse than a longer axis.

**Slip is Cartesian.** ``mhs.tdcs`` builds the rotation to the per-triangle
(strike, dip, tensile) frame, and no function here takes it as an argument,
because cutde's dip direction points *up* rather than down-dip and restating that
convention is how it would eventually be restated wrongly. The caller composes
the rotation explicitly; the gate that checks it is ``verify_cutde_limit``.

**There is no ``eps="auto"``.** That rule needs a mesh spacing ``h``, and these
functions take a vertex array.
"""
from __future__ import annotations

import numpy as np

from . import chunking
from .fullspace.shape import n_nodes
from .kernels import assemble as _asm
from .materials import Material

__all__ = [
    "disp_matrix",
    "stress_matrix",
    "total_stress_matrix",
    "eigenstress_matrix",
    "elastic_strain_matrix",
]


# --------------------------------------------------------------- coercion ----
def _coerce(obs_pts, tris, material, eps):
    """Validate and normalise the four arguments every entry point takes.

    Returns ``(obs, tris, mu, lam, eps_arr)`` with the arrays contiguous
    float64, so no kernel has to defend itself.

    The half-space convention is checked HERE rather than in a kernel: the free
    surface is ``z = 0``, the body is ``z <= 0``, and a source at or above the
    surface is a different problem (the Mindlin construction continues
    analytically there, but a triangle that straddles the surface is almost
    always a sign error in the caller's mesh, not an intention).
    """
    if not isinstance(material, Material):
        raise TypeError(
            f"material must be an mhs.Material(mu, lam), got "
            f"{type(material).__name__}. mhs takes (mu, lam) and never nu -- "
            f"use Material.from_mu_nu(mu, nu) to convert someone else's "
            f"parameters once, at the boundary.")

    obs = np.ascontiguousarray(np.asarray(obs_pts, dtype=np.float64))
    tri = np.ascontiguousarray(np.asarray(tris, dtype=np.float64))
    if obs.ndim != 2 or obs.shape[1] != 3:
        raise ValueError(f"obs_pts must have shape (n_obs, 3), got {obs.shape}")
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(
            f"tris must have shape (n_src, 3, 3) = (triangle, vertex, xyz), "
            f"got {tri.shape}")
    if obs.shape[0] == 0 or tri.shape[0] == 0:
        raise ValueError(
            f"nothing to assemble: {obs.shape[0]} obs points, "
            f"{tri.shape[0]} source triangles")
    if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(tri)):
        raise ValueError("obs_pts and tris must be finite")

    if np.any(obs[:, 2] > 0.0):
        bad = int(np.argmax(obs[:, 2]))
        raise ValueError(
            f"the half space is z <= 0 with the free surface at z = 0, but "
            f"obs_pts[{bad}] has z = {obs[bad, 2]:+.6g} > 0")
    if np.any(tri[:, :, 2] > 0.0):
        s, v = np.unravel_index(int(np.argmax(tri[:, :, 2])), tri.shape[:2])
        raise ValueError(
            f"the half space is z <= 0, but tris[{s}] vertex {v} has "
            f"z = {tri[s, v, 2]:+.6g} > 0")

    n_src = tri.shape[0]
    eps_arr = np.ascontiguousarray(
        np.broadcast_to(np.asarray(eps, dtype=np.float64), (n_src,)))
    if not np.all(np.isfinite(eps_arr)) or np.any(eps_arr <= 0.0):
        raise ValueError(
            "eps must be finite and strictly positive -- one value, or one per "
            "source triangle. eps = 0 is the classical singular kernel, which "
            "is cutde's job, not this package's.")
    return obs, tri, float(material.mu), float(material.lam), eps_arr


def _order(order) -> int:
    """Validate the nodal order. 0, 1, 2 -- P0, P1, P2.

    Checked here rather than deep in the kernels, where the failure would be a
    KeyError on a moment table. The table's degrees are generated for these
    three and nothing else.
    """
    p = int(order)
    if p not in (0, 1, 2):
        raise ValueError(
            f"order must be 0, 1 or 2 (P0, P1, P2); got {order!r}. The nodal "
            f"layout and the generated moment degrees exist for those three.")
    return p


def _coerce_normals(normals, n_obs):
    """``(n_obs, 3)`` unit normals, or one normal broadcast to every observer.

    Normalised HERE rather than trusted: a traction scales linearly with the
    normal's length, so an unnormalised input is not an error anywhere -- it
    silently rescales the answer. A near-zero normal is refused instead,
    because that cannot be rescued by normalising.
    """
    arr = np.atleast_2d(np.asarray(normals, float))
    if arr.shape == (1, 3) and n_obs != 1:
        arr = np.repeat(arr, n_obs, axis=0)
    if arr.shape != (n_obs, 3):
        raise ValueError(
            f"obs_normals has shape {np.shape(normals)}; expected "
            f"({n_obs}, 3) to match obs_pts, or (3,) for one shared normal")
    mag = np.linalg.norm(arr, axis=1)
    if not np.all(mag > 1e-13):
        raise ValueError(
            f"obs_normals[{int(np.argmin(mag))}] has length "
            f"{float(mag.min()):.3e} and so names no plane")
    return arr / mag[:, None]


def _prepare_out(out, shape, n_obs, n_src, kind, parts):
    """Validate a caller-supplied ``out=``, or budget-check an allocation.

    ``out=`` is the escape from the ceiling, so when it is given the ceiling is
    not consulted: the caller has already chosen where the bytes live, and that
    choice may be a memmap larger than RAM.
    """
    if out is None:
        chunking.refuse_if_too_large(n_obs, n_src, kind, np.float64, parts)
        return np.zeros(shape, dtype=np.float64)
    arr = out
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"out must be a numpy array, got {type(arr).__name__}")
    if arr.shape != shape:
        raise ValueError(f"out has shape {arr.shape}, expected {shape}")
    if arr.dtype not in (np.float64, np.float32):
        raise ValueError(
            f"out must be float64 or float32, got {arr.dtype}. The kernels "
            f"compute in float64 registers either way; a float32 out is a "
            f"storage choice that halves the footprint.")
    return arr


# ----------------------------------------------------------------- public ----
def disp_matrix(obs_pts, tris, material, eps, *, order=0, out=None):
    """Displacement Green's function matrix.

    Returns ``(n_obs, 3, n_src, 3)`` where ``[o, i, s, k]`` is the ``i``-th
    displacement component at ``obs_pts[o]`` produced by a unit **Cartesian**
    slip component ``k`` on ``tris[s]``, so ``u = einsum("oisk,sk->oi", G, slip)``.
    """
    obs, tri, mu, lam, eps_arr = _coerce(obs_pts, tris, material, eps)
    p = _order(order)
    n_obs, n_dof = obs.shape[0], tri.shape[0] * n_nodes(p)
    buf = _prepare_out(out, (n_obs, 3, n_dof, 3), n_obs, n_dof, "disp", False)
    _asm.assemble_halfspace_disp(obs, tri, mu, lam, eps_arr, buf, p)
    return buf


def stress_matrix(obs_pts, tris, material, eps, *,
                  subtract_eigenstress=True, parts=False, order=0, out=None):
    """ELASTIC stress Green's function matrix, eigenstress removed by default.

    Returns ``(n_obs, 3, 3, n_src, 3)``; ``[o, m, n, s, k]`` is ``sigma_mn`` from
    a unit Cartesian slip ``k`` on source ``s``.

    Every mollified double layer carries an eigenstress ``C:eps*`` -- the smeared
    slip is literally an anelastic eigenstrain -- and within ~3 eps of the
    element that term dominates, growing like ``1/eps``. Stress presented as
    elastic must have it subtracted, so that is the default here rather than an
    option the caller has to know to ask for. ``total_stress_matrix`` is the raw
    field, named so it cannot be reached by accident.

    ``parts=True`` returns ``(stress, eigenstress)`` from one pass, because a
    readout usually wants both and recovering the second by a separate call would
    redo the dominant kernel.
    """
    obs, tri, mu, lam, eps_arr = _coerce(obs_pts, tris, material, eps)
    p = _order(order)
    n_obs, n_dof = obs.shape[0], tri.shape[0] * n_nodes(p)
    shape = (n_obs, 3, 3, n_dof, 3)
    buf = _prepare_out(out, shape, n_obs, n_dof, "stress", parts)
    if parts and out is not None:
        raise ValueError("parts=True returns two arrays, so out= cannot name "
                         "the destination of both; call twice with out=, or "
                         "drop out=.")
    # _prepare_out returns ONE array whatever `parts` is -- that flag only
    # widens the byte budget -- so the second buffer is allocated here. The
    # first draft unpacked `buf` into two names and silently sliced its leading
    # axis instead.
    sig = buf
    eig = np.zeros(shape, dtype=np.float64) if parts else None
    _asm.assemble_halfspace_total_stress(obs, tri, mu, lam, eps_arr, sig, p)
    if subtract_eigenstress or parts:
        star = eig if parts else np.empty(shape, dtype=np.float64)
        _asm.assemble_eigenstress(obs, tri, mu, lam, eps_arr, star, p)
        if subtract_eigenstress:
            sig -= star
    return (sig, eig) if parts else sig


def traction_matrix(obs_pts, obs_normals, tris, material, eps, *,
                    subtract_eigenstress=True, order=0, out=None):
    """ELASTIC traction on a given plane, ``(n_obs, 3, n_src, 3)``.

    ``[o, i, s, k]`` is Cartesian traction component ``i`` on the plane through
    ``obs_pts[o]`` with normal ``obs_normals[o]``, from unit Cartesian slip
    ``k`` on ``tris[s]``. ``obs_normals`` may be one normal for all observers.

    This is ``stress_matrix`` contracted with the receiver normal, but it never
    allocates the stress: the contraction happens per source inside the
    assembly loop, so the stored object is a THIRD the size. That is not a
    micro-optimisation -- a 10k x 10k stress matrix is 20.1 GiB and the ceiling
    refuses it, where this is 6.7 GiB.

    Signs follow the normal you pass. Traction is ``sigma . n``, so flipping
    ``obs_normals[o]`` flips that row; mhs does not orient it for you, because
    which side of a surface you mean is a property of your mesh. ``mhs.tdcs``
    builds the upward-oriented frame if you want cutde's convention.
    """
    obs, tri, mu, lam, eps_arr = _coerce(obs_pts, tris, material, eps)
    p = _order(order)
    n_obs, n_dof = obs.shape[0], tri.shape[0] * n_nodes(p)
    nrm = _coerce_normals(obs_normals, n_obs)
    buf = _prepare_out(out, (n_obs, 3, n_dof, 3), n_obs, n_dof, "traction",
                       False)
    _asm.assemble_halfspace_traction(obs, nrm, tri, mu, lam, eps_arr, buf,
                                     subtract_eigenstress, p)
    return buf


#: How far a nodal collocation point is pulled toward its element's centroid,
#: in barycentric coordinates, for ``interaction_matrix`` at order > 0.
#:
#: At P1 the nodes ARE the vertices and at P2 three of them are edge midpoints,
#: so reading stress at a raw node means reading it on the element's own
#: boundary -- where the clearance from the edge is zero and the mollified
#: kernel is at its hardest. Shrinking by 1/2 puts a P1 vertex at barycentric
#: (2/3, 1/6, 1/6), i.e. a clearance of h/6 against the centroid's h/3, which
#: is the configuration ``docs/derivation.md`` measures the kernel at: the
#: minimum |D3|/sqrt(A) over a surface-breaking element is 1.67 at P1/P2
#: against 3.33 at P0. Exposed as an argument because it is a modelling choice,
#: not a property of the kernel, and 0 (raw nodes) is legitimate if a caller
#: wants it.
COLLOCATION_SHRINK = 0.5


def collocation_points(tris, order: int = 0,
                       shrink: float = COLLOCATION_SHRINK) -> np.ndarray:
    """``(n_src * K, 3)`` nodal collocation points, element-major like the DOFs.

    What ``interaction_matrix`` uses when ``obs_pts`` is None. Public because a
    caller who parallelises over source chunks must compute these ONCE from the
    whole mesh and pass them in -- each chunk would otherwise collocate on its
    own elements alone, which is a different (and much smaller) matrix.
    """
    from .fullspace.shape import lattice
    tri = np.asarray(tris, float).reshape(-1, 3, 3)
    p = _order(order)
    bary = (np.full((1, 3), 1.0 / 3.0) if p == 0
            else lattice(p) / p)                        # (K, 3)
    bary = bary + float(shrink) * (1.0 / 3.0 - bary)    # toward the centroid
    return (bary[None, :, :] @ tri).reshape(-1, 3)


def interaction_matrix(tris, material, eps, *, receiver=("strike", "dip",
                                                         "normal"),
                       source=("strike", "dip", "tensile"),
                       obs_tris=None, obs_pts=None, order=0,
                       shrink=COLLOCATION_SHRINK,
                       subtract_eigenstress=True, out=None):
    """The on-fault interaction matrix, ``(n_rec_dof, n_src_dof, n_a, n_b)``.

    ``[i, j, a, b]`` is the traction on receiver DOF ``i``'s own plane, resolved
    along direction ``a`` of that element's (strike, dip, tensile) frame, from
    unit slip along direction ``b`` of source element ``j``'s frame. This is the
    object Coulomb, rate-and-state and earthquake-cycle work actually wants, and
    it is assembled directly rather than contracted out of a stress matrix,
    because the stress matrix for a production mesh cannot be allocated.

    The frames come from :mod:`mhs.tdcs`, which states cutde's convention in one
    place; ``receiver`` additionally accepts ``"normal"`` for ``"tensile"``,
    because the natural word for a traction's out-of-plane part is the one
    normal to the plane it acts on.

    SELECT COMPONENTS TO SELECT A SIZE. At 10k elements the default
    ``3 x 3`` is 7.2 GiB, two shear components are 3.2 GiB and pure strike-slip
    (``receiver="strike", source="strike"``) is 0.75 GiB. Nothing else about the
    calculation changes, so the selection is purely how much of it you keep.

    RECEIVERS NEED NOT BE THE SOURCES. ``obs_tris`` defaults to ``tris``, which
    is the square self-interaction everyone wants first. Giving it a different
    set makes the matrix RECTANGULAR -- stress on one fault from slip on
    another, and the shape a caller needs when parallelising, because a chunk of
    sources must still see every receiver (``mhs.parallel.by_source``).

    Observers default to the shrunk nodes of ``obs_tris``
    (:func:`collocation_points`), which is where a mollified kernel may be read
    on its own element: the eigenstress is removed and what is left is finite
    there. ``obs_pts`` overrides them and must supply ``K`` per receiver
    element, grouped element-major, because each one is paired with its
    element's plane and frame.
    """
    from . import tdcs
    tri = np.asarray(tris, float).reshape(-1, 3, 3)
    rec_tri = tri if obs_tris is None else np.asarray(
        obs_tris, float).reshape(-1, 3, 3)
    p = _order(order)
    K = n_nodes(p)
    pts = collocation_points(rec_tri, p, shrink) if obs_pts is None else obs_pts
    obs, tri, mu, lam, eps_arr = _coerce(pts, tri, material, eps)
    rec_tri = np.ascontiguousarray(rec_tri)
    n_src_dof = tri.shape[0] * K
    n_rec_dof = rec_tri.shape[0] * K
    if obs.shape[0] != n_rec_dof:
        raise ValueError(
            f"interaction_matrix pairs observers with RECEIVER slip DOFs row by "
            f"row, so at order {p} it needs {K} per receiver element: got "
            f"{obs.shape[0]} observers for {rec_tri.shape[0]} receiver "
            f"triangles ({n_rec_dof} DOFs). For traction at unrelated points "
            f"use traction_matrix(obs_pts, obs_normals, ...).")
    # One frame per ELEMENT, repeated across its nodes: the frame is a property
    # of the plane, and every node of an element shares that plane.
    rec_frame = np.repeat(tdcs.slip_frame(rec_tri), K, axis=0)
    src_frame = tdcs.slip_frame(tri)                    # one row per element
    rec = tdcs.component_index(receiver)
    src = tdcs.component_index(source)
    shape = (n_rec_dof, n_src_dof, len(rec), len(src))
    buf = _prepare_out(out, shape, n_rec_dof, n_src_dof,
                       (len(rec), len(src)), False)
    _asm.assemble_interaction(obs, rec_frame[:, 2, :], rec_frame[:, rec, :],
                              tri, src_frame[:, src, :], mu, lam, eps_arr, buf,
                              subtract_eigenstress, p)
    return buf


def total_stress_matrix(obs_pts, tris, material, eps, *, order=0, out=None):
    """RAW mollified stress: the elastic field PLUS the eigenstress ``C:eps*``.

    This is not the stress of a physical elastic medium near the element. It
    diverges like ``1/eps`` on the source surface (the analytic on-fault peak is
    ``(3/4) mu s / eps``), which is exactly the eigenstress the mollification
    put there. It exists because studying that term is legitimate, and because
    ``total - eigenstress == stress`` is the identity that makes the three-way
    split safe to rely on.
    """
    obs, tri, mu, lam, eps_arr = _coerce(obs_pts, tris, material, eps)
    p = _order(order)
    n_obs, n_dof = obs.shape[0], tri.shape[0] * n_nodes(p)
    buf = _prepare_out(out, (n_obs, 3, 3, n_dof, 3), n_obs, n_dof,
                       "stress", False)
    _asm.assemble_halfspace_total_stress(obs, tri, mu, lam, eps_arr, buf, p)
    return buf


def eigenstress_matrix(obs_pts, tris, material, eps, *, order=0, out=None):
    """The eigenstress term ``C:eps*`` alone, as a matrix.

    ``H*[m,n,k] = Phi_eps (lam d_mn n_k + mu (d_mk n_n + d_nk n_m))`` with
    ``Phi_eps = (15 eps^4 / 8 pi) I7`` the blob integrated over the triangle.

    It depends on the source triangle, its normal, its eps and ``(mu, lam)`` --
    and on nothing else. In particular it does **not** depend on the Green's
    function, which is why the half-space kernel reuses the full-space
    eigenstress verbatim: all the blob content sits in the direct (Kelvin) term,
    while the half-space image correction is smooth in the body. The deep-source
    gate proves that rather than assuming it.

    Identically zero at ``eps = 0``.
    """
    obs, tri, mu, lam, eps_arr = _coerce(obs_pts, tris, material, eps)
    p = _order(order)
    n_obs, n_dof = obs.shape[0], tri.shape[0] * n_nodes(p)
    buf = _prepare_out(out, (n_obs, 3, 3, n_dof, 3), n_obs, n_dof,
                       "eigenstress", False)
    _asm.assemble_eigenstress(obs, tri, mu, lam, eps_arr, buf, p)
    return buf


def elastic_strain_matrix(obs_pts, tris, material, eps, *,
                          subtract_eigenstress=True, order=0, out=None):
    """ELASTIC strain Green's function matrix, as a full ``(3, 3)`` tensor.

    Named ``elastic_`` rather than ``strain_`` on purpose. cutde's idiom is
    ``strain_matrix`` followed by ``strain_to_stress``, and for a mollified
    kernel that path silently returns the TOTAL stress -- eigenstress included --
    which is the single failure this package exists to prevent. There is no bare
    ``strain_matrix`` here for that route to start from.
    """
    obs, tri, mu, lam, eps_arr = _coerce(obs_pts, tris, material, eps)
    p = _order(order)
    n_obs, n_dof = obs.shape[0], tri.shape[0] * n_nodes(p)
    buf = _prepare_out(out, (n_obs, 3, 3, n_dof, 3), n_obs, n_dof,
                       "strain", False)
    _asm.assemble_halfspace_total_stress(obs, tri, mu, lam, eps_arr, buf, p)
    if subtract_eigenstress:
        star = np.empty_like(buf)
        _asm.assemble_eigenstress(obs, tri, mu, lam, eps_arr, star, p)
        buf -= star
    # strain from stress, inverting Hooke with (mu, lam) and never through a
    # 1/(1-2nu) intermediate: tr(e) = tr(sig)/(3 lam + 2 mu), then
    # e = (sig - lam tr(e) I) / (2 mu).
    tr = buf[:, 0, 0, :, :] + buf[:, 1, 1, :, :] + buf[:, 2, 2, :, :]
    tre = tr / (3.0 * lam + 2.0 * mu)
    for a in range(3):
        buf[:, a, a, :, :] -= lam * tre
    buf /= (2.0 * mu)
    return buf
