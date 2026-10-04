"""The public surface: Green's function MATRICES for a mollified half space.

Every function here is a thin Python wrapper. All coercion, shape and convention
checking, the byte budget and the ``out=`` handling happen in this module, so the
numba kernels below ``mhs.kernels`` only ever see contiguous float64 arrays and
plain floats -- never ``None``, never a dataclass, never an array of unknown
layout or dtype.

**Index order** is ``(n_obs, <output>, n_src, 3)``, matching cutde's
``(n_obs, vec_dim, n_src, 3)`` so an existing call site recognises it, except
that a tensor output keeps its full ``(3, 3)`` rather than collapsing to Voigt-6.
That is deliberate: Voigt forces a component ordering and a factor-of-two
question on the shear entries, and keeping the tensor means **no Voigt convention
is stated anywhere in mhs**. A caller who wants Voigt packs it in one line.

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
def disp_matrix(obs_pts, tris, material, eps, *, out=None):
    """Displacement Green's function matrix.

    Returns ``(n_obs, 3, n_src, 3)`` where ``[o, i, s, k]`` is the ``i``-th
    displacement component at ``obs_pts[o]`` produced by a unit **Cartesian**
    slip component ``k`` on ``tris[s]``, so ``u = einsum("oisk,sk->oi", G, slip)``.
    """
    obs, tri, mu, lam, eps_arr = _coerce(obs_pts, tris, material, eps)
    n_obs, n_src = obs.shape[0], tri.shape[0]
    buf = _prepare_out(out, (n_obs, 3, n_src, 3), n_obs, n_src, "disp", False)
    _asm.assemble_halfspace_disp(obs, tri, mu, lam, eps_arr, buf)
    return buf


def stress_matrix(obs_pts, tris, material, eps, *,
                  subtract_eigenstress=True, parts=False, out=None):
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
    n_obs, n_src = obs.shape[0], tri.shape[0]
    shape = (n_obs, 3, 3, n_src, 3)
    buf = _prepare_out(out, shape, n_obs, n_src, "stress", parts)
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
    _asm.assemble_halfspace_total_stress(obs, tri, mu, lam, eps_arr, sig)
    if subtract_eigenstress or parts:
        star = eig if parts else np.empty(shape, dtype=np.float64)
        _asm.assemble_eigenstress(obs, tri, mu, lam, eps_arr, star)
        if subtract_eigenstress:
            sig -= star
    return (sig, eig) if parts else sig


def total_stress_matrix(obs_pts, tris, material, eps, *, out=None):
    """RAW mollified stress: the elastic field PLUS the eigenstress ``C:eps*``.

    This is not the stress of a physical elastic medium near the element. It
    diverges like ``1/eps`` on the source surface (the analytic on-fault peak is
    ``(3/4) mu s / eps``), which is exactly the eigenstress the mollification
    put there. It exists because studying that term is legitimate, and because
    ``total - eigenstress == stress`` is the identity that makes the three-way
    split safe to rely on.
    """
    obs, tri, mu, lam, eps_arr = _coerce(obs_pts, tris, material, eps)
    n_obs, n_src = obs.shape[0], tri.shape[0]
    buf = _prepare_out(out, (n_obs, 3, 3, n_src, 3), n_obs, n_src,
                       "stress", False)
    _asm.assemble_halfspace_total_stress(obs, tri, mu, lam, eps_arr, buf)
    return buf


def eigenstress_matrix(obs_pts, tris, material, eps, *, out=None):
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
    n_obs, n_src = obs.shape[0], tri.shape[0]
    buf = _prepare_out(out, (n_obs, 3, 3, n_src, 3), n_obs, n_src,
                       "eigenstress", False)
    _asm.assemble_eigenstress(obs, tri, mu, lam, eps_arr, buf)
    return buf


def elastic_strain_matrix(obs_pts, tris, material, eps, *,
                          subtract_eigenstress=True, out=None):
    """ELASTIC strain Green's function matrix, as a full ``(3, 3)`` tensor.

    Named ``elastic_`` rather than ``strain_`` on purpose. cutde's idiom is
    ``strain_matrix`` followed by ``strain_to_stress``, and for a mollified
    kernel that path silently returns the TOTAL stress -- eigenstress included --
    which is the single failure this package exists to prevent. There is no bare
    ``strain_matrix`` here for that route to start from.
    """
    obs, tri, mu, lam, eps_arr = _coerce(obs_pts, tris, material, eps)
    n_obs, n_src = obs.shape[0], tri.shape[0]
    buf = _prepare_out(out, (n_obs, 3, 3, n_src, 3), n_obs, n_src,
                       "strain", False)
    _asm.assemble_halfspace_total_stress(obs, tri, mu, lam, eps_arr, buf)
    if subtract_eigenstress:
        star = np.empty_like(buf)
        _asm.assemble_eigenstress(obs, tri, mu, lam, eps_arr, star)
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
