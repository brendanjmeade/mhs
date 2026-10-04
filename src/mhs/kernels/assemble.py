"""Fill the matrix forms, one source triangle at a time.

The loop is over **sources**, not observers, for the same reason the reference
tree's assemblers are: the per-source work -- the local frame, the edge geometry,
the moment-table setup -- is invariant across observers, so hoisting it out of
the inner dimension is the whole saving. Each source also writes a disjoint
column slice ``out[..., s*K:(s+1)*K, :]``, so there is no race when this loop
becomes a ``prange``.

**THE SOURCE AXIS IS SLIP DEGREES OF FREEDOM, NOT ELEMENTS.** At order ``p``
each element carries ``K = n_nodes(p)`` nodal slip vectors (1, 3, 6 for P0, P1,
P2), and element ``s``'s nodes occupy ``s*K .. (s+1)*K`` -- element-major, node
index fastest, so a caller reshapes to ``(n_src, K, 3)``. At P0 ``K == 1``, so
every shape, caller and gate written for elements is unchanged; higher order
only makes the axis longer, which is why the node axis is flattened rather than
added (a return rank that depends on an argument would be worse).

The inner call is ``mhs.fullspace`` (numpy, batched over observers). The loop is
shaped so those calls can be replaced by numba kernels one at a time behind the
parity gate, which is why the per-source body is a single function call rather
than inlined arithmetic.

**The (mu, lam) seam.** ``mhs`` carries ``(mu, lam)`` everywhere, because
``1/(1 - 2 nu)`` diverges toward incompressibility and a lam/mu swap is invisible
at ``nu = 1/4``. The vendored full-space engine takes ``(mu, nu)`` and derives
``lam`` internally, so the conversion happens HERE, in one function, and nowhere
else. Reparametrising the vendored engine to ``(mu, lam)`` is tracked work; until
then this is the one place the forbidden quotient is formed, and it is formed
once per call rather than once per kernel term.
"""
from __future__ import annotations

import numpy as np

from .image import image_total

from ..fullspace import kernels as _fs_kernels
from ..fullspace.shape import n_nodes

#: Which nodal tensor carries which readout, in the vendored engine's naming.
#:   U -> displacement from a slip (dislocation) source
#:   H -> TOTAL stress from the same source (eigenstress included)
#:   E -> the scalar blob weight the eigenstress is built from
WANT_FOR = {"disp": ("U",), "stress": ("H",), "eigenstress": ("E",)}


def _nu_from(mu: float, lam: float) -> float:
    """Poisson's ratio, formed at the single seam with the vendored engine.

    Finite for every admissible ``(mu, lam)``: ``lam > -2 mu / 3`` gives
    ``lam + mu > mu / 3 > 0``, so the denominator cannot vanish. This is the
    inverse direction from the forbidden one -- going ``(mu, lam) -> nu`` is
    always safe; it is ``nu -> lam`` that blows up at ``nu = 1/2``.
    """
    return 0.5 * lam / (lam + mu)


def fullspace_blocks(obs: np.ndarray, tri: np.ndarray, mu: float, lam: float,
                     eps: float, want=("U",), order: int = 0) -> dict:
    """The vendored full-space nodal tensors of ONE triangle at every ``obs``.

    A thin adapter, deliberately: it is the seam where the numba rewrite will
    land, so it has no arithmetic of its own beyond the material conversion.
    """
    return _fs_kernels.nodal_influence(obs, tri, order, float(mu),
                                       _nu_from(mu, lam), float(eps),
                                       want=tuple(want))


def assemble_fullspace_disp(obs: np.ndarray, tris: np.ndarray, mu: float,
                            lam: float, eps_arr: np.ndarray,
                            out: np.ndarray, order: int = 0) -> np.ndarray:
    """FULL-SPACE displacement matrix into ``out`` (n_obs, 3, n_src, 3).

    This is the direct (Kelvin) half of the half-space kernel. It is *not* the
    half-space answer and is not exposed as ``mhs.disp_matrix``: the image
    correction is the other half, and a full-space number under a half-space name
    would be wrong by the entire free-surface effect -- a factor of about two in
    surface displacement, which is exactly the size that looks plausible.
    """
    K = n_nodes(order)
    for s in range(tris.shape[0]):
        # (n_obs, K, 3, 3): [obs, node, i, slip]
        U = np.asarray(fullspace_blocks(obs, tris[s], mu, lam, eps_arr[s],
                                        want=("U",), order=order)["U"])
        out[:, :, s * K:(s + 1) * K, :] = np.moveaxis(U, 1, 2)
    return out


def assemble_fullspace_total_stress(obs: np.ndarray, tris: np.ndarray,
                                    mu: float, lam: float,
                                    eps_arr: np.ndarray, out: np.ndarray,
                                    order: int = 0) -> np.ndarray:
    """FULL-SPACE **total** stress matrix into ``out`` (n_obs, 3, 3, n_src, 3).

    Total, not elastic: within ~3 eps of the element this is dominated by the
    eigenstress the mollification put there, which grows like ``1/eps``.
    """
    K = n_nodes(order)
    for s in range(tris.shape[0]):
        H = np.asarray(fullspace_blocks(obs, tris[s], mu, lam, eps_arr[s],
                                        want=("H",), order=order)["H"])
        out[:, :, :, s * K:(s + 1) * K, :] = np.moveaxis(H, 1, 3)
    return out


def assemble_halfspace_disp(obs: np.ndarray, tris: np.ndarray, mu: float,
                            lam: float, eps_arr: np.ndarray,
                            out: np.ndarray, order: int = 0) -> np.ndarray:
    """HALF-SPACE displacement matrix into ``out`` (n_obs, 3, n_src, 3).

    Direct (Kelvin, closed form) plus the image correction -- closed-form
    R-family plus quadrature Q-family, which is the split
    ``verify_vertical_fault`` clauses [d] and [e] measure. The two halves ADD
    with no sign fix: clq's and moss's conventions agree to 3.5e-16 on the same
    direct term, which is checked rather than assumed because a relative sign
    error between them would be invisible at a symmetric configuration.
    """
    assemble_fullspace_disp(obs, tris, mu, lam, eps_arr, out, order)
    K = n_nodes(order)
    for s in range(tris.shape[0]):
        img = image_total(obs, tris[s], order, float(mu), float(lam),
                          float(eps_arr[s]), want=("U",))
        out[:, :, s * K:(s + 1) * K, :] += np.moveaxis(img["U"], 1, 2)
    return out


def assemble_halfspace_total_stress(obs: np.ndarray, tris: np.ndarray,
                                    mu: float, lam: float,
                                    eps_arr: np.ndarray, out: np.ndarray,
                                    order: int = 0) -> np.ndarray:
    """HALF-SPACE **total** stress matrix into ``out`` (n_obs, 3, 3, n_src, 3).

    Total, not elastic: within ~3 eps of the element the eigenstress the
    mollification put there dominates, and it grows like ``1/eps``. Subtract
    :func:`assemble_eigenstress` for the elastic stress -- which is what
    ``mhs.stress_matrix`` returns, and what any stress presented as elastic
    must have had removed.
    """
    assemble_fullspace_total_stress(obs, tris, mu, lam, eps_arr, out, order)
    K = n_nodes(order)
    for s in range(tris.shape[0]):
        img = image_total(obs, tris[s], order, float(mu), float(lam),
                          float(eps_arr[s]), want=("H",))
        out[:, :, :, s * K:(s + 1) * K, :] += np.moveaxis(img["H"], 1, 3)
    return out


def _eigenstress_traction(blk, tri, nrm, mu: float, lam: float) -> np.ndarray:
    """``C:eps*`` contracted onto the receiver planes, ``(n_obs, K, 3, 3)``.

    The same tensor :func:`assemble_eigenstress` writes, but never materialised
    as ``(3, 3)`` per pair -- which is the only reason the contracted forms cost
    less than contracting their output afterwards would.
    """
    eye = np.eye(3)
    phi = np.asarray(blk["E"])                             # (n_obs, K)
    n = _unit_normal(tri)
    C = (lam * np.einsum("mn,k->mnk", eye, n)
         + mu * (np.einsum("mk,n->mnk", eye, n)
                 + np.einsum("nk,m->mnk", eye, n)))
    return (phi[:, :, None, None]
            * np.einsum("ijk,oj->oik", C, nrm)[:, None, :, :])


def _halfspace_stress_block(obs, tri, mu, lam, eps, want, order=0) -> dict:
    """Total stress of ONE source at every observer: direct PLUS image.

    Stated once, so the contracted paths cannot drift from
    :func:`assemble_halfspace_total_stress` in which halves they add.
    """
    blk = fullspace_blocks(obs, tri, mu, lam, eps, want=want, order=order)
    img = image_total(obs, tri, order, float(mu), float(lam), float(eps),
                      want=("H",))
    return {"H": np.asarray(blk["H"]) + img["H"], "E": blk.get("E")}


def assemble_halfspace_traction(obs: np.ndarray, nrm: np.ndarray,
                                tris: np.ndarray, mu: float, lam: float,
                                eps_arr: np.ndarray, out: np.ndarray,
                                subtract_eigenstress: bool = True,
                                order: int = 0) -> np.ndarray:
    """Cartesian traction into ``out`` (n_obs, 3, n_src, 3).

    ``out[o, i, s, k]`` is traction component ``i`` on the plane whose normal is
    ``nrm[o]``, at ``obs[o]``, from unit Cartesian slip ``k`` on ``tris[s]``.

    THE CONTRACTION HAPPENS INSIDE THE SOURCE LOOP, which is the whole point:
    one source's stress at every observer is ``(n_obs, 3, 3, 3)`` and is
    discarded immediately, so the STORED object is a third of the stress
    matrix. Contracting ``mhs.stress_matrix``'s output afterwards would mean
    allocating that larger object first -- 20.1 GiB at 10k x 10k, which the
    ceiling refuses, so the cheaper route is also the only reachable one.
    """
    want = ("H", "E") if subtract_eigenstress else ("H",)
    K = n_nodes(order)
    for s in range(tris.shape[0]):
        blk = _halfspace_stress_block(obs, tris[s], mu, lam, eps_arr[s], want,
                                      order)
        t = np.einsum("okijm,oj->okim", blk["H"], nrm)      # (n_obs, K, 3, 3)
        if subtract_eigenstress:
            t = t - _eigenstress_traction(blk, tris[s], nrm, mu, lam)
        out[:, :, s * K:(s + 1) * K, :] = np.moveaxis(t, 1, 2)
    return out


def assemble_interaction(obs: np.ndarray, nrm: np.ndarray,
                         rec_basis: np.ndarray, tris: np.ndarray,
                         src_basis: np.ndarray, mu: float, lam: float,
                         eps_arr: np.ndarray, out: np.ndarray,
                         subtract_eigenstress: bool = True,
                         order: int = 0) -> np.ndarray:
    """Frame-resolved interaction into ``out`` (n_obs, n_src, n_rec, n_slip).

    ``out[o, s, a, b]`` is traction resolved along ``rec_basis[o, a]`` on the
    plane ``nrm[o]``, from unit slip along ``src_basis[s, b]`` on ``tris[s]``.
    Both bases are Cartesian rows, so this function states no convention of its
    own -- ``mhs.tdcs`` does, in one place, and the caller passes the result.

    Contracts in the source loop for the same reason as above, and here it
    matters more: the full ``(n, 3, 3, n, 3)`` stress is 20.1 GiB at 10k where
    a single-component interaction is 0.75 GiB.
    """
    want = ("H", "E") if subtract_eigenstress else ("H",)
    K = n_nodes(order)
    for s in range(tris.shape[0]):
        blk = _halfspace_stress_block(obs, tris[s], mu, lam, eps_arr[s], want,
                                      order)
        t = np.einsum("okijm,oj->okim", blk["H"], nrm)     # (n_obs, K, 3, slip)
        if subtract_eigenstress:
            t = t - _eigenstress_traction(blk, tris[s], nrm, mu, lam)
        out[:, s * K:(s + 1) * K, :, :] = np.einsum(
            "oai,okim,bm->okab", rec_basis, t, src_basis[s])
    return out


def assemble_eigenstress(obs: np.ndarray, tris: np.ndarray, mu: float,
                         lam: float, eps_arr: np.ndarray, out: np.ndarray,
                         order: int = 0) -> np.ndarray:
    """The eigenstress matrix ``C:eps*`` into ``out`` (n_obs, 3, 3, n_src, 3).

    ``H*[m,n,k] = Phi_eps (lam d_mn n_k + mu (d_mk n_n + d_nk n_m))``, with
    ``Phi_eps`` the blob integrated over the triangle -- the vendored engine's
    ``E`` weight, which is ``(15 eps^4 / 8 pi) M_7^(0,0)``.

    **This is why the half-space kernel can reuse the full-space eigenstress.**
    ``Phi_eps(x) = int_T phi_eps(x - y) dS`` is a pointwise statement about the
    blob and the source triangle: it reads no Green's function, so no free
    surface and no image enter it. All of the blob content sits in the direct
    term; the half-space image correction depends only on the distance to the
    *image*, which never vanishes inside the body, so it is smooth there and
    carries none. The deep-source gate proves that rather than assuming it.
    """
    eye = np.eye(3)
    K = n_nodes(order)
    for s in range(tris.shape[0]):
        blk = fullspace_blocks(obs, tris[s], mu, lam, eps_arr[s], want=("E",),
                               order=order)
        phi = np.asarray(blk["E"])                       # (n_obs, K)
        n = _unit_normal(tris[s])
        # C:sym(slip (x) n) per unit slip component k, as a (3,3,3) constant
        C = (lam * np.einsum("mn,k->mnk", eye, n)
             + mu * (np.einsum("mk,n->mnk", eye, n)
                     + np.einsum("nk,m->mnk", eye, n)))
        out[:, :, :, s * K:(s + 1) * K, :] = (phi[:, None, None, :, None]
                                              * C[None, :, :, None, :])
    return out


def _unit_normal(tri: np.ndarray) -> np.ndarray:
    """Outward unit normal of one triangle, from its stored vertex order.

    The normal's SIGN is a convention of the caller's mesh, not of this package:
    it sets which side of the surface the slip jump refers to. mhs does not flip
    it, and the eigenstress contraction above uses it as given -- the same
    choice the vendored engine makes, so the two cannot disagree.
    """
    v = np.asarray(tri, float)
    n = np.cross(v[1] - v[0], v[2] - v[0])
    norm = np.linalg.norm(n)
    if not (norm > 0.0):
        raise ValueError(f"degenerate triangle, zero area: {v.tolist()}")
    return n / norm
