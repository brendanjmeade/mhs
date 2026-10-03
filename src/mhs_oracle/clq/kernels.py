"""Lift of weighted moments to 3-D tensor moments and the kernel contractions.

With ``d = x - y = -xi_1 e1 - xi_2 e2 + z nhat`` (appendix eq. Req_inplane), a
rank-``r`` tensor moment weighted by a node's shape function is

    T^{[r]}_{i1..ir, n} = int_T N_k d_i1 ... d_ir / R^n dS
                       = sum over slot assignments of (-1)^(a+b) z^(r-a-b)
                         W_k[n][a, b] (basis outer product),

where the assignment puts ``a`` slots on ``e1``, ``b`` on ``e2`` and the rest on
``nhat``.  The mollified kernels (``G^eps`` in the Galerkin/Cortez form with
the ``2(1-nu) eps^2 delta/R^3`` blob term) then give, per node ``k``:

    G1[i,j,m]  = C1 [ -c34 d_ij V3_m + d_im V3_j + d_jm V3_i - 3 T3_5[i,j,m]
                      - 6(1-nu) eps^2 d_ij V5_m ]                     (int dG/dx_m)
    U[i,j]     = -( mu sum_m n_m G1[i,j,m] + lam n_j sum_m G1[i,m,m]
                    + mu sum_m n_m G1[i,m,j] )                          (slip -> displacement)
    D2[r,p,s,q]= C1 { -c34 d_rp [d_sq I3 - 3 T2_5[s,q]] + d_rs [d_pq I3 - 3 T2_5[p,q]]
                      + d_ps [d_rq I3 - 3 T2_5[r,q]]
                      - 3 [d_rq T2_5[p,s] + d_pq T2_5[r,s] + d_sq T2_5[r,p]]
                      + 15 T4_7[r,p,s,q]
                      + 2(1-nu) eps^2 d_rp [-3 d_sq I5 + 15 T2_7[s,q]] }
    B[r,s,j]   = lam n_j sum_p D2[r,p,s,p] + mu sum_q n_q D2[r,j,s,q] + mu sum_p n_p D2[r,p,s,j]
    H[m,l,j]   = -( lam d_ml sum_r B[r,r,j] + mu (B[m,l,j] + B[l,m,j]) )   (slip -> TOTAL stress)
    E          = (15 eps^4 / 8 pi) W_k[7][0, 0]                          (eigenstress weight)

``U`` uses the traction-operator index pairing (slip and normal in the FIRST
index pair of C): the slip-``j`` column is ``-C_{j m p q} n_m dG_{i p}/dx_q``.
This is the form of moss commit f721a6a; msd's pre-fix code has lam and mu
swapped on the first two terms (invisible at nu = 1/4).

The same weighted moments give the kernels of a FORCE source (the Kelvin
single layer), i.e. a force per unit area ``f`` distributed over the triangle:

    G[i,j]     = C1 [ (c34 I1 + 2(1-nu) eps^2 I3) d_ij + T2_3[i,j] ]   (force -> displacement)
    S[i,j,c]   = C_{ijab} G1[a,c,b]
               = lam d_ij sum_a G1[a,c,a] + mu G1[i,c,j] + mu G1[j,c,i]  (force -> stress)

``S`` reuses the very same ``G1`` as ``U`` (hence the shared block below; keep
that expression and the in-place ``G1 *= C1`` untouched so ``U`` stays bitwise
identical).  The force element carries no ``nhat`` dependence at all, and no
eigenstress: a mollified body force is a genuine body force, not an
eigenstrain, so ``S`` is already the elastic stress.  An exact identity ties
the two families together (gated in verify_identities.py):

    U[i,j] = -sum_m n_m S[j,m,i].
"""
from __future__ import annotations

from itertools import product

import numpy as np

from .frame import Frame, local_frame
from .moments import weighted_tables, kernel_degrees, h0_floor
from .shape import nodes


def _lame(mu, nu):
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    return lam


def lift(W: dict[int, np.ndarray], z: np.ndarray, frame: Frame, rank: int, n: int,
         *, floor: int = 0, h0: np.ndarray | None = None) -> np.ndarray:
    """Rank-``rank`` tensor moment (N, K, 3, ..., 3) at ``R^-n`` from the
    weighted table ``W[n]`` (N, K, D+1, D+1).

    ``floor``/``h0``: on the ``h = 0`` rows the table has no entry below the
    degree floor (it is NaN).  Such a slot always carries ``c = rank - (a+b)
    >= 1`` normal indices and ``z`` is exactly ``0.0`` there, so the term is
    bitwise zero -- but ``0.0 * nan`` is NaN, so it is masked rather than
    multiplied."""
    basis = [(-1.0, frame.e1), (-1.0, frame.e2), (None, frame.nhat)]
    Wn = W[n]
    N, K = Wn.shape[:2]
    out = np.zeros((N, K) + (3,) * rank)
    zp = {}
    masked = floor > 0 and h0 is not None and bool(np.any(h0))
    for assign in product(range(3), repeat=rank):
        a = assign.count(0)
        b = assign.count(1)
        c = rank - a - b
        if c not in zp:
            zp[c] = z ** c
        sign = (-1.0) ** (a + b)
        Wab = Wn[:, :, a, b]
        if masked and a + b < floor:
            assert c >= 1, "a sub-floor slot must carry a normal index"
            Wab = np.where(h0[:, None], 0.0, Wab)
        scal = sign * zp[c][:, None] * Wab                      # (N, K)
        vecs = [basis[s][1] for s in assign]
        outer = vecs[0]
        for v in vecs[1:]:
            outer = np.multiply.outer(outer, v)
        out += scal.reshape((N, K) + (1,) * rank) * outer
    return out


def nodal_influence(obs, tri, order: int, mu: float, nu: float, eps: float,
                    want=("U", "H", "E"), far_field: str = "hybrid",
                    check_identity: bool = False):
    """Nodal influence tensors of one triangle at ``obs`` (N, 3).

    Returns a dict with the requested entries -- slip source: ``U`` (N, K, 3, 3),
    ``H`` (N, K, 3, 3, 3), ``E`` (N, K); force source: ``G`` (N, K, 3, 3),
    ``S`` (N, K, 3, 3, 3) -- plus ``nodes``.
    """
    frame = local_frame(tri)
    obs = np.asarray(obs, float).reshape(-1, 3)
    W, z, X, ident, h0 = weighted_tables(frame, obs, eps, order, want, far_field,
                                         check_identity=check_identity)
    floor = h0_floor(kernel_degrees(order, want)) if np.any(h0) else {}
    lam = _lame(mu, nu)
    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    e2 = eps * eps
    n = frame.nhat
    eye = np.eye(3)
    out = {"nodes": nodes(tri, order), "identity_residual": ident}

    if "U" in want or "S" in want:
        V3 = lift(W, z, frame, 1, 3)          # (N,K,3)
        V5 = lift(W, z, frame, 1, 5)
        T35 = lift(W, z, frame, 3, 5)         # (N,K,3,3,3)
        G1 = (-c34 * np.einsum("ij,nkm->nkijm", eye, V3)
              + np.einsum("im,nkj->nkijm", eye, V3)
              + np.einsum("jm,nki->nkijm", eye, V3)
              - 3.0 * T35
              - 6.0 * (1.0 - nu) * e2 * np.einsum("ij,nkm->nkijm", eye, V5))
        G1 *= C1

    if "U" in want:
        term1 = mu * np.einsum("m,nkijm->nkij", n, G1)
        term2 = lam * np.einsum("j,nkimm->nkij", n, G1)
        term3 = mu * np.einsum("m,nkimj->nkij", n, G1)
        out["U"] = -(term1 + term2 + term3)

    if "G" in want:
        # force density -> displacement (Kelvin single layer)
        I1g = W[1][:, :, 0, 0]
        T23 = lift(W, z, frame, 2, 3, floor=floor.get(3, 0), h0=h0)   # (N,K,3,3)
        G = c34 * np.einsum("ij,nk->nkij", eye, I1g) + T23
        if eps != 0.0:
            G = G + 2.0 * (1.0 - nu) * e2 * np.einsum("ij,nk->nkij",
                                                      eye, W[3][:, :, 0, 0])
        out["G"] = C1 * G

    if "S" in want:
        # force density -> stress: S[i,j,c] = C_ijab G1[a,c,b]
        trG1 = np.einsum("nkaca->nkc", G1)
        out["S"] = (lam * np.einsum("ij,nkc->nkijc", eye, trG1)
                    + mu * np.einsum("nkicj->nkijc", G1)
                    + mu * np.einsum("nkjci->nkijc", G1))

    if "H" in want:
        I3 = W[3][:, :, 0, 0]
        I5 = W[5][:, :, 0, 0]
        T25 = lift(W, z, frame, 2, 5)
        T27 = lift(W, z, frame, 2, 7)
        T47 = lift(W, z, frame, 4, 7)
        D2 = np.zeros(T47.shape)
        D2 += -c34 * (np.einsum("rp,sq,nk->nkrpsq", eye, eye, I3)
                      - 3.0 * np.einsum("rp,nksq->nkrpsq", eye, T25))
        D2 += (np.einsum("rs,pq,nk->nkrpsq", eye, eye, I3)
               - 3.0 * np.einsum("rs,nkpq->nkrpsq", eye, T25))
        D2 += (np.einsum("ps,rq,nk->nkrpsq", eye, eye, I3)
               - 3.0 * np.einsum("ps,nkrq->nkrpsq", eye, T25))
        D2 += -3.0 * (np.einsum("rq,nkps->nkrpsq", eye, T25)
                      + np.einsum("pq,nkrs->nkrpsq", eye, T25)
                      + np.einsum("sq,nkrp->nkrpsq", eye, T25))
        D2 += 15.0 * T47
        D2 += 2.0 * (1.0 - nu) * e2 * (
            -3.0 * np.einsum("rp,sq,nk->nkrpsq", eye, eye, I5)
            + 15.0 * np.einsum("rp,nksq->nkrpsq", eye, T27))
        D2 *= C1
        trD = np.einsum("nkrpsp->nkrs", D2)
        B = (lam * np.einsum("j,nkrs->nkrsj", n, trD)
             + mu * np.einsum("q,nkrjsq->nkrsj", n, D2)
             + mu * np.einsum("p,nkrpsj->nkrsj", n, D2))
        trB = np.einsum("nkrrj->nkj", B)
        H = -(lam * np.einsum("ml,nkj->nkmlj", eye, trB)
              + mu * (B + np.swapaxes(B, 2, 3)))
        out["H"] = H

    if "E" in want:
        out["E"] = (15.0 * eps ** 4 / (8.0 * np.pi)) * W[7][:, :, 0, 0]

    return out
