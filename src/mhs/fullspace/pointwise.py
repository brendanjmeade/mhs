"""Point-source mollified kernels (oracle bricks, scalar/vectorised numpy).

``G^eps`` is the Galerkin/Cortez-convolved mollified Kelvin solution

    G_ij(d) = C1 [ (3-4nu) d_ij / R + d_i d_j / R^3 + 2(1-nu) eps^2 d_ij / R^3 ],
    R = sqrt(|d|^2 + eps^2),   C1 = 1/(16 pi mu (1-nu)),   d = x - y,

the exact displacement of a point force smeared by the blob
``phi(r) = 15 eps^4 / (8 pi R^7)`` (``L G^eps = -phi I``).  These routines
are the reference for the closed-form integrals; they use the same
(traction-operator) index pairing as :mod:`clq.kernels`.
"""
from __future__ import annotations

import numpy as np


def blob(d, eps):
    d = np.asarray(d, float).reshape(-1, 3)
    R2 = np.einsum("ni,ni->n", d, d) + eps ** 2
    return 15.0 * eps ** 4 / (8.0 * np.pi * R2 ** 3.5)


def marginal(perp, eps):
    """Fault-normal marginal of the blob integrated over an infinite plane:
    (3/4) eps^4 / (perp^2 + eps^2)^(5/2)."""
    perp = np.asarray(perp, float)
    return 0.75 * eps ** 4 / (perp ** 2 + eps ** 2) ** 2.5


def kelvin_G(d, mu, nu, eps):
    d = np.asarray(d, float).reshape(-1, 3)
    R2 = np.einsum("ni,ni->n", d, d) + eps ** 2
    R = np.sqrt(R2)
    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    cb = 2.0 * (1.0 - nu) * eps ** 2
    eye = np.eye(3)
    G = (c34 / R + cb / R ** 3)[:, None, None] * eye[None] + np.einsum("ni,nj->nij", d, d) / R[:, None, None] ** 3
    return C1 * G


def kelvin_dG(d, mu, nu, eps):
    """DG[n, i, j, m] = dG_ij/dx_m."""
    d = np.asarray(d, float).reshape(-1, 3)
    R2 = np.einsum("ni,ni->n", d, d) + eps ** 2
    R = np.sqrt(R2)
    iR3 = 1.0 / (R * R2)
    iR5 = iR3 / R2
    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    cb = 6.0 * (1.0 - nu) * eps ** 2
    eye = np.eye(3)
    DG = (-c34 * np.einsum("ij,nm,n->nijm", eye, d, iR3)
          + np.einsum("im,nj,n->nijm", eye, d, iR3)
          + np.einsum("jm,ni,n->nijm", eye, d, iR3)
          - 3.0 * np.einsum("ni,nj,nm,n->nijm", d, d, d, iR5)
          - cb * np.einsum("ij,nm,n->nijm", eye, d, iR5))
    return C1 * DG


def force_stress_point(d, mu, nu, eps):
    """S[n, i, j, c] = C_ijab dG_ac/dx_b: stress ij at x per unit point force
    in direction c at y (the Kelvin single layer's stress kernel)."""
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    DG = kelvin_dG(d, mu, nu, eps)                 # [n,i,j,m] = dG_ij/dx_m
    tr = np.einsum("naca->nc", DG)
    eye = np.eye(3)
    return (lam * np.einsum("ij,nc->nijc", eye, tr)
            + mu * np.einsum("nicj->nijc", DG)
            + mu * np.einsum("njci->nijc", DG))


def kelvin_d2G(d, mu, nu, eps):
    """D2G[n, r, p, s, q] = d^2 G_rp / dx_s dx_q."""
    d = np.asarray(d, float).reshape(-1, 3)
    R2 = np.einsum("ni,ni->n", d, d) + eps ** 2
    R = np.sqrt(R2)
    iR3 = 1.0 / (R * R2)
    iR5 = iR3 / R2
    iR7 = iR5 / R2
    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    cb = 2.0 * (1.0 - nu) * eps ** 2
    eye = np.eye(3)
    dd = np.einsum("ni,nj->nij", d, d)
    A3 = np.einsum("sq,n->nsq", eye, iR3) - 3.0 * dd * iR5[:, None, None]   # d_sq/R^3 - 3 d_s d_q/R^5
    D2 = (-c34 * np.einsum("rp,nsq->nrpsq", eye, A3)
          + np.einsum("rs,npq->nrpsq", eye, A3)
          + np.einsum("ps,nrq->nrpsq", eye, A3)
          - 3.0 * (np.einsum("rq,nps,n->nrpsq", eye, dd, iR5)
                   + np.einsum("pq,nrs,n->nrpsq", eye, dd, iR5)
                   + np.einsum("sq,nrp,n->nrpsq", eye, dd, iR5))
          + 15.0 * np.einsum("nr,np,ns,nq,n->nrpsq", d, d, d, d, iR7)
          + cb * np.einsum("rp,nsq->nrpsq", eye,
                           -3.0 * np.einsum("sq,n->nsq", eye, iR5) + 15.0 * dd * iR7[:, None, None]))
    return C1 * D2


def dd_displacement_point(d, nhat, mu, nu, eps):
    """U[n, i, j]: displacement i at x per unit slip component j at a point
    source y with normal nhat (d = x - y).  Traction-operator pairing."""
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    DG = kelvin_dG(d, mu, nu, eps)
    n = np.asarray(nhat, float)
    t1 = mu * np.einsum("m,nijm->nij", n, DG)
    t2 = lam * np.einsum("j,nimm->nij", n, DG)
    t3 = mu * np.einsum("m,nimj->nij", n, DG)
    return -(t1 + t2 + t3)


def dd_stress_point(d, nhat, mu, nu, eps):
    """K[n, m, l, j]: TOTAL stress ml at x per unit slip component j."""
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    D2 = kelvin_d2G(d, mu, nu, eps)
    n = np.asarray(nhat, float)
    eye = np.eye(3)
    trD = np.einsum("nrpsp->nrs", D2)
    B = (lam * np.einsum("j,nrs->nrsj", n, trD)
         + mu * np.einsum("q,nrjsq->nrsj", n, D2)
         + mu * np.einsum("p,nrpsj->nrsj", n, D2))
    trB = np.einsum("nrrj->nj", B)
    return -(lam * np.einsum("ml,nj->nmlj", eye, trB) + mu * (B + np.swapaxes(B, 1, 2)))
