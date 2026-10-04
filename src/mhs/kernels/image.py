"""Closed-form integration of the half-space IMAGE R-family over a triangle.

WHY THIS IS CLOSED FORM AND THE Q-FAMILY IS NOT. The mollified Mindlin kernel
splits by radical into a direct part (``R1``), an image part (``R2``) and a
``Q = R2 - (z + z0)`` part. ``verify_vertical_fault`` clause [e] measures what
each costs for ON-FAULT STRESS on a surface-breaking element, which is the
quantity this package exists to produce:

  - buried element: every family converges at n_quad = 4, because the image is
    far compared with the element and the integrand is smooth.
  - element reaching z = 0, observer on it near the surface trace: the image
    triangle TOUCHES the observer -- for a vertical fault it is coplanar and
    shares that edge -- and the R-family STAGNATES under uniform refinement.
    Sixty-four times the points moves its error from 0.96 to 0.58.

The Q-family does NOT stagnate, but neither is its error eps-independent, which
an earlier version of this module claimed on the strength of a buried-element
measurement. It obeys a budget law (:func:`q_gauss_orders`) and is integrated
at a per-observer order from it.

The escape quadrature has is to raise eps until the band hides the touching
image, and eps is a resolution floor, so that buys accuracy by smearing the
fault. This module removes that trade for the R-family, which is the strongly
singular one: it reaches ``R2^-9`` where the Q-family reaches ``Q^-3``.

THE REDUCTION, in one line, with the derivation in ``tools/gen_image_table.py``:
``D = x - y'`` for ``y'`` the reflected source IS the Mindlin convention's
``(x, y, z + z0)``, so ``z0 = D3 - z`` and every R-family term is
``coeff(z, mu, nu) * D1^a D2^b D3^c / R2^n`` with ``n`` odd. Reflection is an
isometry, so integrating that over the element equals integrating over the
REFLECTED element -- which is exactly a tensor moment from clq's hierarchy, at
no new primitives. ``n`` reaches 9 (three derivatives act on the potentials'
``R2^-3`` term), which is why the vertical seed ladder in ``fullspace.moments``
is a loop rather than a case list.

SIGN CONVENTION, measured rather than argued. The contractions here follow the
moss oracle's ``_stiffness_contract`` / ``_stiffness_contract_DDG``, which carry
no leading minus where clq's ``U`` and ``H`` expressions do -- which looks like
a mismatch and is not: clq's minus is already absorbed into how it builds
``G1``/``D2``, and the two agree to 3.5e-16 (U) and 1.1e-15 (H) on the same
direct term. So the direct and image halves simply ADD, with no sign fix
anywhere. This paragraph exists because the first draft asserted the opposite,
and a relative sign error between the two halves would be invisible at a
symmetric configuration.
"""
from __future__ import annotations

import numpy as np

from .. import defaults
from ..fullspace.frame import local_frame
from ..fullspace.kernels import lift
from ..fullspace.moments import gauss_triangle, weighted_tables
from ..fullspace.shape import shape_coefficients
from ._image_table import (DDG_RECORDS, DG_RECORDS, MAX_RANK, N_DDG, N_DG,
                           N_ORDERS, N_Q_DDG, N_Q_DG, Q_DDG_RECORDS,
                           Q_DG_RECORDS)

#: Reflection in the free surface. The free surface is z = 0 and the body is
#: z <= 0 (checked in matrices._coerce), so this is the whole of the geometry.
_REFLECT = np.array([1.0, 1.0, -1.0])

# A truncated or hand-edited table would silently halve a kernel, so the
# consumer asserts the shape it was generated with.
assert len(DG_RECORDS) == N_DG, "DG_RECORDS truncated"
assert len(DDG_RECORDS) == N_DDG, "DDG_RECORDS truncated"
assert len(Q_DG_RECORDS) == N_Q_DG, "Q_DG_RECORDS truncated"
assert len(Q_DDG_RECORDS) == N_Q_DDG, "Q_DDG_RECORDS truncated"


def _degrees() -> dict[int, int]:
    """Constant-slip table degree per ``n``, read off the generated records.

    Derived rather than written down: the table is generated, so a hand-kept
    degree map would be a second statement of the same fact and could go stale
    against it.
    """
    need: dict[int, int] = {}
    for rec in DG_RECORDS:
        _, _, _, a, b, c, n = rec[:7]
        need[n] = max(need.get(n, -1), a + b + c)
    for rec in DDG_RECORDS:
        _, _, _, _, a, b, c, n = rec[:8]
        need[n] = max(need.get(n, -1), a + b + c)
    return need


DEGREES = _degrees()
assert sorted(DEGREES) == list(N_ORDERS), "table orders disagree with N_ORDERS"
assert max(DEGREES.values()) == MAX_RANK, "table rank disagrees with MAX_RANK"


def _coeff(terms, zp: np.ndarray, nu: float, scale: np.ndarray) -> np.ndarray:
    """``sum num/den * z^pz * nu^pnu`` times the common scale, as an (N,) array.

    ``scale`` carries ``1/(pi mu (1 - nu))`` built from ``(mu, lam)`` -- see
    :func:`image_influence` for why it is not formed from ``nu``.
    """
    out = np.zeros_like(zp)
    for pz, pnu, num, den in terms:
        out += (num / den) * (zp ** pz) * (nu ** pnu)
    return out * scale


def _contract(DG, DDG, nrm, mu: float, lam: float, want) -> dict:
    """The traction-operator contractions, shared by the R and Q paths.

    ``DG[n,k,i,j,m] = int N_k dG_ij/d src_m`` gives

        U[i,k] = mu n_l DG[i,k,l] + mu n_l DG[i,l,k] + lam n_k DG[i,m,m]

    and ``DDG[n,k,i,j,p,m]`` gives the displacement-gradient kernel
    ``S[i,p,k]``, from which Hooke on the symmetrised gradient gives the stress.
    Slip pairs with the normal in the FIRST index pair of C, matching the moss
    oracle's ``_stiffness_contract``; note that convention carries no leading
    minus where clq's ``U``/``H`` do.

    Written as explicit loops over the free indices rather than one einsum: the
    lam term contracts DG's own j/m pair and then multiplies ``n_k`` on the SLIP
    index, which an einsum spelling makes easy to get subtly wrong and hard to
    read. One copy, because a contraction written twice is a convention written
    twice.
    """
    out: dict[str, np.ndarray] = {}
    if "U" in want:
        n_obs, n_node = DG.shape[:2]
        tr = sum(DG[:, :, :, m, m] for m in range(3))         # (N, K, i)
        U = np.zeros((n_obs, n_node, 3, 3))
        for i in range(3):
            for k in range(3):
                U[:, :, i, k] = (
                    mu * sum(nrm[l] * DG[:, :, i, k, l] for l in range(3))
                    + mu * sum(nrm[l] * DG[:, :, i, l, k] for l in range(3))
                    + lam * nrm[k] * tr[:, :, i])
        out["U"] = U
    if "H" in want:
        n_obs, n_node = DDG.shape[:2]
        S = np.zeros((n_obs, n_node, 3, 3, 3))
        for i in range(3):
            for p in range(3):
                tr_m = sum(DDG[:, :, i, m, p, m] for m in range(3))
                for k in range(3):
                    S[:, :, i, p, k] = (
                        mu * sum(nrm[l] * DDG[:, :, i, k, p, l]
                                 for l in range(3))
                        + mu * sum(nrm[l] * DDG[:, :, i, l, p, k]
                                   for l in range(3))
                        + lam * nrm[k] * tr_m)
        H = np.zeros((n_obs, n_node, 3, 3, 3))
        for k in range(3):
            g = S[:, :, :, :, k]                              # (N, K, i, p)
            e = 0.5 * (g + np.swapaxes(g, -1, -2))
            trace = e[:, :, 0, 0] + e[:, :, 1, 1] + e[:, :, 2, 2]
            for a_ in range(3):
                for b_ in range(3):
                    H[:, :, a_, b_, k] = 2.0 * mu * e[:, :, a_, b_]
                    if a_ == b_:
                        H[:, :, a_, b_, k] += lam * trace
        out["H"] = H
    return out


def _unit_normal(tri: np.ndarray) -> np.ndarray:
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    return n / np.linalg.norm(n)


def triangle_distance(obs: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Exact distance from each observer to the closed triangle, as ``(N,)``.

    Exact rather than a vertex/centroid surrogate, because a surrogate
    OVERestimates the distance while the quadrature budget is inversely
    proportional to it -- so an overestimate silently starves the rule. The
    plane distance would be a safe UNDERestimate but is useless here: for a
    vertical fault the image triangle is COPLANAR with the element, so it is
    zero for every observer and would demand the maximum order everywhere.
    """
    obs = np.asarray(obs, float).reshape(-1, 3)
    a, b, c = tri[0], tri[1], tri[2]
    ab, ac = b - a, c - a
    nrm = np.cross(ab, ac)
    ap = obs - a
    d20 = ap @ ab
    d21 = ap @ ac
    d00 = float(ab @ ab)
    d01 = float(ab @ ac)
    d11 = float(ac @ ac)
    den = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    u = 1.0 - v - w
    inside = (u >= 0.0) & (v >= 0.0) & (w >= 0.0)
    out = np.empty(obs.shape[0])
    if np.any(inside):
        out[inside] = np.abs(ap[inside] @ nrm) / np.linalg.norm(nrm)
    if np.any(~inside):
        sel = ~inside
        best = np.full(int(sel.sum()), np.inf)
        for p0, p1 in ((a, b), (b, c), (c, a)):
            e = p1 - p0
            t = np.clip(((obs[sel] - p0) @ e) / float(e @ e), 0.0, 1.0)
            closest = p0[None, :] + t[:, None] * e[None, :]
            best = np.minimum(best,
                              np.linalg.norm(obs[sel] - closest, axis=1))
        out[sel] = best
    return out


def q_gauss_orders(obs: np.ndarray, tri_img: np.ndarray, eps: float,
                   frame_L: float) -> np.ndarray:
    """Gauss order PER OBSERVER, from the measured budget law

        n_quad ~ C * L / sqrt(delta^2 + eps^2)

    with ``delta`` the distance to the IMAGE triangle. This is the SAME law the
    direct term obeys -- ``n_quad ~ 8 L / eps``, gated in
    ``oracle/verify_vertical_fault`` clause [b] -- and with the same constant:
    the direct term's observer sits ON its own element, so ``delta = 0`` and
    the scale is ``eps``. Here the observer is ``delta`` from the image, so the
    scale is ``sqrt(delta^2 + eps^2)``. One law, two cases, which is why the
    constant did not have to be fitted separately.

    Measured ``C`` over delta/h in {0.02 .. 0.33} and eps/h in {0.01 .. 0.1}:
    5.8 to 8.4 for 1e-9 relative. ``defaults.IMAGE_Q_BUDGET_C`` carries
    headroom over the MAXIMUM rather than the mean, because starving this rule
    is silent: the first shipped default was a flat 16 chosen from a BURIED
    element, and it left the P1/P2 collocation point at 7e-4.
    """
    delta = triangle_distance(obs, tri_img)
    scale = np.sqrt(delta * delta + float(eps) ** 2)
    nq = np.ceil(defaults.IMAGE_Q_BUDGET_C * float(frame_L) / scale)
    return np.clip(nq, defaults.IMAGE_Q_GAUSS_MIN,
                   defaults.IMAGE_Q_GAUSS_MAX).astype(int)


def _q_block(obs, tri, frame, order, nu, scale_c, eps, nq, want, n_node):
    """Q-family moments for one group of observers at ONE Gauss order.

    Returns ``(DG, DDG)``, either possibly ``None`` if not requested.
    """
    x1, x2, w = gauss_triangle(nq)
    bary = np.stack([1.0 - x1 - x2, x1, x2], axis=1)          # (Q, 3)
    ypts = bary @ tri                                         # (Q, 3)
    wq = w * (2.0 * frame.area)
    # N_k at the quadrature points, in the centroid frame -- the same device
    # moments.quadrature_weighted_tables uses, so the nodal layout is clq's.
    eta = bary @ frame.p                                      # (Q, 2)
    c0 = shape_coefficients(frame, order, np.zeros((1, 2)))[0]   # (K, D, D)
    D = c0.shape[1]
    Nq = np.zeros((ypts.shape[0], n_node))
    for a in range(D):
        for b in range(D - a):
            Nq += c0[:, a, b][None, :] * (eta[:, 0] ** a
                                          * eta[:, 1] ** b)[:, None]

    # D = obs - reflected source, so D3 = obs_z + y_z
    d1 = obs[:, 0:1] - ypts[None, :, 0]
    d2 = obs[:, 1:2] - ypts[None, :, 1]
    d3 = obs[:, 2:3] + ypts[None, :, 2]
    r2 = np.sqrt(d1 * d1 + d2 * d2 + d3 * d3 + eps ** 2)
    qq = r2 - d3
    z_obs = obs[:, 2:3]

    def integrand(terms, a, b, c, n, q):
        cv = np.zeros_like(z_obs)
        for pz, pnu, num, den in terms:
            cv += (num / den) * (z_obs ** pz) * (nu ** pnu)
        return (cv * scale_c) * (d1 ** a) * (d2 ** b) * (d3 ** c) \
            / (r2 ** n * qq ** q)

    n_sub = obs.shape[0]
    dg = ddg = None
    if "U" in want:
        dg = np.zeros((n_sub, n_node, 3, 3, 3))
        for rec in Q_DG_RECORDS:
            i, j, m, a, b, c, n, q = rec[:8]
            dg[:, :, i, j, m] += (integrand(rec[8], a, b, c, n, q)
                                  * wq[None, :]) @ Nq
    if "H" in want:
        ddg = np.zeros((n_sub, n_node, 3, 3, 3, 3))
        for rec in Q_DDG_RECORDS:
            i, j, p, m, a, b, c, n, q = rec[:9]
            ddg[:, :, i, j, p, m] += (integrand(rec[9], a, b, c, n, q)
                                      * wq[None, :]) @ Nq
    return dg, ddg


def image_q_influence(obs, tri, order: int, mu: float, lam: float, eps: float,
                      want=("U", "H"), n_quad: int | None = None):
    """Q-family contribution, by Gauss quadrature on the element.

    NOT closed form. It does not have to be, because unlike the R-family it
    does not stagnate: it CONVERGES, at an order set by the budget law in
    :func:`q_gauss_orders`. ``n_quad=None`` (the default) applies that law per
    observer and groups by the required order; an explicit ``n_quad`` forces
    one order everywhere, which is what the gates sweep.

    The order genuinely varies: at eps/h = 0.01 on a surface-breaking element
    the law asks for 34 at the P0 collocation point, 67 at the P1/P2 one, and
    the ceiling close to the trace. A flat 16 -- which this module shipped
    first, chosen from a BURIED element where the error really is
    eps-independent -- left the P1/P2 point at 7e-4 and a readout 0.03 h below
    the trace at O(1).

    Omitting the family is not an option either: near the trace it is roughly a
    fifth of the on-fault stress, so dropping it would be wrong by a
    plausible-looking amount -- the failure ``matrices._not_yet`` refused.
    """
    obs = np.asarray(obs, float).reshape(-1, 3)
    tri = np.asarray(tri, float).reshape(3, 3)
    mu = float(mu)
    lam = float(lam)
    nu = lam / (2.0 * (lam + mu))
    scale_c = 2.0 * (lam + mu) / (np.pi * mu * (lam + 2.0 * mu))

    frame = local_frame(tri)
    tri_img = tri * _REFLECT
    n_obs = obs.shape[0]
    n_node = (order + 1) * (order + 2) // 2

    # Per-observer Gauss order from the budget law, unless the caller forces
    # one. Grouping by the required order is the same device weighted_tables
    # uses for its far-field orders: accuracy where the geometry needs it,
    # without paying for it at every observer.
    if n_quad is None:
        orders = q_gauss_orders(obs, tri_img, eps, frame.L)
    else:
        orders = np.full(n_obs, int(n_quad))

    DG = np.zeros((n_obs, n_node, 3, 3, 3)) if "U" in want else None
    DDG = np.zeros((n_obs, n_node, 3, 3, 3, 3)) if "H" in want else None
    for nq in np.unique(orders):
        sel = orders == nq
        dg, ddg = _q_block(obs[sel], tri, frame, order, nu, scale_c,
                           float(eps), int(nq), want, n_node)
        if DG is not None:
            DG[sel] = dg
        if DDG is not None:
            DDG[sel] = ddg
    return _contract(DG, DDG, _unit_normal(tri), mu, lam, want)


def image_influence(obs, tri, order: int, mu: float, lam: float, eps: float,
                    want=("U", "H"), far_field: str = "hybrid"):
    """Image R-family contribution, integrated in closed form over ``tri``.

    Returns ``{"U": (N, K, 3, 3), "H": (N, K, 3, 3, 3)}`` for the requested
    kernels, with ``U[n, k, i, j]`` the displacement ``i`` at ``obs[n]`` from a
    unit Cartesian slip ``j`` on node ``k``, and ``H[n, k, a, b, j]`` the
    stress. Only the entries named in ``want`` are built.

    Material input is ``(mu, lam)`` and never ``nu`` -- but the generated table
    is written in ``nu``, so the conversion happens HERE, once, in the safe
    direction: ``nu = lam / (2 (lam + mu))`` and
    ``1/(pi mu (1 - nu)) = 2 (lam + mu) / (pi mu (lam + 2 mu))``. Neither has a
    pole for ``mu > 0, lam >= 0``; it is ``nu -> lam`` that diverges at
    ``nu = 1/2``, which is the direction this avoids.
    """
    obs = np.asarray(obs, float).reshape(-1, 3)
    tri = np.asarray(tri, float).reshape(3, 3)
    mu = float(mu)
    lam = float(lam)
    nu = lam / (2.0 * (lam + mu))
    scale_c = 2.0 * (lam + mu) / (np.pi * mu * (lam + 2.0 * mu))

    tri_img = tri * _REFLECT
    frame_img = local_frame(tri_img)
    # `want=()` because the degree spec is supplied directly: the full-space
    # module is given a requirement, not told whose kernel it is.
    W, zp, _X, _ident, _h0 = weighted_tables(frame_img, obs, eps, order, (),
                                             far_field=far_field,
                                             degrees=DEGREES)
    n_obs = obs.shape[0]
    n_node = next(iter(W.values())).shape[1]
    z_obs = obs[:, 2]                      # ABSOLUTE depth, not plane height
    scale = np.full(n_obs, scale_c)

    cache: dict[tuple[int, int], np.ndarray] = {}

    def moment(a: int, b: int, c: int, n: int) -> np.ndarray:
        """int N_k D1^a D2^b D3^c / R2^n dS, as (N, K)."""
        rank = a + b + c
        if rank == 0:
            return W[n][:, :, 0, 0]        # a scalar moment has no tensor slots
        if (rank, n) not in cache:
            cache[(rank, n)] = lift(W, zp, frame_img, rank, n)
        idx = (0,) * a + (1,) * b + (2,) * c
        return cache[(rank, n)][(slice(None), slice(None)) + idx]

    nrm = _unit_normal(tri)                # the REAL element's normal

    DG = DDG = None
    if "U" in want:
        DG = np.zeros((n_obs, n_node, 3, 3, 3))
        for rec in DG_RECORDS:
            i, j, m, a, b, c, n = rec[:7]
            DG[:, :, i, j, m] += (_coeff(rec[7], z_obs, nu, scale)[:, None]
                                  * moment(a, b, c, n))
    if "H" in want:
        DDG = np.zeros((n_obs, n_node, 3, 3, 3, 3))
        for rec in DDG_RECORDS:
            i, j, p, m, a, b, c, n = rec[:8]
            DDG[:, :, i, j, p, m] += (_coeff(rec[8], z_obs, nu, scale)[:, None]
                                      * moment(a, b, c, n))
    return _contract(DG, DDG, nrm, mu, lam, want)


def image_total(obs, tri, order: int, mu: float, lam: float, eps: float,
                want=("U", "H"), far_field: str = "hybrid",
                n_quad: int | None = None):
    """The whole image correction: closed-form R-family plus quadrature Q.

    This is what the half-space kernel actually needs, and the split is not a
    convenience -- it is the measured one. See ``image_influence`` for the
    family that must be closed form and ``image_q_influence`` for the one that
    must not be omitted.
    """
    r = image_influence(obs, tri, order, mu, lam, eps, want=want,
                        far_field=far_field)
    q = image_q_influence(obs, tri, order, mu, lam, eps, want=want,
                          n_quad=n_quad)
    return {k: r[k] + q[k] for k in r}
