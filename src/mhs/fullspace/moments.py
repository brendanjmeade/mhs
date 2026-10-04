"""Two-dimensional moment tables over a triangle.

For one triangle ``T`` and ``N`` observation points ``x`` the kernels need

    M_n^{(a,b)}(x) = int_T xi_1^a xi_2^b / R_eps^n dS,
    R_eps^2 = xi_1^2 + xi_2^2 + h_eps^2,   h_eps^2 = z^2 + eps^2,

with ``(xi_1, xi_2)`` in-plane coordinates about the projection of ``x`` and
``z`` its signed height.  The table is built by the divergence-theorem
recursion of the manuscript appendix (eqs. Mrec_x / Mrec_y), generalised to
arbitrary slip order and every odd ``n`` (negative ``n`` = positive powers of
``R``, seeded downward by the vertical identity):

    M_n^{(a+1,b)} = [ a M_{n-2}^{(a-1,b)} - B_{n-2}^{1}(a,b) ] / (n - 2),
    M_n^{(0,b+1)} = [ b M_{n-2}^{(0,b-1)} - B_{n-2}^{2}(0,b) ] / (n - 2),
    B_m^{alpha}(a,b) = sum_edges nu_alpha int_edge xi_1^a xi_2^b / R^m du,

seeded by the solid angle ``I_3 = -Omega(x_eff)/h`` and the vertical identity
``(2-n) I_n + n h^2 I_{n+2} = E_n`` (``I_1 = E_1 - h^2 I_3``,
``I_5 = (E_3 + I_3)/(3 h^2)``, ``I_7 = (E_5 + 3 I_5)/(5 h^2)``,
``I_{-1} = (E_{-1} + h^2 I_1)/3``).  Edge integrals come from
:func:`clq.primitives.edge_table`.

Per-node WEIGHTED tables ``W_k[n][a, b] = sum_{a',b'} c_k[a',b'] M_n^{(a+a',b+b')}``
carry the shape-function polynomial ``N_k(xi) = sum c_k[a',b'] xi_1^a' xi_2^b'``
(:func:`clq.shape.shape_coefficients`); every kernel is a fixed contraction of
``W_k`` (:mod:`clq.kernels`), identical for every slip order.

Far field: the closed form is exact in exact arithmetic, but in floating
point it loses digits roughly like ``(R/L)^4-5`` for quadratic weights.  The
loss is intrinsic to the divergence theorem; see docs/derivation.md and the
README table.  Measured worst relative error for well-shaped triangles at
eps = 0.05 L: ~2e-9 at R = 10 L, 2e-4 at 100 L, O(1) by 500 L.  It is worse
for smaller eps and much worse for thin triangles.  ``far_field="analytic"``
is therefore unusable beyond ~20 L.  In the default "hybrid" mode,
:func:`weighted_tables` switches to :func:`quadrature_weighted_tables`
(Gauss product rule of the smooth integrand, same output, ~1e-14) for
observers with ``sqrt(|x - centroid|^2 + eps^2) > D_STAR * L``.
"""
from __future__ import annotations

import functools

from math import comb

import numpy as np

from . import defaults
from .frame import Frame
from .primitives import edge_table, solid_angle
from .shape import shape_coefficients


# ---------------------------------------------------------------------------
# Degree bookkeeping
# ---------------------------------------------------------------------------

#: kernels whose h = 0 (eps = 0 on the element) limit exists -- the Kelvin
#: single layer is weakly singular there, unlike every dislocation kernel.
ON_PLANE_LIMIT_KERNELS = frozenset({"G"})


def kernel_degrees(order: int, want) -> dict[int, int]:
    """Required table degree ``a+b`` per ``n`` for the requested kernels with
    nodal order ``order``.

    Slip (displacement-discontinuity) source: ``"U"`` displacement, ``"H"``
    stress, ``"E"`` eigenstress.  Force (Kelvin single-layer) source: ``"G"``
    displacement, ``"S"`` stress.  Constant-density needs:
    U -> {3: 1, 5: 3};  H -> {3: 0, 5: 2, 7: 4};  E -> {7: 0};
    G -> {1: 0, 3: 2};  S -> {3: 1, 5: 3};  each raised by ``order``.

    ``"S"`` closes to exactly the ``"U"`` degrees (both read the integrated
    gradient ``G1``), so requesting them together costs no extra table work.
    ``"G"`` is the only kernel that reads the ``n = 1`` row at nonzero degree;
    from ``order >= 1`` it pulls in the ``m = -1`` edge primitives (``J_{-1}``,
    ``K_{-1}``)."""
    need: dict[int, int] = {}
    def add(n, d):
        need[n] = max(need.get(n, -1), d)
    if "U" in want:
        add(3, 1 + order); add(5, 3 + order)
    if "H" in want:
        add(3, 0 + order); add(5, 2 + order); add(7, 4 + order)
    if "E" in want:
        add(7, 0 + order)
    if "G" in want:
        add(1, 0 + order); add(3, 2 + order)
    if "S" in want:
        add(3, 1 + order); add(5, 3 + order)
    return need


def _close_degrees(need: dict[int, int]) -> dict[int, int]:
    """Add the lower-n degrees the recursion consumes: M_n of degree D needs
    M_{n-2} of degree D-2, down to whatever odd n (including negative n,
    i.e. positive powers of R) that requires; plus the vertical seeds
    I_3 -> I_5 -> I_7 upward and I_3 -> I_1 -> I_{-1} -> I_{-3} ... downward."""
    deg = dict(need)
    n = max(deg)
    while True:
        d = deg.get(n, -1)
        if d >= 2:
            deg[n - 2] = max(deg.get(n - 2, -1), d - 2)
        if n - 2 < min(deg):
            break
        n -= 2
    deg[3] = max(deg.get(3, -1), 0)
    for n in list(deg):
        if n >= 5:
            for m in range(3, n, 2):
                deg[m] = max(deg.get(m, -1), 0)
        if n <= 1:
            for m in range(n, 4, 2):
                deg[m] = max(deg.get(m, -1), 0)
    return {n: d for n, d in deg.items() if d >= 0}


def on_plane_tolerance(frame: Frame, obs: np.ndarray) -> float:
    """Absolute |z| below which an observer counts as on the plane: relative
    to the triangle size AND to the coordinate magnitude (the plane itself is
    only defined to ~eps_machine * |coordinates|)."""
    scale = max(frame.L, float(np.max(np.abs(obs))) if obs.size else 0.0,
                float(np.max(np.abs(frame.v))))
    return defaults.ON_PLANE_REL * scale


def check_h_positive(frame: Frame, obs: np.ndarray, eps: float, *,
                     allow_on_plane: bool = False) -> None:
    """Raise if eps = 0 and any observer lies on the plane (h_eps = 0).

    ``allow_on_plane`` is set only for the weakly singular force (single-layer)
    kernel, whose ``h = 0`` limit exists; see :func:`h0_floor`."""
    if eps == 0.0 and not allow_on_plane:
        z, _ = frame.to_plane(obs)
        if np.any(np.abs(z) <= on_plane_tolerance(frame, obs)):
            raise ValueError("h_eps = sqrt(z^2 + eps^2) must be > 0: eps = 0 is "
                             "only allowed for observation points off the plane")


def on_plane_mask(frame: Frame, obs: np.ndarray, eps: float) -> np.ndarray:
    """(N,) bool: rows to be treated as ``h_eps = 0`` exactly.

    The same predicate :func:`check_h_positive` rejects, so there is no gap
    between "rejected as on-plane" and "handled by the h = 0 path"."""
    obs = np.asarray(obs, float).reshape(-1, 3)
    if eps != 0.0:
        return np.zeros(obs.shape[0], dtype=bool)
    z, _ = frame.to_plane(obs)
    return np.abs(z) <= on_plane_tolerance(frame, obs)


def h0_floor(need: dict[int, int]) -> dict[int, int]:
    """Lowest AVAILABLE degree ``a+b`` per ``n`` at ``h = 0``.

    In polar coordinates about the observer ``M_n^(a,b)`` integrates
    ``r^(a+b-n+1)``, so with ``g = a+b`` the index ``g - (n-2)`` decides:
    ``>= 1`` absolutely convergent, ``= 0`` a principal value, ``<= -1``
    divergent.  The index is invariant under the horizontal recursion, so the
    floor ``g >= n-1`` (absolute convergence) is self-consistent.  Entries
    below it are left NaN and must not be read: for the force kernel the lift
    multiplies each of them by ``z^c`` with ``c >= 1``, and ``z`` is exactly
    zero there."""
    return {n: max(0, n - 1) for n in need}


# ---------------------------------------------------------------------------
# Raw moment table
# ---------------------------------------------------------------------------

class MomentTable:
    """Raw 2-D moments for one triangle and a batch of observation points.

    Attributes
    ----------
    z : (N,) signed height;  h2 : (N,) ``z^2 + eps^2``;  X : (N, 2) in-plane
        centroid-frame coordinates of the observation points.
    M : {n: (N, D_n+1, D_n+1)} with ``M[n][:, a, b]`` valid for ``a + b <= D_n``.
    """

    def __init__(self, frame: Frame, obs: np.ndarray, eps: float,
                 degrees: dict[int, int], check_identity: bool = False,
                 floor: dict[int, int] | None = None):
        obs = np.asarray(obs, float).reshape(-1, 3)
        N = obs.shape[0]
        self.frame = frame
        self.eps = float(eps)
        self.z, self.X = frame.to_plane(obs)
        self.h2 = self.z * self.z + self.eps ** 2
        # h = 0 rows (eps = 0 on the plane): allowed only when the caller
        # supplied a degree floor, i.e. for the weakly singular force kernel.
        self.h0 = (on_plane_mask(frame, obs, self.eps) if floor is not None
                   else np.zeros(N, dtype=bool))
        check_h_positive(frame, obs, self.eps, allow_on_plane=bool(self.h0.any()))
        if self.h0.any():
            # snap to the limit so z^c is bitwise zero in the lift
            self.z = np.where(self.h0, 0.0, self.z)
            self.h2 = np.where(self.h0, 0.0, self.h2)
        self.floor = dict(floor or {})
        h = np.sqrt(self.h2)
        self.degrees = _close_degrees(degrees)
        deg = self.degrees
        if self.h0.any() and max(deg) >= 5:
            raise ValueError(
                "eps = 0 with observers on the element is available only for the "
                "force (single-layer) kernel: the requested kernels need I_5/I_7, "
                "which diverge as h_eps -> 0")

        # projected vertices about the observation foot
        p = frame.p[None, :, :] - self.X[:, None, :]      # (N, 3, 2)

        # --- solid angle seed
        obs_eff = frame.centroid[None, :] + self.X[:, :1] * frame.e1 + self.X[:, 1:] * frame.e2 \
            + h[:, None] * frame.nhat[None, :]
        fin = ~self.h0
        I3 = np.full(N, np.nan)
        if np.any(fin):
            Omega = solid_angle(frame.v[0], frame.v[1], frame.v[2], obs_eff[fin])
            I3[fin] = -Omega / h[fin]

        # --- edge integrals needed: m = n - 2 with edge degree D_n - 1 for the
        #     horizontal recursion, plus BD(0,0,m) for the vertical seeds:
        #     I_5, I_7 need m = 3, 5;  I_n for n <= 1 needs m = n.
        edge_deg: dict[int, int] = {}
        for n, d in deg.items():
            if d >= 1:
                edge_deg[n - 2] = max(edge_deg.get(n - 2, -1), d - 1)
        for n in deg:
            if n >= 5:
                edge_deg[n - 2] = max(edge_deg.get(n - 2, -1), 0)
            if n <= 1:
                edge_deg[n] = max(edge_deg.get(n, -1), 0)
        self.edge_deg = edge_deg
        # lowest edge degree the floored recursion consumes: below it the entry
        # is the divergent principal value (+-inf at h = 0), so never form it.
        edge_min: dict[int, int] = {}
        for n, f in self.floor.items():
            if f > 0:
                edge_min[n - 2] = max(edge_min.get(n - 2, 0), f - 1)
        self.edge_min = edge_min

        BN1: dict[tuple[int, int, int], np.ndarray] = {}
        BN2: dict[tuple[int, int, int], np.ndarray] = {}
        BD: dict[int, np.ndarray] = {}
        for e in range(3):
            pa = p[:, e, :]
            pb = p[:, (e + 1) % 3, :]
            ev = pb - pa
            L = np.linalg.norm(ev, axis=1)
            t = ev / L[:, None]
            nout = np.stack([t[:, 1], -t[:, 0]], axis=1)
            dp = np.einsum("ni,ni->n", pa, nout)
            ua = np.einsum("ni,ni->n", pa, t)
            ub = np.einsum("ni,ni->n", pb, t)
            rho2 = dp * dp + self.h2
            c1, c2 = nout[:, 0], nout[:, 1]
            s1, s2 = t[:, 0], t[:, 1]
            spec = {m: kmax for m, kmax in edge_deg.items()}
            tab = edge_table(ua, ub, rho2, spec)
            # precompute powers
            nzdp = dp != 0.0

            def _term(coeff: np.ndarray, col: np.ndarray, pw: int) -> np.ndarray:
                """``coeff * col``, where ``coeff`` already carries ``dp**pw``.

                Where ``dp == 0`` and ``pw > 0`` the coefficient is zero while
                ``col`` may be ``+inf`` (a divergent edge principal value at
                ``rho = 0``), so the product is forced to its ``dp -> 0`` limit
                of 0 instead of evaluating ``0 * inf = nan``.

                The grouping is deliberately ``(comb * dp**pw * c^.. * s^..) *
                P`` -- the form used before the force element existed.  Keep it:
                regrouping the product moves ``U``/``H``/``E`` by ~1e-16, which
                is harmless numerically but breaks the bitwise baseline that
                ``verify_baseline_bitwise.py`` pins."""
                if pw == 0:
                    return coeff * col
                out = np.zeros(N)
                out[nzdp] = coeff[nzdp] * col[nzdp]
                return out

            for m, kmax in edge_deg.items():
                P = tab[m]                                   # (N, kmax+1)
                BD[m] = BD.get(m, 0.0) + _term(dp, P[:, 0], 1)
                lo = edge_min.get(m, 0)
                for a in range(kmax + 1):
                    for b in range(kmax + 1 - a):
                        if a + b < lo:
                            continue          # divergent PV, below the floor
                        val = np.zeros(N)
                        for i in range(a + 1):
                            for j in range(b + 1):
                                pw = a + b - i - j
                                coeff = (comb(a, i) * comb(b, j)
                                         * dp ** pw
                                         * c1 ** (a - i) * s1 ** i
                                         * c2 ** (b - j) * s2 ** j)
                                val += _term(coeff, P[:, i + j], pw)
                        BN1[(a, b, m)] = BN1.get((a, b, m), 0.0) + c1 * val
                        BN2[(a, b, m)] = BN2.get((a, b, m), 0.0) + c2 * val
        self._BN1, self._BN2, self._BD = BN1, BN2, BD

        # --- seeds
        M: dict[int, np.ndarray] = {}
        for n, d in deg.items():
            M[n] = (np.full((N, d + 1, d + 1), np.nan) if self.floor.get(n, 0) > 0
                    else np.zeros((N, d + 1, d + 1)))
        M[3][:, 0, 0] = I3
        # Upward seeds from the vertical identity (2-n) I_n + n h^2 I_{n+2} = E_n,
        # rearranged as  I_{n+2} = (E_n + (n-2) I_n) / (n h^2).  ONE LOOP rather
        # than a case per n, because the half-space IMAGE kernel needs I_9: the
        # Papkovich-Neuber form already takes one derivative and the slip-to-
        # stress readout two more, so three derivatives act on the potentials'
        # R^-3 term.  A fourth hardcoded case would be this identity written a
        # fourth time.  Bitwise identical to the n = 5 and n = 7 cases it
        # replaces -- (n-2) and n are exact small integers and the grouping is
        # unchanged, which matters because the kernels are pinned bitwise.
        n = 3
        while n + 2 in M:
            M[n + 2][:, 0, 0] = ((BD[n] + (n - 2) * M[n][:, 0, 0])
                                 / (n * self.h2))
            n += 2
        # downward seeds from the vertical identity (2-n) I_n + n h^2 I_{n+2} = E_n:
        #   I_1 = E_1 - h^2 I_3,  I_n = (E_n - n h^2 I_{n+2}) / (2 - n)  for n = -1, -3, ...
        # At h = 0 the coupling term vanishes identically (h^2 I_3 = -Omega h -> 0)
        # and the chain decouples: I_1 = E_1, I_{-1} = E_{-1}/3, ...  The masked
        # product below is what keeps that exact instead of 0 * inf = nan.
        n = 1
        while n in M:
            corr = np.zeros(N)
            up = M[n + 2][:, 0, 0]
            src = fin if (n + 2) == 3 else np.ones(N, dtype=bool)
            corr[src] = n * self.h2[src] * up[src]
            M[n][:, 0, 0] = (BD[n] - corr) / (2.0 - n)
            n -= 2

        # --- horizontal recursion, ascending n
        for n in sorted(M):
            d = deg[n]
            if d == 0:
                continue
            m = n - 2
            for q in range(max(1, self.floor.get(n, 0)), d + 1):
                for a in range(q, -1, -1):
                    b = q - a
                    if a >= 1:
                        val = -BN1[(a - 1, b, m)]
                        if a >= 2:
                            val = val + (a - 1) * M[m][:, a - 2, b]
                    else:
                        val = -BN2[(0, b - 1, m)]
                        if b >= 2:
                            val = val + (b - 1) * M[m][:, 0, b - 2]
                    M[n][:, a, b] = val / (n - 2)
        self.M = M

        if check_identity:
            # two-route identity for M^{(1,1)}: e1-route vs e2-route
            for n in M:
                if deg[n] >= 2 and (0, 1, n - 2) in BN1 and (1, 0, n - 2) in BN2:
                    r1 = -BN1[(0, 1, n - 2)] / (n - 2)
                    r2 = -BN2[(1, 0, n - 2)] / (n - 2)
                    self.identity_residual = np.max(np.abs(r1 - r2))


# ---------------------------------------------------------------------------
# Weighted tables
# ---------------------------------------------------------------------------

def weighted_from_table(table: MomentTable, coeffs: np.ndarray, need: dict[int, int]):
    """``W[n][:, k, a, b] = sum_{a',b'} coeffs[:, k, a', b'] M_n^{(a+a', b+b')}``.

    ``coeffs`` has shape (N, K, p+1, p+1); returns {n: (N, K, D+1, D+1)} for
    the kernel degrees ``need`` (constant-density degrees).  Slots below the
    table's ``h = 0`` floor stay NaN: ``W_n^{(a,b)}`` reads ``M_n`` at total
    degree ``>= a+b``, so the floor carries over unchanged."""
    N, K, D1, _ = coeffs.shape
    p = D1 - 1
    floor = getattr(table, "floor", {}) or {}
    W = {}
    for n, d in need.items():
        f = floor.get(n, 0)
        Wn = np.full((N, K, d + 1, d + 1), np.nan) if f > 0 else np.zeros((N, K, d + 1, d + 1))
        Mn = table.M[n]
        for a in range(d + 1):
            for b in range(d + 1 - a):
                if a + b < f:
                    continue                      # below the floor: leave NaN
                acc = np.zeros((N, K))
                for ap in range(p + 1):
                    for bp in range(p + 1 - ap):
                        c = coeffs[:, :, ap, bp]                  # (N, K)
                        if not np.any(c):
                            continue
                        acc += c * Mn[:, a + ap, b + bp][:, None]
                Wn[:, :, a, b] = acc
        W[n] = Wn
    return W


@functools.lru_cache(maxsize=64)
def gauss_triangle(n: int):
    """Collapsed product Gauss-Legendre rule on the reference triangle
    (0,0),(1,0),(0,1): returns (xi1, xi2, w) with sum(w) = 1/2.

    MEMOISED: a pure function of ``n`` that runs a Legendre eigenvalue solve
    and two meshgrids, and the producers call it on every block -- 64 times for
    one 512 x 16 matrix, over a handful of distinct orders.

    The returned arrays are READ-ONLY, because a memoised array factory that
    hands out mutable references invites a caller to poison the cache for
    every later one. No caller writes to them; this makes that a loud failure
    rather than a quiet one.
    """
    g, w = np.polynomial.legendre.leggauss(n)
    g = 0.5 * (g + 1.0)
    w = 0.5 * w
    gi, gj = np.meshgrid(g, g, indexing="ij")
    wi, wj = np.meshgrid(w, w, indexing="ij")
    out = (gi.ravel(), (gj * (1.0 - gi)).ravel(),
           (wi * wj * (1.0 - gi)).ravel())
    for arr in out:
        arr.flags.writeable = False
    return out


def quadrature_weighted_tables(frame: Frame, obs, eps: float, order: int,
                               need: dict[int, int], n_gauss: int):
    """Far-field producer of the same weighted tables by Gauss quadrature of
    ``N_k(eta_q) xi_1^a xi_2^b R^-n`` (smooth for distant observers).
    Also returns (z, X) for the lift."""
    obs = np.asarray(obs, float).reshape(-1, 3)
    z, X = frame.to_plane(obs)
    h2 = z * z + eps ** 2
    x1, x2, w = gauss_triangle(n_gauss)
    lam = np.stack([1.0 - x1 - x2, x1, x2], axis=1)          # (Q, 3)
    eta = lam @ frame.p                                       # (Q, 2) centroid frame
    wq = w * (2.0 * frame.area)                               # area weights
    # shape functions at the quadrature points (obs independent): evaluate the
    # polynomial about X = 0 at xi = eta
    c0 = shape_coefficients(frame, order, np.zeros((1, 2)))[0]   # (K, p+1, p+1)
    K, D1, _ = c0.shape
    Nq = np.zeros((x1.shape[0], K))
    for a in range(D1):
        for b in range(D1 - a):
            Nq += c0[:, a, b][None, :] * (eta[:, 0] ** a * eta[:, 1] ** b)[:, None]
    xi = eta[None, :, :] - X[:, None, :]                      # (N, Q, 2)
    xi0 = xi[..., 0]
    xi1_ = xi[..., 1]
    R2 = xi0 * xi0 + xi1_ * xi1_ + h2[:, None]
    if not need:
        return {}, z, X

    # THE IN-PLANE MONOMIALS DO NOT DEPEND ON n, and were being rebuilt inside
    # the loop over it: the image spec {3:1, 5:3, 7:4, 9:5} has 49 (n, a, b)
    # slots over the 21 distinct monomials of its top degree. Built once here,
    # by power LADDERS rather than `**` -- the exponents are small fixed
    # integers, where `**` is a pow() call per element.
    dmax = max(need.values())
    p0 = _ladder(xi0, dmax)
    p1 = _ladder(xi1_, dmax)
    mono: dict[tuple[int, int], np.ndarray] = {}
    for a in range(dmax + 1):
        for b in range(dmax + 1 - a):
            if a and b:
                mono[(a, b)] = wq[None, :] * p0[a] * p1[b]
            elif a:
                mono[(a, b)] = wq[None, :] * p0[a]
            elif b:
                mono[(a, b)] = wq[None, :] * p1[b]
            else:
                mono[(a, b)] = np.broadcast_to(wq[None, :], R2.shape)

    # R^-n off ONE reciprocal square root and a ladder in R^-2, replacing a
    # fractional pow() per element per n. Every n the kernels and the image
    # table ask for is ODD (kernel_degrees gives {1, 3, 5, 7}, the image table
    # {3, 5, 7, 9}), which is what lets the ladder step by two; asserted rather
    # than assumed, because an even n would otherwise miss the dict and fail
    # as a KeyError far from its cause.
    assert all(n % 2 == 1 for n in need), f"even n in the far-field spec: {need}"
    rpow: dict[int, np.ndarray] = {}
    inv = 1.0 / np.sqrt(R2)
    inv2 = inv * inv
    cur = inv                                                 # n = 1
    for n in range(1, max(need) + 1, 2):
        if n in need:
            rpow[n] = cur
        cur = cur * inv2

    W = {}
    for n, d in need.items():
        Rn = rpow[n]
        Wn = np.zeros((X.shape[0], K, d + 1, d + 1))
        for a in range(d + 1):
            for b in range(d + 1 - a):
                Wn[:, :, a, b] = (mono[(a, b)] * Rn) @ Nq
        W[n] = Wn
    return W, z, X


def _ladder(arr: np.ndarray, kmax: int) -> list:
    """``[1.0, arr, arr**2, ...]`` by repeated multiplication.

    Slot 0 is the scalar ``1.0`` and is never read: a zero exponent is skipped
    by the caller rather than multiplied by an array of ones.
    """
    out = [1.0]
    cur = arr
    for _ in range(kmax):
        out.append(cur)
        cur = cur * arr
    return out


def weighted_tables(frame: Frame, obs, eps: float, order: int, want,
                    far_field: str = "hybrid", check_identity: bool = False,
                    degrees: dict[int, int] | None = None):
    """Per-node weighted moment tables for all observation points.

    Returns ``(W, z, X)`` with ``W = {n: (N, K, D+1, D+1)}`` at the
    constant-slip degrees of the requested kernels.

    ``degrees`` supplies those constant-slip degrees directly, for a caller
    whose kernel is not one of the named full-space ones -- the half-space
    IMAGE family, whose degree spec comes from its generated monomial table.
    Passing a spec rather than a kernel NAME is what keeps this module ignorant
    of the image kernel: it is given a degree requirement, not knowledge of who
    needs it. Everything else -- the near/far split at ``D_STAR``, the Gauss
    fallback, the separate floored table for on-plane rows -- is shared, which
    is the reason this is a parameter instead of a second dispatcher."""
    obs = np.asarray(obs, float).reshape(-1, 3)
    N = obs.shape[0]
    need0 = dict(degrees) if degrees is not None else kernel_degrees(0, want)
    if far_field not in ("hybrid", "analytic", "quadrature"):
        raise ValueError("far_field must be 'hybrid', 'analytic' or 'quadrature'")
    # eps = 0 on the plane is admissible only when every requested kernel is
    # weakly singular there -- i.e. the force displacement kernel alone.  The
    # permission is derived from `want`, never from a user-facing flag.
    allow = bool(want) and set(want) <= ON_PLANE_LIMIT_KERNELS
    check_h_positive(frame, obs, float(eps), allow_on_plane=allow)
    h0 = on_plane_mask(frame, obs, float(eps)) if allow else np.zeros(N, dtype=bool)
    if h0.any() and far_field == "quadrature":
        raise ValueError("far_field='quadrature' cannot integrate through the "
                         "eps = 0 singularity of an on-element observer; use "
                         "'hybrid' (the default) or 'analytic'")
    if far_field == "analytic":
        near = np.ones(N, dtype=bool)
    elif far_field == "quadrature":
        near = np.zeros(N, dtype=bool)
    else:
        # effective distance: the closed form loses digits with R/L whether R
        # is large because the observer is far or because eps is large
        D = np.sqrt(np.sum((obs - frame.centroid) ** 2, axis=1) + float(eps) ** 2)
        near = D <= defaults.D_STAR * frame.L
    z = np.empty(N)
    X = np.empty((N, 2))
    W = {n: np.empty((N, (order + 1) * (order + 2) // 2, d + 1, d + 1))
         for n, d in need0.items()}
    identity_residual = 0.0
    if np.any(near):
        need = ({n: d + order for n, d in degrees.items()}
                if degrees is not None else kernel_degrees(order, want))
        # A FLOORED table leaves its sub-floor slots NaN, which is correct only
        # for on-plane rows -- `lift` masks them there because z is exactly 0.
        # An off-plane row in the same table would multiply that NaN by a
        # nonzero z^c and silently return NaN, so the two regimes get SEPARATE
        # tables.  A batch that is all-on-plane or all-off-plane still builds
        # exactly one, as before.
        h0_near = h0 & near
        if allow and h0_near.any():
            groups = [(h0_near, h0_floor(need))]
            rest = near & ~h0_near
            if rest.any():
                groups.append((rest, None))
        else:
            groups = [(near, None)]
        for sel, fl in groups:
            tab = MomentTable(frame, obs[sel], eps, need,
                              check_identity=check_identity, floor=fl)
            coeffs = shape_coefficients(frame, order, tab.X)
            Wn = weighted_from_table(tab, coeffs, need0)
            for n in W:
                W[n][sel] = Wn[n]
            z[sel] = tab.z
            X[sel] = tab.X
            if check_identity:
                identity_residual = max(
                    identity_residual, getattr(tab, "identity_residual", 0.0))
    if np.any(~near):
        far = ~near
        Dfar = np.sqrt(np.sum((obs[far] - frame.centroid) ** 2, axis=1) + float(eps) ** 2)
        ng = np.where(Dfar > defaults.D_STAR_DISTANT * frame.L,
                      defaults.FAR_GAUSS_N_DISTANT, defaults.FAR_GAUSS_N)
        for g in np.unique(ng):
            sel = np.zeros(N, dtype=bool)
            sel[np.flatnonzero(far)[ng == g]] = True
            Wf, zf, Xf = quadrature_weighted_tables(frame, obs[sel], eps, order, need0, int(g))
            for n in W:
                W[n][sel] = Wf[n]
            z[sel] = zf
            X[sel] = Xf
    if h0.any():
        # far rows are never floored; keep the mask consistent with `near`
        h0 = h0 & near
        z = np.where(h0, 0.0, z)
    return W, z, X, identity_residual, h0
