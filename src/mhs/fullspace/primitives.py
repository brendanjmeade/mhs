"""One-dimensional edge antiderivatives and the solid angle.

Every triangle integral in ``clq`` reduces (divergence theorem) to edge
integrals of the form

    P_k^m(u_a, u_b) = int_{u_a}^{u_b} u^k / (u^2 + rho^2)^(m/2) du,

with ``rho^2 = d_perp^2 + h_eps^2 > 0`` constant along the edge, ``m`` odd
(negative ``m`` means positive powers of ``R = sqrt(u^2 + rho^2)``), and
``k`` up to a few units.  Two layers are provided:

* :func:`antiderivative` -- the closed-form antiderivative ``P_k^m(u)`` as a
  function of ``u`` (works with numpy arrays or sympy symbols through the
  ``mod`` argument).  It is the reference the difference layer is verified
  against (``verify/verify_primitives.py``), built from

      J_1 = log(u + R),      J_3 = u / (rho^2 R),
      J_{m+2} = [(m - 1) J_m + u R^{-m}] / (m rho^2)              (m >= 1),
      J_{-1} = (u R + rho^2 log(u + R)) / 2,
      J_{-p} = u R^p / (p + 1) + p rho^2 / (p + 1) J_{-(p-2)}     (p >= 3),
      K_m = -R^{2-m} / (m - 2)                                    (m != 2),
      P_k^m = P_{k-2}^{m-2} - rho^2 P_{k-2}^m,   P_0^m = J_m,  P_1^m = K_m.

* :func:`edge_table` -- the DIFFERENCES ``P_k^m(u_b) - P_k^m(u_a)`` for a
  batch of edges, computed without cancellation:

  - closed-form regime (``u_a u_b <= 0`` or ``min|u| < SERIES_U_OVER_RHO rho``):
    conjugate forms for ``Delta R``, ``Delta J_1`` (log1p, reflected for
    negative ``u``, product form for mixed signs), ``Delta J_3``, factored
    ``Delta R^q`` for ``Delta K_m``, and the ``J`` recursions in difference
    form;
  - small-|u| regime (``max|u| <= SMALL_U_OVER_RHO rho``): the binomial
    series in ``(u/rho)^2``, ``rho^-m sum_j C(-m/2, j) rho^-2j [u^p/p]``,
    ``p = k + 2j + 1`` (the closed form's ``u^k`` reduction would lose
    ``(rho/u)^2`` per level there);
  - ``rho = 0`` regime (an observer on the edge's line at ``h_eps = 0``):
    ``R = |u|`` and the integrals are elementary powers, ``|u|^q / q`` with
    ``q = k - m + 1`` (``log|u|`` at ``q = 0``, ``+-inf`` when the edge
    straddles the observer -- a genuinely divergent principal value, never a
    quiet zero).  Exact test ``rho2 == 0.0``, no tolerance;
  - large-|u| regime (same sign, ``min|u| >= SERIES_U_OVER_RHO rho``): the
    large-``|u|`` binomial series
    ``sum_j C(-m/2, j) rho^(2j) [u^q/q]_{u_a}^{u_b}``, ``q = k - m + 1 - 2j``,
    with each power difference evaluated as ``u_a^q expm1(q log(u_b/u_a))``.

  The naive differences lose ``(u/rho)^(m-1)`` digits for observers many
  mollification widths along an edge; the forms above are exact to
  rounding in every regime (see docs/derivation.md, numerics).
"""
from __future__ import annotations

from math import comb

import numpy as np

from . import defaults


# ---------------------------------------------------------------------------
# Value layer (numpy or sympy)
# ---------------------------------------------------------------------------

def _binom_half(m: int, j: int):
    """Generalised binomial C(-m/2, j) as an exact Fraction-free float."""
    from fractions import Fraction
    a = Fraction(-m, 2)
    out = Fraction(1)
    for i in range(j):
        out *= (a - i) / (i + 1)
    return out


def antiderivative(k: int, m: int, u, rho2, mod=np):
    """Antiderivative of ``u^k / (u^2 + rho2)^(m/2)`` with respect to ``u``.

    ``m`` must be odd.  ``mod`` supplies ``sqrt`` and ``log`` (``numpy`` or
    ``sympy``).  Any ``k >= 0`` is accepted (recursive reduction).
    """
    if m % 2 == 0:
        raise ValueError("m must be odd")
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return _J(m, u, rho2, mod)
    if k == 1:
        return _K(m, u, rho2, mod)
    return (antiderivative(k - 2, m - 2, u, rho2, mod)
            - rho2 * antiderivative(k - 2, m, u, rho2, mod))


def _J(m: int, u, rho2, mod=np):
    R = mod.sqrt(u * u + rho2)
    if m == 1:
        return mod.log(u + R)
    if m == 3:
        return u / (rho2 * R)
    if m >= 5:
        mm = m - 2
        return ((mm - 1) * _J(mm, u, rho2, mod) + u * R ** (-mm)) / (mm * rho2)
    if m == -1:
        return (u * R + rho2 * mod.log(u + R)) / 2
    # m <= -3 : int R^p du with p = -m
    p = -m
    return u * R ** p / (p + 1) + p * rho2 / (p + 1) * _J(m + 2, u, rho2, mod)


def _K(m: int, u, rho2, mod=np):
    R = mod.sqrt(u * u + rho2)
    q = 2 - m
    return R ** q / q


# ---------------------------------------------------------------------------
# Difference layer (vectorised over edges)
# ---------------------------------------------------------------------------

class _EdgeDiffs:
    """Cancellation-free differences P_k^m(u_b) - P_k^m(u_a) for a batch of
    edges in the closed-form regime.  All inputs are (N,) arrays."""

    def __init__(self, ua, ub, rho2):
        self.ua = ua
        self.ub = ub
        self.rho2 = rho2
        self.Ra = np.sqrt(ua * ua + rho2)
        self.Rb = np.sqrt(ub * ub + rho2)
        # R_b - R_a without cancellation
        self.dR = (ub - ua) * (ub + ua) / (self.Ra + self.Rb)
        self._J: dict[int, np.ndarray] = {}
        self._K: dict[int, np.ndarray] = {}
        self._P: dict[tuple[int, int], np.ndarray] = {}

    # --- Delta J_m -------------------------------------------------------
    def dJ(self, m: int) -> np.ndarray:
        if m in self._J:
            return self._J[m]
        ua, ub, Ra, Rb, rho2, dR = self.ua, self.ub, self.Ra, self.Rb, self.rho2, self.dR
        if m == 1:
            pos = ua >= 0.0
            neg = ub <= 0.0
            mixed = ~(pos | neg)
            out = np.empty_like(ua)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[pos] = np.log1p(((ub - ua) + dR)[pos] / (ua + Ra)[pos])
                out[neg] = np.log1p(((ub - ua) - dR)[neg] / (Rb - ub)[neg])
                out[mixed] = np.log(((ub + Rb) * (Ra - ua))[mixed] / rho2[mixed])
        elif m == 3:
            same = (ua * ub) > 0.0
            out = np.empty_like(ua)
            with np.errstate(divide="ignore", invalid="ignore"):
                den = (Ra * Rb * (ub * Ra + ua * Rb))
                out[same] = ((ub - ua) * (ub + ua))[same] / den[same]
                out[~same] = (ub / Rb - ua / Ra)[~same] / rho2[~same]
        elif m >= 5:
            mm = m - 2
            duRm = ub * Rb ** (-mm) - ua * Ra ** (-mm)
            out = (duRm + (mm - 1) * self.dJ(mm)) / (mm * rho2)
        elif m == -1:
            duR = 0.5 * (ub - ua) * (Ra + Rb) + 0.5 * (ua + ub) * dR   # u_b R_b - u_a R_a
            out = 0.5 * (duR + rho2 * self.dJ(1))
        else:  # m <= -3
            p = -m
            duRp = ub * Rb ** p - ua * Ra ** p
            out = duRp / (p + 1) + p * rho2 / (p + 1) * self.dJ(m + 2)
        self._J[m] = out
        return out

    # --- Delta K_m -------------------------------------------------------
    def dK(self, m: int) -> np.ndarray:
        if m in self._K:
            return self._K[m]
        Ra, Rb, dR = self.Ra, self.Rb, self.dR
        q = 2 - m
        if q > 0:
            # (R_b^q - R_a^q)/q = dR * sum_i R_a^i R_b^(q-1-i) / q
            S = sum(Ra ** i * Rb ** (q - 1 - i) for i in range(q))
            out = dR * S / q
        else:
            p = -q
            # (R_b^-p - R_a^-p)/(-p) = dR * sum_i R_a^i R_b^(p-1-i) / (p (R_a R_b)^p)
            S = sum(Ra ** i * Rb ** (p - 1 - i) for i in range(p))
            out = dR * S / (p * (Ra * Rb) ** p)
        self._K[m] = out
        return out

    # --- Delta P_k^m -----------------------------------------------------
    def dP(self, k: int, m: int) -> np.ndarray:
        key = (k, m)
        if key in self._P:
            return self._P[key]
        if k == 0:
            out = self.dJ(m)
        elif k == 1:
            out = self.dK(m)
        else:
            out = self.dP(k - 2, m - 2) - self.rho2 * self.dP(k - 2, m)
        self._P[key] = out
        return out


def _dpow(xa, xb, q, log_r, close):
    """(xb^q - xa^q) for arrays with 0 < xa < xb: the expm1 form where the
    endpoints are close (ratio < 2, avoids cancellation), the direct
    difference otherwise (no overflow: |x| <= 1 in the small-u series and the
    negative-q powers are < 1 in the large-u series).  q = 0 -> log(xb/xa)."""
    if q == 0:
        return log_r
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        e = np.where(close, xa ** q * np.expm1(q * log_r), xb ** q - xa ** q)
    return e


def _series_dP(k: int, m: int, ua, ub, rho2, n_terms: int) -> np.ndarray:
    """Large-|u| series for int_{ua}^{ub} u^k R^-m du, requires 0 < ua < ub and
    rho2/ua^2 <= 1/4.  Scale-free form: with x = rho^2/ua^2 and t = u/ua,

        = ua^(k-m+1) sum_j C(-m/2, j) x^j [t^q / q]_1^{ub/ua},  q = k - m + 1 - 2j.
    """
    x = rho2 / (ua * ua)
    r = ub / ua
    log_r = np.log1p((ub - ua) / ua)
    close = r < 2.0
    ones = np.ones_like(ua)
    base = ua ** (k - m + 1)
    out = np.zeros_like(ua)
    xj = np.ones_like(ua)
    for j in range(n_terms):
        q = k - m + 1 - 2 * j
        c = float(_binom_half(m, j))
        term = _dpow(ones, r, q, log_r, close)
        if q != 0:
            term = term / q
        out = out + c * xj * term
        xj = xj * x
    return base * out


def _small_u_series_dP(k: int, m: int, ua, ub, rho2, n_terms: int) -> np.ndarray:
    """Small-|u| series for int_{ua}^{ub} u^k R^-m du, requires
    max(|ua|, |ub|) <= rho/2.  Any signs.  Scale-free form: with y = u/rho,

        = rho^(k+1-m) sum_j C(-m/2, j) [y^p / p]_{ya}^{yb},  p = k + 2j + 1.
    """
    rho = np.sqrt(rho2)
    ya = ua / rho
    yb = ub / rho
    same = (ua * ub) > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_r = np.where(same, np.log1p((ub - ua) / ua), 0.0)
        ratio = np.where(same, ub / ua, np.inf)
    close = same & (np.abs(ratio) < 2.0) & (np.abs(ratio) > 0.5)
    scale = rho ** (k + 1 - m)
    out = np.zeros_like(ua)
    for j in range(n_terms):
        pw = k + 2 * j + 1
        c = float(_binom_half(m, j))
        with np.errstate(over="ignore", invalid="ignore"):
            d_close = ya ** pw * np.expm1(pw * log_r)
            d_direct = yb ** pw - ya ** pw
        dpow = np.where(close, d_close, d_direct)
        out = out + c * dpow / pw
    return scale * out


def _rho0_dP(k: int, m: int, ua, ub):
    """``int_{ua}^{ub} u^k |u|^{-m} du`` -- the rho = 0 limit of ``P_k^m``.

    With ``q = k - m + 1`` the antiderivative is ``|u|^q / q`` for ``u >= 0``
    and ``(-1)^(k+1) |u|^q / q`` for ``u < 0`` (``log|u|`` with the same sign
    convention at ``q = 0``, which diverges when the edge straddles ``u = 0``:
    that is the genuinely infinite principal-value moment, returned as +-inf,
    never as a quiet zero).
    """
    q = k - m + 1

    def F(u):
        au = np.abs(u)
        if q == 0:
            with np.errstate(divide="ignore"):
                f = np.log(au)
        else:
            with np.errstate(divide="ignore"):
                f = au ** float(q) / q
        return np.where(u < 0.0, (-1.0) ** (k + 1) * f, f)

    out = F(ub) - F(ua)
    if q <= 0:
        # |u|^(k-m) is not integrable at u = 0: any edge whose closed span
        # contains the observer's foot diverges.  The antiderivative
        # difference would cancel into a finite, WRONG number, so say +inf.
        out = np.where((ua <= 0.0) & (ub >= 0.0), np.inf, out)
    return out


def edge_table(ua, ub, rho2, spec: dict[int, int]) -> dict[int, np.ndarray]:
    """Differences ``int_{ua}^{ub} u^k / R^m du`` for a batch of edges.

    Parameters
    ----------
    ua, ub, rho2 : (N,) arrays with ``ub > ua`` and ``rho2 >= 0``.  ``rho2 = 0``
        (an observer on the edge's LINE with ``h_eps = 0``, which happens
        bit-exactly at every vertex and edge midpoint of an axis-aligned
        triangle) is handled by the elementary ``R = |u|`` forms of
        :func:`_rho0_dP`; every other row is untouched.
    spec : {m: kmax} -- for each odd ``m`` the highest power ``k`` needed.

    Returns
    -------
    {m: array (N, kmax_m + 1)} with ``[:, k]`` the integral for power ``k``.
    """
    ua = np.asarray(ua, float)
    ub = np.asarray(ub, float)
    rho2 = np.asarray(rho2, float)
    rho = np.sqrt(rho2)
    out = {m: np.empty((ua.shape[0], kmax + 1)) for m, kmax in spec.items()}

    # rho = 0 rows are excluded from the general machinery (which forms 1/rho^2)
    # and filled from the elementary closed form below.
    zero = rho2 == 0.0
    fin = ~zero
    if np.any(fin):
        diffs = _EdgeDiffs(ua[fin], ub[fin], rho2[fin])
        for m, kmax in spec.items():
            for k in range(kmax + 1):
                out[m][fin, k] = diffs.dP(k, m)
    if np.any(zero):
        for m, kmax in spec.items():
            for k in range(kmax + 1):
                out[m][zero, k] = _rho0_dP(k, m, ua[zero], ub[zero])

    # Small-|u| regime: both |u| <= SMALL_U_OVER_RHO * rho (rho >> |u|).
    umax = np.maximum(np.abs(ua), np.abs(ub))
    small = fin & (umax <= defaults.SMALL_U_OVER_RHO * rho)
    if np.any(small):
        a = ua[small]
        b = ub[small]
        r2 = rho2[small]
        for m, kmax in spec.items():
            for k in range(kmax + 1):
                out[m][small, k] = _small_u_series_dP(k, m, a, b, r2, defaults.SERIES_TERMS)

    # Large-|u| series regime: same sign, both |u| >= SERIES_U_OVER_RHO * rho.
    same = (ua * ub) > 0.0
    umin = np.minimum(np.abs(ua), np.abs(ub))
    ser = fin & same & (umin >= defaults.SERIES_U_OVER_RHO * rho)
    if np.any(ser):
        a = ua[ser]
        b = ub[ser]
        r2 = rho2[ser]
        negative = a < 0.0
        # map negative edges to positive: (a, b) -> (-b, -a), factor (-1)^k
        a_pos = np.where(negative, -b, a)
        b_pos = np.where(negative, -a, b)
        for m, kmax in spec.items():
            for k in range(kmax + 1):
                val = _series_dP(k, m, a_pos, b_pos, r2, defaults.SERIES_TERMS)
                if k % 2 == 1:
                    val = np.where(negative, -val, val)
                out[m][ser, k] = val
    return out


# ---------------------------------------------------------------------------
# Solid angle
# ---------------------------------------------------------------------------

def solid_angle(v1, v2, v3, obs) -> np.ndarray:
    """Van Oosterom-Strackee signed solid angle of triangle (v1, v2, v3) seen
    from each point of ``obs`` (N, 3).  Negative for an observer on the +nhat
    side of a counter-clockwise triangle (the Van Oosterom triple product
    r1.(r2 x r3) with r_i = v_i - obs), so that ``I_3 = -Omega/h > 0``."""
    obs = np.asarray(obs, float).reshape(-1, 3)
    r1 = v1[None, :] - obs
    r2 = v2[None, :] - obs
    r3 = v3[None, :] - obs
    R1 = np.linalg.norm(r1, axis=1)
    R2 = np.linalg.norm(r2, axis=1)
    R3 = np.linalg.norm(r3, axis=1)
    numer = np.einsum("ni,ni->n", r1, np.cross(r2, r3))
    denom = (R1 * R2 * R3
             + R3 * np.einsum("ni,ni->n", r1, r2)
             + R1 * np.einsum("ni,ni->n", r2, r3)
             + R2 * np.einsum("ni,ni->n", r1, r3))
    return 2.0 * np.arctan2(numer, denom)
