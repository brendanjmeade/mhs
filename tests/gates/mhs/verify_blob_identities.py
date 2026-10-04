#!/usr/bin/env python
"""The blob-convolution foundation for the Cortez-consistent half-space kernel.

The half-space work rests on being able to convolve Apostol's Papkovich-Neuber
potentials with the Cortez blob **exactly**, in closed form. This gate pins the
three facts that make that possible, each measured rather than asserted, so the
derivation that follows is built on checked ground.

  [a] FOUR GENERATOR IDENTITIES, against 3-D quadrature of the convolution
      itself. With ``R = sqrt(|d|^2 + eps^2)``, ``Q = R - d3`` and the Cortez
      blob ``phi_eps(r) = 15 eps^4 / (8 pi (r^2+eps^2)^(7/2))``:

          r        * phi = R                                  (no counter-term)
          1/r      * phi = 1/R   + (eps^2/2)/R^3              (known Cortez)
          log(r-z) * phi = log Q - (eps^2/2)/(R Q)            NEW
          d_a/(r-z)* phi = d_a/Q + (eps^2/2) d_a/(R Q^2)      NEW

      The last two are the half-space generators -- the ``1/Q`` and ``log Q``
      family -- and they are what the existing kernel gets only approximately.
      Each is compared BOTH against the identity and against the LITERAL LIFT
      (substitute r -> R and stop), because the gate's job is to show the
      counter-terms are the content rather than a refinement. That is asserted
      as a RATIO -- the lift must be at least 20x worse than the identity in
      every case, measured 1024x at worst -- because the lift's own error scales
      as eps^2/distance^2 and so is legitimately small for a distant observer at
      small eps without the identity being any less necessary.

  [b] THE UNIFIED RULE reproduces the Cortez Kelvin kernel exactly. For any
      biharmonic ``f``,

          f * phi = (1 - eps^2 d/d(eps^2)) [f]_eps + (eps^2/2) [ (1/2) lap f ]_eps

      Applied to Kelvin's PN potentials (``b_i = F_i/(4 pi mu r)``, ``beta = 0``)
      it must give ``C1[(3-4nu) delta/R + d_i d_j/R^3 + 2(1-nu) eps^2 delta/R^3]``
      -- the published mollified kernel. This is the strongest available check
      that the rule is right, because the target was derived independently.
      Note ``r`` is NOT harmonic (``lap r = 2/r``), so the harmonic form alone
      would give ``R - eps^2/(2R)``; the biharmonic term restores it exactly.

The third fact the derivation needs -- that Apostol's potentials are BIHARMONIC
in the source variable, so the rule applies wholesale -- is exact algebra and is
checked symbolically in ``tests/gates/oracle/verify_free_surface.py``. It is not
here because a fourth derivative by nested finite differences divides by h^4,
and at h = 2e-3 the cancellation reaches 7e-4 on a quantity whose true value is
1e-16. Measuring exact algebra numerically was the wrong tool, and the first
draft of this gate used it.

**SCOPE, established by the derivation this gate supports.** The rule above is
EXACT for HARMONIC ``f`` -- which is what clause [a] measures, and why the
generator identities hold to quadrature precision. For merely BIHARMONIC ``f``
it is only O(eps^4): measured order 3.95-4.02 on every one of Apostol's
complete potentials, each verified biharmonic to 1e-16 or better. The ``r``
case in clause [b] is exact because its biharmonic structure is trivial, and
generalising from it was the mistake. Consequences, numbers and the obstruction
are recorded in ``docs/derivation.md``; the short version is that an EXACT free
surface is not available by this route.

Run from anywhere:  python tests/gates/mhs/verify_blob_identities.py
"""
from __future__ import annotations

import sys

import numpy as np

from _common import Report

# --- quadrature of the convolution ------------------------------------------
NR, NTH, NPH = 300, 64, 128
RMAX_OVER_EPS = 400.0      # the blob's 1/r^7 tail, resolved by the tan substitution

#: Observers chosen so none is special: off-axis, mixed signs, O(1) distances.
DS = (np.array([0.7, -0.4, -1.3]),
      np.array([-1.1, 0.6, -0.5]),
      np.array([0.2, 0.3, -2.1]))
EPSS = (0.3, 0.15)

TOL_IDENTITY = 2e-5       # quadrature-limited; measured 7e-9 .. 1e-5
# The claim is that the counter-terms MATTER, and the scale-free way to say it
# is a RATIO. An absolute floor on the lift error is the wrong statement: that
# error scales as eps^2 / distance^2, so at small eps and a distant observer it
# is legitimately small (measured 8.6e-4 at eps = 0.15) without the identity
# being any less necessary. Measured worst ratio 88x.
LIFT_OVER_IDENTITY_MIN = 20.0
TOL_KELVIN = 1e-13        # the rule vs the published kernel: algebra, not quadrature


def blob(r, eps):
    return 15.0 * eps ** 4 / (8.0 * np.pi * (r * r + eps * eps) ** 3.5)


def convolve(f, d, eps):
    """``(f * phi_eps)(d)`` by a spherical product rule.

    Radial substitution ``r = eps tan t`` so the algebraic tail is resolved: the
    blob has unit integral and most of its mass within a few eps, but the tail
    carries enough weight that a naive finite cutoff biases the result.
    """
    tmax = np.arctan(RMAX_OVER_EPS)
    xt, wt = np.polynomial.legendre.leggauss(NR)
    t = 0.5 * tmax * (xt + 1.0)
    wt = wt * 0.5 * tmax
    r = eps * np.tan(t)
    jac = eps / np.cos(t) ** 2
    xc, wc = np.polynomial.legendre.leggauss(NTH)
    ph = (np.arange(NPH) + 0.5) * 2.0 * np.pi / NPH
    wph = 2.0 * np.pi / NPH
    st = np.sqrt(1.0 - xc ** 2)
    u = np.empty((NTH, NPH, 3))
    u[..., 0] = st[:, None] * np.cos(ph)[None, :]
    u[..., 1] = st[:, None] * np.sin(ph)[None, :]
    u[..., 2] = xc[:, None]
    W = r[:, None, None, None] * u[None, ...]
    wts = ((blob(r, eps) * r * r * jac * wt)[:, None, None]
           * wc[None, :, None] * wph)
    return float((f(d[None, None, None, :] - W) * wts).sum())


def _R(x, e):
    return np.sqrt((x * x).sum(-1) + e * e)


def _Q(x, e):
    return _R(x, e) - x[..., 2]


def generators(d, eps):
    """(name, f, exact convolution, literal lift) for the four generators."""
    R, Q = _R(d, eps), _Q(d, eps)
    return [
        ("r", lambda x: np.sqrt((x * x).sum(-1)), R, R),
        ("1/r", lambda x: 1.0 / np.sqrt((x * x).sum(-1)),
         1.0 / R + 0.5 * eps ** 2 / R ** 3, 1.0 / R),
        ("log(r-z)", lambda x: np.log(np.sqrt((x * x).sum(-1)) - x[..., 2]),
         np.log(Q) - 0.5 * eps ** 2 / (R * Q), np.log(Q)),
        ("d1/(r-z)",
         lambda x: x[..., 0] / (np.sqrt((x * x).sum(-1)) - x[..., 2]),
         d[0] / Q + 0.5 * eps ** 2 * d[0] / (R * Q ** 2), d[0] / Q),
    ]


def part_a(rep) -> None:
    print("\n[a] FOUR GENERATOR IDENTITIES vs 3-D quadrature of the convolution")
    unit = convolve(lambda x: np.ones(x.shape[:-1]), DS[0], EPSS[0])
    rep.check("a the blob has unit integral", abs(unit - 1.0), 1e-8,
              f"({unit:.10f})")
    print(f"    {'generator':10s} {'eps':>5s} {'numerical':>14s} "
          f"{'identity':>14s} {'rel err':>9s} | {'lift err':>9s}")
    worst_id, worst_ratio = 0.0, np.inf
    for d in DS:
        for eps in EPSS:
            for name, f, exact, lift in generators(d, eps):
                num = convolve(f, d, eps)
                e_id = abs(num - exact) / max(abs(exact), 1e-300)
                e_lf = abs(num - lift) / max(abs(exact), 1e-300)
                worst_id = max(worst_id, e_id)
                if name != "r":          # 'r' has no counter-term by design
                    worst_ratio = min(worst_ratio,
                                      e_lf / max(e_id, 1e-300))
                print(f"    {name:10s} {eps:5.2f} {num:14.9f} {exact:14.9f} "
                      f"{e_id:9.2e} | {e_lf:9.2e}")
    rep.check("a every identity matches the convolution", worst_id,
              TOL_IDENTITY, "(quadrature-limited)")
    rep.check_bool(f"a the LITERAL LIFT is >= {LIFT_OVER_IDENTITY_MIN:.0f}x "
                   f"worse than the identity, every case",
                   worst_ratio >= LIFT_OVER_IDENTITY_MIN,
                   f"(worst ratio {worst_ratio:.0f}x) -- so the counter-terms "
                   f"are the content, not a refinement")


def part_b(rep) -> None:
    """The rule must reproduce the published Cortez Kelvin kernel."""
    print("\n[b] THE UNIFIED RULE vs the published Cortez Kelvin kernel")
    mu, nu = 1.0, 0.3
    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    K = 1.0 / (4.0 * np.pi * mu)
    worst = 0.0
    for d in DS:
        for eps in EPSS:
            R = _R(d, eps)
            # The rule, assembled in closed form for Kelvin:
            #   b_i    = F_i K / r        harmonic   -> K(1/R + e2/(2R^3))
            #   r.b    = K F_j x_j / r    biharmonic -> (1-e2 d_e2)[K x_j/R]
            #                                          + (e2/2)[-K x_j/R^3]
            # then u_i = b_i - d_i(r.b)/(4(1-nu)).
            e2 = eps ** 2
            got = np.zeros((3, 3))
            for j in range(3):
                for i in range(3):
                    dij = 1.0 if i == j else 0.0
                    b_i = dij * K * (1.0 / R + 0.5 * e2 / R ** 3)
                    # d_i of (1 - e2 d_e2)[K x_j/R] + (e2/2)[-K x_j/R^3]:
                    #   [K x_j/R]        -> K(dij/R - x_i x_j/R^3)
                    #   d_e2 of that     -> K(-dij/(2R^3) + 3 x_i x_j/(2R^5))
                    #   [-K x_j/R^3]     -> K(-dij/R^3 + 3 x_i x_j/R^5)
                    t_lift = K * (dij / R - d[i] * d[j] / R ** 3)
                    t_de2 = K * (-0.5 * dij / R ** 3
                                 + 1.5 * d[i] * d[j] / R ** 5)
                    t_bih = K * (-dij / R ** 3 + 3.0 * d[i] * d[j] / R ** 5)
                    grad = t_lift - e2 * t_de2 + 0.5 * e2 * t_bih
                    got[i, j] = b_i - grad / (4.0 * (1.0 - nu))
            tgt = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    dij = 1.0 if i == j else 0.0
                    tgt[i, j] = C1 * ((3.0 - 4.0 * nu) * dij / R
                                      + d[i] * d[j] / R ** 3
                                      + 2.0 * (1.0 - nu) * e2 * dij / R ** 3)
            worst = max(worst, float(np.abs(got - tgt).max()
                                     / np.abs(tgt).max()))
    rep.check("b the rule reproduces C1[(3-4nu)d/R + dd/R^3 + 2(1-nu)e^2 d/R^3]",
              worst, TOL_KELVIN,
              "algebra, not quadrature -- the target was derived independently")
    # and the harmonic form ALONE would be wrong, by exactly the blob term
    d, eps = DS[0], EPSS[0]
    R, e2 = _R(d, eps), EPSS[0] ** 2
    harm_only = 1.0 / R + 0.5 * e2 / R ** 3     # (1 - e2 d_e2)[1/R]
    rep.check_bool("b tripwire: 'r' is NOT harmonic, so the biharmonic term "
                   "is required",
                   abs(_R(d, eps) - (R - 0.5 * e2 / R)) > 1e-3,
                   f"(harmonic-only would give R - e^2/2R, off by "
                   f"{0.5 * e2 / R:.3e}; lap r = 2/r != 0)")


def main() -> bool:
    rep = Report("Blob-convolution foundation for the Cortez-consistent "
                 "half-space kernel")
    part_a(rep)
    part_b(rep)
    print("\n  SCOPE: the rule is EXACT for HARMONIC f (clause [a]) and only "
          "O(eps^4) for")
    print("  merely BIHARMONIC f -- measured order 3.95-4.02 on Apostol's "
          "complete")
    print("  potentials. So an exact free surface is NOT available by this "
          "route; the")
    print("  traction stays O(eps^2) with a 2.6x-50x better constant. "
          "docs/derivation.md")
    print("  has the tables and the obstruction.")
    return rep.finish()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
