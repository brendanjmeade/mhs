#!/usr/bin/env python
"""The free-surface traction residual -- the baseline the derivation must beat.

**This gate exists before the thing it measures.** The Cortez-consistent image
mollification is the next piece of work, and its whole claim is that the
free-surface condition goes from approximate to exact. A number claimed after
the fact is worth much less than one measured against a baseline recorded
first, so this gate records the baseline now, in the form the improvement will
be visible in.

The vendored kernel mollifies by the literal lift -- ``r -> sqrt(r^2 + eps^2)``
on both the direct and the image distance -- plus an explicit Cortez blob on the
direct part, so that half satisfies ``L G = -delta phi_eps`` exactly while the
image correction is only approximate. Classical Mindlin gives ``t_3k = 0`` at
``z = 0`` identically; this kernel gives it only to ``O(eps^2)``.

Four clauses, and the third is the one that matters:

  [a] THE RESIDUAL, ``max|sigma_3k(z=0)| / max|sigma(bulk)|``, over nu in
      {0, 0.25, 0.49} and eps in {0.5, 0.2, 0.1, 0.05}. Gated at the measured
      worst with headroom, so it catches a regression in the vendored kernel.
      The normalisation is stated rather than assumed: the numerator is the
      traction on the free surface (which must vanish) and the denominator the
      peak stress one unit below it, so the ratio is "how large is the thing
      that should be zero, next to the thing that should not be".

  [b] THE ORDER IN eps, which must be ~2 and is the structural signature of the
      literal lift. Measured 1.90-2.00 across every nu. If this ever came out
      at 1 the mollification would have changed character, which no absolute
      tolerance would reveal.

  [c] A TRIPWIRE ASSERTING THE DEFECT IS STILL THERE. The residual must be
      eps-DEPENDENT -- order above 1.5 and the coarsest rung above 1e-3. That
      clause is deliberately the wrong way round for a gate, and it is the
      point: when the Cortez-consistent derivation lands, the residual should
      become eps-INDEPENDENT at machine precision, this clause will FAIL, and
      the commit that lands the derivation must invert it. An improvement that
      forces a gate to be rewritten is an improvement nobody can merge without
      noticing. (Upstream uses the same shape -- "P0 top: median ABOVE
      tripwire" -- to prove a check sees the defect it is about.)

      Target for the inverted clause, stated now so it is not negotiated later:
      eps-independent, below 1e-11 relative, at every nu including 0.49.

  [d] APOSTOL'S POTENTIALS ARE BIHARMONIC IN THE SOURCE VARIABLE, checked
      symbolically. This is the foundation the Cortez-consistent derivation
      stands on: it lets ONE unified rule be applied to the whole potential set
      rather than classifying term by term. It lives here rather than in the
      numpy-only gate because it is exact algebra, and a fourth derivative by
      nested finite differences divides by h^4 -- at h = 2e-3 the cancellation
      reaches 7e-4 on a quantity whose true value is 1e-16.

NOTE ON nu -> 1/2. The residual at nu = 0.49 is roughly twenty times the one at
nu = 0.25 and grows as nu rises. That is not incidental: the image terms
carrying the ``(1-2nu)`` prefactor are exactly the ``1/Q`` and ``log Q`` family
whose mollification is the approximate part. So nu -> 1/2 is simultaneously the
stress case and the diagnostic for whether the new derivation is right.

Needs the ``[oracle]`` extra; importing the Mindlin kernels costs ~14 s.

Run from anywhere:  python tests/gates/oracle/verify_free_surface.py
"""
from __future__ import annotations

import sys

import numpy as np

from _oracle import MU, Report, order_in_eps, sigma_of

NUS = (0.0, 0.25, 0.49)
EPSS = (0.5, 0.2, 0.1, 0.05)
SRC = np.array([0.0, 0.0, -2.0])        # a buried point force
RING = 2.0                              # observer ring radius
N_RING = 6

# Measured on this configuration; gated with headroom so a regression in the
# vendored kernel is caught but the natural nu -> 1/2 growth is not flagged.
TOL_RESIDUAL = 0.60      # worst measured 4.41e-1, at nu = 0.49, eps = 0.5
TOL_ORDER_MIN = 1.80     # measured 1.90 .. 2.00
TRIP_ORDER = 1.50        # the defect: still eps-dependent
TRIP_COARSE = 1.0e-3     # the defect: still large at the coarsest rung
TARGET_EXACT = 1.0e-11   # what the Cortez-consistent version must reach


def main() -> bool:
    from mhs_oracle.moss.mindlin_kernels import mindlin_G

    rep = Report("Free-surface traction residual of the vendored Mindlin "
                 "kernel (the baseline)")
    th = np.linspace(0.0, 2.0 * np.pi, N_RING + 1)[:-1]
    surf = np.stack([RING * np.cos(th), RING * np.sin(th),
                     np.zeros_like(th)], axis=1)
    bulk = surf.copy()
    bulk[:, 2] = -1.0
    print(f"  source at {SRC.tolist()}, {N_RING} observers on a ring of "
          f"radius {RING:g} at z = 0,")
    print(f"  bulk scale from the same ring at z = -1. "
          f"mu = {MU:g}. Classical Mindlin gives 0.")
    print(f"\n  {'nu':>5} {'eps':>7} {'max|t_3k|':>12} {'bulk scale':>12} "
          f"{'relative':>11}")

    res: dict[tuple[float, float], float] = {}
    for nu in NUS:
        for eps in EPSS:
            t3 = max(float(np.abs(sigma_of(mindlin_G, p, SRC, MU, nu,
                                           eps)[2, :, :]).max())
                     for p in surf)
            sb = max(float(np.abs(sigma_of(mindlin_G, p, SRC, MU, nu,
                                           eps)).max()) for p in bulk)
            res[(nu, eps)] = t3 / sb
            print(f"  {nu:5.2f} {eps:7.3f} {t3:12.3e} {sb:12.3e} "
                  f"{t3 / sb:11.3e}")

    worst = max(res.values())
    rep.check("a residual is bounded (regression guard)", worst,
              TOL_RESIDUAL, f"worst at nu=0.49, eps=0.5")

    print("\n  order in eps (consecutive rungs):")
    orders: dict[float, list[float]] = {}
    for nu in NUS:
        vals = [res[(nu, e)] for e in EPSS]
        orders[nu] = order_in_eps(vals, list(EPSS))
        print(f"    nu={nu:4.2f}: "
              + ", ".join(f"{o:.2f}" for o in orders[nu]))
    for nu in NUS:
        rep.check_bool(f"b nu={nu:.2f}: order in eps >= {TOL_ORDER_MIN}",
                       min(orders[nu]) >= TOL_ORDER_MIN,
                       f"(min {min(orders[nu]):.2f}; the literal lift is "
                       f"O(eps^2))")

    print(f"\n  [c] TRIPWIRE: the defect this gate exists to watch")
    print(f"      The residual is still eps-DEPENDENT. When the "
          f"Cortez-consistent image")
    print(f"      mollification lands it must become eps-INDEPENDENT below "
          f"{TARGET_EXACT:.0e},")
    print(f"      these two clauses will FAIL, and the commit that lands it "
          f"must invert")
    print(f"      them. That is the intended behaviour, not a bug in the gate.")
    for nu in NUS:
        rep.check_bool(f"c nu={nu:.2f}: STILL eps-dependent "
                       f"(order > {TRIP_ORDER})",
                       min(orders[nu]) > TRIP_ORDER,
                       f"(min order {min(orders[nu]):.2f})")
    coarse = max(res[(nu, EPSS[0])] for nu in NUS)
    rep.check_bool(f"c STILL far from exact at eps = {EPSS[0]:g} "
                   f"(> {TRIP_COARSE:.0e})", coarse > TRIP_COARSE,
                   f"({coarse:.3e}; target is < {TARGET_EXACT:.0e}, "
                   f"eps-independent)")

    # nu -> 1/2 is the stress case AND the diagnostic: the image terms carrying
    # (1-2nu) are exactly the ones whose mollification is approximate.
    ratio = res[(0.49, 0.1)] / res[(0.25, 0.1)]
    rep.check_bool("c nu = 0.49 is materially worse than nu = 0.25",
                   ratio > 5.0,
                   f"({ratio:.1f}x at eps = 0.1 -- the (1-2nu) image terms "
                   f"are the approximate ones)")
    part_d(rep)
    return rep.finish()


def part_d(rep) -> None:
    """Apostol's potentials are BIHARMONIC in the SOURCE variable.

    This is what lets the Cortez-consistent derivation apply one unified rule to
    the whole potential set instead of classifying term by term:

        f * phi_eps = (1 - eps^2 d/d(eps^2))[f]_eps + (eps^2/2)[(1/2) lap f]_eps

    holds for any biharmonic f. The mollification is a SOURCE convolution, so
    the relevant Laplacian is in the source variable, which in the half-space
    parametrisation (x, y relative in-plane; z, z0 absolute depths) is
    ``d2/dx2 + d2/dy2 + d2/dz0^2``.

    Checked SYMBOLICALLY and then evaluated, not by finite differences: a fourth
    derivative by nested 5-point stencils divides by h^4, and at h = 2e-3 the
    cancellation reaches 7e-4 on a quantity whose true value is 1e-16. The first
    draft of this clause lived in the numpy-only gate and measured exactly that
    cancellation.
    """
    import sympy as sp

    print("\n[d] APOSTOL'S POTENTIALS ARE BIHARMONIC IN THE SOURCE VARIABLE")
    print("    lap_src = d2/dx2 + d2/dy2 + d2/dz0^2, differentiated "
          "symbolically")
    x, y, z, z0 = sp.symbols("x y z z0", real=True)
    nu = sp.Symbol("nu", real=True)
    mu_s = sp.Symbol("mu", positive=True)
    r1 = sp.sqrt(x ** 2 + y ** 2 + (z - z0) ** 2)
    r2 = sp.sqrt(x ** 2 + y ** 2 + (z + z0) ** 2)
    Q = r2 - (z + z0)
    K = 1 / (4 * sp.pi * mu_s)
    a0 = -z0                      # |z0|, analytic because z0 <= 0

    pots = {
        "vertical bz": K * (1 / r1 + (3 - 4 * nu) / r2
                            + 2 * z0 * (z + z0) / r2 ** 3),
        "vertical beta": K * (a0 / r1 + (3 - 4 * nu) * a0 / r2
                              - 4 * (1 - nu) * (1 - 2 * nu) * sp.log(Q)),
        "horiz-x bx": K * (1 / r1 + 1 / r2),
        "horiz-x bz": 2 * K * (a0 / r2 ** 2
                               - (1 - 2 * nu) / Q) * x / r2,
        "horiz-x beta": (2 * K * (1 - 2 * nu)
                         * (a0 / r2 - (1 - 2 * nu)) * x / Q),
    }

    def lap_src(f):
        return sp.diff(f, x, 2) + sp.diff(f, y, 2) + sp.diff(f, z0, 2)

    args = (x, y, z, z0, mu_s, nu)
    vals = (0.7, -0.4, -1.1, -2.3, 1.0, 0.3)
    print(f"    {'potential':16s} {'|f|':>12s} {'|lap_src f|':>13s} "
          f"{'|lap^2_src f|':>14s}  verdict")
    worst = 0.0
    for name, f in pots.items():
        L1 = lap_src(f)
        L2 = lap_src(L1)
        fn = abs(float(sp.lambdify(args, f, "numpy")(*vals)))
        f1 = abs(float(sp.lambdify(args, L1, "numpy")(*vals)))
        f2 = abs(float(sp.lambdify(args, L2, "numpy")(*vals)))
        kind = "harmonic" if f1 < 1e-10 else "biharmonic"
        worst = max(worst, f2 / fn)
        print(f"    {name:16s} {fn:12.4e} {f1:13.4e} {f2:14.4e}  {kind}")
    rep.check("d every potential satisfies lap_src^2 f = 0", worst, 1e-12,
              "so the unified rule applies wholesale -- no term-by-term "
              "classification, no new function class")


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
