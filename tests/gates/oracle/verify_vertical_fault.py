#!/usr/bin/env python
"""The vertical fault: a coplanar image, and the quadrature budget it costs.

The plan calls this the most likely unpleasant surprise, so it gets a gate
before any closed form exists to be surprised by.

**The geometric fact.** Reflect a triangle in the free surface. For a DIPPING
triangle the image lies in a different plane, comfortably away from the
original. For a VERTICAL one the image is **coplanar with the original** -- the
perpendicular offset from an on-fault observer to the image plane is exactly
zero -- so an on-element observer sits in the image's own plane, displaced
in-plane by about twice the depth. That is the regime where the in-plane moment
machinery is weakest: the effective height collapses to eps while the in-plane
distance is large, which is exactly the ``(D/h_eps)`` ratio whose digit loss the
upstream derivation notes document, and where the ``I7`` seed's accuracy bound
is ABSOLUTE rather than relative. Vertical faults are the commonest geometry in
practice, so this is not a corner case.

Four clauses, and the fourth retires a motivation the roadmap had wrong:

  [a] THE GEOMETRY, measured rather than assumed: the vertical triangle's image
      plane contains the observer (offset 0 to rounding) and the dipping one's
      does not. If a future change to the frame convention broke this, every
      conclusion below would be about a different configuration.

  [b] THE QUADRATURE BUDGET, which is the quantitative statement of what a
      closed form is worth. The oracle's Gauss rule resolves the kernel only
      when its point count grows with ``L/eps``: measured, ``n_quad`` of about
      ``8 L / eps`` is needed for machine precision. Halving that costs about
      six orders of accuracy (3e-13 -> 3.5e-7 at every rung) and quartering it
      costs ten (8.8e-3). This is gated as a LAW, not a single number -- the
      rule must converge at each eps once given enough points, and must visibly
      degrade when starved.

      It is also a warning about using quadrature as an oracle: a comparison run
      at fixed ``n_quad`` across an eps ladder silently stops being a
      comparison. The first draft of this measurement read the non-monotonic
      tail of an under-resolved reference as the method failing.

  [c] THE COPLANAR IMAGE IS NOT, BY ITSELF, FATAL. Once the rule is resolved,
      the vertical case is as accurate as the dipping one. So the risk flagged in
      the plan is a CONDITIONING risk in the closed form yet to be written, not
      an inherited defect in the kernel -- and when that closed form lands, this
      configuration at small eps/L is precisely where it has to earn its keep.

  [d] A SURFACE-REACHING FAULT IS NOT THE HARD CASE, measured. The roadmap's
      stated reason for wanting closed-form image integration was that the
      near-singular surface-breaking configuration is reachable only in closed
      form. It is not: the hybrid decomposition -- closed-form direct term plus
      quadrature on the image CORRECTION -- is machine-accurate at n_quad = 16
      for a triangle whose vertex sits exactly on z = 0, and the VERTICAL
      geometry of clause [a] is the better of the two. See part_d for why, for
      the eps-independence that explains it, and for the domain boundary that
      the first draft of this measurement mistook for a near-singularity.

Needs the ``[oracle]`` extra; importing the Mindlin kernels costs ~14 s.

Run from anywhere:  python tests/gates/oracle/verify_vertical_fault.py
"""
from __future__ import annotations

import sys

import numpy as np

from _oracle import MU, Report, relmax

NU = 0.3
DEPTH = 2.0

#: Vertical: in the x = 0 plane, so its mirror is the same plane.
VERT = np.array([[0.0, 0.0, -DEPTH],
                 [0.0, 1.0, -DEPTH],
                 [0.0, 0.4, -DEPTH + 0.9]])
#: Dipping, for contrast: the mirror is a different plane.
DIP = np.array([[0.0, 0.0, -DEPTH],
                [1.0, 0.0, -DEPTH - 0.3],
                [0.3, 0.9, -DEPTH + 0.5]])

#: (eps, n_quad that should resolve it, n_quad that should NOT). The resolved
#: column follows the measured law n_quad ~ 8 L / eps, with L ~ 1.08 here.
BUDGET = ((0.4, 24, None), (0.2, 48, 24), (0.1, 96, 48), (0.05, 160, 96))
TOL_RESOLVED = 1e-9       # measured 1.6e-14 .. 3.3e-13 at the resolved rung
# What starving actually costs, measured against a PROPERLY resolved reference:
# halving n_quad below the resolved point gives a consistent ~3.5e-7 at every
# rung -- about SIX orders worse than the 3e-13 a resolved rule delivers -- and
# quartering it gives 8.8e-3. So the penalty is steep in DIGITS rather than
# large in absolute terms, and the first draft of this clause said the latter
# because it compared against an under-resolved reference of its own.
STARVED_MIN = 1e-8
TOL_COPLANAR = 1e-12      # the image-plane offset of a vertical triangle
# Clause [d]: the hybrid (closed-form direct + quadrature image) ON the surface.
# Measured 8.1e-15 vertical, 1.4e-14 dipping, at n_quad = 16.
TOL_SURFACE = 1e-12
TOL_EPS_SPREAD = 10.0     # measured spread over a 16x eps range; see [d]


def unit_normal(t):
    n = np.cross(t[1] - t[0], t[2] - t[0])
    return n / np.linalg.norm(n)


def mirror(t):
    m = t.copy()
    m[:, 2] *= -1.0
    return m


def main() -> bool:
    from mhs_oracle.moss.mindlin_triangle import integrate_mindlin_dd_kernel

    rep = Report("Vertical fault: the coplanar image and its quadrature budget")

    # ----------------------------------------------------------------- [a] ---
    print("\n[a] THE GEOMETRY: where the image plane sits relative to the "
          "observer")
    offsets = {}
    for name, t in (("vertical", VERT), ("dipping", DIP)):
        img = mirror(t)
        inrm = unit_normal(img)
        cen = t.mean(axis=0)
        perp = abs(float((cen - img.mean(axis=0)) @ inrm))
        inpl = float(np.linalg.norm((cen - img.mean(axis=0)) - perp * inrm))
        offsets[name] = perp
        print(f"    {name:9s} normal {np.round(unit_normal(t), 3)}  "
              f"perp offset to the image plane {perp:10.3e}  "
              f"in-plane distance {inpl:.3f}")
    rep.check("a a VERTICAL triangle's image plane contains the observer",
              offsets["vertical"], TOL_COPLANAR,
              "so the effective height collapses to eps")
    rep.check_bool("a a DIPPING triangle's does not (the contrast is real)",
                   offsets["dipping"] > 1.0,
                   f"(offset {offsets['dipping']:.3f})")

    # ----------------------------------------------------------------- [b] ---
    L = max(float(np.linalg.norm(VERT[i] - VERT[j]))
            for i in range(3) for j in range(3))
    cen = VERT.mean(axis=0)
    n = unit_normal(VERT)
    print(f"\n[b] THE QUADRATURE BUDGET on the vertical element "
          f"(L = {L:.3f}, observer ON it)")
    print(f"    {'eps':>7} {'eps/L':>7} {'8 L/eps':>9} "
          f"{'resolved n_quad':>16} {'err':>10} {'starved':>9} {'err':>10}")
    for eps, nq_ok, nq_bad in BUDGET:
        ref = integrate_mindlin_dd_kernel(cen, VERT[0], VERT[1], VERT[2], n,
                                          MU, NU, eps, n_quad=2 * nq_ok)
        got = integrate_mindlin_dd_kernel(cen, VERT[0], VERT[1], VERT[2], n,
                                          MU, NU, eps, n_quad=nq_ok)
        e_ok = relmax(got, ref)
        if nq_bad is None:
            print(f"    {eps:7.3f} {eps / L:7.3f} {8 * L / eps:9.0f} "
                  f"{nq_ok:16d} {e_ok:10.1e} {'--':>9} {'--':>10}")
            rep.check(f"b eps={eps:g}: n_quad={nq_ok} resolves it", e_ok,
                      TOL_RESOLVED)
            continue
        bad = integrate_mindlin_dd_kernel(cen, VERT[0], VERT[1], VERT[2], n,
                                          MU, NU, eps, n_quad=nq_bad)
        e_bad = relmax(bad, ref)
        print(f"    {eps:7.3f} {eps / L:7.3f} {8 * L / eps:9.0f} "
              f"{nq_ok:16d} {e_ok:10.1e} {nq_bad:9d} {e_bad:10.1e}")
        rep.check(f"b eps={eps:g}: n_quad={nq_ok} resolves it", e_ok,
                  TOL_RESOLVED)
        rep.check_bool(f"b eps={eps:g}: n_quad={nq_bad} is STARVED "
                       f"(> {STARVED_MIN:.0e})", e_bad > STARVED_MIN,
                       f"({e_bad:.1e}) -- the budget is a law, not a "
                       f"preference")

    # ----------------------------------------------------------------- [c] ---
    print("\n[c] IS THE COPLANAR IMAGE ITSELF FATAL? Compare vertical against "
          "dipping")
    print("    at a RESOLVED rule, so the comparison is about geometry rather "
          "than budget.")
    eps = 0.2
    pair = {}
    for name, t in (("vertical", VERT), ("dipping", DIP)):
        nn, c = unit_normal(t), t.mean(axis=0)
        ref = integrate_mindlin_dd_kernel(c, t[0], t[1], t[2], nn, MU, NU, eps,
                                          n_quad=96)
        got = integrate_mindlin_dd_kernel(c, t[0], t[1], t[2], nn, MU, NU, eps,
                                          n_quad=48)
        pair[name] = relmax(got, ref)
        print(f"    {name:9s} self-consistency at n_quad 48 vs 96: "
              f"{pair[name]:.2e}")
    rep.check("c vertical is as accurate as dipping once resolved",
              pair["vertical"], max(100.0 * pair["dipping"], 1e-12),
              "so the plan's R4 is a conditioning risk in the closed form "
              "still to be written, not an inherited defect")
    part_d(rep)
    return rep.finish()


def part_d(rep) -> None:
    """A SURFACE-REACHING fault, which was supposed to be the hard case.

    The roadmap's stated motivation for closed-form image integration was that
    "the near-singular surface-breaking case" is reachable only in closed form,
    because the reflected triangle approaches the real one. **Measured, that is
    false**, and this clause is here to keep it false rather than let the claim
    come back.

    Three things, and the first is the one that matters:

      [d] THE HYBRID IS MACHINE-ACCURATE AT THE SURFACE. Closed-form direct term
          plus Gauss quadrature on the image CORRECTION only (the decomposition
          ``integrate_mindlin_dd_kernel_analytical`` implements). For a triangle
          whose shallowest vertex sits exactly on ``z = 0``, n_quad = 16 lands at
          8.1e-15 for the vertical geometry and 1.4e-14 for the dipping one --
          and the vertical case, the one R4 flags, is the BETTER of the two.
          So the coplanar image plus a ``log Q`` singularity at ``z = 0`` costs
          nothing once the direct term is analytic.

          Why it works: the singularity is ``log Q`` with ``Q -> 0`` on a set of
          measure zero, which is integrable. Refining a vertex to 1e-9 below the
          surface moves the integral by 2e-15 .. 9e-15 between n_quad 16 and 256
          -- rounding, not convergence. Quadrature does not notice it.

      [d] THE IMAGE QUADRATURE IS eps-INDEPENDENT, at fixed n_quad, over an eps
          ladder. This one the roadmap got right, and it is the structural
          reason d-above holds: the image integrand's length scale is set by
          DEPTH, not by eps, so no eps-dependent budget appears. Contrast
          clause [b], where the direct term needs n_quad ~ 8 L / eps.

      [d] A STRADDLING TRIANGLE IS SILENTLY ACCEPTED BY THE ORACLE, and that
          is why the half-space convention is checked on the VERTICES in
          ``mhs.matrices._coerce`` rather than left to a kernel. Push one vertex
          to ``z = +1.0``, well outside the body, and the hybrid returns
          0.2381 at n_quad = 16 and 0.2384 at 32: wrong, and CONVERGED, which
          is the one failure mode a refinement study cannot see. The mechanism
          is an asymmetry in the oracle -- ``mindlin_G``,
          ``mindlin_DG_source`` and ``mindlin_DDG_obs_source`` all check
          ``source[2] < 0``, the three ``*_correction`` functions do not, and
          the hybrid calls one of those. Gated rather than fixed: the oracle is
          frozen, and the shipped package already refuses the geometry.

          "Surface-breaking" therefore has to mean REACHING the surface, not
          crossing it. The first draft of this measurement used a triangle with
          a vertex at z = +0.0038, read the resulting exception from a
          DIFFERENT entry point as a near-singularity, and concluded the
          surface case was unreachable -- the opposite of what d-above shows.

    What survives of the case for closed-form image integration is therefore
    PERFORMANCE, not capability, and it has to be costed against the SHIPPED
    direct term (``mhs.fullspace``, vectorised over observers, 3.4-8.0 us per
    obs/source pair) rather than this oracle's scalar Python recursion at
    2895 us. Costed against the latter the ceiling looks like 5.5x; against the
    former the image quadrature dominates and the ceiling is an order of
    magnitude. Only the second comparison means anything.
    """
    from mhs_oracle.moss.mindlin_triangle import (
        integrate_mindlin_dd_kernel_analytical as hybrid)

    #: Shallowest vertex exactly ON the free surface, vertical and dipping.
    surf = {
        "vertical": np.array([[0.0, 0.0, -1.0],
                              [0.0, 1.0, -1.0],
                              [0.0, 0.4, 0.0]]),
        "dipping": np.array([[0.0, 0.0, -1.0],
                             [1.0, 0.0, -1.1],
                             [0.3, 0.9, 0.0]]),
    }
    eps = 0.1
    print("\n[d] A SURFACE-REACHING FAULT: the case the roadmap called "
          "unreachable")
    print("    Closed-form direct term + quadrature on the IMAGE CORRECTION "
          "only.")
    print(f"    {'geometry':10s} {'n_quad=4':>11} {'8':>11} {'16':>11}  "
          f"(vs n_quad=96)")
    worst16 = 0.0
    for name, t in surf.items():
        nn, c = unit_normal(t), t.mean(axis=0)
        ref = hybrid(c, t[0], t[1], t[2], nn, MU, NU, eps, n_quad=96)
        errs = [relmax(hybrid(c, t[0], t[1], t[2], nn, MU, NU, eps, n_quad=q),
                       ref) for q in (4, 8, 16)]
        worst16 = max(worst16, errs[-1])
        print(f"    {name:10s} " + "".join(f"{e:11.2e}" for e in errs))
    rep.check("d the hybrid is machine-accurate ON the free surface at "
              "n_quad=16", worst16, TOL_SURFACE,
              "so closed-form image integration is a PERFORMANCE question, "
              "not a capability one")

    # eps-independence at fixed n_quad: the structural reason d-above holds.
    print("\n    the image quadrature's error does not depend on eps "
          "(n_quad = 8 throughout):")
    t = surf["dipping"]
    nn, c = unit_normal(t), t.mean(axis=0)
    ladder = {}
    for ev in (0.4, 0.1, 0.025):
        ref = hybrid(c, t[0], t[1], t[2], nn, MU, NU, ev, n_quad=96)
        ladder[ev] = relmax(hybrid(c, t[0], t[1], t[2], nn, MU, NU, ev,
                                   n_quad=8), ref)
        print(f"      eps = {ev:6.3f}   err = {ladder[ev]:.2e}")
    spread = max(ladder.values()) / max(min(ladder.values()), 1e-300)
    rep.check_bool(f"d that error is eps-INDEPENDENT (spread < "
                   f"{TOL_EPS_SPREAD:g}x over a 16x eps range)",
                   spread < TOL_EPS_SPREAD,
                   f"({spread:.1f}x) -- the image integrand's scale is DEPTH, "
                   f"not eps, unlike the direct term of clause [b]")

    # Why the half-space convention is checked at the GEOMETRY level in
    # mhs.matrices._coerce and not left to a kernel.
    print("\n    a triangle STRADDLING the surface (one vertex above z = 0):")
    print(f"      {'vertex z':>9} {'oracle hybrid n_quad=16':>24} "
          f"{'n_quad=32':>12}")
    straddle = []
    for zv in (0.05, 1.0):
        above = surf["dipping"].copy()
        above[2, 2] = zv                     # a vertex OUT of the half space
        nn, c = unit_normal(above), above.mean(axis=0)
        vals = []
        for q in (16, 32):
            try:
                vals.append(float(np.abs(hybrid(c, above[0], above[1],
                                                above[2], nn, MU, NU, eps,
                                                n_quad=q)).max()))
            except ValueError:
                vals.append(None)
        straddle.append(vals)
        fmt = [("RAISES" if v is None else f"{v:.6f}") for v in vals]
        print(f"      {zv:9.2f} {fmt[0]:>24} {fmt[1]:>12}")
    accepted = all(v is not None for vals in straddle for v in vals)
    converged = all(abs(v[0] - v[1]) / max(v[1], 1e-300) < 1e-2
                    for v in straddle if None not in v)
    rep.check_bool("d the ORACLE silently accepts a straddling triangle",
                   accepted and converged,
                   "and the answer it returns is not merely wrong but "
                   "CONVERGED in n_quad, which is the failure mode a "
                   "refinement study cannot detect. Three of the oracle's "
                   "pointwise entry points check source[2] < 0; the three "
                   "*_correction ones do not, and the hybrid calls one of "
                   "those. Gated, NOT fixed: the oracle is frozen.")

    from mhs import Material
    from mhs.matrices import disp_matrix
    above = surf["dipping"].copy()
    above[2, 2] = +0.05
    try:
        disp_matrix(np.array([[0.0, 0.0, -0.5]]), above[None, ...],
                    Material(mu=MU, lam=2.0 * MU * NU / (1.0 - 2.0 * NU)),
                    eps)
        caught = ""
    except ValueError as exc:
        caught = str(exc)
    except NotImplementedError:
        caught = ""          # validation ran and passed it through: a failure
    rep.check_bool("d but mhs REFUSES it, at the geometry level",
                   "z <= 0" in caught and "vertex 2" in caught,
                   f"({caught[:70] or 'NOT refused'}) -- which is the whole "
                   f"reason _coerce checks the convention on the vertices "
                   f"instead of trusting the kernel to notice")


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
