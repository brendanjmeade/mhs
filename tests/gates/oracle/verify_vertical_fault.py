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

Three clauses:

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
    return rep.finish()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
