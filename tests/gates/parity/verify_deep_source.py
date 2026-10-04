#!/usr/bin/env python
"""The deep-source limit, and the proof that the eigenstress reuse is legitimate.

``mhs`` reuses the FULL-SPACE eigenstress verbatim for the half-space kernel.
That is the load-bearing assumption of the whole decomposition, and so far it
has been an argument rather than a measurement. The argument:
``Phi_eps(x) = int_T phi_eps(x - y) dS`` is a pointwise statement about the blob
and the source triangle, reading no Green's function, so no free surface and no
image can enter it; all the blob content sits in the direct term, while the
image correction depends only on the distance to the image, which never
vanishes inside the body.

This gate turns each half of that into a check.

  [a] THE LIMIT. Far below the free surface the half-space kernel must become
      the full-space one. Oracle quadrature of the Mindlin triangle against the
      SHIPPED full-space closed form, displacement and stress. If the image
      terms failed to vanish with depth this is where it shows.

  [b] THE EIGENSTRESS READS NO GREEN'S FUNCTION. Translate the same triangle and
      observer together to a different depth: the full-space eigenstress must be
      unchanged, because it depends on the triangle's geometry, its normal, eps
      and (mu, lam) and on nothing else. A quantity that secretly knew about the
      free surface could not survive this.

      It is NOT bitwise, and the reason is worth recording rather than hiding
      behind a loose tolerance: translating to depth 2000 is exact in real
      arithmetic but not in floating point, because coordinates of magnitude
      2000 against a triangle of size 1 lose absolute precision inside the
      moment integrals. Measured 1.1e-11 absolute on a quantity of size 21.5 --
      5e-13 relative, which is rounding, not depth dependence. Gated relatively
      and labelled as such; the first draft of this clause asserted bitwise
      equality and was wrong.

  [c] THE REUSE ITSELF, and this is the clause the decomposition rests on. At a
      MODERATE depth, where the image correction is fully alive, drive eps down
      on the source triangle and watch the half-space stress:

        - the RAW (total) half-space stress diverges, like 1/eps;
        - the same stress MINUS the full-space eigenstress stays bounded.

      If the half-space kernel carried any blob content the full-space
      eigenstress did not account for, the subtraction would leave a divergent
      remainder. It does not. That is the measurement the reuse argument needed,
      and it is made where the image term matters rather than in the limit where
      it has switched off.

Needs the ``[oracle]`` extra; importing the Mindlin kernels costs ~14 s.

Run from anywhere:  python tests/gates/parity/verify_deep_source.py
"""
from __future__ import annotations

import sys

import numpy as np

MU, LAM = 1.0, 1.5          # nu = 0.3, not 1/4
NU = 0.5 * LAM / (LAM + MU)

#: A tilted triangle, so no axis is special, placed at a depth below.
TRI0 = np.array([[0.00, 0.00, 0.00],
                 [1.00, 0.00, -0.10],
                 [0.30, 0.90, 0.05]])
SLIP = np.array([0.7, -0.4, 0.2])

DEEP = 2.0e3                # depth at which the image must be negligible
MODERATE = 1.5              # depth at which the image is fully alive
EPS_LADDER = (0.4, 0.2, 0.1, 0.05, 0.025)

# The deep-source clause is limited by the ORACLE's quadrature, not by anything
# mhs does: the reference IS a Gauss rule on the Mindlin kernel. Measured
# 3.4e-7 at n_quad = 24, and upstream's own clause on this same comparison
# gates it at 1e-5 for exactly that reason. 1e-6 is three times the
# measurement and still two orders inside upstream's choice.
TOL_DEEP_U = 1e-6
TOL_DEEP_S = 1e-6
# Translating a triangle to depth 2000 is translation-invariant in exact
# arithmetic but NOT in floating point: coordinates of magnitude 2000 against a
# triangle of size 1 lose absolute precision in the moment integrals. Measured
# 1.1e-11 absolute on a quantity of size 21.5, i.e. ~5e-13 relative -- rounding,
# not a dependence on depth. Gated relatively and labelled as such; "bitwise"
# would have been an overclaim.
TOL_EIG_TRANSLATE = 1e-11
RAW_GROWTH_MIN = 1.7        # the raw stress must grow per eps halving
# The COARSEST rung is exempt, with the reason: at eps = 0.4 on a triangle of
# unit scale the ~4 eps / 3 mollified fault-zone width is comparable to the
# inradius, so there is no interior to be bounded in yet. Upstream's full-space
# ladder exempts its own coarsest rung for the same reason. What is asserted
# instead is stronger than a band: the ratios must DECREASE toward 1.
SUB_RATIO_BAND = (0.8, 1.30)


class Report:
    def __init__(self, title):
        self.rows = []
        self.title = title
        print("=" * 76)
        print(title)
        print("=" * 76)

    def check(self, name, value, tol, extra=""):
        ok = bool(value < tol)
        self.rows.append(ok)
        print(f"  [{'ok' if ok else 'XX'}] {name:52s} {value:10.3e}  "
              f"(tol {tol:.0e}) {extra}")
        return ok

    def check_bool(self, name, ok, extra=""):
        ok = bool(ok)
        self.rows.append(ok)
        print(f"  [{'ok' if ok else 'XX'}] {name:52s} {extra}")
        return ok

    def finish(self):
        ok = all(self.rows)
        print("-" * 76)
        print(f"{'PASS' if ok else 'FAIL'}: {self.title} "
              f"({len(self.rows)} checks)")
        return ok


def relmax(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ref = np.max(np.abs(b))
    return (np.max(np.abs(a - b)) / ref if ref > 1e-300
            else np.max(np.abs(a - b)))


def at_depth(d):
    """The triangle shifted so its first vertex sits at depth ``d``."""
    tri = TRI0.copy()
    tri[:, 2] -= d
    return tri


def shipped_fullspace(obs, tri, eps, want):
    from mhs.fullspace.kernels import nodal_influence
    return nodal_influence(obs, tri, 0, MU, NU, eps, want=want,
                           far_field="analytic")


def main() -> bool:
    from mhs.kernels import assemble as asm
    from mhs_oracle.moss.mindlin_triangle import (
        integrate_mindlin_dd_kernel, integrate_mindlin_stress_kernel_analytical)

    rep = Report("Deep-source limit, and the eigenstress reuse")
    print(f"  mu = {MU:g}, lam = {LAM:g} (nu = {NU:.3f}, not 1/4)")

    # ----------------------------------------------------------------- [a] ---
    print(f"\n[a] THE LIMIT: at depth {DEEP:g} the half space must become the "
          f"full space")
    tri = at_depth(DEEP)
    n = asm._unit_normal(tri)
    obs = np.array([[0.5, 0.4, 1.5 - DEEP], [0.9, 0.2, -1.2 - DEEP]])
    eps = 0.1
    for i, o in enumerate(obs):
        Uhs = integrate_mindlin_dd_kernel(o, tri[0], tri[1], tri[2], n,
                                          MU, NU, eps, n_quad=24)
        Ufs = np.asarray(shipped_fullspace(o[None], TRI0 - [0, 0, DEEP], eps,
                                           ("U",))["U"])[0, 0]
        rep.check(f"a obs {i}: half-space DD kernel -> full space",
                  relmax(Uhs, Ufs), TOL_DEEP_U)
        Hhs = integrate_mindlin_stress_kernel_analytical(
            o, tri[0], tri[1], tri[2], n, MU, NU, eps, n_quad=8)
        Hfs = np.asarray(shipped_fullspace(o[None], TRI0 - [0, 0, DEEP], eps,
                                           ("H",))["H"])[0, 0]
        rep.check(f"a obs {i}: half-space stress kernel -> full space",
                  relmax(Hhs, Hfs), TOL_DEEP_S)

    # ----------------------------------------------------------------- [b] ---
    print("\n[b] THE EIGENSTRESS READS NO GREEN'S FUNCTION: translating the")
    print("    triangle and observer together must leave it unchanged")
    eps_arr = np.full(1, 0.1)
    probe = np.array([[0.35, 0.30, 0.02]])
    eig = {}
    for d in (MODERATE, DEEP):
        tri_d = at_depth(d)
        q = probe.copy()
        q[:, 2] -= d
        out = np.zeros((1, 3, 3, 1, 3))
        asm.assemble_eigenstress(q, tri_d[None], MU, LAM, eps_arr, out)
        eig[d] = out
    scale = float(np.abs(eig[MODERATE]).max())
    diff = float(np.abs(eig[MODERATE] - eig[DEEP]).max())
    rep.check(f"b eigenstress at depth {MODERATE:g} == at depth {DEEP:g}",
              diff / scale, TOL_EIG_TRANSLATE,
              f"[abs {diff:.1e} on a quantity of size {scale:.1f}; the "
              f"residue is coordinate rounding at depth 2000, not depth "
              f"dependence]")
    rep.check_bool("b and it is non-zero there (not a vacuous equality)",
                   float(np.abs(eig[MODERATE]).max()) > 1e-6,
                   f"max |C:eps*| = {np.abs(eig[MODERATE]).max():.3e}")

    # ----------------------------------------------------------------- [c] ---
    print(f"\n[c] THE REUSE, at depth {MODERATE:g} where the image term is "
          f"fully alive")
    tri = at_depth(MODERATE)
    n = asm._unit_normal(tri)
    centroid = tri.mean(axis=0)
    print(f"      {'eps':>7} {'raw |sigma|':>13} {'raw ratio':>10} "
          f"{'subtracted':>13} {'sub ratio':>10}")
    raw_list, sub_list = [], []
    for eps in EPS_LADDER:
        H = integrate_mindlin_stress_kernel_analytical(
            centroid, tri[0], tri[1], tri[2], n, MU, NU, eps, n_quad=8)
        raw = np.einsum("mnk,k->mn", H, SLIP)
        out = np.zeros((1, 3, 3, 1, 3))
        asm.assemble_eigenstress(centroid[None], tri[None], MU, LAM,
                                 np.full(1, eps), out)
        eigv = np.einsum("omnsk,sk->omn", out, SLIP[None])[0]
        raw_list.append(float(np.linalg.norm(raw)))
        sub_list.append(float(np.linalg.norm(raw - eigv)))
        rr = (f"{raw_list[-1] / raw_list[-2]:10.3f}"
              if len(raw_list) > 1 else " " * 10)
        sr = (f"{sub_list[-1] / sub_list[-2]:10.3f}"
              if len(sub_list) > 1 else " " * 10)
        print(f"      {eps:7.4f} {raw_list[-1]:13.5f} {rr} "
              f"{sub_list[-1]:13.5f} {sr}")

    raw_ratio = np.array(raw_list[1:]) / np.array(raw_list[:-1])
    sub_ratio = np.array(sub_list[1:]) / np.array(sub_list[:-1])
    rep.check_bool(f"c RAW half-space stress diverges "
                   f"(>= {RAW_GROWTH_MIN} per halving)",
                   bool(np.all(raw_ratio >= RAW_GROWTH_MIN)),
                   f"(min {raw_ratio.min():.3f}) -- the 1/eps eigenstress")
    lo, hi = SUB_RATIO_BAND
    # The coarsest rung is excluded; see the constant's comment.
    fine = sub_ratio[1:]
    rep.check_bool(f"c MINUS the FULL-SPACE eigenstress it stays bounded "
                   f"({lo}, {hi}) for the fine rungs",
                   bool(np.all((fine > lo) & (fine < hi))),
                   f"(range {fine.min():.3f}..{fine.max():.3f}; coarsest rung "
                   f"{sub_ratio[0]:.3f} exempt -- no interior at eps = "
                   f"{EPS_LADDER[0]:g})")
    rep.check_bool("c and the ratios DECREASE toward 1 along the ladder",
                   bool(np.all(np.diff(sub_ratio) < 0.0)
                        and sub_ratio[-1] < 1.05),
                   f"({', '.join(f'{r:.3f}' for r in sub_ratio)}) -- "
                   f"converging, which a band alone would not show")
    growth = raw_list[-1] / raw_list[0]
    bounded = sub_list[-1] / sub_list[0]
    rep.check_bool("c the two behaviours are unmistakably different",
                   growth / max(bounded, 1e-300) > 5.0,
                   f"raw grew {growth:.1f}x over the ladder, subtracted "
                   f"{bounded:.2f}x")
    return rep.finish()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
