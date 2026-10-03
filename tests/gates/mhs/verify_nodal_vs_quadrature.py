#!/usr/bin/env python
"""The assembled U/H/E and G/S kernels vs the Gauss product-rule oracle.

``verify_moments`` checks the moment TABLE against quadrature; this checks the
KERNELS built from it, which is a different claim -- the lift from in-plane
moments to the 3-D tensors, the stiffness contractions, and the ``(3-4nu)``,
``2(1-nu)`` and ``eps^2`` coefficients that ride on them. A moment table can be
perfect and the kernel still wrong by a transposed index or a swapped lam/mu.

Everything runs at ``nu = 0.3``, never 1/4, where ``lam == mu`` and a swap is
invisible. That is not hypothetical: upstream records the Mindlin triangle code
keeping lam and mu swapped until 2026-09-17, and four files carrying the error
because every gate ran at 1/4.

Four batteries:

  * OFF-PLANE, p = 0/1/2, against a 40x40 rule at 1e-9.
  * ON-PLANE and mollified -- an interior point and an EDGE MIDPOINT, where the
    edge primitives switch regime -- at h = eps = 0.15 L against 160x160. The
    FORCE kernels are gated six orders tighter than the slip ones: their
    highest inverse power is R^-5 at degree 5, so the rule resolves them and
    they are not quadrature-limited. The slip kernels' 1e-6 is legacy headroom
    from a smaller eps, kept as upstream had it.
  * THE FAR-FIELD PRODUCER at D = 3 L in an oblique direction, closed form vs
    quadrature. The closed form loses digits as (R/L)^{4-5}, which is why a
    far-field producer exists at all.
  * **HYBRID SWITCHING**, bitwise. Within ``D_STAR`` L the hybrid must BE the
    closed form, beyond it must BE quadrature, on a mixed batch and on pure
    near/far batches. This battery matters more here than upstream: the same
    architecture is the stated fallback if the half-space ``Q_eps`` closed form
    turns out numerically worse than quadrature in some regime, so the
    switching machinery has to be trustworthy before it is relied on.

    One documented non-bitwise case: against a FULL-BATCH quadrature call the
    far rows agree only to ~1e-15, because the BLAS matmul inside the weighted
    tables rounds differently for a different batch size. Checked at 1e-13 and
    compared bitwise against quadrature evaluated on exactly those rows, which
    is what the hybrid actually does. A batch-size artefact, not a kernel issue.

Tolerances are upstream's, unchanged.

Run from anywhere:  python tests/gates/mhs/verify_nodal_vs_quadrature.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

from _common import MU, TRI, Report, relmax, shipped

NU = 0.3
KEYS = ("U", "H", "E")            # slip (displacement-discontinuity) source
KEYS_F = ("G", "S")               # force (Kelvin single-layer) source
ALL_KEYS = KEYS + KEYS_F
TOL_ON_PLANE_SLIP = 1e-6          # legacy headroom; measured ~2.5e-14 at 0.15 L
TOL_ON_PLANE_FORCE = 1e-12        # measured ~2.1e-14: not quadrature-limited


def compare(rep, label, got, ref, tol, keys=KEYS) -> float:
    worst = 0.0
    for key in keys:
        r = relmax(got[key], ref[key])
        worst = max(worst, r)
        rep.check(f"{label}: {key}", r, tol)
    return worst


def bitwise(rep, label, a, b, mask, keys=ALL_KEYS) -> None:
    """Rows ``mask`` of every tensor must be bitwise identical (relmax == 0)."""
    for key in keys:
        r = relmax(a[key][mask], b[key][mask])
        rep.check_bool(f"{label}: {key} bitwise (relmax == 0)", r == 0.0,
                       f"(relmax {r:.1e})")


def main() -> bool:
    from mhs_oracle.clq.quadrature import quadrature_influence

    nodal_influence = shipped("kernels").nodal_influence
    local_frame = shipped("frame").local_frame
    defaults = shipped("defaults")

    rep = Report("Assembled nodal kernels vs the Gauss quadrature oracle "
                 "(mhs.fullspace)")
    fr = local_frame(TRI)
    v1, v2, v3 = TRI
    L = fr.L
    t0 = time.perf_counter()

    # ------------------------------------------------------------- off-plane --
    obs = np.array([[0.60, -0.10, 0.60],
                    [2.00, 1.50, 3.00],
                    [-0.40, 0.00, -0.80]])
    eps = 0.3
    worst_off = 0.0
    for p in (0, 1, 2):
        ana = nodal_influence(obs, TRI, p, MU, NU, eps, want=ALL_KEYS,
                              far_field="analytic")
        quad = quadrature_influence(obs, TRI, p, MU, NU, eps, n_gauss=40,
                                    want=ALL_KEYS)
        rep.check_bool(f"p={p} off-plane: node coordinates agree",
                       np.array_equal(ana["nodes"], quad["nodes"]))
        worst_off = max(worst_off,
                        compare(rep, f"p={p} off-plane eps=0.3 vs 40x40",
                                ana, quad, 1e-9, keys=ALL_KEYS))

    # -------------------------------------------------------------- on-plane --
    obs_on = np.array([v1 + 0.3 * (v2 - v1) + 0.3 * (v3 - v1),   # inside
                       0.5 * (v2 + v3)])                          # edge midpoint
    z_on, _ = fr.to_plane(obs_on)
    rep.check("on-plane observers: max |z| / L",
              float(np.max(np.abs(z_on)) / L), 1e-12)
    eps_on = 0.15 * L
    worst_on = worst_on_force = 0.0
    for p in (0, 1, 2):
        ana = nodal_influence(obs_on, TRI, p, MU, NU, eps_on, want=ALL_KEYS,
                              far_field="analytic")
        quad = quadrature_influence(obs_on, TRI, p, MU, NU, eps_on,
                                    n_gauss=160, want=ALL_KEYS)
        worst_on = max(worst_on,
                       compare(rep, f"p={p} on-plane h=eps=0.15L vs 160x160",
                               ana, quad, TOL_ON_PLANE_SLIP))
        worst_on_force = max(
            worst_on_force,
            compare(rep, f"p={p} on-plane h=eps=0.15L vs 160x160",
                    ana, quad, TOL_ON_PLANE_FORCE, keys=KEYS_F))

    # ------------------------------------------------------ far-field producer
    direction = np.array([0.55, -0.35, 0.76])
    direction /= np.linalg.norm(direction)
    obs_far = (fr.centroid + 3.0 * L * direction)[None, :]
    D_far = float(np.linalg.norm(obs_far[0] - fr.centroid) / L)
    rep.check_bool("far observer at D = 3 L (oblique)", abs(D_far - 3.0) < 1e-12,
                   f"(D/L = {D_far:.6f}, |cos| to nhat = "
                   f"{abs(direction @ fr.nhat):.3f})")
    ana = nodal_influence(obs_far, TRI, 2, MU, NU, eps, want=ALL_KEYS,
                          far_field="analytic")
    quad = nodal_influence(obs_far, TRI, 2, MU, NU, eps, want=ALL_KEYS,
                           far_field="quadrature")
    worst_far = compare(rep, "p=2 D=3L far_field=quadrature vs analytic",
                        quad, ana, 1e-9, keys=ALL_KEYS)

    # ------------------------------------------------------- hybrid switching
    D_STAR = defaults.D_STAR
    rng = np.random.default_rng(7)
    dirs = rng.standard_normal((6, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    Ds = np.array([0.7, 3.0, 0.999 * D_STAR, 1.001 * D_STAR, 25.0,
                   1.5 * defaults.D_STAR_DISTANT]) * L
    obs_mix = fr.centroid + Ds[:, None] * dirs
    obs_mix = np.vstack([obs_mix, obs])                  # plus the 3 near points
    Dmix = np.linalg.norm(obs_mix - fr.centroid, axis=1)
    near = Dmix <= D_STAR * L
    far = ~near
    rep.check_bool("hybrid batch has both near and far observers",
                   near.sum() >= 2 and far.sum() >= 2,
                   f"({near.sum()} near, {far.sum()} far)")
    for p in (0, 2):
        hyb = nodal_influence(obs_mix, TRI, p, MU, NU, eps, want=ALL_KEYS,
                              far_field="hybrid")
        ana = nodal_influence(obs_mix, TRI, p, MU, NU, eps, want=ALL_KEYS,
                              far_field="analytic")
        quad = nodal_influence(obs_mix, TRI, p, MU, NU, eps, want=ALL_KEYS,
                               far_field="quadrature")
        quad_sub = nodal_influence(obs_mix[far], TRI, p, MU, NU, eps,
                                   want=ALL_KEYS, far_field="quadrature")
        bitwise(rep, f"p={p} hybrid == analytic within D_STAR L (mixed)",
                hyb, ana, near)
        bitwise(rep, f"p={p} hybrid == quadrature on the far rows (mixed)",
                {k: hyb[k][far] for k in ALL_KEYS}, quad_sub,
                np.ones(far.sum(), dtype=bool))
        for key in ALL_KEYS:   # full-batch producer: BLAS batch-size roundoff
            rep.check(f"p={p} hybrid vs full-batch quadrature beyond D_STAR L: "
                      f"{key}", relmax(hyb[key][far], quad[key][far]), 1e-13)
        hyb_n = nodal_influence(obs_mix[near], TRI, p, MU, NU, eps,
                                want=ALL_KEYS, far_field="hybrid")
        ana_n = nodal_influence(obs_mix[near], TRI, p, MU, NU, eps,
                                want=ALL_KEYS, far_field="analytic")
        bitwise(rep, f"p={p} hybrid == analytic (pure near batch)",
                hyb_n, ana_n, np.ones(near.sum(), dtype=bool))
        hyb_f = nodal_influence(obs_mix[far], TRI, p, MU, NU, eps,
                                want=ALL_KEYS, far_field="hybrid")
        bitwise(rep, f"p={p} hybrid == quadrature (pure far batch)",
                hyb_f, quad_sub, np.ones(far.sum(), dtype=bool))

    print(f"  worst: off-plane {worst_off:.3e}, on-plane slip {worst_on:.3e}, "
          f"on-plane force {worst_on_force:.3e}, far producer {worst_far:.3e}"
          f"   ({time.perf_counter() - t0:.1f} s)")
    return rep.finish()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
