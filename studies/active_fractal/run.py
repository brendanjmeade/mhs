#!/usr/bin/env python
"""Driver for the mechanically-active-fractal experiment.

    python studies/active_fractal/run.py assemble --config network
    python studies/active_fractal/run.py assemble --config patch --workers 12
    python studies/active_fractal/run.py geometry --config network

TWO CONFIGURATIONS, because the questions want different geometries and one
cannot serve both. The element budget is fixed by what a dense interaction
matrix can hold, so faults-per-network and elements-per-fault trade directly
against each other:

    network   a deep cascade, many faults, ~4 elements each. The arrangement's
              dimension is known in closed form, the measuring window fits
              inside the cascade's own range, and the active-set dimension is
              measurable. Slip is near-uniform on a fault, so an event is
              counted in whole faults.
    patch     a shallower cascade, fewer faults, ~20 elements each. Slip varies
              across a fault and tapers against its own locked edges, so
              moment is finely graded and a barrier can sit WITHIN a fault.
              The dimension window is thinner, which is why the other config
              exists.

Agreement between the two is itself a check: a result that holds at both
discretizations is not an artefact of either.

WHY THERE IS A ``__main__`` GUARD AND NO LAMBDAS. ``mhs`` has no numba kernels;
it is numpy, and it parallelises over SPAWNED PROCESSES
(``mhs.parallel.by_source``), because threading its per-source loop measures
1.28x at two threads and then gets worse on the GIL. Spawned workers re-import
this module, so the guard is mandatory and every callable handed to a worker
has to be picklable.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import friction as Fr                                        # noqa: E402
import kernel as K                                          # noqa: E402
import measure as M                                         # noqa: E402
import network as N                                         # noqa: E402

#: Material, in the project's units. nu = 0.25 at mu = lam, which is the value
#: every mhs gate runs at -- and the free-surface residual is forty times worse
#: as nu approaches 1/2, so a network experiment has no business near there.
MU, LAM = 30.0, 30.0

#: A 40 km cube: crustal thickness, and ISOTROPIC because a cascade measured
#: with cubic boxes needs an isotropic domain or its dimension stops being
#: 3 log(keep)/log(base). ``build_network`` refuses anything else.
BOX_XY, Z_TOP = 40.0, -2.0

CONFIGS = {
    # --- fault networks: coherence IMPOSED, one rectangle per fault ---------
    # ~4096 faults at ~4 elements each. D_geom = 1.893 exactly.
    "network": dict(kind="faults", cascade=(2, 3, 4), h=0.2, a=1.5),
    # ~512 faults at ~20 elements each. D_geom = 1.500 exactly.
    "patch": dict(kind="faults", cascade=(2, 4, 3), h=0.1, a=1.5),

    # --- dense clouds: the element IS the fracture, coherence EMERGENT ------
    # The control: homogeneous background, so any localization is mechanical,
    # and the no-interaction null is spatially unstructured by construction.
    # n_elements is a REQUEST: the face-clearance floor prunes ~30 %, so this
    # asks for 30000 to keep ~20000. s/h = 0.7 puts B near 0.55 at that size,
    # subcritical with a margin and in the slightly-subcritical regime real
    # seismicity is inferred to occupy.
    "cloud_uniform": dict(kind="cloud", centers="uniform", n_elements=30000,
                          h=0.5, spacing_over_h=0.7),
    # The test: the same elements on a cascade, asking whether mechanics
    # amplifies, ignores or overrides pre-existing geometric clustering.
    "cloud_fractal": dict(kind="cloud", centers="cascade", cascade=(3, 4, 3),
                          h=0.5, spacing_over_h=0.7),
}


def build(config: str, seed: int = 1, **over):
    """The geometry for a named configuration, fault network or cloud."""
    if config not in CONFIGS:
        raise SystemExit(f"no config {config!r}; have {', '.join(CONFIGS)}")
    kw = dict(CONFIGS[config])
    kind = kw.pop("kind")
    kw.update(over)
    if kind == "cloud":
        return N.build_cloud(seed=seed, z_top=Z_TOP, **kw)
    return N.build_network(seed=seed, n_elements_target=10 ** 9,
                           centers="cascade", box_xy=BOX_XY, z_top=Z_TOP,
                           z_bottom=Z_TOP - BOX_XY, aspect=2.0, **kw)


def window_of(net) -> dict:
    """The dimension window for either geometry. One statement, two callers.

    The lower bound is 2 x the FINEST FEATURE, and the feature differs by
    experiment: for a fault network it is the cascade cell that the faults sit
    in, for a cloud it is the element edge itself, because the element IS the
    fracture and there is nothing finer. Below that bound a box resolves a
    single planar patch and the slope goes to 2 regardless of how the patches
    are arranged, which is a statement about the primitive rather than the
    network.
    """
    m = net.meta
    feature = net.h if m.get("kind") == "cloud" else m["cascade_cell_km"]
    return M.window_for(feature, m["box_xy_km"])


def describe(net) -> None:
    m = net.meta
    if m.get("kind") == "cloud":
        nn = m["nearest_neighbour_km"]
        print(f"  CLOUD: {net.n_elements} single-triangle elements, "
              f"h {net.h} km, eps {net.eps:.4f} km")
        print(f"  centers {m['centers']}, spacing {m['spacing_km']:.3f} km "
              f"= {m['spacing_over_h']:.2f} h, box {m['box_xy_km']:.2f} km, "
              f"P32 {m['P32_per_km']:.3f} /km")
        if nn:
            print(f"  nearest neighbour mean {nn['mean'] / net.h:.2f} h, "
                  f"min {nn['min'] / net.h:.2f} h "
                  f"(Poisson puts the mean at 0.554 x spacing)")
        print(f"  fabric S {[round(v, 4) for v in m['fabric_S']]}, "
              f"|S1-1/3| {m['fabric_anisotropy']:.4f} "
              f"(isotropic by construction), pruned "
              f"{m['pruned_fraction']:.2%}")
    else:
        print(f"  {net.n_faults} faults, {net.n_elements} elements "
              f"({net.n_elements / net.n_faults:.1f} per fault)")
        print(f"  cascade {m['cascade']}  cell {m['cascade_cell_km']:.4f} km  "
              f"D_geom known {m['d_geom_known']:.4f}")
        print(f"  L {net.length.min():.3f}-{net.length.max():.3f} km, "
              f"h {net.h} km, eps {net.eps:.4f} km, "
              f"pruned {m['pruned_fraction']:.1%}")
    print(f"  matrix would be {K.required_bytes(net.n_elements) / 1024 ** 3:.2f}"
          f" GiB")
    try:
        win = window_of(net)
        print(f"  dimension window r = {win['r_lo']:.3f}-{win['r_hi']:.3f} km "
              f"= {math.log10(win['r_hi'] / win['r_lo']):.2f} decades")
    except ValueError as e:
        print(f"  dimension window UNUSABLE: {e}")


def cmd_calibrate(a) -> int:
    """Sweep the cloud spacing and report the branching ratio.

    THIS IS WHAT FIXES THE PRODUCTION SPACING. The coupling goes as the cube of
    the spacing, so a factor of two in `spacing_over_h` is a factor of eight in
    B, and B is the quantity that decides whether cascades terminate at all.
    Target `B ~ 0.7-0.9`: below 1 so they terminate, near enough to 1 that
    clustering is possible. Small clouds, so this runs in minutes.
    """
    if a.face_clearance is not None:
        N.MIN_FACE_CLEARANCE_OVER_EPS = a.face_clearance
    print(f"[calibrate] {a.elements} elements, h {a.h} km, "
          f"centers {a.centers}, face clearance "
          f"{N.MIN_FACE_CLEARANCE_OVER_EPS} eps")
    print(f"{'s/h':>5} {'box':>7} {'d_nn/h':>7} {'pruned':>7} {'B mean':>8} "
          f"{'B p90':>8} {'B max':>8} {'min eig sym(-P)':>16}")
    for soh in a.spacings:
        net = N.build_cloud(seed=a.seed, n_elements=a.elements, h=a.h,
                            spacing_over_h=soh, centers=a.centers,
                            z_top=Z_TOP)
        mat, _ = K.assemble(net, mu=MU, lam=LAM, workers=a.workers,
                            cache_dir=a.cache, verbose=False)
        fric = Fr.build_friction(net)
        P = K.project(mat, fric.slip_dir, fric.slip_dir)
        B = K.branching_ratio(P)
        ev = np.linalg.eigvalsh(-0.5 * (P + P.T))
        nn = net.meta["nearest_neighbour_km"]
        dnn = nn["mean"] / net.h if nn else float("nan")
        print(f"{soh:5.2f} {net.meta['box_xy_km']:7.2f} {dnn:7.2f} "
              f"{net.meta['pruned_fraction']:7.2%} {B.mean():8.4f} "
              f"{np.percentile(B, 90):8.4f} {B.max():8.4f} "
              f"{ev.min():16.4e}  n={net.n_elements} "
              f"neg={int((ev < 0).sum())}")
        del mat, P
    return 0


def cmd_geometry(a) -> int:
    """Build a network and measure its geometry. No matrix, so it is cheap."""
    net = build(a.config, a.seed)
    print(f"[{a.config}] seed {a.seed}")
    describe(net)
    try:
        win = window_of(net)
    except ValueError as e:
        print(f"  no usable dimension window: {e}")
        return 0
    fit = M.dimension(net.tris, n_radii=a.radii, **win)
    known = net.meta["d_geom_known"]
    print(f"  D_raw {fit.dimension:.3f}  D_calibrated "
          f"{M.calibrated(fit):.3f}  known {known:.4f}  "
          f"err {M.calibrated(fit) - known:+.3f}")
    print(f"  local slopes: " + " ".join(f"{v:.2f}" for v in fit.local[1:]))
    return 0


def cmd_assemble(a) -> int:
    """Assemble and cache the interaction matrix."""
    net = build(a.config, a.seed)
    print(f"[{a.config}] seed {a.seed}")
    describe(net)
    mat, info = K.assemble(net, mu=MU, lam=LAM, workers=a.workers,
                           cache_dir=a.cache)
    n = net.n_elements
    idx = np.arange(n)
    print("  self-interaction diagonal (slip must relieve its own shear):")
    for b, nm in enumerate(K.SOURCE):
        d = mat[idx, b, K.RECEIVER.index(nm), idx]
        print(f"    {nm:6s} {d.min():.4e} .. {d.max():.4e}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("geometry", cmd_geometry), ("assemble", cmd_assemble)):
        p = sub.add_parser(name)
        p.add_argument("--config", default="network",
                       choices=sorted(CONFIGS))
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--radii", type=int, default=10)
        p.add_argument("--workers", type=int, default=None)
        p.add_argument("--cache", default=None)
        p.set_defaults(fn=fn)
    c = sub.add_parser("calibrate")
    c.add_argument("--elements", type=int, default=2000)
    c.add_argument("--h", type=float, default=0.5)
    c.add_argument("--centers", default="uniform",
                   choices=("uniform", "cascade"))
    c.add_argument("--spacings", type=float, nargs="+",
                   default=[1.0, 1.5, 2.0, 2.5, 3.0])
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--workers", type=int, default=None)
    c.add_argument("--cache", default=None)
    c.add_argument("--face-clearance", type=float, default=None,
                   help="override the face-clearance floor, in units of eps")
    c.set_defaults(fn=cmd_calibrate)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
