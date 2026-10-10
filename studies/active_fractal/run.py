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
    # ~4096 faults at ~4 elements each. D_geom = 1.893 exactly.
    "network": dict(cascade=(2, 3, 4), h=0.2, a=1.5),
    # ~512 faults at ~20 elements each. D_geom = 1.500 exactly.
    "patch": dict(cascade=(2, 4, 3), h=0.1, a=1.5),
}


def build(config: str, seed: int = 1):
    """The network for a named configuration."""
    if config not in CONFIGS:
        raise SystemExit(f"no config {config!r}; have {', '.join(CONFIGS)}")
    return N.build_network(seed=seed, n_elements_target=10 ** 9,
                           centers="cascade", box_xy=BOX_XY, z_top=Z_TOP,
                           z_bottom=Z_TOP - BOX_XY, aspect=2.0,
                           **CONFIGS[config])


def describe(net) -> None:
    m = net.meta
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
        win = M.window_for(m["cascade_cell_km"], BOX_XY)
        print(f"  dimension window r = {win['r_lo']:.3f}-{win['r_hi']:.3f} km "
              f"= {math.log10(win['r_hi'] / win['r_lo']):.2f} decades")
    except ValueError as e:
        print(f"  dimension window UNUSABLE: {e}")


def cmd_geometry(a) -> int:
    """Build a network and measure its geometry. No matrix, so it is cheap."""
    net = build(a.config, a.seed)
    print(f"[{a.config}] seed {a.seed}")
    describe(net)
    cell = net.meta["cascade_cell_km"]
    win = M.window_for(cell, BOX_XY)
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
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
