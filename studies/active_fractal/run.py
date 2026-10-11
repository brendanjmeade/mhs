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

import cascade as C                                         # noqa: E402
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

    # --- the geologically recognizable one: power-law lengths, big faults ---
    # Faults 2-40 km in a 100 km box, so the largest spans 38 % of the domain
    # against 1.6 % for the cascade configs above. 538 faults at 38.6 elements
    # each: enough faults for statistics AND enough elements per fault for a
    # rupture to be partial. This is the configuration the brief described; it
    # got displaced by the cascade while chasing a known D_geom, and a known
    # D_geom is worth nothing if the picture is dust.
    "faults_pl": dict(kind="faults_pl", centers="uniform", n_elements_target=20000,
                      l_min=2.0, l_max=40.0, h=1.0, box_xy=100.0,
                      z_top=-2.0, z_bottom=-42.0, a=1.5),

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
    if kind == "faults_pl":
        return N.build_network(seed=seed, aspect=2.0, **kw)
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
    if m.get("kind") == "cloud":
        feature = net.h
    elif m.get("cascade_cell_km") is not None:
        feature = m["cascade_cell_km"]
    else:
        # A power-law network has no cascade cell; the finest feature is then
        # the LARGEST fault, since below that a box resolves one planar patch.
        feature = float(net.length.max())
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
        if m.get("cascade_cell_km") is not None:
            print(f"  cascade {m['cascade']}  cell "
                  f"{m['cascade_cell_km']:.4f} km  "
                  f"D_geom known {m['d_geom_known']:.4f}")
        else:
            print(f"  centers {m['centers']}, box {m['box_xy_km']:.0f} km, "
                  f"a {m['a_nominal']} -> {m['a_realized']:.3f} realized, "
                  f"P32 {m['total_area_km2'] / m['box_xy_km'] ** 3:.4f} /km")
            print(f"  largest fault is {net.length.max() / m['box_xy_km']:.2f}"
                  f" of the domain (the cascade configs: 0.016)")
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


def cmd_localize(a) -> int:
    """Load a cloud quasi-statically and ask whether slip ALIGNS and LOCALIZES.

    THE DECOMPOSITION, which is sharper than it first looks. Global
    moment-tensor coherence depends only on the orientations and slip
    directions of the active elements, NOT on where they are -- so a
    position-randomized null is identical to the active set for it, and global
    coherence tests exactly one thing: did interaction select more mutually
    aligned ORIENTATIONS than the loading does on its own? The spatial question
    is separate and is answered by the radial profile C(r) against a
    position-shuffled active set, and by the participation ratio and the
    clusters against matched-cardinality nulls.

    So:
      global C      active vs no-interaction null   -> orientation selection
      C(r)          active vs position-shuffled     -> spatial organization
      PR, clusters  active vs both nulls            -> spatial localization
    """
    net = build(a.config, a.seed)
    print(f"[{a.config}] seed {a.seed}")
    describe(net)
    mat, info = K.assemble(net, mu=MU, lam=LAM, workers=a.workers,
                           cache_dir=a.cache)
    fric = Fr.build_friction(net, drop_ratio=a.drop_ratio)
    P = K.project(mat, fric.slip_dir, fric.slip_dir)
    B = K.branching_ratio(P)
    ev_min = float(np.linalg.eigvalsh(-0.5 * (P + P.T)).min())
    print(f"  branching ratio B: mean {B.mean():.4f} p90 "
          f"{np.percentile(B, 90):.4f} max {B.max():.4f}")
    print(f"  min eig sym(-P) {ev_min:.4g} "
          f"({'positive definite' if ev_min > 0 else 'INDEFINITE -- '
             'uniqueness not guaranteed, relying on solver-independence'})")

    lam0 = float(np.nanmin(fric.first_failure_lambda()))
    st, ev = C.run_history(net, fric, P, shear_modulus=MU,
                           lam_max=a.lam_factor * lam0)
    sz = np.array([e.n_slipped for e in ev]) if ev else np.array([0])
    act = st.s > 0.0
    n_act = int(act.sum())
    print(f"  {len(ev)} events to lam {st.lam:.5f} "
          f"({a.lam_factor}x first failure), all converged "
          f"{all(e.converged for e in ev)}")
    print(f"  active {n_act}/{net.n_elements} = {n_act / net.n_elements:.1%}; "
          f"event size mean {sz.mean():.2f} max {sz.max()}")
    if n_act < 50:
        print("  too few active elements to say anything; stopping.")
        return 0

    # AVALANCHE SIZES. An event here is one relaxation to equilibrium after a
    # single element is pushed to failure, so its size is the number of
    # elements that ended up slipping -- a cascade, not a dynamic rupture.
    # Reported as a survival function rather than a fitted exponent: a
    # power-law fit needs a defended s_min and these sizes are bounded above by
    # the elements per fault, so quoting one slope would overstate the claim.
    q = np.percentile(sz, [50, 90, 99, 99.9])
    print(f"  event sizes: median {q[0]:.0f} p90 {q[1]:.0f} p99 {q[2]:.0f} "
          f"p99.9 {q[3]:.0f}; {100 * (sz >= 2).mean():.1f}% multi-element")
    thr = [1, 2, 5, 10, 50, 100, 500]
    frac = "  ".join(f">={t}: {100 * (sz >= t).mean():.2f}%"
                     for t in thr if t <= sz.max())
    print(f"  P(size >= s):  {frac}")

    cent = net.tris.mean(axis=1)
    from mhs import tdcs
    nrm = tdcs.unit_normals(net.tris)

    def stats(mask, label, weight=None):
        """TWO participation ratios, because they answer different questions.

        ``PR`` weights every member equally and asks WHERE THE ACTIVE ELEMENTS
        ARE. It is the only one comparable across rows, because a null has no
        slip solution and weighting a null by moment is undefined.

        ``PRm`` weights by ``||M_i||`` and asks where the MOMENT is, which is a
        statement about slip heterogeneity WITHIN a set rather than about the
        set's spatial extent.

        ONE COLUMN FOR BOTH WAS AN ERROR WORTH NAMING. The active row was
        moment-weighted while every null row was unit-weighted, so the printed
        gap mixed "interaction concentrates the active set" together with "slip
        is heterogeneous among active elements" and credited all of it to the
        first. Unit weight on the active set is what the nulls can be compared
        against; the moment column is read against the active set's own unit
        column, not across rows.
        """
        n_m = int(mask.sum())
        Mt = M.moment_tensors(net.tris[mask], st.s[mask] if weight is None
                              else weight[mask], fric.slip_dir[mask], MU)
        coh = M.coherence(Mt)
        unit = M.moment_tensors(net.tris[mask], np.ones(n_m),
                                fric.slip_dir[mask], MU)
        pr = M.participation_ratio(cent[mask],
                                   np.linalg.norm(unit, axis=(1, 2)),
                                   a.cell, available=cent)
        pr_m = M.participation_ratio(cent[mask],
                                     np.linalg.norm(Mt, axis=(1, 2)),
                                     a.cell, available=cent)
        S = N.fabric(nrm[mask])
        lab = M.connected_clusters(cent[mask], a.reach * net.h)
        sizes = np.bincount(lab)
        big = lab == int(np.argmax(sizes))
        pla, lin = M.shape_indices(cent[mask][big])
        loc = np.zeros((n_m, 3))
        loc[:, :2] = fric.slip_dir[mask]
        disp = M.slip_dispersion(tdcs.to_cartesian(net.tris[mask], loc))
        prom = M.promoting_share(P, mask)
        print(f"    {label:22s} prom {prom:.4f}  C {coh:.4f}  PR {pr:.4f}  "
              f"PRm {pr_m:.4f}  "
              f"S1 {S[0]:.4f}  clusters {sizes.size:5d}  "
              f"largest {sizes.max():5d}  planarity {pla:.3f}  "
              f"linearity {lin:.3f}  slip disp {disp:.3f}")
        return coh, pr, S[0], pr_m

    print("\n  ACTIVE SET against its nulls "
          f"(cell {a.cell} km, cluster reach {a.reach} h):")
    c_act, pr_act, s_act, prm_act = stats(act, "active")

    # Null 2: the no-interaction set, same cardinality, chosen by the load
    # alone. This is the one the headline claim has to beat.
    score = -fric.first_failure_lambda()
    order = np.argsort(-score)[:n_act]
    m2 = np.zeros(net.n_elements, bool)
    m2[order] = True
    c_n2, pr_n2, s_n2, _ = stats(m2, "null: no interaction",
                                 weight=np.ones(net.n_elements))

    # Null 1: a random subset of the same size.
    rng = np.random.default_rng(a.seed)
    c1 = []
    for k in range(3):
        m1 = np.zeros(net.n_elements, bool)
        m1[rng.choice(net.n_elements, n_act, replace=False)] = True
        c1.append(stats(m1, f"null: random {k + 1}",
                        weight=np.ones(net.n_elements)))
    c_n1 = float(np.mean([v[0] for v in c1]))
    pr_n1 = float(np.mean([v[1] for v in c1]))

    print(f"\n  THE COMPARISONS:")
    print(f"    coherence   active {c_act:.4f}  vs no-interaction {c_n2:.4f}"
          f"  -> {c_act - c_n2:+.4f}")
    print(f"    (orientation selection by interaction, beyond the load)")
    print(f"    PR   unit   active {pr_act:.4f}  vs no-interaction {pr_n2:.4f}"
          f"  vs random {pr_n1:.4f}")
    print(f"    (WHICH ELEMENTS are active, matched cardinality; below the "
          f"nulls means spatially MORE concentrated)")
    print(f"    PR   moment active {prm_act:.4f}  vs its own unit "
          f"{pr_act:.4f}  -> {prm_act - pr_act:+.4f}")
    print(f"    (WHERE THE MOMENT IS within the active set -- slip "
          f"heterogeneity, NOT comparable to the nulls)")

    # HOW SIMILAR ARE THE TWO SETS AT ALL? Printed before any statistic that
    # compares them, because it bounds every such comparison: two sets that
    # overlap by 99 % cannot differ much in anything measured on membership,
    # and a reader shown only the differences would not know that.
    inter = int((act & m2).sum())
    print(f"\n  active vs no-interaction null: {inter} shared of {n_act} "
          f"({inter / n_act:.1%}); symmetric difference {2 * (n_act - inter)}")

    # C(r): spatial organization of coherence.
    #
    # THREE ROWS, BECAUSE THE POSITION SHUFFLE ALONE IS NOT A CONTROL. Shuffling
    # positions among active elements destroys fault membership, so its excess
    # reports "a fault is a coherent object" -- which is true by construction --
    # on top of anything mechanical. The no-interaction null holds the fault
    # structure fixed and changes only which elements are active, so it is the
    # row that isolates interaction. Both are computed at UNIT slip, for the
    # same reason the participation ratio is: weighting one row by real slip and
    # the other by ones measures the weighting, not the mechanics.
    radii = np.geomspace(2.0 * net.h, net.meta["box_xy_km"] / 3.0, 6)

    def cr_unit(mask):
        Mu = M.moment_tensors(net.tris[mask], np.ones(int(mask.sum())),
                              fric.slip_dir[mask], MU)
        return M.coherence_vs_radius(cent[mask], Mu, radii, seed=a.seed)

    Mt = M.moment_tensors(net.tris[act], st.s[act], fric.slip_dir[act], MU)
    cr_mom = M.coherence_vs_radius(cent[act], Mt, radii, seed=a.seed)
    perm = rng.permutation(n_act)
    cr_shuf = M.coherence_vs_radius(cent[act][perm], Mt, radii, seed=a.seed)
    cr_act = cr_unit(act)
    cr_null = cr_unit(m2)

    print(f"\n  C(r): is coherence ORGANIZED IN SPACE?")
    print(f"    {'r km':>7} {'act(mom)':>9} {'shuffled':>9} | "
          f"{'act(unit)':>9} {'null(unit)':>10} {'excess':>9}")
    for r, m_, s_, a_, n_ in zip(radii, cr_mom, cr_shuf, cr_act, cr_null):
        print(f"    {r:7.2f} {m_:9.4f} {s_:9.4f} | "
              f"{a_:9.4f} {n_:10.4f} {a_ - n_:+9.4f}")
    print(f"    (the LAST column is the mechanical one: same fault geometry, "
          f"same weighting, only the active set differs)")
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
    lz = sub.add_parser("localize")
    lz.add_argument("--config", default="cloud_uniform", choices=sorted(CONFIGS))
    lz.add_argument("--seed", type=int, default=1)
    lz.add_argument("--workers", type=int, default=12)
    lz.add_argument("--cache", default=None)
    lz.add_argument("--drop-ratio", type=float, default=0.9)
    lz.add_argument("--lam-factor", type=float, default=3.0)
    lz.add_argument("--cell", type=float, default=2.0,
                    help="km, the cell for the participation ratio")
    lz.add_argument("--reach", type=float, default=1.5,
                    help="cluster adjacency radius, in units of h")
    lz.set_defaults(fn=cmd_localize)
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
