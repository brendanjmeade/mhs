#!/usr/bin/env python
"""The study's own gates. Run by path, before any result is believed.

    python studies/active_fractal/selfcheck.py            # all clauses
    python studies/active_fractal/selfcheck.py -k estimator

WHY HERE AND NOT IN tests/gates/. The package's suites cover ``src/mhs/`` and
their per-suite counts are PINNED in ``tests/run_all.py`` -- a green empty
suite being the one failure a runner must not be able to report. This is study
code run by path, so adding to those counts would make the package's gate
inventory a function of which studies exist. The convention is kept though:
``main() -> bool``, ``PASS:``/``FAIL:`` at column 0 as the last such line, and
the exit code carries it.

The negative self-interaction diagonal is NOT re-gated here as a physics claim;
``tests/gates/mhs/verify_image_kernel`` clause [h] already owns it. What this
file checks is that the NETWORK this study builds does not violate it -- a
different question, because a network has mixed orientations and a separation
floor, and clause [h] runs on one tidy dipping fault.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import pathlib
import sys
import tempfile

import numpy as np

import mhs

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cascade as Csc                                        # noqa: E402
import friction as Fr                                        # noqa: E402
import kernel as Kmod                                        # noqa: E402
import measure as M                                          # noqa: E402
import network as N                                          # noqa: E402
import spectrum as Sp                                        # noqa: E402
import growth as Gr                                          # noqa: E402
import static as St                                          # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []

# One window for every dimension in this file, so differences are comparable.
# It is measure.window_for's rule at a unit finest feature in an 81 km domain,
# written out so the calibration constant and these clauses cannot drift apart.
CAL_WINDOW = dict(**M.window_for(1.0, 81.0), n_radii=10)

#: Tolerance on a DIFFERENCE of dimensions between two sets measured in one
#: window. 0.08 against a measured worst of 0.059. An earlier version used 0.02
#: on the strength of a single window where the error happened to be 0.003;
#: across three windows and two set pairs the real spread is -0.012 to -0.059,
#: so 0.003 was luck and 0.02 would have been a gate built on it.
#:
#: This is the number that sets what the experiment can conclude. The
#: cardinality artefact it has to beat is -0.28 at a 5 % subset, about six
#: times this, so a real localization signal of ~0.1 or more is resolvable and
#: anything under ~0.05 is not.
TOL_DIM_DIFF = 0.08

#: The null's OWN reproducibility: independent random subsets of one set at
#: matched size. Measured 0.0007 (30 %), 0.0036 (10 %), 0.0090 (5 %), so the
#: null is far more repeatable than the systematic difference error above --
#: which is why the null is the right reference and the absolute is not.
TOL_NULL_SPREAD = 0.02

#: The subset artefact at 5 %, which must be LARGE for the nulls to be
#: necessary. Gated as a floor rather than a ceiling: if a 5 % subset of a flat
#: sheet ever stopped reading ~0.36 low, the null-model argument would need
#: rewriting rather than quietly passing.
MIN_SUBSET_ARTEFACT_5PCT = 0.20

#: Placement distortion of the length exponent, in units of the MLE's OWN
#: standard error at the realized fault count. Gated this way rather than as an
#: absolute offset: the first version used 0.08 and failed on a nominal 1.5
#: that realized 1.618 at 152 faults, where the MLE's standard error is
#: a/sqrt(n) = 0.12. That gate was measuring the random number generator. The
#: control is an undistorted draw of the SAME number of faults, so what is left
#: is placement alone.
MAX_EXPONENT_BIAS_SIGMA = 3.0

#: Jaccard index between active sets found under different update rules. Gated
#: at 1.0 exactly, because that is what was measured: sequential, two random
#: orderings and synchronous Jacobi all produced the IDENTICAL set of elements
#: that ever slipped (symmetric difference 0). The event statistics are not that
#: robust -- Jacobi shifts event counts by ~7 % -- so this clause pins the
#: quantity the headline claim rests on and leaves the other one carrying its
#: stated uncertainty.
MIN_ACTIVE_SET_JACCARD = 1.0

#: Coherence of a set of identically-oriented, identically-slipping elements.
#: Gated at float precision, because it is exactly 1 by construction: measured
#: 1.000000000000.
TOL_COHERENCE_EXACT = 1.0e-9

#: Coherence of N randomly oriented elements, in units of the 1/sqrt(N) that
#: theory predicts. A factor of 2 either way: measured ratios 0.95, 0.66, 1.16
#: and 0.94 at N = 10, 100, 1000, 10000 over eight realizations each, so the
#: scaling holds but the scatter is wide at small N.
MAX_COHERENCE_SCALING_FACTOR = 2.0

#: Participation ratio of a random set, against the POISSON prediction
#: lam/(lam+1) rather than against 1. This is the clause that records why: an
#: unlocalized set does NOT give 1, so comparing a measured PR against 1 would
#: read ordinary occupancy scatter as localization. Measured 0.811 against a
#: predicted 0.800 at four elements per cell.
TOL_PR_POISSON = 0.05

#: Minimum eigenvalue of sym(-P) for the projected operator. Gated as a FLOOR:
#: the quasi-static complementarity problem is uniquely solvable on every
#: active set when -P is positive definite, so this is the well-posedness
#: condition and not a quality metric. Measured 14.88 on a 202-element network,
#: against a matrix whose entries are tens of GPa per km of slip.
MIN_EIG_SYM_NEG_P = 1.0

#: An ABSOLUTE calibrated dimension, on a network whose arrangement dimension
#: is known. 0.10 against measured -0.017 (0.83-decade window) and -0.082
#: (0.73 decades). Deliberately looser than the difference tolerance: an
#: absolute is the weaker measurement and the gate should say so.
TOL_D_ABSOLUTE = 0.10


def check(name: str, value: float, tol: float, *, below: bool = True) -> None:
    ok = bool(value < tol) if below else bool(value > tol)
    CHECKS.append((name, ok, f"{value:.4g} {'<' if below else '>'} {tol:g}"))
    print(f"  [{'ok' if ok else 'XX'}] {name:52s} "
          f"{value:.4g} {'<' if below else '>'} {tol:g}")


def check_true(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'ok' if ok else 'XX'}] {name:52s} {detail}")


# ------------------------------------------------------------- [estimator] ---
def c_estimator() -> None:
    """The estimator recovers a known dimension DIFFERENCE, and R^2 does not
    detect the absolute bias."""
    print("\n[estimator] known dimensions, and what survives the bias")
    want_dust = 3.0 * math.log(2) / math.log(3)
    plane = M.plane_tris(81.0, 1.0)
    dust = M.cantor_dust_tris(4, 81.0, 2, 3, 0.45)
    dp = M.dimension(plane, **CAL_WINDOW)
    dd = M.dimension(dust, **CAL_WINDOW)
    # The experiment's real comparison is between two STRUCTURALLY SIMILAR
    # sets -- two subsets of one network -- not between a sheet and a dust. So
    # the difference is gated on two dusts of the same cell size, which is the
    # nearer analogue; the sheet-vs-dust pair is kept only to show the
    # absolutes are biased.
    dust24 = M.cantor_dust_tris(3, 64.0, 2, 4, 0.45)   # cell 1.0 as well
    want24 = 3.0 * math.log(2) / math.log(4)
    sim = dict(**M.window_for(1.0, 64.0), n_radii=10)
    s23 = M.dimension(M.cantor_dust_tris(4, 81.0, 2, 3, 0.45), **sim)
    s24 = M.dimension(dust24, **sim)
    print(f"       two dusts, matched window: want "
          f"{want_dust - want24:+.3f}  got "
          f"{s23.dimension - s24.dimension:+.3f}")
    check("a DIFFERENCE between SIMILAR sets",
          abs((s23.dimension - s24.dimension) - (want_dust - want24)),
          TOL_DIM_DIFF)
    print(f"       flat sheet  want 2.000     got {dp.dimension:.3f}  "
          f"R2 {dp.r2:.5f}")
    print(f"       Cantor dust want {want_dust:.3f}     got {dd.dimension:.3f}  "
          f"R2 {dd.r2:.5f}")
    # The absolutes are biased, and that is the point of gating the difference.
    check_true("absolutes ARE biased low (so do not quote them)",
               dp.dimension < 2.0 - 0.05 and dd.dimension < want_dust - 0.05,
               f"sheet {dp.dimension - 2.0:+.3f}, "
               f"dust {dd.dimension - want_dust:+.3f}")
    check_true("R2 fails to detect that bias", dp.r2 > 0.999 and dd.r2 > 0.999,
               f"R2 {min(dp.r2, dd.r2):.5f} with error up to "
               f"{max(abs(dp.dimension - 2.0), abs(dd.dimension - want_dust)):.3f}")
    # The calibration must remove the bias it was fitted to remove.
    check("calibrated flat sheet", abs(M.calibrated(dp) - 2.0), 0.05)
    check("calibrated Cantor dust", abs(M.calibrated(dd) - want_dust), 0.05)
    # A sliver, as a third construction with a different answer entirely.
    ds = M.dimension(M.line_tris(81.0, 1.0, 0.02), **CAL_WINDOW)
    check("sliver reads 1-dimensional", abs(ds.dimension - 1.0), 0.10)


# --------------------------------------------------------------- [subset] ----
def c_subset() -> None:
    """Cardinality alone moves the dimension, which is why nulls exist."""
    print("\n[subset] the artefact the nulls are built to exclude")
    full = M.plane_tris(81.0, 1.0)
    n = full.shape[0]
    d_full = M.dimension(full, **CAL_WINDOW).dimension
    rng = np.random.default_rng(0)
    shifts = {}
    for frac in (0.5, 0.05):
        k = max(4, int(frac * n))
        vals = [M.dimension(full[rng.choice(n, k, replace=False)],
                            **CAL_WINDOW).dimension for _ in range(3)]
        shifts[frac] = float(np.mean(vals)) - d_full
        print(f"       random {frac:5.0%} of a FLAT sheet -> "
              f"D shift {shifts[frac]:+.3f}")
    check_true("a 50 % subset is nearly unshifted",
               abs(shifts[0.5]) < 0.02, f"{shifts[0.5]:+.3f}")
    # The null's reproducibility, which is what makes it usable as a reference.
    rng2 = np.random.default_rng(3)
    k = int(0.05 * n)
    rep = [M.dimension(full[rng2.choice(n, k, replace=False)],
                       **CAL_WINDOW).dimension for _ in range(6)]
    check("the null is reproducible at matched size",
          float(np.std(rep)), TOL_NULL_SPREAD)
    check(f"a 5 % subset loses >= {MIN_SUBSET_ARTEFACT_5PCT} of a dimension",
          -shifts[0.05], MIN_SUBSET_ARTEFACT_5PCT, below=False)


# -------------------------------------------------------------- [network] ----
def c_network() -> None:
    """The generator realizes the exponent it was asked for, and stays legal."""
    print("\n[network] the independent variable is the REALIZED exponent")
    print("       placement is compared against an UNDISTORTED draw of the")
    print("       same fault count, so sampling noise is not read as bias")
    worst_sigma = 0.0
    for centers in ("levy", "uniform"):
        for a in (1.0, 1.5, 2.5):
            got, ctrl, nf = [], [], 0
            for seed in (3, 4, 5):
                net = N.build_network(seed=seed, n_elements_target=4000, a=a,
                                      centers=centers)
                got.append(net.meta["a_realized"])
                nf += net.n_faults
                rng = np.random.default_rng(1000 + seed)
                ctrl += [N.pareto_mle(
                    N.pareto_lengths(rng, net.n_faults, a, 2.0, 40.0),
                    2.0, 40.0) for _ in range(12)]
            # the control's spread IS the sampling error at this fault count
            sd = float(np.std(ctrl)) or 1e-9
            bias = float(np.mean(got)) - float(np.mean(ctrl))
            sigma = abs(bias) / sd
            worst_sigma = max(worst_sigma, sigma)
            print(f"       {centers:7s} a {a:.1f} -> placed "
                  f"{np.mean(got):.3f}  control {np.mean(ctrl):.3f} "
                  f"+- {sd:.3f}  bias {bias:+.3f} = {sigma:.1f} sigma "
                  f"({nf // 3} faults)")
    check("placement does not bias the exponent",
          worst_sigma, MAX_EXPONENT_BIAS_SIGMA)

    net = N.build_network(seed=3, n_elements_target=4000,
                          centers="uniform")
    check_true("every vertex is below the free surface",
               bool((net.tris[:, :, 2] < 0.0).all()),
               f"max z = {net.tris[:, :, 2].max():.3f} km")
    again = N.build_network(seed=3, n_elements_target=4000,
                            centers="uniform")
    check_true("the generator is deterministic for a seed",
               bool(np.array_equal(net.tris, again.tris)))
    other = N.build_network(seed=4, n_elements_target=4000,
                            centers="uniform")
    check_true("and a different seed gives a different network",
               not bool(np.array_equal(net.tris, other.tris)))
    # The separation floor is what keeps the interaction matrix conditioned.
    cent = net.tris.mean(axis=1)
    d2 = ((cent[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    check_true("no two elements closer than the floor",
               bool(np.sqrt(d2.min()) >= net.meta["min_sep_km"] - 1e-9),
               f"min {math.sqrt(d2.min()):.4f} km vs floor "
               f"{net.meta['min_sep_km']:.4f} km")


# --------------------------------------------------------------- [budget] ----
def c_budget() -> None:
    """The closed-form element budget matches what the generator spends."""
    print("\n[budget] the arithmetic that rules out largest-first placement")
    worst = 0.0
    for a in (1.0, 1.5, 2.5):
        rng = np.random.default_rng(11)
        L = N.pareto_lengths(rng, 400000, a, 2.0, 40.0)
        emp = float(np.mean(2.0 * L * (L / 2.0)))      # aspect 2, h 1
        ana = N.expected_elements_per_fault(a, 2.0, 40.0, 2.0, 1.0)
        rel = abs(emp - ana) / ana
        worst = max(worst, rel)
        print(f"       a {a:.1f}: analytic {ana:7.1f}  sampled {emp:7.1f}  "
              f"rel {rel:.4f}")
    check("closed-form budget matches sampling", worst, 0.02)
    big = N.expected_elements_per_fault(1.5, 2.0, 40.0, 2.0, 1.0)
    one = 2.0 * 40.0 * 20.0            # the largest single fault at h = 1
    check_true("the largest fault alone exceeds 30x the mean",
               one / big > 30.0, f"{one:.0f} vs mean {big:.1f} "
                                 f"= {one / big:.0f}x")


def c_cascade() -> None:
    """A NETWORK of real oriented faults on an arrangement of known dimension.

    The only end-to-end check available: generator, sampler, box counter and
    calibration together, against a number that is known in closed form rather
    than measured. An earlier version of this clause passed at +0.018 while
    SQUASHING the domain in z by 38/324 -- an anisotropic distortion that
    happened to cancel the estimator's bias. The isotropy guard in
    ``build_network`` exists because of that, and this clause runs on a cube.
    """
    print("\n[cascade] the whole pipeline against a known arrangement")
    worst = 0.0
    for cas, h in (((2, 3, 4), 0.2), ((2, 4, 3), 0.1)):
        net = N.build_network(seed=1, n_elements_target=20000,
                              centers="cascade", cascade=cas, box_xy=40.0,
                              z_top=-2.0, z_bottom=-42.0, h=h, aspect=2.0)
        cell = net.meta["cascade_cell_km"]
        known = net.meta["d_geom_known"]
        win = M.window_for(cell, 40.0)
        fit = M.dimension(net.tris, n_radii=10, **win)
        got = M.calibrated(fit)
        worst = max(worst, abs(got - known))
        print(f"       cascade {cas}  D known {known:.3f}  calibrated "
              f"{got:.3f}  err {got - known:+.3f}  "
              f"({net.n_faults} faults, {net.n_elements} elem, "
              f"{math.log10(win['r_hi'] / win['r_lo']):.2f} dec)")
    check("calibrated D of a known arrangement", worst, TOL_D_ABSOLUTE)
    # The isotropy guard is load-bearing, so it is checked rather than trusted.
    try:
        N.build_network(seed=1, n_elements_target=1000, centers="cascade",
                        box_xy=100.0, z_top=-2.0, z_bottom=-42.0, h=0.2)
        ok = False
    except ValueError:
        ok = True
    check_true("an anisotropic cascade domain is refused", ok)


def c_kernel() -> None:
    """The interaction matrix: the sign, the layout, and well-posedness."""
    print("\n[kernel] the sign that makes the cascade convergent")
    net = N.build_network(seed=2, n_elements_target=180, centers="uniform",
                          l_min=2.0, l_max=6.0, h=1.0, box_xy=60.0,
                          z_top=-2.0, z_bottom=-30.0)
    with tempfile.TemporaryDirectory() as tmp:
        K, info = Kmod.assemble(net, workers=1, cache_dir=tmp, verbose=False)
        n = net.n_elements
        idx = np.arange(n)
        worst = -np.inf
        for b, nm in enumerate(Kmod.SOURCE):
            d = K[idx, b, Kmod.RECEIVER.index(nm), idx]
            worst = max(worst, float(d.max()))
            print(f"       self {nm:6s} {d.min():.4e} .. {d.max():.4e}")
        check_true("self-interaction diagonal is negative", worst < 0.0,
                   f"worst {worst:.4e} over {n} elements, both components")

        # The layout is a performance claim, so it is checked as one.
        raw = mhs.interaction_matrix(
            net.tris, mhs.Material(mu=30.0, lam=30.0), net.eps,
            receiver=Kmod.RECEIVER, source=Kmod.SOURCE, obs_tris=net.tris)
        check_true("source-major transpose is exact",
                   bool(np.array_equal(K, np.transpose(raw, (1, 3, 2, 0)))))
        check_true("a source's block is contiguous",
                   bool(K[3].flags["C_CONTIGUOUS"]),
                   "one 48*n byte read per slip event")

        # Well-posedness: -P positive definite => the complementarity problem
        # has a unique solution on EVERY active set, not just the ones hit.
        rec = np.zeros((n, 2)); rec[:, 0] = 1.0
        src = np.zeros((n, 2)); src[:, 0] = 1.0
        P = Kmod.project(K, rec, src)
        ev = np.linalg.eigvalsh(-0.5 * (P + P.T))
        print(f"       sym(-P) eigenvalues {ev.min():.4e} .. {ev.max():.4e}; "
              f"asymmetry {np.abs(P - P.T).max() / np.abs(P).max():.2e}")
        check("sym(-P) is positive definite", float(ev.min()),
              MIN_EIG_SYM_NEG_P, below=False)

        # TRIPWIRE for a real bug. The first version of rect_patch rotated
        # down-dip the wrong way, so every mesh normal pointed DOWN while
        # tdcs's frame points UP; the eigenstress (dominant within a few eps)
        # was then subtracted with the wrong sign and the diagonal came out at
        # +48 instead of -50 on every element. Reversing the vertex order
        # reproduces exactly that, and it must be refused.
        flipped = dataclasses.replace(net, tris=np.ascontiguousarray(
            net.tris[:, [0, 2, 1]]))
        try:
            Kmod.assemble(flipped, workers=1, cache_dir=tmp, verbose=False)
            caught = False
        except SystemExit:
            caught = True
        check_true("tripwire: a reversed mesh normal is REFUSED", caught,
                   "wrong-handed vertex order flips the eigenstress sign")

    # AN EXACTLY VERTICAL PLANE IS A DEGENERATE CASE for the upward-normal
    # convention, and a generator that produces one gets normals decided by
    # float noise. mhs.tdcs flips to the upper hemisphere, so at n_z = 0 the
    # sign is arbitrary: a band built that way came back with 183 of 580 mesh
    # normals ANTI-ALIGNED with their own vertex order, and the
    # self-interaction diagonal went positive on exactly those.
    # ``build_network`` escapes it only because its dip is uniform on [30, 90]
    # and exactly 90 has measure zero -- which is luck, not design, so it is
    # pinned here.
    az = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    for label, nz in (("exactly vertical", 0.0), ("85 deg dip", 0.0872)):
        hz = math.sqrt(max(0.0, 1.0 - nz * nz))
        nn = np.column_stack([np.cos(az) * hz, np.sin(az) * hz,
                              np.full(az.size, nz)])
        cen = np.column_stack([np.arange(az.size) * 5.0,
                               np.zeros(az.size),
                               np.full(az.size, -10.0)])
        tri = N.oriented_triangle(cen, nn, 1.0)
        mesh = mhs.tdcs.unit_normals(tri)
        cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        cr /= np.linalg.norm(cr, axis=1, keepdims=True)
        agree = int((np.abs((cr * mesh).sum(axis=1) - 1.0) < 1e-6).sum())
        if nz == 0.0:
            check_true("an exactly vertical plane IS ambiguous (the trap)",
                       agree < az.size,
                       f"{agree} of {az.size} agree -- the rest are flipped")
        else:
            check_true("a few degrees off vertical is unambiguous",
                       agree == az.size, f"{agree} of {az.size} agree")

    # And the shipped generator must land on the safe side of it.
    net = N.build_aligned(seed=1, n_elements=400, h=0.5, n_planes=1,
                          scatter_over_h=0.0, box_xy=2.0, vertical=True,
                          z_top=-2.0, z_bottom=-26.0)
    mesh = mhs.tdcs.unit_normals(net.tris)
    cr = np.cross(net.tris[:, 1] - net.tris[:, 0],
                  net.tris[:, 2] - net.tris[:, 0])
    cr /= np.linalg.norm(cr, axis=1, keepdims=True)
    check_true("build_aligned's vertical bands are unambiguous",
               int((np.abs((cr * mesh).sum(axis=1) - 1.0) < 1e-6).sum())
               == net.n_elements,
               f"all {net.n_elements} mesh normals match their vertex order")

    # A REGULAR IN-PLANE LATTICE, not a Poisson scatter: Poisson puts pairs
    # 0.25 h apart, whose triangles interpenetrate, and that made sym(-P)
    # indefinite at exactly the coplanar end the experiment is about.
    cen = net.tris.mean(axis=1)
    dmin = np.inf
    for i in range(0, net.n_elements, 7):
        d = np.linalg.norm(cen - cen[i], axis=1)
        d[i] = np.inf
        dmin = min(dmin, float(d.min()))
    check_true("coplanar elements do not interpenetrate",
               dmin > 0.9 * net.h,
               f"closest pair {dmin / net.h:.2f} h apart")

    # THE NORMAL ROW'S SIGN, pinned against an INDEPENDENT route, because
    # nothing crashes if it is backwards. `project_normal` is TENSION
    # positive, the opposite of `friction.resolve_tensor`'s sigma_n, and the
    # two live modules apart; reversed, the coupled solve silently clamps the
    # wrong side of a step, which is the entire content of a stepping
    # asymmetry. The independent route is the full stress tensor from
    # `mhs.stress_matrix`, resolved by resolve_tensor's own convention.
    print("       [sign] the normal row against mhs.stress_matrix")
    from mhs.matrices import COLLOCATION_SHRINK                # noqa: PLC0415
    hs = 0.075
    srcs = N.disc_mesh(0.30, hs, depth=-20.0)
    recs = srcs.copy()
    recs[:, :, 0] += 0.42                      # offset OUT of the source plane
    recs[:, :, 2] += 0.25
    matl = mhs.Material(mu=30.0, lam=30.0)
    frs = mhs.tdcs.slip_frame(srcs)
    xdir = np.stack([frs[:, 0, 0], frs[:, 1, 0]], axis=1)     # unit +x slip
    rws = mhs.interaction_matrix(
        tris=srcs, material=matl, eps=0.1 * hs, receiver=Kmod.RECEIVER,
        source=Kmod.SOURCE, obs_tris=recs)
    row = Kmod.project_normal(Kmod.to_source_major(rws), xdir).sum(axis=0)
    del rws
    cart = np.zeros((srcs.shape[0], 3))
    cart[:, 0] = 1.0
    Sm = mhs.stress_matrix(
        mhs.collocation_points(recs, 0, COLLOCATION_SHRINK), srcs, matl,
        0.1 * hs)
    sg2 = np.einsum("omnsk,sk->omn", Sm, cart)
    del Sm
    nh = mhs.tdcs.slip_frame(recs)[:, 2, :]
    trc = np.einsum("omn,on->om", sg2, nh)
    sn_comp = -np.einsum("om,om->o", trc, nh)  # resolve_tensor's convention
    check("project_normal is tension-positive, i.e. minus sigma_n",
          float(np.abs(row + sn_comp).max() / np.abs(row).max()), 1e-12)


def c_crack() -> None:
    """PHASE 0: the mollified solve against a penny-shaped shear crack.

    The one 3-D crack geometry with a closed form, and therefore the gate that
    says whether growth work can proceed at all. Frictionless, so the LCP's
    answer IS the classical traction-free crack under a uniform stress drop:

        delta(r) = delta_0 sqrt(1 - r^2/a^2),  delta_0 = 24/(7 pi) d_tau a / mu

    at the nu = 1/4 this clause runs at; `growth.penny` carries the general
    nu, which the nu sweep needed and which this clause now reads from.

    WHAT MOLLIFICATION COSTS IS MEASURED HERE AND NOWHERE ELSE in either
    repository. The error is NOT uniform: it concentrates within a few eps of
    the crack front, which is exactly where a growth criterion would be
    evaluated -- and that is the quantitative argument for an ENERGETIC growth
    criterion, a global integral dominated by the accurate interior, over any
    front-local quantity.
    """
    print("\n[crack] a penny-shaped shear crack, which has a closed form")
    import mhs                                                # noqa: PLC0415
    a, mu_, lam_, dtau, depth = 1.0, 30.0, 30.0, 0.01, -20.0
    # The closed forms are nu dependent and `growth.penny` is where that is
    # stated; this clause runs at nu = 1/4, where they take the familiar
    # 24/(7 pi), 8/7 and 12/(7 pi) values that phases 0 and 1 were built on.
    nu_ = mhs.Material(mu=mu_, lam=lam_).nu_if_you_must
    pen = Gr.penny(nu_)
    want_C = pen.slip * dtau / mu_                            # slip per km of a

    got = {}
    for eps_over_h in (0.2, 0.1, 0.05):
        h = 0.1 * a
        eps = eps_over_h * h
        tris = N.disc_mesh(a, h, depth=depth)
        n = tris.shape[0]
        raw = mhs.interaction_matrix(
            tris=tris, material=mhs.Material(mu=mu_, lam=lam_), eps=eps,
            receiver=Kmod.RECEIVER, source=Kmod.SOURCE, obs_tris=tris)
        frame = mhs.tdcs.slip_frame(tris)
        sd = np.stack([frame[:, 0, 0], frame[:, 1, 0]], axis=1)
        sd /= np.linalg.norm(sd, axis=1, keepdims=True)
        P = Kmod.project(Kmod.to_source_major(raw), sd, sd)
        del raw
        sol = St.solve(P, np.full(n, dtau), np.zeros(n), 1.0, max_sweeps=20000)
        del P
        r = np.hypot(*tris.mean(axis=1)[:, :2].T)
        prof = np.sqrt(np.maximum(0.0, a * a - r * r))
        inner = r < 0.5 * a
        C = float((sol.delta[inner] * prof[inner]).sum()
                  / (prof[inner] ** 2).sum())
        pred = C * prof
        r2 = 1.0 - float(((sol.delta - pred) ** 2).sum()) / float(
            ((sol.delta - sol.delta.mean()) ** 2).sum())
        dev = np.abs(sol.delta - pred) / sol.delta.max()
        far = (a - r) / eps > 10.0
        near = ((a - r) / eps >= 1.0) & ((a - r) / eps < 3.0)
        got[eps_over_h] = (C / want_C, r2, float(np.median(dev[far])),
                           float(np.median(dev[near])), n, sol.converged)
        print(f"       eps/h {eps_over_h:4.2f}  n {n:4d}  C/analytic "
              f"{C / want_C:.4f}  R2 {r2:.5f}  dev far {np.median(dev[far]):.5f}"
              f"  near {np.median(dev[near]):.5f}")

    ref = got[0.1]
    check("the Eshelby coefficient is recovered", abs(ref[0] - 1.0), 0.03)
    check("the slip profile is elliptical", 1.0 - ref[1], 0.01)
    check_true("every crack solve converged",
               all(v[5] for v in got.values()), "")

    # CONVERGENCE IN eps, which is the process-zone statement: the mollified
    # solution approaches the sharp-crack answer as the regularization
    # vanishes. Gated as an ORDERING rather than a value, because the floor is
    # set by h and the point is the direction.
    errs = [abs(got[e][0] - 1.0) for e in (0.2, 0.1, 0.05)]
    check_true("error falls as eps falls",
               errs[0] > errs[1] > errs[2],
               " > ".join(f"{v:.4f}" for v in errs))

    # AND THE ERROR IS CONCENTRATED AT THE FRONT. If this ever inverted, a
    # front-local growth criterion would be as good as an energetic one and
    # the argument for work optimization would weaken.
    check_true("error concentrates near the front",
               ref[3] > 4.0 * ref[2],
               f"near {ref[3]:.4f} vs far {ref[2]:.4f}")

    # PHASE 1: the energy, and Irwin. The factor of two in released_energy is
    # the one error that would corrupt every growth decision while leaving
    # every number plausible, so it is pinned against closed forms.
    def solve_crack(aa, hh, ee):
        tr = N.disc_mesh(aa, hh, depth=depth)
        rw = mhs.interaction_matrix(
            tris=tr, material=mhs.Material(mu=mu_, lam=lam_), eps=ee,
            receiver=Kmod.RECEIVER, source=Kmod.SOURCE, obs_tris=tr)
        fr = mhs.tdcs.slip_frame(tr)
        sd = np.stack([fr[:, 0, 0], fr[:, 1, 0]], axis=1)
        sd /= np.linalg.norm(sd, axis=1, keepdims=True)
        Pm = Kmod.project(Kmod.to_source_major(rw), sd, sd)
        del rw
        so = St.solve(Pm, np.full(tr.shape[0], dtau), np.zeros(tr.shape[0]),
                      1.0, max_sweeps=20000)
        ar = 0.5 * np.linalg.norm(np.cross(tr[:, 1] - tr[:, 0],
                                           tr[:, 2] - tr[:, 0]), axis=1)
        en = St.released_energy(Pm, so.delta, ar)
        del Pm
        return en, float(ar.sum()), so.converged

    print(f"       [energy] E = {pen.energy:.5f} d_tau^2 a^3 / mu at nu "
          f"= {nu_:.3f}, and dE/dA vs Irwin")
    ratios, Es, As = [], {}, {}
    for aa in (1.0, 1.4):
        en, ar, okc = solve_crack(aa, 0.1 * aa, 0.01 * aa)
        want_E = pen.energy * dtau ** 2 * aa ** 3 / mu_
        ratios.append(en / want_E)
        Es[aa], As[aa] = en, ar
        print(f"         a {aa:4.2f}  E {en:.5e}  E/analytic {en/want_E:.4f}"
              f"{'' if okc else '  NOT CONVERGED'}")
    check("the energy coefficient is recovered", abs(ratios[0] - 1.0), 0.06)
    # THE SHARP ONE: a wrong factor of two is a constant offset, a wrong power
    # is a drift. Constancy across radii tests the a^3 scaling independently
    # of the coefficient.
    check("E/analytic is constant across radii (the a^3 scaling)",
          abs(ratios[0] - ratios[1]), 0.01)

    # Irwin by the compliance method, at an increment of 2h -- below that the
    # energy difference is dominated by discrete changes in element count
    # rather than by mechanics (measured: 2.3 % at h, 1.37 % at 2h, 1.34 % at
    # 4h).
    en2, ar2, ok2 = solve_crack(1.2, 0.1, 0.01)
    en1, ar1, ok1 = solve_crack(1.0, 0.1, 0.01)
    G = (en2 - en1) / (ar2 - ar1)
    want_G = pen.g * dtau ** 2 * 1.1 / mu_
    print(f"         G by compliance {G:.5e}  G/analytic {G/want_G:.4f}")
    check("dE/dA reproduces Irwin's G", abs(G / want_G - 1.0), 0.08)
    # AND THE BIAS IS SHARED. This is what makes a 1.3 % offset harmless for
    # work optimization: it cancels in the comparison between candidates.
    check("the energy and its derivative carry the SAME bias",
          abs(G / want_G - ratios[0]), 0.03)

    # PHASE 2: growth by work optimization, and the risk that would sink it.
    # Sharp-crack growth is notoriously mesh sensitive, so the same growth is
    # run on a lattice ROTATED BY 30 DEGREES -- the maximal inequivalence for
    # a 6-fold lattice. Gated on the ENERGY rather than the shape: the crude
    # max|x|/max|y| aspect moves 0.100 per cell, so it cannot resolve what
    # this is asking.
    print("       [growth] work-optimized growth on two lattice orientations")
    hh, ee, amax = 0.15, 0.015, 0.9
    out = {}
    for rot in (0.0, 30.0):
        tr = N.disc_mesh(amax, hh, depth=depth)
        if rot:
            t = math.radians(rot)
            c_, s_ = math.cos(t), math.sin(t)
            tr = tr @ np.array([[c_, -s_, 0.0], [s_, c_, 0.0],
                                [0.0, 0.0, 1.0]]).T
            tr[:, :, 2] = depth
        rw = mhs.interaction_matrix(
            tris=tr, material=mhs.Material(mu=mu_, lam=lam_), eps=ee,
            receiver=Kmod.RECEIVER, source=Kmod.SOURCE, obs_tris=tr)
        fr2 = mhs.tdcs.slip_frame(tr)
        sd2 = np.stack([fr2[:, 0, 0], fr2[:, 1, 0]], axis=1)
        sd2 /= np.linalg.norm(sd2, axis=1, keepdims=True)
        Pm = Kmod.project(Kmod.to_source_major(rw), sd2, sd2)
        del rw
        ar2 = 0.5 * np.linalg.norm(np.cross(tr[:, 1] - tr[:, 0],
                                            tr[:, 2] - tr[:, 0]), axis=1)
        cn = tr.mean(axis=1)
        nn = tr.shape[0]
        onm = (cn[:, 0] / 0.60) ** 2 + (cn[:, 1] / 0.45) ** 2 <= 1.0
        dd, gg = np.full(nn, dtau), np.zeros(nn)
        warm, Ee, shp = None, None, None
        for _ in range(3):
            onm, so2, Ee, _, ncand = Gr.step(
                Pm, dd, gg, ar2, cn, onm, pitch=hh, batch=4, warm=warm)
            warm = so2.delta
        shp = Gr.shape_ratio(cn, onm)
        out[rot] = (Ee, shp, int(onm.sum()), nn)
        if rot:
            del Pm
        else:
            base = (Pm, tr, cn, ar2, sd2, dd, gg)      # reused below, at two nu
        print(f"         rot {rot:4.0f} deg  lattice {nn:4d}  active "
              f"{int(onm.sum()):4d}  E {Ee:.5e}  shape {shp:.4f}")
    rel = abs(out[0.0][0] - out[30.0][0]) / out[0.0][0]
    check("growth is mesh independent in ENERGY", rel, 0.03)
    # SHAPE IS NOT GATED HERE, and the reason is resolution rather than
    # taste. The same analytic ellipse discretises to shape 1.2695 on one
    # lattice and 1.3653 on the other -- 0.096 apart BEFORE any growth -- and
    # this clause adds four cells per step out of about a hundred, so it
    # cannot separate what growth did from what the initial condition
    # inherited. Loosening a tolerance until it passed would be tuning the
    # gate to the answer.
    #
    # AND AT PRODUCTION RESOLUTION IT IS RESOLVABLE AND IT DOES NOT AGREE,
    # which an earlier version of this comment got wrong by reading the crude
    # aspect. 2346 cells, h = 0.075, fourteen steps: in `shape_ratio` the two
    # lattices' initial ellipses agree to 0.0058, yet the paths end 0.0952
    # apart -- the divergence is ACCUMULATED, not inherited, and the shape
    # change is +0.054 against +0.144, a factor of 2.6. The crude aspect hid
    # that inside its own 0.0733 initial offset. Energy meanwhile tracks to
    # 0.71 % worst and 0.10 % at the final step, so the energy check below is
    # the one this clause can support -- at ANY resolution, not merely this
    # one. See `growth`'s module docstring for why nu = 1/4 is the worst case.
    print(f"         shape {out[0.0][1]:.4f} vs {out[30.0][1]:.4f} -- NOT "
          f"gated: the initial ellipse already differs by 0.096 here")
    # The elongation itself: a shear crack grows along the slip direction
    # because G_II exceeds G_III by 1/(1-nu). Gated as a SIGN, since the
    # equilibrium aspect is not known in closed form.
    check_true("a shear crack elongates along slip (shape > 1)",
               out[0.0][1] > 1.0, f"shape {out[0.0][1]:.4f}")

    # WHY IT ELONGATES, which the sign above does not establish. Around a
    # circular shear crack the energy release rate must follow
    #
    #     G(phi)/G(0) = cos^2 phi + (1 - nu) sin^2 phi
    #
    # phi from the slip direction -- `growth.penny(nu).front`. That is the one
    # closed form here with no counterpart at fixed nu: the slip, the energy
    # and the global dE/dA are all azimuthal integrals carrying the same
    # (1-nu)/(2-nu), so a criterion that preferred the WRONG direction would
    # reproduce every one of them. This measures dE/dA around a single front,
    # with no growth and no path, and reads the azimuthal coefficient off it.
    #
    # Gated at TWO nu because the prediction contains a sign change, which is
    # much harder to reproduce by accident than a magnitude: at nu = 0 the
    # front is ISOTROPIC and an ellipse rounds up instead of elongating.
    # nu = 1/4 costs no assembly (the rot-0 matrix above is already at it);
    # nu = 0 costs one more on the same lattice, and that is the whole price
    # of the clause.
    print("       [front] dE/dA around a circular front vs cos^2 + (1-nu) sin^2")
    # FIRST THAT THE ESTIMATOR IS UNBIASED, separately from whether the
    # physics is. Synthetic gains built from a known c, carrying the same
    # radial modulation the lattice imposes, must come back exactly -- so a
    # bias in the measured c below is discretization and not the fit. Keeping
    # these two apart is the distinction that cost a run when an expectation
    # and a measurement disagreed and the measurement was right.
    rng_f = np.random.default_rng(0)
    nsyn = 84                      # the production front's candidate count
    psyn = np.linspace(-math.pi, math.pi, nsyn, endpoint=False)
    rsyn = 1.0 + rng_f.uniform(-0.02, 0.02, nsyn)
    ctsyn = np.zeros((nsyn + 1, 3))
    ctsyn[1:, 0] = rsyn * np.cos(psyn)
    ctsyn[1:, 1] = rsyn * np.sin(psyn)
    onsyn = np.zeros(nsyn + 1, bool)
    onsyn[0] = True
    cdsyn = np.arange(1, nsyn + 1)
    arsyn = np.full(nsyn + 1, 0.5)
    sdir = np.array([1.0, 0.0])

    def synth(c_true, arc):
        """Gains built the way `front_gains(arc=w)` measures them."""
        out = np.empty(cdsyn.size)
        for k, cc in enumerate(cdsyn):
            patch = (Gr.arc_patch(ctsyn, cdsyn, cc, arc) if arc > 1
                     else np.array([cc]))
            p = np.arctan2(ctsyn[patch, 1], ctsyn[patch, 0])
            r = np.hypot(ctsyn[patch, 0], ctsyn[patch, 1])
            w = arsyn[patch] / arsyn[patch].sum()
            out[k] = ((w * (np.cos(p) ** 2 + c_true * np.sin(p) ** 2)).sum()
                      * (1.0 - 2.0 * ((w * r).sum() - 1.0)))
        return out

    for arc_s in (1, 9):
        for c_true in (1.0, 0.75, 0.60):
            gn = synth(c_true, arc_s)
            c_s, _ = Gr.front_anisotropy(ctsyn, onsyn, cdsyn, gn, sdir,
                                         arc=arc_s, areas=arsyn)
            check(f"the arc-{arc_s} fit recovers a synthetic c = {c_true:.2f}",
                  abs(c_s - c_true), 0.002)
    # AND THAT THE ARC CORRECTION IS DOING SOMETHING, so the clause above
    # cannot pass by the correction being a no-op. Fitting arc-9 gains as if
    # they were centre values must read HIGH, because averaging an azimuthal
    # ratio over 39 degrees of front compresses it toward one.
    naive, _ = Gr.front_anisotropy(ctsyn, onsyn, cdsyn, synth(0.60, 9), sdir)
    check_true("ignoring the arc biases c high, which is why arc is passed",
               naive > 0.60 + 0.02, f"centre-value fit reads {naive:.4f} "
                                    f"against 0.6000")

    Pm, tr, cn, ar2, sd2, dd, gg = base

    def anisotropy(Pmat, rad=0.60):
        onc = cn[:, 0] ** 2 + cn[:, 1] ** 2 <= rad * rad
        cand, gain, _, _ = Gr.front_gains(Pmat, dd, gg, ar2, cn, onc,
                                          pitch=hh, max_sweeps=20000)
        c_, r2_ = Gr.front_anisotropy(cn, onc, cand, gain, sd2[0])
        return c_, r2_, int(onc.sum()), int(cand.size)

    fronts = {}
    for nu_f in (nu_, 0.0):
        if nu_f == nu_:
            Pf = Pm
        else:
            mat = mhs.Material.from_mu_nu(mu_, nu_f)
            rwf = mhs.interaction_matrix(
                tris=tr, material=mat, eps=ee, receiver=Kmod.RECEIVER,
                source=Kmod.SOURCE, obs_tris=tr)
            Pf = Kmod.project(Kmod.to_source_major(rwf), sd2, sd2)
            del rwf
        c_, r2_, nact, ncd = anisotropy(Pf)
        fronts[nu_f] = c_
        want_f = Gr.penny(nu_f).front
        print(f"         nu {nu_f:4.2f}  front {nact:4d} active, {ncd:3d} "
              f"candidates  c {c_:.4f} vs 1-nu {want_f:.4f} "
              f"({100 * (c_ / want_f - 1):+.1f} %)  R2 {r2_:.4f}")
        if Pf is not Pm:
            del Pf
    del Pm, base

    # PHASE 3: OUT-OF-PLANE CANDIDATES, whose correctness is geometric and so
    # can be pinned without a single solve. Growth out of plane cannot be
    # activation -- a kinked cell is in no lattice -- so candidates are BUILT,
    # and three things must hold or nothing downstream means anything.
    print("       [kink] out-of-plane candidates: the identity, rigidity, sign")
    hk = 0.075
    trk = N.disc_mesh(0.30, hk, depth=depth)
    cnk = trk.mean(axis=1)
    onk = cnk[:, 0] ** 2 + cnk[:, 1] ** 2 <= 0.04
    fr3 = Gr.front_edges(trk, onk)
    c0 = Gr.kink_candidates(fr3, 0.0)

    # (i) theta = 0 REPRODUCES the lattice, vertices and winding both. This is
    # what makes planar growth the special case of out-of-plane growth rather
    # than a separate code path, and the winding half is the one that bites:
    # the reversed order flips the slip frame and the self-interaction's sign
    # while leaving n_z reporting a perfectly innocent 1.
    def cyclic(a, b):
        a = [tuple(np.round(v, 9)) for v in a]
        b = [tuple(np.round(v, 9)) for v in b]
        return any(a == b[i:] + b[:i] for i in range(3))

    exact = 0
    for k in range(fr3.cell.size):
        d2 = ((cnk - c0[k].mean(axis=0)) ** 2).sum(axis=1)
        j = int(np.argmin(d2))
        if d2[j] < 1e-18 and cyclic(trk[j], c0[k]):
            exact += 1
    check_true("theta = 0 reproduces the lattice cell, winding included",
               exact == fr3.cell.size, f"{exact} of {fr3.cell.size}")

    # (ii) THE ROTATION IS RIGID. Edge length and area must not move, or a
    # kink angle is confounded with a change of element size.
    e0 = np.linalg.norm(c0[:, 1] - c0[:, 0], axis=1)
    a0 = 0.5 * np.linalg.norm(np.cross(c0[:, 1] - c0[:, 0],
                                       c0[:, 2] - c0[:, 0]), axis=1)
    worst_e = worst_a = 0.0
    for th in (10.0, 30.0, 60.0):
        ck = Gr.kink_candidates(fr3, math.radians(th))
        ek = np.linalg.norm(ck[:, 1] - ck[:, 0], axis=1)
        ak = 0.5 * np.linalg.norm(np.cross(ck[:, 1] - ck[:, 0],
                                           ck[:, 2] - ck[:, 0]), axis=1)
        worst_e = max(worst_e, float(np.abs(ek / e0 - 1.0).max()))
        worst_a = max(worst_a, float(np.abs(ak / a0 - 1.0).max()))
        # The normal tilts by exactly theta, which is also the statement that
        # the rotation axis is the front edge and nothing else.
        nz = mhs.tdcs.unit_normals(ck)[:, 2]
        check(f"the normal tilts by exactly {th:.0f} deg",
              float(np.abs(nz - math.cos(math.radians(th))).max()), 1e-12)
    check("kinking preserves the front edge", worst_e, 1e-12)
    check("kinking preserves the element area", worst_a, 1e-12)

    # (iii) ONE CANDIDATE PER POSITION. A triangle outside a ragged front can
    # share edges with two active cells, and admitting both puts coincident
    # elements in a trial -- area counted twice, mutual interaction equal to
    # self-interaction. Measured, that misread a tip-to-tip gain ratio by 31 %
    # and invented a four-fold difference between the two tips.
    cc = c0.mean(axis=1)
    dmk = np.linalg.norm(cc[:, None, :] - cc[None, :, :], axis=2)
    np.fill_diagonal(dmk, np.inf)
    check_true("no two candidates coincide",
               float(dmk.min()) > 0.3 * hk,
               f"min separation {dmk.min() / hk:.4f} h")

    # AND THE SIGN SURVIVES ASSEMBLY, which geometry alone cannot promise:
    # slip on a kinked cell must still relieve the shear that drives it.
    sig3 = dtau * (np.outer([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
                   + np.outer([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]))

    def proj_diag(tk):
        tq, _ = Fr.resolve_tensor(tk, sig3)
        sdk = tq / np.maximum(np.linalg.norm(tq, axis=1), 1e-300)[:, None]
        rw3 = mhs.interaction_matrix(
            tris=tk, material=mhs.Material(mu=mu_, lam=lam_), eps=0.1 * hk,
            receiver=Kmod.RECEIVER, source=Kmod.SOURCE, obs_tris=tk)
        return Kmod.project(Kmod.to_source_major(rw3), sdk, sdk).diagonal()

    for th in (0.0, 30.0):
        dg = proj_diag(Gr.kink_candidates(fr3, math.radians(th)))
        check_true(f"kinked cells at {th:.0f} deg relieve their own shear",
                   float(dg.max()) < 0.0, f"worst diagonal {dg.max():+.4e}")

    # (iv) AN ARC MUST TURN AS ONE PIECE, and the quantity that says so is the
    # KINKED NORMAL, not the tangent. `coherent_arc` groups by the SIGNED edge
    # axis because |t_i . t_j| = 1 admits antiparallel partners that tilt the
    # opposite way under one theta. Both halves are gated: the selection is
    # exact, and the naive alternative is MEASURABLY worse, so dropping the
    # sign cannot pass quietly.
    toward = np.array([-1.0, 0.0, 0.0])
    arc = Gr.coherent_arc(fr3, toward, width=7)
    axk = fr3.edge[:, 1] - fr3.edge[:, 0]
    axk = axk / np.linalg.norm(axk, axis=1, keepdims=True)
    mk = 0.5 * (fr3.edge[:, 0] + fr3.edge[:, 1])

    def normals(idx, th):
        return mhs.tdcs.unit_normals(
            Gr.kink_candidates(fr3, math.radians(th), which=idx))

    # The normals are compared as VECTORS, not as an angle: they agree to
    # 1.4e-16 componentwise, and `arccos` near 1 turns that into 1.2e-6
    # degrees, which would make a float-precision check look like a tolerance
    # to be loosened.
    worst = max(float(np.abs(normals(arc, th) - normals(arc, th)[0]).max())
                for th in (10.0, 30.0, 45.0))
    check("a coherent arc's kinked normals agree", worst, 1e-14)
    # AND THE FAILURE MODE HAS A CLOSED FORM, so the signed criterion is gated
    # by an identity rather than by a threshold: an ANTIPARALLEL partner
    # rotates about the reversed axis, so one theta tilts the two cells
    # opposite ways and their normals differ by exactly 2 theta. That is the
    # 60 degrees an unsigned tangent family shows at theta = 30, and the whole
    # reason `coherent_arc` groups by the signed axis.
    anti = np.flatnonzero(axk @ axk[arc[0]] < -1.0 + 1e-9)
    check_true("the front has an antiparallel partner to separate",
               anti.size > 0, f"{anti.size} of {fr3.cell.size} edges")
    worst_anti = 0.0
    for th in (10.0, 30.0):
        nk = normals(np.array([arc[0], anti[0]]), th)
        got = math.degrees(math.acos(float(np.clip(nk[0] @ nk[1], -1.0, 1.0))))
        worst_anti = max(worst_anti, abs(got - 2.0 * th))
    check("an antiparallel partner's normals differ by exactly 2 theta",
          worst_anti, 1e-9)
    # Every selected edge must actually advance toward the target, or "the
    # facing tip" is a position with no direction attached.
    outk = fr3.edge[:, 0] + fr3.edge[:, 1] - fr3.apex - mk
    outk = outk / np.linalg.norm(outk, axis=1, keepdims=True)
    check_true("every arc edge advances toward the target",
               float((outk[arc] @ toward).min()) > 0.0,
               f"worst {float((outk[arc] @ toward).min()):+.4f}")
    # AND THE PRICE IS RECORDED: coherence excludes adjacency on this tiling,
    # so the arc is co-oriented flakes and not a flap. An experiment wanting a
    # turning front has to say so rather than inherit the word from "arc".
    ck = Gr.kink_candidates(fr3, 0.0, which=arc).mean(axis=1)
    dsep = np.linalg.norm(ck[:, None, :] - ck[None, :, :], axis=2) / hk
    np.fill_diagonal(dsep, np.inf)
    check_true("a coherent arc is NOT edge-adjacent",
               float(dsep.min()) > 0.9,
               f"min separation {dsep.min():.3f} h, adjacency 0.577 h")

    # (v) WHY A CRACK KINKS AT ALL, which is a SYMMETRY and its breaking. At
    # sigma_n = 0 with no friction the kink gain is EVEN in theta, because
    # sigma changes sign on reflection through the crack plane and E goes as
    # sigma^2 -- that evenness is the whole content of 75334ee's planar front,
    # and it makes theta* = 0 a consequence of the loading rather than of the
    # elasticity. Coulomb breaks it: sigma_n(theta) = sigma_c - d_tau sin
    # 2theta carries an ODD part, so mu_f sigma_n is odd and the gain is not
    # symmetric. The growth path follows -- the crack turns once, to 17.26
    # degrees against the Coulomb prediction of 16.60, and is planar after.
    # The path costs 200 solves and is not gateable; this mechanism is four.
    arc1 = Gr.coherent_arc(fr3, toward, width=1)
    e_on = np.ones(trk.shape[0], bool)

    def kink_gain(theta_deg, mu_f, sig_c):
        sg = sig3 - sig_c * np.eye(3)
        cand = Gr.kink_candidates(fr3, math.radians(theta_deg), which=arc1)

        def coupled(t):
            tq, sn = Fr.resolve_tensor(t, sg)
            mg = np.linalg.norm(tq, axis=1)
            sd = tq / np.maximum(mg, 1e-300)[:, None]
            sm = Kmod.to_source_major(mhs.interaction_matrix(
                tris=t, material=mhs.Material(mu=mu_, lam=lam_), eps=0.1 * hk,
                receiver=Kmod.RECEIVER, source=Kmod.SOURCE, obs_tris=t))
            return (Kmod.project(sm, sd, sd) + mu_f * Kmod.project_normal(
                sm, sd), mg, mu_f * sn)

        ar = 0.5 * np.linalg.norm(np.cross(trk[:, 1] - trk[:, 0],
                                           trk[:, 2] - trk[:, 0]), axis=1)
        P0, m0, g0_ = coupled(trk)
        s0 = Gr._converged(P0, m0, g0_, e_on, None, St.MAX_SWEEPS)
        E0 = St.released_energy(P0, s0.delta, ar)
        big = np.concatenate([trk, cand])
        P1, m1, g1 = coupled(big)
        a1 = 0.5 * np.linalg.norm(np.cross(big[:, 1] - big[:, 0],
                                           big[:, 2] - big[:, 0]), axis=1)
        s1 = Gr._converged(P1, m1, g1, np.ones(big.shape[0], bool),
                           np.concatenate([s0.delta,
                                           np.zeros(cand.shape[0])]),
                           St.MAX_SWEEPS)
        return ((St.released_energy(P1, s1.delta, a1) - E0)
                / a1[trk.shape[0]:].sum())

    gp, gm = kink_gain(+10.0, 0.0, 0.0), kink_gain(-10.0, 0.0, 0.0)
    # THE EVENNESS MUST NOT BE 0 == 0, which is how this clause could pass
    # while measuring nothing at all: two locked candidates agree perfectly.
    check_true("on frictionless gains that are actually there",
               min(gp, gm) > 1e-10, f"gains {gp:.4e} and {gm:.4e}")
    check("frictionless, the kink gain is EVEN in theta",
          abs(gp - gm) / abs(gp), 1e-6)
    # WITH COULOMB THE POSITIVE TILT TAKES NO SLIP HERE, so the asymmetry
    # saturates at 1.0000. That is the symmetry breaking in its strongest form,
    # and the part worth recording is that it CANNOT BE PREDICTED FROM THE
    # REMOTE STRESS: against sigma alone both tilts are admissible, at a
    # driving-force ratio of 3.05, and it is the PARENT's own slip -- through
    # the coupled sigma_n -- that clamps the positive one to zero. A first
    # version of this clause tried to force a graded comparison by lowering
    # sigma_c, which instead drove the positive gain NEGATIVE (-3.1e-08 at
    # 0.2 d_tau): growth there costs energy rather than releasing it, so the
    # ratio test it was built on stopped meaning anything. Only the solve
    # decides, which is why this clause solves rather than resolving a traction.
    fp, fm = kink_gain(+10.0, 0.6, dtau), kink_gain(-10.0, 0.6, dtau)
    check_true("Coulomb breaks that evenness",
               abs(fp - fm) / max(abs(fp), abs(fm)) > 0.2,
               f"asymmetry {abs(fp - fm) / max(abs(fp), abs(fm)):.4f} "
               f"against {abs(gp - gm) / abs(gp):.2e} frictionless")
    # AND IN THE DIRECTION THAT UNCLAMPS, not merely in some direction: the
    # favoured tilt must be the one whose own sigma_n is lower, or the
    # asymmetry would be a sign convention rather than friction.
    sn_hi = Fr.resolve_tensor(
        Gr.kink_candidates(fr3, math.radians(+10.0), which=arc1),
        sig3 - dtau * np.eye(3))[1][0]
    sn_lo = Fr.resolve_tensor(
        Gr.kink_candidates(fr3, math.radians(-10.0), which=arc1),
        sig3 - dtau * np.eye(3))[1][0]
    check_true("and favours the tilt that unclamps",
               (fp > fm) == (sn_hi < sn_lo),
               f"gain {'+' if fp > fm else '-'}10 larger, sigma_n "
               f"{float(sn_hi):+.5f} vs {float(sn_lo):+.5f}")

    # THE SIGN, which no discretization produces on its own: more energy is
    # released growing ALONG slip than across it, whenever nu > 0.
    check_true("dE/dA is larger along slip than across it (c < 1)",
               fronts[nu_] < 1.0, f"c {fronts[nu_]:.4f} at nu {nu_:.2f}")
    # AND IT VANISHES AT nu = 0, the prediction's own sign change and the
    # reason this is mode mixity rather than a directional bug -- a bug does
    # not know about nu. This one IS gated against the closed form, because
    # c = 1 at nu = 0 is where the measurement is accurate: production reads
    # 1.0370 by this fit and 1.0102 by the wedge, against 1.
    check("the front is isotropic at nu = 0", abs(fronts[0.0] - 1.0), 0.15)
    # THE ORDERING rather than the value at nu > 0, and the reason is a
    # measured bias, not coarse resolution. A one-cell gain carries the
    # bump's own self-interaction, which the lattice's fixed cell aspect
    # keeps alive under refinement: c plateaus at 0.4900 for a/h = 12 and 18
    # alike against a closed form of 0.6000, and spreading the perturbation
    # over a 24-51 degree arc recovers 0.545-0.576. So the slope of c in nu
    # comes out 35 % steep everywhere and gating the value would gate the
    # bias. `growth.front_anisotropy` carries the numbers.
    #
    # THIS CLAUSE'S OWN -2.8 % AT nu = 1/4 IS FORTUITOUS and must not be read
    # as the measurement being accurate here. It is the coarsest lattice of
    # any run (a/h = 4) and has the largest eps/a = 0.025; production, eight
    # times better resolved, reads -8.7 %. Falling eps moves c AWAY from the
    # closed form while falling h moves it toward, so a tolerance tuned to
    # this combination would pass for the wrong reason and break on a
    # resolution change that improved the physics.
    #
    # AND THE FIX IS NOT EXERCISED ON REAL GAINS HERE, deliberately, because
    # this clause cannot afford the resolution it needs: the front row holds
    # about 25 cells, so an arc of 9 would span 130 degrees and stop being a
    # local measurement at all. The synthetic clauses above gate the
    # estimator; the fix's effect on the physics is measured at production
    # resolution, where arc 9 and 15 bracket the closed form at -1.0 % and
    # +1.2 % against the one-cell -18.3 %.
    check_true("the anisotropy strengthens with nu",
               fronts[0.0] > fronts[nu_],
               f"c {fronts[0.0]:.4f} at nu 0 vs {fronts[nu_]:.4f} at {nu_:.2f}")


def c_load() -> None:
    """The FULL stress tensor: the loading path, its ceiling, and the locked set.

    The regional stress is ``sigma_c I + lambda (D + k I)``. Two things here are
    physics rather than bookkeeping and both were found by asking what happens
    as lambda grows without bound.

    THE LOCKED SET. The load clamps as well as shears, so strength grows with
    lambda too and an element with ``|tau| <= mu sigma_n`` per unit load never
    fails AT ANY LOAD. That set is fixed by orientation and mu alone.

    THE COMPRESSIVE CEILING. A deviatoric D is traceless, so at k = 0 the
    tension-favoured planes lose normal stress as fast as they gain shear and
    go into effective TENSION past ``lambda ~ sigma_c``, where Coulomb friction
    does not apply and the negative strength that results is met by any shear
    at all. ``k >= 1`` removes the ceiling entirely because the minimum
    eigenvalue of ``D + k I`` is then non-negative.
    """
    print("\n[load] the full stress tensor and the loading path")
    net = N.build_network(seed=1, n_elements_target=4000, l_min=2.0,
                          l_max=20.0, h=1.0, box_xy=60.0, z_top=-2.0,
                          z_bottom=-42.0, a=1.5, aspect=2.0, centers="uniform")

    f0 = Fr.build_friction(net, mean_stress_ratio=0.0)
    f1 = Fr.build_friction(net, mean_stress_ratio=1.0)

    # An isotropic tensor resolves the same normal stress onto every plane and
    # NO shear onto any, so k must move sigma_n by exactly k and leave the
    # shear and the slip direction untouched. If this ever fails, k is leaking
    # into the driving stress and every locked fraction below is wrong.
    check("k adds no shear", float(np.abs(f1.tau_load_mag
                                          - f0.tau_load_mag).max()), 1e-12)
    check("k shifts sigma_n by exactly k",
          float(np.abs((f1.sigma_n_load - f0.sigma_n_load) - 1.0).max()), 1e-12)
    check("k leaves the slip direction alone",
          float(np.abs(f1.slip_dir - f0.slip_dir).max()), 1e-12)

    # k >= 1 can never unclamp a plane, so the ceiling is infinite and the
    # minimum resolved normal is non-negative.
    check_true("k >= 1 is compressive at every lambda",
               not np.isfinite(f1.max_lambda_compressive())
               and float(f1.sigma_n_load.min()) >= 0.0,
               f"min sigma_n_load {f1.sigma_n_load.min():+.4f}, ceiling "
               f"{f1.max_lambda_compressive()}")

    # At k = 0 the ceiling is finite and is exactly where sigma_eff hits zero.
    cap = f0.max_lambda_compressive()
    sig_at_cap = f0.sigma_eff_0 + cap * f0.sigma_n_load
    check_true("k = 0 has a finite ceiling", np.isfinite(cap),
               f"lambda {cap:.5f} = "
               f"{cap / float(np.nanmin(f0.first_failure_lambda())):.2f}x "
               f"first failure")
    check("the ceiling is exactly where sigma_eff reaches zero",
          float(np.abs(sig_at_cap.min())), 1e-12)

    # The locked fraction must RISE with both k and mu: both raise the normal
    # stress term relative to the shear. A non-monotone answer means a sign
    # error somewhere in the resolution.
    byk = [Fr.build_friction(net, mean_stress_ratio=k).locked_fraction()
           for k in (0.0, 1 / 3, 2 / 3, 1.0)]
    bymu = [Fr.build_friction(net, mu_mean=m).locked_fraction()
            for m in (0.2, 0.4, 0.6, 0.8)]
    print(f"       locked vs k  {['%.3f' % v for v in byk]}")
    print(f"       locked vs mu {['%.3f' % v for v in bymu]}")
    check_true("the locked fraction rises with k",
               all(b >= a - 1e-12 for a, b in zip(byk, byk[1:])), "monotone")
    check_true("the locked fraction rises with mu",
               all(b >= a - 1e-12 for a, b in zip(bymu, bymu[1:])), "monotone")

    # A locked element must stay locked at ANY lambda, which is the whole
    # claim. Checked directly against the yield function rather than against
    # the formula it was derived from.
    locked = f0.tau_load_mag - f0.mu * f0.sigma_n_load <= 0.0
    worst = -np.inf
    for lam in (1.0, 1e3, 1e8):
        F = lam * f0.tau_load_mag[locked] - f0.mu[locked] * (
            f0.sigma_eff_0[locked] + lam * f0.sigma_n_load[locked])
        worst = max(worst, float(F.max()))
    check_true("a locked element never yields, out to lambda = 1e8",
               worst <= 0.0, f"worst F {worst:.3e} <= 0 over "
                             f"{int(locked.sum())} locked elements")

    # THE TENSOR AND THE PER-ELEMENT RESOLUTION MUST AGREE. Every plane's
    # effective normal stress is a Rayleigh quotient of the total stress, so
    # it lies between the extreme principal values -- for ANY orientation.
    # This ties friction.resolve_tensor to the tensor the page draws, and it
    # is how a sign error in the drawn one was caught: flipping the isotropic
    # part along with the deviatoric reported two negative principal stresses
    # on a load that is compressive at every element.
    import payload as PL                                      # noqa: PLC0415
    for kk in (0.0, 2 / 3):
        fk = Fr.build_friction(net, mean_stress_ratio=kk)
        lk = 2.0 * float(np.nanmin(fk.first_failure_lambda()))
        pr = PL._principal(fk, lk, kk)["principal_GPa"]
        se = fk.sigma_eff_0 + lk * fk.sigma_n_load
        check_true(f"k={kk:.2f}: sigma_eff lies inside the principal range",
                   bool(se.min() >= min(pr) - 1e-9
                        and se.max() <= max(pr) + 1e-9),
                   f"[{se.min():+.4f}, {se.max():+.4f}] within "
                   f"[{min(pr):+.4f}, {max(pr):+.4f}]")

    # first_failure_lambda must agree with the yield function it encodes:
    # at its own lambda an unlocked element is exactly at failure.
    lf = f0.first_failure_lambda()
    ok = np.isfinite(lf)
    F0 = lf[ok] * f0.tau_load_mag[ok] - f0.mu[ok] * (
        f0.sigma_eff_0[ok] + lf[ok] * f0.sigma_n_load[ok])
    check("first_failure_lambda sits exactly on the yield surface",
          float(np.abs(F0).max()), 1e-12)
    check_true("every element is locked or has a finite failure lambda",
               bool(np.all(locked ^ ok)),
               f"{int(ok.sum())} fail, {int(locked.sum())} never")


def c_static() -> None:
    """The static solve: unique by theorem, and checked against that.

    "Which faults slip under this stress field" is a Linear Complementarity
    Problem with a P-matrix, so it has exactly one solution and needs no
    strength drop, no loading path and no update rule. These checks pin that
    the implementation finds THE solution rather than A fixed point.
    """
    print("\n[static] the LCP solution at one load")
    rng = np.random.default_rng(0)
    n = 60
    # A random P with negative diagonal and PD negative symmetric part: the
    # structure kernel.project produces, without needing an assembly.
    B = rng.normal(size=(n, n)) * 0.05
    P = -(np.eye(n) * 1.0 + B @ B.T * 0.1)
    ev = float(np.linalg.eigvalsh(-0.5 * (P + P.T)).min())
    check("test operator has PD symmetric part", ev, 0.1, below=False)

    d = np.abs(rng.normal(1.0, 0.3, n))
    g = np.abs(rng.normal(1.0, 0.2, n))
    lam_lo = 0.5 * float((g / d).min())
    lam_hi = 3.0 * float((g / d).min())

    # [a] Below the first failure NOTHING slips, and exactly zero rather than
    # merely small.
    lo = St.solve(P, d, g, lam_lo)
    check_true("no slip below the first failure load",
               lo.converged and float(np.abs(lo.delta).max()) == 0.0,
               f"max slip {np.abs(lo.delta).max():.3e}")

    # [b] Complementarity: locked and below strength, or slipping and AT it.
    hi = St.solve(P, d, g, lam_hi)
    F = lam_hi * d + P.T @ hi.delta - g
    check_true("converged", hi.converged,
               f"{hi.sweeps} sweeps, residual {hi.residual:.2e}")
    check("no element exceeds its strength", float(F.max()), 1e-8)
    check("slip is non-negative", float(-hi.delta.min()), 1e-12)
    check("slipping elements sit exactly at strength",
          float(np.abs(F[hi.active]).max()) if hi.active.any() else 0.0, 1e-8)

    # [c] UNIQUENESS, which is the whole claim: permuting the problem must
    # permute the solution and change nothing else.
    pi = rng.permutation(n)
    inv = np.argsort(pi)
    pm = St.solve(P[np.ix_(pi, pi)], d[pi], g[pi], lam_hi)
    check("a permuted problem gives the permuted solution",
          float(np.abs(pm.delta[inv] - hi.delta).max()), 1e-7)

    # [d] A WARM START cannot change the answer, only the sweep count. If it
    # ever does, the solve is finding a fixed point rather than the solution.
    warm = St.solve(P, d, g, lam_hi, warm=hi.delta * 0.5 + 0.01)
    check("a warm start reaches the same solution",
          float(np.abs(warm.delta - hi.delta).max()), 1e-7)
    check_true("warm start is cheaper", warm.sweeps <= hi.sweeps,
               f"{warm.sweeps} sweeps vs {hi.sweeps} cold")

    # [e] `include` must mean ABSENT, not merely zero: a subnetwork's answer
    # has to equal the answer for that submatrix solved on its own.
    keep = np.zeros(n, bool)
    keep[rng.choice(n, n // 2, replace=False)] = True
    sub = St.solve(P, d, g, lam_hi, include=keep)
    k = np.flatnonzero(keep)
    alone = St.solve(P[np.ix_(k, k)], d[k], g[k], lam_hi)
    check("a masked subnetwork equals the standalone submatrix",
          float(np.abs(sub.delta[keep] - alone.delta).max()), 1e-7)
    check_true("masked-out elements carry exactly zero slip",
               float(np.abs(sub.delta[~keep]).max()) == 0.0, "exactly 0")

    # [f] One element has a closed form: it slips exactly back to strength.
    one = St.solve(np.array([[-2.0]]), np.array([1.0]), np.array([1.0]), 3.0)
    check("one element matches the closed form",
          abs(float(one.delta[0]) - (3.0 - 1.0) / 2.0), 1e-12)

    # [g] sigma_n COUPLED TO SLIP, which a stepping asymmetry needs and the
    # frictionless solve above cannot represent at all. Two cracks offset by
    # +dz or -dz are the two stepping senses, and the whole effect rests on one
    # symmetry: the SHEAR they exchange is EVEN in that sense while the NORMAL
    # traction is ODD. That is a mirror symmetry of the elastic problem, so a
    # frictionless criterion is blind to left- versus right-stepping EXACTLY
    # rather than weakly, and any measured asymmetry is the coupling alone.
    # Gated because the claim is structural: were P to stop being even, the
    # asymmetry would stop meaning what it gets reported to mean.
    print("       [coupled] sigma_n coupled to slip: the stepping symmetry")
    import mhs                                                # noqa: PLC0415
    # THE ENERGY SPLIT'S SUM IDENTITY, which needs no assembly and is the
    # standing check on how the asymmetry gets attributed. Only
    # shear + clamp = total is forced; everything the decomposition claims
    # beyond that is measured. Passing P_eff where the BARE P belongs
    # double-counts the clamping and the halves cancel, which once turned a
    # -11.9 % asymmetry into 0.00 % -- so the identity is pinned here and the
    # misuse cannot survive it.
    rng_e = np.random.default_rng(11)
    ne = 60
    Pe_ = -np.eye(ne) * 5.0 + 0.2 * rng_e.standard_normal((ne, ne))
    Ne_ = 0.4 * rng_e.standard_normal((ne, ne))
    de_ = np.abs(rng_e.standard_normal(ne))
    ae_ = 0.5 + rng_e.random(ne)
    mue = 0.3 + 0.4 * rng_e.random(ne)
    e_sh, e_cl = St.energy_split(Pe_, Ne_, mue, de_, ae_)
    want_e = St.released_energy(Pe_ + mue[None, :] * Ne_, de_, ae_)
    check("the energy split sums to the released energy",
          abs(float(e_sh.sum() + e_cl.sum()) - want_e) / abs(want_e), 1e-13)
    # And the misuse it guards against is not a no-op: feeding P_eff in as
    # the shear operator must give a DIFFERENT number, or the check is idle.
    bad_sh, bad_cl = St.energy_split(Pe_ + mue[None, :] * Ne_, Ne_, mue,
                                     de_, ae_)
    check_true("double-counting the clamp term really does change the answer",
               abs(float(bad_sh.sum() + bad_cl.sum()) - want_e)
               > 0.01 * abs(want_e),
               f"{float(bad_sh.sum() + bad_cl.sum()):.5e} vs {want_e:.5e}")

    hc, ac_, mu_c = 0.075, 0.30, 30.0
    base = N.disc_mesh(ac_, hc, depth=-20.0)
    sg = 0.01 * (np.outer([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
                 + np.outer([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]))

    def pair(dz):
        t = np.concatenate([base, base.copy()])
        t[base.shape[0]:, :, 0] += 1.4 * ac_
        t[base.shape[0]:, :, 2] += dz
        tq, _ = Fr.resolve_tensor(t, sg)
        sdp = tq / np.maximum(np.linalg.norm(tq, axis=1), 1e-300)[:, None]
        rw = mhs.interaction_matrix(
            tris=t, material=mhs.Material(mu=mu_c, lam=mu_c), eps=0.1 * hc,
            receiver=Kmod.RECEIVER, source=Kmod.SOURCE, obs_tris=t)
        sm = Kmod.to_source_major(rw)
        out = (Kmod.project(sm, sdp, sdp), Kmod.project_normal(sm, sdp))
        del rw, sm
        return out


    Pp, Np_ = pair(+0.25)
    Pm, Nm_ = pair(-0.25)
    check("shear transfer is EVEN in the stepping sense",
          float(np.abs(Pp - Pm).max() / np.abs(Pp).max()), 1e-9)
    check("normal transfer is ODD in the stepping sense",
          float(np.abs(Np_ + Nm_).max() / np.abs(Np_).max()), 1e-6)
    # In-plane slip induces no normal traction on a COPLANAR element, so the
    # coupling cannot act on a single flat crack at all: it lives entirely
    # between non-parallel elements. That is also why the coupled operator
    # keeps its P-matrix property on this geometry.
    check("coplanar slip induces no normal traction on itself",
          float(np.abs(np.diagonal(Np_)).max() / np.abs(Np_).max()), 1e-15)

    # AND THE COUPLED OPERATOR MUST STAY A P-MATRIX, or `solve` loses the
    # uniqueness every claim in this module rests on. Letting the operator
    # depend on the solution is exactly when bifurcation becomes possible, so
    # it is checked rather than assumed -- on a network of mixed orientations
    # it is an open question.
    print(f"         min eig sym(-P) uncoupled "
          f"{np.linalg.eigvalsh(-0.5 * (Pp + Pp.T))[0]:+.4f}")
    for muf in (0.3, 0.6):
        Pe = Pp + muf * Np_
        em = float(np.linalg.eigvalsh(-0.5 * (Pe + Pe.T))[0])
        check_true(f"the coupled operator is a P-matrix at mu_f {muf}",
                   em > 0.0 and float(Pe.diagonal().max()) < 0.0,
                   f"min eig {em:+.4f}, max diag {Pe.diagonal().max():+.3e}")

    # WHICH STEP IS EXTENSIONAL IS MEASURED, NEVER NAMED. Left and right
    # stepping are a chirality, so the labelling is read off the step's own
    # induced sigma_n instead of mapped onto the classical map view by hand.
    # Two facts have to hold or the labelling means nothing: the two senses
    # must be OPPOSITE at a given offset (the mirror symmetry again), and the
    # sign must FLIP with offset -- which is why the linkage asymmetry's
    # reversal is largely a relabelling rather than a change of physics.
    print("       [steps] which step is extensional, measured not named")
    gap_s = 2.2 * ac_
    matl_s = mhs.Material(mu=mu_c, lam=mu_c)

    def step_sign(dz):
        t = np.concatenate([base, base.copy()])
        t[base.shape[0]:, :, 0] += gap_s
        t[base.shape[0]:, :, 2] += dz
        tq, sn_s = Fr.resolve_tensor(t, sg - 0.01 * np.eye(3))
        mg = np.linalg.norm(tq, axis=1)
        sdl = tq / np.maximum(mg, 1e-300)[:, None]
        rw = mhs.interaction_matrix(
            tris=t, material=matl_s, eps=0.1 * hc, receiver=Kmod.RECEIVER,
            source=Kmod.SOURCE, obs_tris=t)
        smj = Kmod.to_source_major(rw)
        Pl = Kmod.project(smj, sdl, sdl) + 0.6 * Kmod.project_normal(smj, sdl)
        del rw, smj
        so = St.solve(Pl, mg, 0.6 * sn_s, 1.0, max_sweeps=20000)
        del Pl
        # a grid in the gap BETWEEN the facing tips
        xs = np.linspace(ac_ + 0.15 * (gap_s - 2 * ac_),
                         gap_s - ac_ - 0.15 * (gap_s - 2 * ac_), 4)
        zs = (np.linspace(0.25 * dz, 0.75 * dz, 4) if abs(dz) > 1e-9
              else np.zeros(1))
        X, Y, Z = np.meshgrid(xs, np.linspace(-0.2 * ac_, 0.2 * ac_, 4),
                              zs, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), -20.0 + Z.ravel()], axis=1)
        return float(Fr.induced_normal_stress(
            t, Fr.slip_cartesian(t, sdl, so.delta), pts,
            [0.0, 0.0, 1.0], matl_s, 0.1 * hc).mean())

    near, far = 0.18 * ac_, 0.60 * ac_        # straddling the crossing
    s_np, s_nm = step_sign(+near), step_sign(-near)
    s_fp, s_fm = step_sign(+far), step_sign(-far)
    print(f"         dz/a 0.18: {s_np:+.3e} / {s_nm:+.3e}   "
          f"dz/a 0.60: {s_fp:+.3e} / {s_fm:+.3e}")
    check("the two senses' step stress is equal and opposite (near)",
          abs(s_np + s_nm) / max(abs(s_np), 1e-300), 0.05)
    check("the two senses' step stress is equal and opposite (far)",
          abs(s_fp + s_fm) / max(abs(s_fp), 1e-300), 0.05)
    check_true("the step's own sigma_n FLIPS sign with offset",
               s_np * s_fp < 0.0,
               f"{s_np:+.3e} at dz/a 0.18 against {s_fp:+.3e} at 0.60")

    # AND THE EXTENSIONAL STEP MAKES THE FACING FRONT MORE COMPETITIVE, which
    # is the quantity that decides whether two cracks link. The growth path
    # itself is far too expensive to gate -- twenty increments on two cracks
    # -- but the mechanism is one `front_gains` call: the gain on the front
    # FACING the neighbour, against the gain on the front pointing away. The
    # extensional sense must raise that ratio. Gated as an ORDERING because
    # the ratio's own value is threshold-sensitive, which is exactly what
    # made a four-increment run read the path as insensitive.
    print("       [link] does the extensional step favour the facing front?")
    amax_l = 0.40
    latt = N.disc_mesh(amax_l, hc, depth=-20.0)
    gap_l = 2.2 * ac_

    def facing_ratio(dz):
        t = np.concatenate([latt, latt.copy()])
        t[latt.shape[0]:, :, 0] += gap_l
        t[latt.shape[0]:, :, 2] += dz
        nl = latt.shape[0]
        cn = t.mean(axis=1)
        tq, sn_l = Fr.resolve_tensor(t, sg - 0.01 * np.eye(3))
        mg = np.linalg.norm(tq, axis=1)
        sdl = tq / np.maximum(mg, 1e-300)[:, None]
        rw = mhs.interaction_matrix(
            tris=t, material=matl_s, eps=0.1 * hc, receiver=Kmod.RECEIVER,
            source=Kmod.SOURCE, obs_tris=t)
        smj = Kmod.to_source_major(rw)
        Pl = Kmod.project(smj, sdl, sdl) + 0.6 * Kmod.project_normal(smj, sdl)
        del rw, smj
        arl = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0],
                                            t[:, 2] - t[:, 0]), axis=1)
        onl = np.concatenate([
            np.hypot(cn[:nl, 0], cn[:nl, 1]) <= ac_,
            np.hypot(cn[nl:, 0] - gap_l, cn[nl:, 1]) <= ac_])
        cd, gn, _, _ = Gr.front_gains(Pl, mg, 0.6 * sn_l, arl, cn, onl,
                                      pitch=hc, max_sweeps=20000, arc=5)
        del Pl
        isa = cd < nl
        ph = np.degrees(np.arctan2(cn[cd[isa], 1], cn[cd[isa], 0]))
        ga = gn[isa]
        fc = np.abs(ph) < 35.0                       # front facing the other
        aw = np.abs(np.abs(ph) - 180.0) < 35.0       # front pointing away
        return float(ga[fc].mean() / ga[aw].mean())

    r_con, r_ext = facing_ratio(+0.60 * ac_), facing_ratio(-0.60 * ac_)
    print(f"         facing/away gain: {r_con:.4f} at +dz against "
          f"{r_ext:.4f} at -dz")
    check_true("the extensional step favours the facing front",
               r_ext > r_con,
               f"{r_ext:.4f} extensional against {r_con:.4f} contractional")

    # ---- the remote load relaxing as slip accumulates ---------------------
    # EVERY RESULT BEFORE THIS ONE HELD THE REMOTE STRESS FIXED, so a crack's
    # slip never reduced what drove it and G grew without bound with length:
    # a disc's front beats the same patch in the far field by 4.67, 9.34 and
    # 13.42 at radii 3, 6, 9 h, linear in L. `load_feedback` makes the
    # reservoir finite. Its properties are closed form, so they are gated as
    # identities and not as tolerances to be tuned.
    print("       [load] the remote load relaxing as slip accumulates")
    mu_l, dtau_l = 30.0, 0.01
    trl = N.disc_mesh(0.30, 0.075, depth=-20.0)
    sigl = dtau_l * (np.outer([1.0, 0, 0], [0, 0, 1.0])
                     + np.outer([0, 0, 1.0], [1.0, 0, 0]))
    taul, _ = Fr.resolve_tensor(trl, sigl)
    sdl = taul / np.maximum(np.linalg.norm(taul, axis=1), 1e-300)[:, None]
    sxyz = Fr.slip_cartesian(trl, sdl, np.ones(trl.shape[0]))
    nrml = mhs.tdcs.unit_normals(trl)
    arl = 0.5 * np.linalg.norm(np.cross(trl[:, 1] - trl[:, 0],
                                        trl[:, 2] - trl[:, 0]), axis=1)
    vol = 1.0
    dP, dN = St.load_feedback(sxyz, nrml, arl, mu=mu_l, volume=vol)

    # C : m = 2 mu m holds ONLY because slip is tangential and m traceless.
    # A normal component would leave the lambda term alive and the operator
    # silently wrong, so this is checked and not assumed.
    check("slip is tangential, so m is traceless",
          float(np.abs((sxyz * nrml).sum(axis=1)).max()), 1e-12)
    # The self term has a closed form: m : m = 1/2 for a unit tangential slip.
    check("the self term is exactly -mu A / V",
          float(np.abs(np.diag(dP) / (-mu_l * arl / vol) - 1.0).max()),
          1e-12)
    # Tangential slip induces no normal stress on its own plane, so dN's
    # diagonal must vanish identically -- the same fact that keeps the
    # coplanar coupling term at 8e-24 above.
    check("tangential slip clamps nothing on its own plane",
          float(np.abs(np.diag(dN)).max()), 1e-18)
    # dP/A symmetric is what makes the energy a quadratic form at all.
    qa = dP / arl[:, None]
    check("dP / A is symmetric", float(np.abs(qa - qa.T).max()), 1e-12)
    # AND THE ENERGY IT ADDS IS THE ELASTIC ENERGY OF THE AVERAGE STRAIN,
    # (mu/V) |sum_i A_i d_i m_i|^2, which is the derivation in one line.
    rngl = np.random.default_rng(5)
    dl = rngl.random(trl.shape[0])
    mten = np.zeros((3, 3))
    for i in range(trl.shape[0]):
        mten += arl[i] * dl[i] * 0.5 * (np.outer(sxyz[i], nrml[i])
                                        + np.outer(nrml[i], sxyz[i]))
    check("the energy it adds is (mu/V) |sum A d m|^2",
          abs(float(-0.5 * (arl * (dP.T @ dl) * dl).sum())
              / (mu_l / vol * float((mten * mten).sum())) - 1.0), 1e-12)
    # THE CORRECTION MUST NOT BREAK THE LCP: -dP/A positive semidefinite keeps
    # `solve`'s uniqueness argument intact rather than merely probably intact.
    check_true("it leaves -P more definite, not less",
               float(np.linalg.eigvalsh(0.5 * (-qa - qa.T)).min()) > -1e-9,
               f"min eig {np.linalg.eigvalsh(0.5 * (-qa - qa.T)).min():+.3e}")
    # THE RANK IS AT MOST 5 -- the m span the traceless symmetrics -- and a
    # PLANAR crack realises 1, since every element shares one m. Both ends are
    # checked because the bound alone would pass on an operator that is
    # accidentally zero. The mixed set below reaches 4 rather than 5: one
    # theta through `kink_candidates` rotates each edge about its OWN signed
    # axis and so yields several normals, not two, but not enough of them to
    # span the whole traceless space. 60 isotropically oriented cells do reach
    # 5, measured, which is what shows the bound is the real one.
    rk = np.linalg.matrix_rank(dP, tol=1e-9 * float(np.abs(dP).max()))
    check_true("a planar crack's feedback is rank 1", rk == 1, f"rank {rk}")
    mix = np.concatenate([trl, Gr.kink_candidates(
        Gr.front_edges(trl, np.ones(trl.shape[0], bool)), math.radians(35.0))])
    tm, _ = Fr.resolve_tensor(mix, sigl)
    sdm = tm / np.maximum(np.linalg.norm(tm, axis=1), 1e-300)[:, None]
    am = 0.5 * np.linalg.norm(np.cross(mix[:, 1] - mix[:, 0],
                                       mix[:, 2] - mix[:, 0]), axis=1)
    dPm, _ = St.load_feedback(
        Fr.slip_cartesian(mix, sdm, np.ones(mix.shape[0])),
        mhs.tdcs.unit_normals(mix), am, mu=mu_l, volume=vol)
    rkm = np.linalg.matrix_rank(dPm, tol=1e-9 * float(np.abs(dPm).max()))
    check_true("mixed orientations raise the rank, within the bound",
               1 < rkm <= 5,
               f"rank {rkm} on {mix.shape[0]} elements, bound 5")
    # AND IT VANISHES AS 1/V EXACTLY, which is the statement that the
    # infinite reservoir of every earlier commit is this operator's own limit
    # and not a separate code path. Gated as the identity rather than as "it
    # gets small at some large V", which would pass on any decaying error.
    ten, _ = St.load_feedback(sxyz, nrml, arl, mu=mu_l, volume=10.0 * vol)
    check("the feedback scales exactly as 1 / V",
          float(np.abs(ten * 10.0 / dP - 1.0).max()), 1e-15)


def c_spectrum() -> None:
    """Soft modes: the limits are known, so they are checked rather than eyeballed.

    The claim the modes carry is that the lowest eigenvector of ``sym(-P)``
    picks out the mutually-promoting subset -- the structure the network would
    localize onto. These pin the machinery that claim rests on.
    """
    print("\n[spectrum] soft modes of the elastic operator")

    # [a] NO INTERACTION, NO SOFT MODES. A diagonal operator has every
    # eigenvalue equal to the self-stiffness, so there is nothing to localize
    # onto. This is the control for the whole idea: if soft modes survived
    # here they would not be an interaction effect.
    n = 40
    d = np.full(n, 3.0)
    Pd = -np.diag(d)
    w, V = Sp.soft_modes(Pd, 5)
    check("a diagonal operator has no soft end",
          float(w.max() - w.min()), 1e-10)
    check("its eigenvalue is the self-stiffness", abs(float(w[0]) - 3.0), 1e-10)

    # [b] A MUTUALLY PROMOTING BLOCK IS SOFT, and softer the larger it is.
    # Promoting means P_ij > 0, hence -P_ij < 0: a cooperative off-diagonal.
    # The all-ones vector on such a block is its softest pattern, with
    # eigenvalue d - (m-1)c, which is checked in closed form.
    soft = []
    for m in (2, 4, 8):
        P = -np.diag(np.full(n, 3.0))
        c = 0.2
        for i in range(m):
            for j in range(m):
                if i != j:
                    P[i, j] = c                      # promoting
        w, V = Sp.soft_modes(P, 3)
        want = 3.0 - (m - 1) * c
        check(f"promoting block of {m}: eigenvalue matches d-(m-1)c",
              abs(float(w[0]) - want), 1e-9)
        soft.append(float(w[0]))
        # the softest mode must live ON the block, and all of one sign
        v = V[:, 0]
        on = float((v[:m] ** 2).sum())
        check_true(f"promoting block of {m}: mode sits on the block",
                   on > 0.99, f"{on:.4f} of the energy")
        st = Sp.measure(w, V, _FakeNet(n))[0]
        check_true(f"promoting block of {m}: elements slip TOGETHER",
                   st.sign_coherence > 0.99,
                   f"sign coherence {st.sign_coherence:.4f}")
        check_true(f"promoting block of {m}: n_effective ~ {m}",
                   abs(st.n_effective - m) < 0.15 * m,
                   f"n_eff {st.n_effective:.2f}")
    check_true("a bigger promoting block is softer",
               soft[0] > soft[1] > soft[2],
               " > ".join(f"{v:.3f}" for v in soft))

    # [c] SHADOWING SOFTENS A DIFFERENT MODE, and this is the check that
    # sharpened the claim. A first version asserted that a shadowing block is
    # simply stiff; it is not. With positive diagonal d and off-diagonal
    # -P_ij = +c, the block's eigenvalues are d + (m-1)c for the all-together
    # pattern and d - c for the m-1 patterns in which elements slip AGAINST
    # each other. So shadowing does produce modes below d -- just oscillatory
    # ones, and only by c rather than by (m-1)c.
    #
    # The physically meaningful separation is therefore SIGN COHERENCE, not
    # the eigenvalue alone: a localizing structure is one whose soft mode has
    # its elements slipping TOGETHER. That is what these three checks pin.
    m, c = 4, 0.2
    Pp = -np.diag(np.full(n, 3.0))
    Ps = -np.diag(np.full(n, 3.0))
    for i in range(m):
        for j in range(m):
            if i != j:
                Pp[i, j] = c                         # promoting
                Ps[i, j] = -c                        # shadowing
    wp, Vp = Sp.soft_modes(Pp, 3)
    ws, Vs = Sp.soft_modes(Ps, 3)
    sp = Sp.measure(wp, Vp, _FakeNet(n))[0]
    ss = Sp.measure(ws, Vs, _FakeNet(n))[0]
    check_true("shadowing's softest mode is NOT cooperative",
               ss.sign_coherence < 0.2,
               f"sign coherence {ss.sign_coherence:.4f} against promoting's "
               f"{sp.sign_coherence:.4f}")
    check_true("promoting is softer than shadowing at equal |c|",
               float(wp[0]) < float(ws[0]) - 1e-9,
               f"{float(wp[0]):.3f} < {float(ws[0]):.3f}")
    # The all-together pattern on a shadowing block is STIFFER than uncoupled,
    # which is the statement "shadowing resists collective slip".
    ones = np.zeros(n); ones[:m] = 1.0 / np.sqrt(m)
    stiff = float(ones @ (-0.5 * (Ps + Ps.T)) @ ones)
    check_true("shadowing makes the all-together pattern STIFFER",
               stiff > 3.0 + 1e-9,
               f"{stiff:.3f} against the uncoupled 3.0")

    # [d] The quadratic form sees only the symmetric part, which is why a
    # non-symmetric P is still legitimately handled.
    rng = np.random.default_rng(0)
    A = rng.normal(size=(12, 12))
    x = rng.normal(size=12)
    check("v^T A v equals v^T sym(A) v",
          abs(float(x @ A @ x) - float(x @ (0.5 * (A + A.T)) @ x)), 1e-10)


class _FakeNet:
    """Minimal stand-in: measure() needs centroids and fault ids only."""

    def __init__(self, n):
        self.n_elements = n
        self.n_faults = n
        self.fault_id = np.arange(n)
        self.tris = np.zeros((n, 3, 3))
        self.tris[:, :, 0] = np.arange(n)[:, None]


def c_slip() -> None:
    """The quasi-static cascade: equilibrium, finiteness, and rule robustness."""
    print("\n[slip] quasi-static Coulomb loading")
    net = N.build_network(seed=5, n_elements_target=250, centers="uniform",
                          l_min=2.0, l_max=6.0, h=1.0, box_xy=60.0,
                          z_top=-2.0, z_bottom=-30.0)
    with tempfile.TemporaryDirectory() as tmp:
        K, _ = Kmod.assemble(net, workers=1, cache_dir=tmp, verbose=False)
    fric = Fr.build_friction(net, drop_ratio=0.9)
    P = Kmod.project(K, fric.slip_dir, fric.slip_dir)
    d, g_s, g_d = Csc.coefficients(fric)
    lam_max = 4.0 * float(np.nanmin(fric.first_failure_lambda()))

    sets, slips, stats = {}, {}, {}
    for order, seed in (("sequential", 0), ("random", 1), ("random", 2),
                        ("sync", 0)):
        st, ev = Csc.run_history(net, fric, P, lam_max=lam_max, order=order,
                                 seed=seed)
        key = f"{order}{seed if order == 'random' else ''}"
        sets[key] = st.s > 0.0
        slips[key] = st.s
        sz = np.array([e.n_slipped for e in ev])
        stats[key] = (len(ev), float(sz.mean()), int(sz.max()),
                      float(st.s.sum()), all(e.converged for e in ev))
        print(f"       {key:12s} {len(ev):5d} events  mean size "
              f"{sz.mean():5.3f}  max {sz.max():3d}  "
              f"active {int(sets[key].sum()):4d}  "
              f"total slip {st.s.sum():.4f} km")

    ref = "sequential"
    st, ev = Csc.run_history(net, fric, P, lam_max=lam_max)
    check_true("every event converged",
               all(e.converged for e in ev), f"{len(ev)} events")
    check_true("every event moved at least one element",
               all(e.n_slipped >= 1 for e in ev),
               f"min {min(e.n_slipped for e in ev)}")
    check_true("cascades are genuinely multi-element",
               max(e.n_slipped for e in ev) > 1,
               f"largest event {max(e.n_slipped for e in ev)} elements")
    check("equilibrium: max(tau - static strength) <= 0",
          Csc.equilibrium_residual(st, d, g_s), 0.0)
    check_true("cumulative slip is non-negative",
               bool((st.s >= 0.0).all()), f"min {st.s.min():.3e}")
    check_true("released moment is positive",
               all(e.moment_Nm > 0.0 for e in ev),
               f"total {sum(e.moment_Nm for e in ev):.3e} N m")

    # THE ROBUSTNESS CLAIM the dimension result depends on.
    worst_j, worst_slip = 1.0, 0.0
    for key, act in sets.items():
        if key == ref:
            continue
        inter = float((sets[ref] & act).sum())
        union = float((sets[ref] | act).sum())
        worst_j = min(worst_j, inter / union)
        worst_slip = max(worst_slip, float(
            np.abs(slips[ref] - slips[key]).max() / slips[ref].max()))
    check("active set is identical under every update rule",
          worst_j, MIN_ACTIVE_SET_JACCARD - 1e-12, below=False)
    print(f"       worst per-element slip difference across rules "
          f"{worst_slip:.2e} of the maximum")
    spread = max(abs(stats[k][0] / stats[ref][0] - 1.0) for k in stats)
    print(f"       event COUNT spread across rules {spread:.1%} -- not "
          f"robust, and reported as such")

    # TRIPWIRE: no strength drop means no mechanism, and must be refused.
    try:
        Fr.build_friction(net, drop_ratio=1.0)
        caught = False
    except ValueError:
        caught = True
    check_true("tripwire: drop_ratio = 1 is REFUSED", caught,
               "without weakening, quasi-static slip is continuous "
               "and there are no events")


def c_align() -> None:
    """The alignment statistics, against answers known in closed form.

    None of these existed in either repository, so each is pinned to a limit it
    must reproduce rather than to a number someone looked at once.
    """
    print("\n[align] coherence, localization, shape, clustering")
    rng = np.random.default_rng(0)

    # Coherence is EXACTLY 1 when every element pulls the same way.
    n = 400
    nrm = np.repeat(np.array([[0.3, 0.4, math.sqrt(0.75)]]), n, axis=0)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    cen = rng.uniform(-10, 10, (n, 3))
    cen[:, 2] = -20.0
    tris = N.oriented_triangle(cen, nrm, 1.0)
    Mt = M.moment_tensors(tris, np.ones(n), np.tile([1.0, 0.0], (n, 1)))
    check("coherence of an identical set is 1",
          abs(M.coherence(Mt) - 1.0), TOL_COHERENCE_EXACT)

    # And falls as 1/sqrt(N) for random orientations, which is the floor any
    # claim of emergent coherence has to clear.
    worst = 1.0
    for nn in (100, 1000):
        vals = []
        for k in range(6):
            r2 = np.random.default_rng(500 + k)
            nr = N.isotropic_normals(r2, nn)
            cc = r2.uniform(-10, 10, (nn, 3))
            cc[:, 2] = -20.0
            ang = r2.uniform(0.0, 2 * math.pi, nn)
            sd = np.stack([np.cos(ang), np.sin(ang)], axis=1)
            vals.append(M.coherence(M.moment_tensors(
                N.oriented_triangle(cc, nr, 1.0), np.ones(nn), sd)))
        ratio = float(np.mean(vals)) * math.sqrt(nn)
        print(f"       N {nn:5d}: C {np.mean(vals):.4f}  "
              f"x sqrt(N) = {ratio:.3f}")
        worst = min(worst, min(ratio, 1.0 / ratio))
    check("random coherence scales as 1/sqrt(N)",
          1.0 / worst, MAX_COHERENCE_SCALING_FACTOR)

    # Participation ratio: the baseline is lam/(lam+1), NOT 1.
    pts = rng.uniform(0.0, 10.0, (4000, 3))
    pr = M.participation_ratio(pts, np.ones(4000), 1.0)
    lam = 4000.0 / np.unique(np.floor(pts).astype(np.int64), axis=0).shape[0]
    want = lam / (lam + 1.0)
    print(f"       PR of a random set {pr:.4f}, Poisson predicts "
          f"lam/(lam+1) = {want:.4f} at lam = {lam:.2f}")
    check("PR of a random set matches the Poisson baseline",
          abs(pr - want), TOL_PR_POISSON)
    inside = pts[np.all((pts > 0.0) & (pts < 1.0), axis=1)]
    ncell = np.unique(np.floor(pts).astype(np.int64), axis=0).shape[0]
    pr1 = M.participation_ratio(inside, np.ones(len(inside)), 1.0,
                                available=pts)
    check("PR of a one-cell set is 1/N_cells",
          abs(pr1 - 1.0 / ncell) * ncell, 0.05)

    # Shape indices separate a plane from a line, which a single flatness
    # number would not.
    pl = np.c_[rng.uniform(-1, 1, 1500), rng.uniform(-1, 1, 1500),
               np.full(1500, 1e-9)]
    ln = np.c_[rng.uniform(-1, 1, 1500), np.full(1500, 1e-9),
               np.full(1500, 1e-9)]
    bl = rng.normal(size=(1500, 3))
    p_pl, l_pl = M.shape_indices(pl)
    p_ln, l_ln = M.shape_indices(ln)
    p_bl, l_bl = M.shape_indices(bl)
    print(f"       plane ({p_pl:.3f}, {l_pl:.3f})  line ({p_ln:.3f}, "
          f"{l_ln:.3f})  blob ({p_bl:.3f}, {l_bl:.3f})")
    check("a plane reads planar", 1.0 - p_pl, 0.10)
    check("a line reads linear and NOT planar", max(1.0 - l_ln, p_ln), 0.10)
    check("a blob reads neither", max(p_bl, l_bl), 0.15)

    # Clustering.
    a = rng.normal(0.0, 0.3, (200, 3))
    b = rng.normal(0.0, 0.3, (200, 3)) + 20.0
    both = np.vstack([a, b])
    check_true("two separated blobs give two clusters",
               int(M.connected_clusters(both, 1.0).max()) + 1 == 2)
    check_true("one blob gives one cluster",
               int(M.connected_clusters(a, 1.0).max()) + 1 == 1)
    check_true("a reach spanning both gives one",
               int(M.connected_clusters(both, 50.0).max()) + 1 == 1)

    # Slip dispersion.
    check("aligned slip has zero dispersion",
          M.slip_dispersion(np.repeat([[1.0, 0.0, 0.0]], 500, axis=0)), 1e-12)
    check("random slip has dispersion near 1",
          1.0 - M.slip_dispersion(rng.normal(size=(2000, 3))), 0.15)


def c_cloud() -> None:
    """The dense cloud generator: isotropy, orientation, and the two samplers."""
    print("\n[cloud] a dense cloud of independently-oriented patches")
    rng = np.random.default_rng(0)

    iso = N.isotropic_normals(rng, 100000)
    S = N.fabric(iso)
    print(f"       isotropic_normals  S {np.round(S, 4)}")
    check("isotropic_normals is isotropic", abs(S[0] - 1.0 / 3.0), 0.005)

    # The truncated dip range is ANISOTROPIC, and by a measured amount. Gated
    # as a difference between the two samplers rather than a failure of one:
    # the truncation is a legitimate modelling choice that the fault network
    # makes deliberately, and only the cloud needs full isotropy.
    for lo, hi, want in ((30.0, 90.0, 0.020), (0.0, 90.0, 0.166)):
        d = np.radians(rng.uniform(lo, hi, 100000))
        t = np.radians(rng.uniform(0.0, 360.0, 100000))
        nr = np.stack([-np.sin(t) * np.sin(d), np.cos(t) * np.sin(d),
                       np.cos(d)], axis=1)
        got = abs(N.fabric(nr)[0] - 1.0 / 3.0)
        print(f"       dip U[{lo:.0f},{hi:.0f}]       "
              f"|S1-1/3| {got:.4f} (measured {want})")
        check(f"dip U[{lo:.0f},{hi:.0f}] anisotropy is {want}",
              abs(got - want), 0.01)

    net = N.build_cloud(seed=4, n_elements=1500, h=0.5, spacing_over_h=1.5)
    e1 = net.tris[:, 1] - net.tris[:, 0]
    e2 = net.tris[:, 2] - net.tris[:, 0]
    mn = np.cross(e1, e2)
    mn /= np.linalg.norm(mn, axis=1, keepdims=True)
    up = mhs.tdcs.unit_normals(net.tris)
    check_true("every mesh normal points UP and matches the tdcs frame",
               bool((mn[:, 2] >= -1e-12).all())
               and bool((np.einsum("ij,ij->i", mn, up) > 0.999999).all()),
               f"min n_z {mn[:, 2].min():+.2e}")
    check_true("every vertex is below the free surface",
               bool((net.tris[:, :, 2] < 0.0).all()),
               f"max z {net.tris[:, :, 2].max():.4f} km")
    ar = net.areas()
    check("all elements have the same area",
          float(ar.std() / ar.mean()), 1e-12)
    again = N.build_cloud(seed=4, n_elements=1500, h=0.5, spacing_over_h=1.5)
    check_true("the cloud is deterministic for a seed",
               bool(np.array_equal(net.tris, again.tris)))
    # The Poisson nearest-neighbour factor, which is why spacing_over_h is a
    # density and not a neighbour distance.
    nn = net.meta["nearest_neighbour_km"]
    ratio = nn["mean"] / net.meta["spacing_km"]
    print(f"       nearest neighbour / spacing {ratio:.3f} "
          f"(Poisson predicts 0.554)")
    check("nearest-neighbour spacing matches Poisson", abs(ratio - 0.554), 0.06)

    # ---- orientation spread, and the published payload's encodings --------
    # SUBPARALLEL IS WHAT BUYS THE COUPLING DENSITY COULD NOT. B counts only
    # the positive transfer, so differently-oriented neighbours cancel; at 2000
    # elements and s/h = 0.65 the budget runs 0.581 isotropic to 0.762 parallel
    # while the eigenvalue margin falls 21.8 to 6.2, which is why the shipped
    # config sits at 20 degrees and not at 0.
    mn = np.array([0.2669, 0.0, 0.9637])
    s_tight = N.fabric(N.subparallel_normals(rng, 40000, mn, 5.0))
    s_wide = N.fabric(N.subparallel_normals(rng, 40000, mn, 45.0))
    s_iso = N.fabric(N.isotropic_normals(rng, 40000))
    print(f"       S1 at spread 5 / 45 / isotropic: {s_tight[0]:.4f} "
          f"{s_wide[0]:.4f} {s_iso[0]:.4f}")
    check_true("a tight spread is nearly parallel", s_tight[0] > 0.98,
               f"S1 {s_tight[0]:.4f}")
    check_true("the spread orders the fabric", s_tight[0] > s_wide[0] > s_iso[0],
               f"{s_tight[0]:.4f} > {s_wide[0]:.4f} > {s_iso[0]:.4f}")
    # Spread 0 must be EXACTLY the mean direction, or "subparallel" is a
    # distribution whose own limit it does not reach.
    # Against the NORMALISED mean, since the function normalises and the
    # literal above has norm 0.99998 -- comparing to the raw vector measures
    # that rounding and nothing else.
    check("spread 0 is exactly the mean normal",
          float(np.abs(N.subparallel_normals(rng, 100, mn, 0.0)
                       - mn / np.linalg.norm(mn)).max()), 1e-12)

    # THE PUBLISHED PAYLOAD'S THREE RAMPS, each held to its own bound. The log
    # ramp is for slip, which spans decades and reserves 0 for "never
    # slipped"; planarity is linear on [0, 1] where 0 is MEASURED; and the
    # active-against-null comparison is labels, where arithmetic between codes
    # is meaningless. Pushing the labels down the linear ramp lands them on
    # bytes 0, 126 and 253 and decodes them to 0, 0.996 and 2.0 -- distinct but
    # inexact -- which is why they get identity instead.
    import payload as Pl                                      # noqa: PLC0415
    lin = rng.random(5000)
    bl, ml = Pl.encode_linear_u8(lin, 0.0, 1.0)
    gl, bd = Pl.roundtrip(lin, bl, ml)
    check_true("the linear ramp round trips inside its bound",
               gl <= Pl.ROUNDTRIP_MARGIN * bd,
               f"{gl:.3e} against {bd:.3e}")
    cats = np.array([0, 1, 2] * 300)
    bc, mc = Pl.encode_category_u8(cats, ["a", "b", "c"])
    gc, _ = Pl.roundtrip(cats, bc, mc)
    check_true("labels survive the category encoding exactly", gc == 0.0,
               f"worst departure {gc}")
    check_true("and keep their own code values",
               sorted(set(np.frombuffer(bc, np.uint8).tolist())) == [0, 1, 2],
               f"{sorted(set(np.frombuffer(bc, np.uint8).tolist()))}")
    # The failure this replaced, kept as a tripwire: a ramp would NOT be exact.
    bw, mw = Pl.encode_linear_u8(cats.astype(float), 0.0, 2.0)
    check_true("a ramp would have been inexact for labels",
               float(np.abs(Pl.decode_linear_u8(bw, mw) - cats).max()) > 1e-6,
               f"{float(np.abs(Pl.decode_linear_u8(bw, mw) - cats).max()):.3e}")


# ----------------------------------------------------------------- [rough] ---
def c_rough() -> None:
    """The rough CONTIGUOUS crack: the tiling survives, and the flat crack's
    coupling obeys a sum rule that gives the sweep an analytic baseline."""
    print("\n[rough] a contiguous crack with displaced shared vertices")
    h, a, depth = 0.5, 4.0, -20.0
    s30, c30 = math.sin(math.radians(30.0)), math.cos(math.radians(30.0))
    nrm = (s30, 0.0, c30)

    # -- the vertex table ----------------------------------------------------
    flat = N.disc_mesh(a, h, depth=depth, normal=nrm)
    counts = {tol: N.weld_vertices(flat, tol * h)[0].shape[0]
              for tol in (1e-12, 1e-9, 1e-6)}
    print(f"       weld counts by tolerance {counts}")
    check_true("welding is tolerance independent over six decades",
               len(set(counts.values())) == 1, f"{counts}")
    # A TOLERANCE IS NEEDED AT ALL: exact comparison over-counts, because
    # disc_mesh reaches a shared vertex by two arithmetic paths.
    exact = np.unique(flat.reshape(-1, 3), axis=0).shape[0]
    check_true("and exact comparison would have over-counted",
               exact > counts[1e-9], f"{exact} exact against {counts[1e-9]}")

    # -- the flat limit, to MACHINE PRECISION, not bit-identically -----------
    r0 = N.rough_disc(a, h, rms_slope=0.0, corr_over_h=8.0, depth=depth,
                      normal=nrm)
    gap = float(np.abs(r0.tris - flat).max())
    check(f"rms_slope=0 reproduces disc_mesh ({gap:.1e} km)", gap / h, 1e-13)

    # -- contiguity, which the whole experiment rests on ---------------------
    for sl in (0.0, 0.1, 0.25, 0.5):
        r = N.rough_disc(a, h, rms_slope=sl, corr_over_h=8.0, depth=depth,
                         normal=nrm, seed=1)
        share = N.edge_sharing(r.idx)
        nv = np.cross(r.tris[:, 1] - r.tris[:, 0], r.tris[:, 2] - r.tris[:, 0])
        nv /= np.linalg.norm(nv, axis=1, keepdims=True)
        n_e = int(share[1] + share[2])
        euler = r.verts.shape[0] - n_e + r.tris.shape[0]
        check_true(f"slope {sl}: no edge shared by three triangles",
                   len(share) <= 3, f"sharing histogram {list(share)}")
        check_true(f"slope {sl}: Euler V-E+F = 1 (a disc)", euler == 1,
                   f"{r.verts.shape[0]} - {n_e} + {r.tris.shape[0]} = {euler}")
        check_true(f"slope {sl}: every vertex stays below z = 0",
                   bool(np.all(r.verts[:, 2] < 0.0)),
                   f"max z {r.verts[:, 2].max():.3f} km")
        # The measured failure mode is a POSITIVE self-interaction while
        # tdcs.unit_normals still reports an innocent n_z, so the normal alone
        # is not a sufficient check -- but a hemisphere flip is what causes it.
        check_true(f"slope {sl}: all normals in one hemisphere",
                   bool(np.all(nv[:, 2] > 0.0)),
                   f"min n_z {nv[:, 2].min():.3f}")
        check(f"slope {sl}: realized rms slope is the requested one",
              abs(r.slope_rms - sl), 1e-12)

    # The calibration is a single exact rescale, so the sweep's x-axis must not
    # move with the seed -- the fiction smooth_field's docstring records.
    got = [N.rough_disc(a, h, rms_slope=0.2, corr_over_h=8.0, depth=depth,
                        normal=nrm, seed=sd).slope_rms for sd in range(1, 6)]
    check("realized slope is seed independent", float(np.std(got)), 1e-14)

    # -- the guards ----------------------------------------------------------
    for kw, why in ((dict(rms_slope=0.1, corr_over_h=2.0), "below Nyquist"),
                    (dict(rms_slope=0.1, corr_over_h=40.0), "longer than 2a"),
                    (dict(rms_slope=-0.1, corr_over_h=8.0), "negative slope")):
        try:
            N.rough_disc(a, h, depth=depth, normal=nrm, **kw)
            check_true(f"rough_disc refuses a roughness {why}", False,
                       "NOT REFUSED")
        except ValueError:
            check_true(f"rough_disc refuses a roughness {why}", True)

    # -- THE SUM RULE, which is the sweep's analytic baseline ----------------
    # Uniform slip on an infinite plane is a rigid translation and produces no
    # stress, so sum_j P_ij = 0 and the triggering budget is B = 1 exactly.
    # A finite crack loses a perimeter fraction out its free edge, measured as
    # B = 1 - 0.40 h/a. The consequence worth gating is the SIGN STRUCTURE:
    # the diagonal is negative, every interior off-diagonal is positive, and
    # so a coplanar contiguous crack has NO STRESS SHADOW AT ALL.
    net = N.Network(tris=flat, fault_id=np.zeros(flat.shape[0], np.int64),
                    length=np.array([0.0]), width=np.array([0.0]),
                    azimuth=np.array([0.0]), dip=np.array([0.0]),
                    center=flat.reshape(-1, 3).mean(axis=0)[None, :],
                    h=h, eps=0.1 * h, meta=dict(kind="disc"))
    mat, _ = Kmod.assemble(net, mu=30.0, lam=30.0, workers=1, verbose=False)
    fric = Fr.build_friction(net, regime="thrust", drop_ratio=0.9,
                             mean_stress_ratio=0.6667)
    P = Kmod.project(mat, fric.slip_dir, fric.slip_dir)
    del mat
    cen = flat.mean(axis=1)
    deep = np.linalg.norm(cen - cen.mean(axis=0), axis=1) < 0.5 * a
    dg = np.abs(np.diag(P))
    off = P - np.diag(np.diag(P))
    neg = float((np.minimum(off, 0.0).sum(axis=1) / dg)[deep].max())
    rowsum = float((P.sum(axis=1) / dg)[deep].mean())
    B = Kmod.branching_ratio(P)
    print(f"       interior rowsum/|P_ii| {rowsum:+.5f}, B {B[deep].mean():.5f}"
          f", deficit*a/h {-rowsum * a / h:.3f}")
    check_true("the self-interaction diagonal is negative",
               bool(np.all(np.diag(P) < 0.0)),
               f"worst {float(np.diag(P).max()):.4e}")
    check("a coplanar contiguous crack has NO stress shadow", abs(neg), 1e-12)
    check("B = 1 + sum_j P_ij/|P_ii| exactly",
          float(np.abs(B[deep] - (1.0 + (P.sum(axis=1) / dg)[deep])).max()),
          1e-12)

    # -- region_concentration's exact limits, and the weighting trap ---------
    # Both limits, because the statistic is the sweep's headline and neither
    # was gated when it was written.
    pts = np.array([[0.1, 0.1, -1.0], [0.2, 0.2, -1.0], [0.15, 0.1, -1.0]])
    check("all weight in one cell gives exactly 1 domain",
          abs(M.region_concentration(pts, np.ones(3), 1.0) - 1.0), 1e-12)
    m = 5
    spread = np.stack([2.0 * np.arange(m), np.zeros(m), -np.ones(m)], axis=1)
    check(f"weight spread over {m} cells gives exactly {m}",
          abs(M.region_concentration(spread, np.ones(m), 1.0) - m), 1e-12)
    # THE TRAP, kept as a tripwire: a varying weight against a FLAT weight is
    # not a null, it is a weighting artefact. Comparing the same positions
    # under the two weightings must give different answers, so the two can
    # never be quoted as though interchangeable.
    w_vary = np.array([100.0, 1.0, 1.0, 1.0, 1.0])
    check_true("a varying weight is NOT interchangeable with a flat one",
               abs(M.region_concentration(spread, w_vary, 1.0)
                   - M.region_concentration(spread, np.ones(m), 1.0)) > 1.0,
               f"{M.region_concentration(spread, w_vary, 1.0):.3f} against "
               f"{M.region_concentration(spread, np.ones(m), 1.0):.3f}")


# ------------------------------------------------------------------ [iso] ---
def c_iso() -> None:
    """The isosurface pipeline against a closed form: a level set whose area,
    radius and normals are known, and a round trip through the GLB writer."""
    print("\n[iso] contouring, normals and the glTF writer")
    import isosurf as Iso                                    # noqa: PLC0415

    # An ELLIPSOID, not a sphere. A sphere centred in a cubic grid is symmetric
    # under swapping two axes, so the index-order check below would pass on a
    # transposed grid and prove nothing -- measured: 1.7e-7 either way.
    n, h, R = 41, 0.05, 0.5
    ax = h * np.arange(n)
    c = np.full(3, ax.mean())
    abc = np.array([1.0, 0.72, 0.5])
    K, J, I = np.meshgrid(ax, ax, ax, indexing="ij")          # x fastest
    q = np.stack([(I - c[0]) / abc[0], (J - c[1]) / abc[1], (K - c[2]) / abc[2]])
    f = np.sqrt((q ** 2).sum(0)).ravel()
    dims = np.array([n, n, n])
    got = Iso.contour(f, dims, h, R)
    pts, tri, nrm = got["points"], got["triangles"], got["normals"]
    check_true("the level set is non-empty", tri.shape[0] > 100,
               f"{tri.shape[0]} triangles, {pts.shape[0]} vertices")

    # -- geometry against the closed form ------------------------------------
    rr = np.sqrt((((got["raw_points"] - c) / abc) ** 2).sum(1))
    check("every vertex sits on the level surface", float(np.abs(rr - R).max() / R),
          1e-2)
    # Through surface_extent, because that is the area the manifest PUBLISHES
    # as the kernel discriminator; a second copy of the arithmetic here would
    # check nothing about the number that ships.
    area, pieces = Iso.surface_extent(got["polydata"])
    sm = (abc * R) ** 1.6075                     # Thomsen, p = 1.6075
    want = 4 * np.pi * ((sm[0] * sm[1] + sm[0] * sm[2] + sm[1] * sm[2]) / 3
                        ) ** (1 / 1.6075)
    check(f"surface area ({area:.4f} against {want:.4f})",
          abs(area / want - 1.0), 2e-2)
    check_true("one closed ellipsoid is one piece", pieces == 1, f"{pieces}")

    # -- the unbounded-node fill ---------------------------------------------
    # The classical kernel hands back inf at a node that lands on an element
    # edge (one in 203,391 on the published detail grid), and marching cubes
    # must still put that node on a side of the level. The fill takes the
    # largest-magnitude finite face neighbour, so what is checked is that it
    # picks exactly that, that it leaves a clean grid alone, and that it refuses
    # a neighbourhood it cannot resolve rather than guessing through it.
    same, nf0, _ = Iso.fill_nonfinite(f, dims)
    check_true("a finite grid is returned unchanged", nf0 == 0
               and np.array_equal(same, f), f"{nf0} filled")
    blown = f.reshape(n, n, n).copy()
    iz, iy, ix = n // 3, n // 2, n // 4
    nb = [blown[iz + 1, iy, ix], blown[iz - 1, iy, ix], blown[iz, iy + 1, ix],
          blown[iz, iy - 1, ix], blown[iz, iy, ix + 1], blown[iz, iy, ix - 1]]
    want_fill = max(nb, key=abs)
    blown[iz, iy, ix] = np.inf
    filled, nf1, mag = Iso.fill_nonfinite(blown.ravel(), dims)
    check_true("one unbounded node is filled, and only it", nf1 == 1
               and np.isfinite(filled).all(), f"{nf1} filled")
    check("and it takes the most extreme finite neighbour",
          abs(filled.reshape(n, n, n)[iz, iy, ix] - want_fill), 1e-12)
    check("and reports the magnitude it used", abs(mag - abs(want_fill)), 1e-12)
    walled = f.reshape(n, n, n).copy()
    walled[iz - 1:iz + 2, iy - 1:iy + 2, ix - 1:ix + 2] = np.nan
    try:
        Iso.fill_nonfinite(walled.ravel(), dims)
        caught = False
    except SystemExit:
        caught = True
    check_true("and it REFUSES a node with no finite neighbour", caught)

    # -- the tube radius, on a cylinder whose radius is known ----------------
    # This is the number that tests r = A / L on the detail scene, so it is
    # pinned on geometry with an answer: points scattered on a cylinder of
    # radius r0 about a tilted segment must measure r0. The last case is the
    # bias the estimator exists to avoid -- one stray vertex hugging an
    # ENDPOINT, where several edges meet on the real mesh, moves a plain
    # minimum by an order of magnitude and must not move the radius at all.
    rng = np.random.default_rng(3)
    ends = np.array([[0.11, -0.20, 0.30], [0.60, 0.44, 1.05]])
    ax = ends[1] - ends[0]
    e1 = np.cross(ax, [0.0, 0.0, 1.0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1); e2 /= np.linalg.norm(e2)
    r0 = 0.037
    tt = rng.uniform(0.0, 1.0, 6000)
    th = rng.uniform(0.0, 2 * np.pi, 6000)
    cyl = (ends[0] + tt[:, None] * ax
           + r0 * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2))
    rad, close = Iso.tube_radius(cyl, ends)
    check(f"tube radius on a known cylinder ({rad:.4f} for {r0})",
          abs(rad / r0 - 1.0), 2e-2)
    far = np.vstack([cyl, ends[0] + 0.5 * ax + 9 * r0 * e1])
    check("and a distant vertex does not move it",
          abs(Iso.tube_radius(far, ends)[0] / rad - 1.0), 1e-12)
    spike = np.vstack([cyl, ends[0] + 0.1 * r0 * e1])
    sr, sc = Iso.tube_radius(spike, ends)
    check("and one hugging an endpoint does not either",
          abs(sr / rad - 1.0), 1e-12)
    check_true("though it DOES move a plain minimum", sc < 0.2 * close,
               f"{sc:.5f} against {close:.5f}")

    # -- the level check, and that it can fail -------------------------------
    err = Iso.level_error(got["raw_points"], f, dims, h, R)
    check("the contour is at the level it claims", err, 1e-4)
    for name, g in (("y<->z", f.reshape(n, n, n).transpose(1, 0, 2).ravel()),
                    ("x<->z", f.reshape(n, n, n).transpose(2, 1, 0).ravel())):
        bad = Iso.level_error(got["raw_points"], g, dims, h, R)
        check_true(f"and it CATCHES a {name} transposed grid", bad > 0.1,
                   f"{bad:.3f} against {err:.1e}")
    bad = Iso.level_error(got["raw_points"], f, dims, h * 1.1, R)
    check_true("and a 10 % error in the spacing", bad > 0.1, f"{bad:.3f}")

    # -- normals: unit, and along -grad f ------------------------------------
    # The failure this pins: where the level passes exactly through a grid node,
    # flying edges emits coincident vertices and zero-area triangles, and the
    # normal filter hands back a ZERO normal -- 84 of 1064 vertices here before
    # the cleaning pass, which in a lit scene is black speckle.
    check("normals are unit length",
          float(np.abs(np.linalg.norm(nrm, axis=1) - 1.0).max()), 1e-4)
    grad = (pts - c) / abc ** 2
    grad /= np.linalg.norm(grad, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(
        -np.einsum("ij,ij->i", nrm, grad), -1.0, 1.0)))
    check(f"and point along -grad f (worst {ang.max():.2f} deg)",
          float(ang.max()), 10.0)

    # -- decimation reports what it cost -------------------------------------
    small = Iso.contour(f, dims, h, R, max_tris=300)
    check_true("decimation reaches its budget",
               small["triangles"].shape[0] <= 320,
               f"{small['n_raw']} -> {small['triangles'].shape[0]}")
    check(f"and its deviation is measured "
          f"({small['deviation_km'] / h:.3f} cells)",
          small["deviation_km"] / h, 1.0)

    # -- the glTF writer round trips -----------------------------------------
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "iso.glb"
        nb = Iso.write_glb(path, [dict(name="iso", points=pts, normals=nrm,
                                       triangles=tri, color=(0.5, 0.5, 0.5))])
        back = Iso.read_glb(path)["meshes"][0]
        check_true(f"glTF round trip ({nb} bytes)",
                   back["POSITION"].shape[0] == pts.shape[0]
                   and back["INDICES"].shape[0] == tri.shape[0]
                   and np.array_equal(back["INDICES"], tri),
                   f"{back['POSITION'].shape[0]} verts, "
                   f"{back['INDICES'].shape[0]} triangles")
        # float32 is exact for these magnitudes, so this is equality, not a
        # tolerance: a drift here would be the writer's offsets, not rounding.
        check("positions survive the writer exactly",
              float(np.abs(back["POSITION"] - pts.astype(np.float32)).max()),
              1e-12)
        check("and so do the normals",
              float(np.abs(back["NORMAL"] - nrm.astype(np.float32)).max()), 1e-12)


CLAUSES = {"estimator": c_estimator, "subset": c_subset,
           "network": c_network, "budget": c_budget, "cascade": c_cascade,
           "kernel": c_kernel, "crack": c_crack, "load": c_load,
           "static": c_static,
           "spectrum": c_spectrum, "slip": c_slip, "align": c_align,
           "cloud": c_cloud, "rough": c_rough, "iso": c_iso}


def main() -> bool:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-k", "--only", default=None,
                    help=f"one clause: {', '.join(CLAUSES)}")
    a = ap.parse_args()
    run = ({a.only: CLAUSES[a.only]} if a.only else CLAUSES)
    if a.only and a.only not in CLAUSES:
        raise SystemExit(f"no clause {a.only!r}; have {', '.join(CLAUSES)}")
    print("active_fractal selfcheck")
    print(f"window: r = {CAL_WINDOW['r_lo']}-{CAL_WINDOW['r_hi']} km "
          f"({math.log10(CAL_WINDOW['r_hi'] / CAL_WINDOW['r_lo']):.2f} decades)"
          f", {M.N_ORIGINS} grid phases")
    for fn in run.values():
        fn()
    bad = [n for n, ok, _ in CHECKS if not ok]
    print("-" * 70)
    if bad:
        for n in bad:
            print(f"  failed: {n}")
        print(f"FAIL: {len(bad)} of {len(CHECKS)} checks")
        return False
    print(f"PASS: active_fractal selfcheck ({len(CHECKS)} checks)")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
