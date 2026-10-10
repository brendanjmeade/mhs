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


CLAUSES = {"estimator": c_estimator, "subset": c_subset,
           "network": c_network, "budget": c_budget, "cascade": c_cascade,
           "kernel": c_kernel, "slip": c_slip}


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
