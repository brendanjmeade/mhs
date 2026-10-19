#!/usr/bin/env python
"""Stick-slip loading histories on ONE rough fault: the on-fault page's A.

    python studies/active_fractal/avalanche.py assemble A --workers 12
    python studies/active_fractal/avalanche.py run A [--spread 0.02] [--coupled]
    python studies/active_fractal/avalanche.py lcp A
    python studies/active_fractal/avalanche.py figure

THE QUESTION. The three-fault system of the on-fault page interacts through
a shadow: no fault's onset moves, the neighbours only throttle completion
(``onfault.py thresholds``). Local, positive transfer -- one element's slip
bringing its neighbours to failure -- needs no second fault: a single
near-planar crack under a uniform load is a critical branching process by
geometry (the interior row sums of its self-stress operator nearly vanish,
``kernel.branching_ratio`` 0.97-0.99 on every variant here), so one rough
disc is the avalanche system. The history is ``cascade.run_history``'s: an
element reaching static strength slips to dynamic strength and re-locks,
events are differences of equilibria, there is no time. The static LCP at
the two strengths brackets what the history activates (``static.solve``).

THE VARIANTS. ``A`` is the page's fault alone, on the page's mesh (one-fault
spec, so no junction constraint); ``A_rough`` and ``A_corr2`` change the
roughness; ``stadium`` pulls A apart along strike by an 18 km run
(``system.outline``'s stadium); ``A_h*`` is the RESOLUTION LADDER, the same
fault at h 0.35, 0.25, 0.18 and 0.13 km. Every size here is in elements and
the slip of one failure is the strength drop over a self-stiffness that
scales like 1/h, so the ladder is what decides whether the exponent and the
cutoff belong to the fault or to the mesh. On the ladder the taper belt is
held at the page's 1.5 km in KILOMETRES (``TAPER_KM``), not at 6 h: the
near-planar belt is what fails first, and a belt that widened with h would
change the fault between rungs. At h = 0.25 that is 6 h, the page's mesh.

THE LCP BESIDE IT. ``lcp`` scans the load with the static LCP (one strength,
no drop, one solution at each load) at the static and at the dynamic
strength, recording each element's activation load and the slip, so the
history's first-failure map has its path-free counterpart: the history lies
between the two LCPs element by element, in activation load and in slip
(measured on A: 3748 of 3748 both ways), and the LCP has no events because
its slip is continuous in the load.

Outputs go to ``out/avalanche_<tag>.npz`` (one per run, gitignored) and the
figures to ``website/src/figures/`` for the site's /avalanche/ page. One
spawn pool per process (``assemble``), BLAS pinned in the parent first.
"""
from __future__ import annotations

import os
import sys

_CMD = sys.argv[1] if len(sys.argv) > 1 else ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1" if _CMD == "assemble" else "8")

import argparse                                              # noqa: E402
import json                                                  # noqa: E402
import pathlib                                               # noqa: E402
import time                                                  # noqa: E402

import numpy as np                                           # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cascade as C                                          # noqa: E402
import friction as Fr                                        # noqa: E402
import kernel as K                                           # noqa: E402
import onfault as OF                                         # noqa: E402
import static as St                                          # noqa: E402
import system as Sy                                          # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
FIGURES = HERE.parents[1] / "website" / "src" / "figures"

#: The page's taper belt, in km: 6 h at the page's h. Fixed in km across the
#: resolution ladder (module docstring).
TAPER_KM = Sy.TAPER_OVER_H * OF.H_KM

#: Events of at least this many elements keep their footprint.
BIG = 10

#: The size the power-law fit starts at, and the load beyond which the
#: history is steady (the first event at lam0 is the taper belt, a third
#: of the fault; the transient is over by 1.15 lam0 on every variant).
S_MIN = 10
STEADY_OVER_LAM0 = 1.15

A_SPEC = OF.FAULTS[0]                          # a 6, dip 30, trace (0, 0), v_c 4

VARIANTS = {
    "A":       dict(h=OF.H_KM, slope=OF.SLOPE, corr=OF.CORR_KM, run=0.0),
    "A_h0.35": dict(h=0.35, slope=OF.SLOPE, corr=OF.CORR_KM, run=0.0),
    "A_h0.18": dict(h=0.18, slope=OF.SLOPE, corr=OF.CORR_KM, run=0.0),
    "A_h0.13": dict(h=0.13, slope=OF.SLOPE, corr=OF.CORR_KM, run=0.0),
    "A_rough": dict(h=OF.H_KM, slope=0.40, corr=OF.CORR_KM, run=0.0),
    "A_corr2": dict(h=OF.H_KM, slope=OF.SLOPE, corr=2.0, run=0.0),
    "stadium": dict(h=OF.H_KM, slope=OF.SLOPE, corr=OF.CORR_KM, run=18.0),
}
LADDER = ("A_h0.35", "A", "A_h0.18", "A_h0.13")
SHAPES = ("A", "A_rough", "A_corr2", "stadium")
LABELS = {"A": "A, as on the page", "A_rough": "A, rougher (slope 0.40)",
          "A_corr2": "A, finer roughness (corr 2 km)",
          "stadium": "stadium: A pulled 18 km apart"}
SHORT = {"A": "A", "A_rough": "A rougher", "A_corr2": "A finer", "stadium": "stadium"}


# ---------------------------------------------------------------- build --
def build(name: str):
    """``(net, record)`` for a variant, through ``system.build_system`` with
    the page's gates; the record holds the in-plane tiling for the maps."""
    v = VARIANTS[name]
    spec = dict(A_SPEC, name=name, run=v["run"])
    records, net = Sy.build_system((spec,), v["h"], rms_slope=v["slope"],
                                   corr=v["corr"], eps=0.1 * v["h"],
                                   min_nz=OF.MIN_NZ,
                                   clearance_over_h=OF.CLEARANCE_OVER_H,
                                   taper_over_h=TAPER_KM / v["h"])
    OF.frame_gate(net.tris)
    r = records[0]
    return net, dict(uv=r["uv"], idx=r["idx"], w=r["w"],
                     meta=dict(net.meta["faults"][0], h=v["h"],
                               taper_km=TAPER_KM, min_nz=net.meta["min_nz"]))


def describe(name: str, net, rec) -> None:
    m = rec["meta"]
    z = net.tris[:, :, 2]
    print(f"  {name:8s} {net.n_elements:6d} elements, "
          f"{int((z.max(axis=1) >= -1e-12).sum()):4d} at the trace, h {m['h']} km, "
          f"median edge {m['median_edge_km']:.3f} km, tilt rms "
          f"{m['tilt_rms_deg']:.1f} deg, height rms {m['height_rms_km']:.3f} km, "
          f"slope rms {m['slope_rms']:.3f}, min n_z {m['min_nz']:.3f}, area "
          f"{net.areas().sum():.1f} km^2", flush=True)


def cmd_assemble(a) -> int:
    net, rec = build(a.name)
    describe(a.name, net, rec)
    t0 = time.time()
    Km, info = K.assemble(net, mu=OF.MU, lam=OF.LAM, workers=a.workers,
                          verbose=True)
    print(f"  {a.name}: matrix {Km.shape} ready in {time.time() - t0:.0f} s "
          f"(key {info['key']})", flush=True)
    return 0


# ------------------------------------------------------------------ run --
class Recorder:
    """What ``cascade.run_history`` does not keep: who slipped first and at
    what load, how often each element failed, the trigger of every event,
    the elements new to each event, and the footprint of every event of
    ``BIG`` elements or more."""

    def __init__(self, n: int):
        self.lam_first = np.full(n, np.nan)
        self.count = np.zeros(n, np.int64)
        self.n_new, self.trigger = [], []
        self.foot_idx, self.foot_off, self.foot_k = [], [0], []

    def __call__(self, k, lam, j, moved, ds):
        new = moved & np.isnan(self.lam_first)
        self.lam_first[new] = lam
        self.count[moved] += 1
        self.n_new.append(int(new.sum()))
        self.trigger.append(int(j))
        if int(moved.sum()) >= BIG:
            idx = np.flatnonzero(moved)
            self.foot_idx.append(idx)
            self.foot_off.append(self.foot_off[-1] + idx.size)
            self.foot_k.append(k - 1)

    def footprint(self):
        idx = (np.concatenate(self.foot_idx) if self.foot_idx
               else np.zeros(0, np.int64))
        return idx, np.array(self.foot_off), np.array(self.foot_k)


def cmd_run(a) -> int:
    tag = (a.name + (f"_spread{a.spread:g}" if a.spread > 0 else "")
           + ("_coupled" if a.coupled else ""))
    print(f"=== {tag}", flush=True)
    net, rec = build(a.name)
    describe(a.name, net, rec)
    n = net.n_elements
    areas = net.areas()
    Km, _ = K.assemble(net, mu=OF.MU, lam=OF.LAM, workers=1, verbose=True)
    fric = Fr.build_friction(
        net, regime=OF.REGIME, drop_ratio=OF.DROP_RATIO,
        mean_stress_ratio=OF.MEAN_STRESS,
        mu_mode="per_element" if a.spread > 0 else "uniform",
        mu_spread=a.spread, seed=0)
    d, g_s, g_d = C.coefficients(fric)
    P = K.project(Km, fric.slip_dir, fric.slip_dir)
    if a.coupled:
        Nrm = K.project_normal(Km, fric.slip_dir)
        P, d2, g2 = St.coulomb_coupled(P, Nrm, fric)
        if not (np.allclose(d2, d) and np.allclose(g2, g_s)):
            raise SystemExit("coulomb_coupled changed the load coefficients")
        del Nrm
    del Km
    diag = P[np.arange(n), np.arange(n)]
    if diag.max() >= 0.0:
        raise SystemExit(f"{tag}: a non-negative self-term ({diag.max():.3e})")
    B = K.branching_ratio(P)
    mineig = float(np.linalg.eigvalsh(-0.5 * (P + P.T)).min())
    ff = fric.first_failure_lambda()              # the remote load alone
    lam0 = float(np.nanmin(ff))
    print(f"  {n} elements; lam0 {lam0:.5f} GPa; B mean {B.mean():.3f} median "
          f"{np.median(B):.3f}; min eig sym(-P) {mineig:+.3f}; drop "
          f"{(g_s - g_d).mean() * 1e3:.1f} MPa; g/d p50 "
          f"{np.nanpercentile(ff, 50) / lam0:.3f} p95 "
          f"{np.nanpercentile(ff, 95) / lam0:.3f} lam0, never "
          f"{int(np.isinf(ff).sum())}", flush=True)
    recorder = Recorder(n)
    t0 = time.time()
    state, events = C.run_history(net, fric, P, shear_modulus=OF.MU,
                                  lam_max=a.lam_max * lam0,
                                  max_events=a.max_events, order=a.order,
                                  verbose=True, record=recorder)
    dt = time.time() - t0
    runaway = not events[-1].converged
    ev = dict(lam=np.array([e.lam for e in events]),
              inc=np.array([e.lam_increment for e in events]),
              n=np.array([e.n_slipped for e in events]),
              n_new=np.array(recorder.n_new),
              moment=np.array([e.moment_Nm for e in events]),
              max_slip=np.array([e.max_slip_km for e in events]),
              sweeps=np.array([e.sweeps for e in events]),
              trigger=np.array(recorder.trigger))
    foot = recorder.footprint()
    count, lam_first = recorder.count, recorder.lam_first
    ever = count > 0
    sol_s = St.solve(P, d, g_s, state.lam, warm=state.s)
    sol_d = St.solve(P, d, g_d, state.lam, warm=state.s)
    adv = (ff - lam_first) / lam0          # > 0: failed EARLIER than alone
    fin = np.isfinite(adv)
    early = fin & (adv > 1e-9)
    late = fin & (adv < -1e-9)
    sizes = ev["n"]
    steady = ev["lam"] > STEADY_OVER_LAM0 * lam0
    alpha, n_tail = power_law_alpha(sizes[steady])
    big = np.flatnonzero(sizes >= 0.25 * n)
    gaps = np.diff(ev["lam"][big]) / lam0
    summary = dict(
        tag=tag, name=a.name, spread=a.spread, coupled=a.coupled, n=n,
        h=VARIANTS[a.name]["h"], area_km2=float(areas.sum()), lam0=lam0,
        lam_end=float(state.lam), lam_max_over_lam0=a.lam_max,
        runaway=runaway, seconds=dt, B_mean=float(B.mean()),
        B_median=float(np.median(B)), mineig=mineig,
        n_events=int(sizes.size), total_slips=int(sizes.sum()),
        mean_size=float(sizes.mean()), max_size=int(sizes.max()),
        max_size_frac=float(sizes.max() / n),
        transfer_fraction=float((sizes - 1).sum() / sizes.sum()),
        n_ge_10=int((sizes >= 10).sum()), n_ge_100=int((sizes >= 100).sum()),
        n_fault_scale=int(big.size),
        gap_mean=float(gaps.mean()) if gaps.size else float("nan"),
        gap_cv=float(gaps.std() / gaps.mean()) if gaps.size > 1 else float("nan"),
        steady_events=int(steady.sum()), alpha=alpha, n_tail=n_tail,
        n_early=int(early.sum()), n_late=int(late.sum()), n_first=int(fin.sum()),
        early_median_lam0=float(np.median(adv[early])) if early.any() else 0.0,
        early_p90_lam0=float(np.percentile(adv[early], 90)) if early.any() else 0.0,
        total_moment_Nm=float(ev["moment"].sum()),
        max_event_moment_Nm=float(ev["moment"].max()),
        failures_per_element=float(count.mean()),
        static_active=[sol_s.n_active, sol_d.n_active],
        bracket_violations=[int((sol_s.active & ~ever).sum()),
                            int((ever & ~sol_d.active).sum())],
        static_converged=[bool(sol_s.converged), bool(sol_d.converged)],
        meta=rec["meta"])
    print("  " + json.dumps({k: v for k, v in summary.items() if k != "meta"},
                            default=float), flush=True)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"avalanche_{tag}.npz"
    np.savez_compressed(path, tris=net.tris, uv=rec["uv"], idx=rec["idx"],
                        areas=areas, lam0=lam0, lam_first=lam_first, ff=ff,
                        count=count, slip=state.s, B=B,
                        static_s=sol_s.delta, static_d=sol_d.delta,
                        foot_idx=foot[0], foot_off=foot[1], foot_k=foot[2],
                        summary=json.dumps(summary, default=float),
                        **{"ev_" + k: v for k, v in ev.items()})
    print(f"  saved {path} ({dt:.0f} s of history)", flush=True)
    return 0


def power_law_alpha(sizes, s_min: int = S_MIN) -> tuple[float, int]:
    """Discrete power-law MLE (Clauset, Shalizi and Newman 2009) for
    ``P(s) ~ s^-alpha`` on ``s >= s_min``: ``(alpha, n_tail)``."""
    s = np.asarray(sizes, float)
    s = s[s >= s_min]
    if s.size < 10:
        return float("nan"), int(s.size)
    return 1.0 + s.size / np.log(s / (s_min - 0.5)).sum(), int(s.size)


# --------------------------------------------------------------- figure --
def load(path):
    z = np.load(path, allow_pickle=False)
    return z, json.loads(str(z["summary"]))


def ccdf(x):
    x = np.sort(np.asarray(x, float))
    return x, 1.0 - np.arange(x.size) / x.size


def footprint_aspect(z, s, s_min: int = 100) -> float:
    """Median along-strike / down-dip extent of the events of ``s_min`` or
    more elements (nan when there are none)."""
    n = z["ev_n"]
    cen = z["uv"][z["idx"]].mean(axis=1)
    fk, fo, fi = z["foot_k"], z["foot_off"], z["foot_idx"]
    asp = []
    for w, k in enumerate(fk):
        if n[k] >= s_min:
            els = fi[fo[w]:fo[w + 1]]
            asp.append(np.ptp(cen[els, 0]) / max(np.ptp(cen[els, 1]), 1e-9))
    return float(np.median(asp)) if asp else float("nan")


COLS = ["tag", "n", "h", "B_mean", "mineig", "n_events", "total_slips",
        "failures_per_element", "transfer_fraction", "mean_size", "max_size",
        "max_size_frac", "n_ge_100", "n_fault_scale", "gap_mean", "gap_cv",
        "alpha", "n_tail", "n_early", "n_late", "early_median_lam0",
        "early_p90_lam0", "max_event_moment_Nm", "bracket_violations",
        "runaway", "aspect"]


def table(runs) -> list[dict]:
    rows = []
    for z, s in runs:
        row = {c: s.get(c) for c in COLS}
        row["aspect"] = footprint_aspect(z, s)
        rows.append(row)
    print("\t".join(COLS))
    for row in rows:
        print("\t".join(f"{v:.4g}" if isinstance(v, float) else str(v)
                        for v in row.values()))
    return rows


def _style(ax, *, log_x=False, log_y=False):
    ax.tick_params(direction="out", length=3, width=0.8)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")


def _letter(ax, s):
    ax.text(0.97, 0.95, s, transform=ax.transAxes, ha="right", va="top")


def _decades(ax, axis, lo, hi):
    ticks = [10.0 ** k for k in range(int(np.floor(np.log10(lo))),
                                     int(np.ceil(np.log10(hi))) + 1)]
    (ax.set_xticks if axis == "x" else ax.set_yticks)([ticks[0], ticks[-1]])
    (ax.set_xticks if axis == "x" else ax.set_yticks)(ticks, minor=True)
    (ax.set_xlim if axis == "x" else ax.set_ylim)(ticks[0], ticks[-1])


SHAPE_COL = {"A": "#1f77b4", "A_rough": "#ff7f0e", "A_corr2": "#2ca02c",
             "stadium": "#9467bd"}


def stats_figure(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = {s["name"]: (z, s) for z, s in runs
            if s["spread"] == 0 and not s["coupled"] and s["name"] in SHAPES}
    names = [nm for nm in SHAPES if nm in base]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6),
                             gridspec_kw=dict(wspace=0.32, hspace=0.42))
    for j, nm in enumerate(names):
        z, s = base[nm]
        ax = axes[0, j]
        lam = z["ev_lam"] / s["lam0"]
        n = z["ev_n"]
        ax.scatter(lam, n, s=2 + 8 * (n >= 10), c=SHAPE_COL[nm], alpha=0.5, lw=0)
        ax.axhline(s["n"], color="0.5", lw=0.6, ls="--")
        _style(ax, log_y=True)
        ax.set_ylim(0.8, 2.0e4)
        ax.set_yticks([1, 1e2, 1e4])
        ax.set_xlim(1.0, 2.0)
        ax.set_xticks([1.0, 1.5, 2.0])
        ax.set_title(f"{LABELS[nm]}\n{s['n']} elements, $B$ = {s['B_mean']:.3f}",
                     fontsize=8)
        ax.set_xlabel(r"$\lambda / \lambda_0$")
        _letter(ax, "abcd"[j])
    axes[0, 0].set_ylabel("$s$ (elements in one event)")
    for j in range(len(names), 4):
        axes[0, j].set_visible(False)

    ax = axes[1, 0]
    for nm in names:
        z, s = base[nm]
        lam = z["ev_lam"] / s["lam0"]
        x, y = ccdf(z["ev_n"][lam > STEADY_OVER_LAM0])
        ax.step(x, y, where="post", color=SHAPE_COL[nm], lw=1.4,
                label=f"{nm}, $\\alpha$ = {s['alpha']:.2f}")
    xg = np.array([10.0, 1000.0])
    ax.plot(xg, 0.3 * (xg / 10.0) ** -0.6, color="0.4", lw=0.8, ls=":")
    _style(ax, log_x=True, log_y=True)
    _decades(ax, "x", 1, 1e4)
    _decades(ax, "y", 1e-5, 1)
    ax.set_xlabel("$s$ (elements)")
    ax.set_ylabel(r"$P(\geq s)$, $\lambda > 1.15\,\lambda_0$")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    _letter(ax, "e")

    ax = axes[1, 1]
    for nm in names:
        z, s = base[nm]
        adv = (z["ff"] - z["lam_first"]) / s["lam0"]
        fin = np.isfinite(adv)
        x, y = ccdf(adv[fin & (adv > 0)])
        ax.plot(x, y * (fin & (adv > 0)).sum() / fin.sum(), color=SHAPE_COL[nm],
                lw=1.4, label=f"{nm}: {100 * s['n_early'] / s['n_first']:.0f} % early")
    _style(ax, log_x=True)
    _decades(ax, "x", 1e-4, 10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"first failure earlier than alone, $\Delta\lambda / \lambda_0$")
    ax.set_ylabel("fraction of elements")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    _letter(ax, "f")

    ax = axes[1, 2]
    for nm in names:
        z, s = base[nm]
        lam = z["ev_lam"] / s["lam0"]
        ax.plot(lam, np.cumsum(z["ev_n_new"]) / s["n"], color=SHAPE_COL[nm],
                lw=1.4, label=nm)
    _style(ax)
    ax.set_xlim(1.0, 1.6)
    ax.set_xticks([1.0, 1.3, 1.6])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"$\lambda / \lambda_0$")
    ax.set_ylabel("fraction of the fault that has slipped")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    _letter(ax, "g")

    ax = axes[1, 3]
    for nm in names:
        z, s = base[nm]
        lam = z["ev_lam"] / s["lam0"]
        x, y = ccdf(z["ev_moment"][lam > STEADY_OVER_LAM0])
        ax.step(x, y, where="post", color=SHAPE_COL[nm], lw=1.4, label=nm)
    _style(ax, log_x=True, log_y=True)
    _decades(ax, "x", 1e13, 1e18)
    _decades(ax, "y", 1e-5, 1)
    ax.set_xlabel("$M_0$ (N m)")
    ax.set_ylabel(r"$P(\geq M_0)$, $\lambda > 1.15\,\lambda_0$")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    _letter(ax, "h")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def resolution_figure(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rungs = {s["name"]: (z, s) for z, s in runs
             if s["spread"] == 0 and not s["coupled"] and s["name"] in LADDER}
    names = [nm for nm in LADDER if nm in rungs]
    if len(names) < 2:
        print("  resolution figure skipped: fewer than two rungs")
        return
    hs = [rungs[nm][1]["h"] for nm in names]
    cmap = plt.get_cmap("viridis")
    col = {nm: cmap(0.1 + 0.8 * i / max(len(names) - 1, 1))
           for i, nm in enumerate(names)}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4),
                             gridspec_kw=dict(wspace=0.36))
    ax = axes[0]
    for nm in names:
        z, s = rungs[nm]
        lam = z["ev_lam"] / s["lam0"]
        x, y = ccdf(z["ev_n"][lam > STEADY_OVER_LAM0])
        ax.step(x, y, where="post", color=col[nm], lw=1.4,
                label=f"$h$ = {s['h']:.2f} km, {s['n']} elements")
    _style(ax, log_x=True, log_y=True)
    _decades(ax, "x", 1, 1e4)
    _decades(ax, "y", 1e-5, 1)
    ax.set_xlabel("$s$ (elements)")
    ax.set_ylabel(r"$P(\geq s)$, $\lambda > 1.15\,\lambda_0$")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    _letter(ax, "a")
    ax = axes[1]
    for nm in names:
        z, s = rungs[nm]
        lam = z["ev_lam"] / s["lam0"]
        area = z["ev_n"] * s["area_km2"] / s["n"]
        x, y = ccdf(area[lam > STEADY_OVER_LAM0])
        ax.step(x, y, where="post", color=col[nm], lw=1.4)
    ax.axvline(rungs[names[0]][1]["area_km2"], color="0.5", lw=0.6, ls="--")
    _style(ax, log_x=True, log_y=True)
    _decades(ax, "x", 1e-2, 1e3)
    _decades(ax, "y", 1e-5, 1)
    ax.set_xlabel("event area (km$^2$)")
    ax.set_ylabel(r"$P(\geq$ area$)$")
    _letter(ax, "b")
    ax = axes[2]
    for nm in names:
        z, s = rungs[nm]
        lam = z["ev_lam"] / s["lam0"]
        x, y = ccdf(z["ev_moment"][lam > STEADY_OVER_LAM0])
        ax.step(x, y, where="post", color=col[nm], lw=1.4)
    _style(ax, log_x=True, log_y=True)
    _decades(ax, "x", 1e13, 1e18)
    _decades(ax, "y", 1e-5, 1)
    ax.set_xlabel("$M_0$ (N m)")
    ax.set_ylabel(r"$P(\geq M_0)$")
    _letter(ax, "c")
    ax = axes[3]
    al = [rungs[nm][1]["alpha"] for nm in names]
    mx = [rungs[nm][1]["max_size_frac"] for nm in names]
    Bm = [rungs[nm][1]["B_mean"] for nm in names]
    fe = [rungs[nm][1]["failures_per_element"] for nm in names]
    ax.plot(hs, al, "o-", color="#1f77b4", lw=1, ms=5, label=r"exponent $\alpha$")
    ax.plot(hs, mx, "s-", color="#ff7f0e", lw=1, ms=5, label="largest event / fault")
    ax.plot(hs, Bm, "^-", color="#2ca02c", lw=1, ms=5, label="$B$")
    ax.plot(hs, np.array(fe) / 1000.0, "d-", color="#9467bd", lw=1, ms=5,
            label="failures per element / 1000")
    _style(ax)
    ax.set_xlim(0.1, 0.4)
    ax.set_xticks([0.1, 0.25, 0.4])
    ax.set_ylim(0, 2.5)
    ax.set_yticks([0, 1, 2])
    ax.set_xlabel("$h$ (km)")
    ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=2)
    _letter(ax, "d")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def maps_figure(runs, out, n_big: int = 3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    base = {s["name"]: (z, s) for z, s in runs
            if s["spread"] == 0 and not s["coupled"] and s["name"] in SHAPES}
    discs = [nm for nm in SHAPES if nm in base and nm != "stadium"]
    stad = ["stadium"] if "stadium" in base else []
    nrow = (1 if discs else 0) + (1 if stad else 0)
    fig = plt.figure(figsize=(13, 4.4 * nrow))
    gs = fig.add_gridspec(nrow, max(len(discs), 1), hspace=0.02, wspace=0.08)
    axes = [fig.add_subplot(gs[0, j]) for j in range(len(discs))]
    if stad:
        axes.append(fig.add_subplot(gs[nrow - 1, :]))
    pc = None
    for i, (ax, nm) in enumerate(zip(axes, discs + stad)):
        z, s = base[nm]
        uv, idx = z["uv"], z["idx"]
        lf = z["lam_first"] / s["lam0"]
        polys = uv[idx]
        pc = PolyCollection(polys, array=np.where(np.isfinite(lf), lf, np.nan),
                            cmap="viridis", clim=(1.0, 1.3), edgecolors="none")
        ax.add_collection(pc)
        sizes = z["ev_n"]
        fk, fo, fi = z["foot_k"], z["foot_off"], z["foot_idx"]
        cols = ["#d62728", "#ff7f0e", "#ffffff"]
        words = []
        for r, k in enumerate(np.argsort(sizes)[::-1][:n_big]):
            w = np.flatnonzero(fk == k)
            if w.size == 0:
                continue
            els = fi[fo[w[0]]:fo[w[0] + 1]]
            ax.add_collection(PolyCollection(polys[els], facecolors="none",
                                             edgecolors=cols[r], linewidths=0.6))
            words.append(f"{100 * sizes[k] / s['n']:.0f} %")
        ax.set_xlim(uv[:, 0].min() - 0.3, uv[:, 0].max() + 0.3)
        ax.set_ylim(uv[:, 1].max() + 0.3, -0.3)
        ax.set_aspect("equal")
        _style(ax)
        ax.set_xticks([round(uv[:, 0].min()), 0, round(uv[:, 0].max())])
        ax.set_yticks([0, 5, 10])
        ax.set_title(f"{LABELS[nm]}\nlargest events {', '.join(words)}", fontsize=8)
        _letter(ax, "abcd"[i])
        if ax is axes[0] or nm == "stadium":
            ax.set_ylabel("$v$, down dip (km)")
        if nm == "stadium" or not stad:
            ax.set_xlabel("$u$, along strike (km); the trace is $v$ = 0")
    cb = fig.colorbar(pc, ax=axes, shrink=0.5, pad=0.02, ticks=[1.0, 1.1, 1.2, 1.3])
    cb.set_label(r"load at first failure, $\lambda / \lambda_0$")
    fig.savefig(out, dpi=150, bbox_inches="tight")      # 24k polygons: 150 keeps it under 2 MB
    plt.close(fig)


# ------------------------------------------------------------------ lcp --
#: The load scan of ``lcp``: fine through the activation of the fault, then
#: coarse to the history's end load.
LCP_LOADS = np.concatenate([np.arange(1.0, 1.30, 0.0025),
                            np.arange(1.30, 2.0001, 0.01)])


def cmd_lcp(a) -> int:
    print(f"=== lcp {a.name}", flush=True)
    net, rec = build(a.name)
    describe(a.name, net, rec)
    n = net.n_elements
    areas = net.areas()
    Km, _ = K.assemble(net, mu=OF.MU, lam=OF.LAM, workers=1, verbose=True)
    fric = Fr.build_friction(net, regime=OF.REGIME, drop_ratio=OF.DROP_RATIO,
                             mean_stress_ratio=OF.MEAN_STRESS)
    d, g_s, g_d = C.coefficients(fric)
    P = K.project(Km, fric.slip_dir, fric.slip_dir)
    del Km
    lam0 = float(np.nanmin(fric.first_failure_lambda()))
    lams = LCP_LOADS * lam0
    out = {}
    for name, g in (("static", g_s), ("dynamic", g_d)):
        act = np.full(n, np.nan)
        frac, mom = [], []
        warm = None
        t0 = time.time()
        for lam in lams:
            sol = St.solve(P, d, g, lam, warm=warm)
            if not sol.converged:
                raise SystemExit(f"{name}: the LCP did not converge at "
                                 f"{lam / lam0:.4f} lam0")
            warm = sol.delta
            new = (sol.delta > 0.0) & np.isnan(act)
            act[new] = lam
            frac.append(float((sol.delta > 0.0).mean()))
            mom.append(float((areas * sol.delta).sum() * OF.MU * 1e18))
        out[name] = dict(act=act, frac=np.array(frac), mom=np.array(mom),
                         slip=sol.delta)
        print(f"  {name} strength: {lams.size} loads in {time.time() - t0:.0f} s; "
              f"active at {lams[-1] / lam0:.1f} lam0 {int((sol.delta > 0).sum())} "
              f"of {n}, moment {mom[-1]:.3e} N m", flush=True)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"avalanche_lcp_{a.name}.npz"
    np.savez_compressed(path, lams=lams, lam0=lam0, uv=rec["uv"], idx=rec["idx"],
                        areas=areas, **{f"{k}_{q}": v for k, o in out.items()
                                        for q, v in o.items()})
    print(f"  saved {path}", flush=True)
    return 0


def lcp_figure(runs, out):
    """The history's first-failure map between the two LCPs' activation
    maps, the slip at the end load for the three, and the curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    hist = {s["name"]: (z, s) for z, s in runs
            if s["spread"] == 0 and not s["coupled"]}
    files = sorted(OUT.glob("avalanche_lcp_*.npz"))
    pairs = [(f.stem[len("avalanche_lcp_"):], f) for f in files]
    pairs = [(nm, f) for nm, f in pairs if nm in hist]
    if not pairs:
        print("  lcp figure skipped: no lcp scan beside a history")
        return
    nm, f = pairs[0]
    z, s = hist[nm]
    L = np.load(f, allow_pickle=False)
    lam0 = float(L["lam0"])
    uv, idx = L["uv"], L["idx"]
    polys = uv[idx]
    cols = [("static LCP, strength $\\mu_s$", L["static_act"], L["static_slip"], "#1f77b4"),
            ("stick-slip history", z["lam_first"], z["slip"], "#d62728"),
            ("static LCP, strength $\\mu_d$", L["dynamic_act"], L["dynamic_slip"], "#2ca02c")]
    smax = max(float(np.nanmax(c[2])) for c in cols) * 1e3
    fig = plt.figure(figsize=(13, 9.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.85], hspace=0.3, wspace=0.3)

    def frame(ax):
        ax.set_xlim(uv[:, 0].min() - 0.3, uv[:, 0].max() + 0.3)
        ax.set_ylim(uv[:, 1].max() + 0.3, -0.3)
        ax.set_aspect("equal")
        _style(ax)
        ax.set_xticks([round(uv[:, 0].min()), 0, round(uv[:, 0].max())])
        ax.set_yticks([0, 5, 10])

    pc1 = pc2 = None
    for j, (title, act, slip, _) in enumerate(cols):
        ax = fig.add_subplot(gs[0, j])
        pc1 = PolyCollection(polys, array=np.where(np.isfinite(act), act / lam0, np.nan),
                             cmap="viridis", clim=(1.0, 1.3), edgecolors="none")
        ax.add_collection(pc1)
        frame(ax)
        ax.set_title(title, fontsize=10)
        _letter(ax, "abc"[j])
        if j == 0:
            ax.set_ylabel("$v$, down dip (km)")
        ax = fig.add_subplot(gs[1, j])
        pc2 = PolyCollection(polys, array=slip * 1e3, cmap="magma", clim=(0.0, smax),
                             edgecolors="none")
        ax.add_collection(pc2)
        frame(ax)
        ax.set_xlabel("$u$, along strike (km)")
        ax.text(0.97, 0.95, "def"[j], transform=ax.transAxes, ha="right", va="top", color="w")
        if j == 0:
            ax.set_ylabel("$v$, down dip (km)")
    fig.colorbar(pc1, ax=fig.axes[0:5:2], shrink=0.7, pad=0.02, ticks=[1.0, 1.1, 1.2, 1.3],
                 label=r"activation load, $\lambda / \lambda_0$")
    fig.colorbar(pc2, ax=fig.axes[1:6:2], shrink=0.7, pad=0.02,
                 label=f"slip at {L['lams'][-1] / lam0:.0f} $\\lambda_0$ (m)")
    x = L["lams"] / lam0
    xh = z["ev_lam"] / lam0
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(x, L["static_frac"], color=cols[0][3], lw=1.4, label="static LCP, $\\mu_s$")
    ax.plot(xh, np.cumsum(z["ev_n_new"]) / s["n"], color=cols[1][3], lw=1.4, label="history")
    ax.plot(x, L["dynamic_frac"], color=cols[2][3], lw=1.4, label="static LCP, $\\mu_d$")
    _style(ax)
    ax.set_xlim(1.0, 1.5)
    ax.set_xticks([1.0, 1.25, 1.5])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"$\lambda / \lambda_0$")
    ax.set_ylabel("active fraction of the fault")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _letter(ax, "g")
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(x, L["static_mom"] / 1e19, color=cols[0][3], lw=1.4)
    ax.plot(xh, np.cumsum(z["ev_moment"]) / 1e19, color=cols[1][3], lw=1.4)
    ax.plot(x, L["dynamic_mom"] / 1e19, color=cols[2][3], lw=1.4)
    _style(ax)
    ax.set_xlim(1.0, 2.0)
    ax.set_xticks([1.0, 1.5, 2.0])
    ax.set_ylim(0, 6)
    ax.set_yticks([0, 3, 6])
    ax.set_xlabel(r"$\lambda / \lambda_0$")
    ax.set_ylabel("cumulative moment ($10^{19}$ N m)")
    _letter(ax, "h")
    ax = fig.add_subplot(gs[2, 2])
    gap_s = (L["static_act"] - z["lam_first"]) / lam0      # how much earlier the history is
    gap_d = (z["lam_first"] - L["dynamic_act"]) / lam0     # how much later than the dynamic LCP
    for g_, c, lab in ((gap_s, cols[0][3], "static LCP activates later by"),
                       (gap_d, cols[2][3], "dynamic LCP activates earlier by")):
        v = np.sort(g_[np.isfinite(g_)])
        ax.step(v, 1.0 - np.arange(v.size) / v.size, where="post", color=c, lw=1.4, label=lab)
    _style(ax, log_x=True)
    _decades(ax, "x", 1e-4, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"$\Delta\lambda / \lambda_0$ against the history, per element")
    ax.set_ylabel("fraction of elements")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    _letter(ax, "i")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    n_out = int(((z["slip"] > L["dynamic_slip"] + 1e-12) | (z["slip"] < L["static_slip"] - 1e-12)).sum())
    print(f"  lcp figure on {nm}: history outside the slip bracket on {n_out} of {s['n']} elements; "
          f"moment at the end load static {L['static_mom'][-1]:.3e}, history "
          f"{np.cumsum(z['ev_moment'])[-1]:.3e}, dynamic {L['dynamic_mom'][-1]:.3e} N m")


def cmd_figure(a) -> int:
    files = sorted(f for f in OUT.glob("avalanche_*.npz")
                   if not f.name.startswith("avalanche_lcp_"))
    if not files:
        raise SystemExit(f"no runs in {OUT}")
    runs = [load(f) for f in files]
    order = {nm: i for i, nm in enumerate(list(VARIANTS))}
    runs.sort(key=lambda r: (order.get(r[1]["name"], 99), r[1]["spread"],
                             r[1]["coupled"]))
    rows = table(runs)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats_figure(runs, out / "avalanche_stats.png")
    resolution_figure(runs, out / "avalanche_resolution.png")
    maps_figure(runs, out / "avalanche_maps.png")
    lcp_figure(runs, out / "avalanche_lcp.png")
    (OUT / "avalanche_table.json").write_text(json.dumps(rows, indent=1,
                                                         default=float))
    print(f"  figures -> {out}; table -> {OUT / 'avalanche_table.json'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("assemble", help="one variant's matrix into the cache, "
                                        "in its own process")
    s.add_argument("name", choices=sorted(VARIANTS))
    s.add_argument("--workers", type=int, default=12)
    r = sub.add_parser("run", help="the loading history of one variant")
    r.add_argument("name", choices=sorted(VARIANTS))
    r.add_argument("--spread", type=float, default=0.0,
                   help="per-element friction spread, std / mean")
    r.add_argument("--coupled", action="store_true",
                   help="normal stress coupled to slip (static.coulomb_coupled)")
    r.add_argument("--lam-max", type=float, default=2.0,
                   help="in units of lam0, the first failure with no interaction")
    r.add_argument("--max-events", type=int, default=5_000_000)
    r.add_argument("--order", default="sequential")
    l_ = sub.add_parser("lcp", help="the static LCP's load scan at both strengths, "
                                    "the history's path-free counterpart")
    l_.add_argument("name", choices=sorted(VARIANTS))
    f = sub.add_parser("figure", help="table and figures from out/avalanche_*.npz")
    f.add_argument("--out", default=str(FIGURES))
    a = ap.parse_args(argv)
    if a.cmd == "assemble":
        return cmd_assemble(a)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "lcp":
        return cmd_lcp(a)
    return cmd_figure(a)


if __name__ == "__main__":
    raise SystemExit(main())
