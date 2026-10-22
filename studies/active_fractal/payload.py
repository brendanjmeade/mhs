#!/usr/bin/env python
"""Encode a loading history for the website's animation.

    python studies/active_fractal/payload.py --config patch --frames 120

WHAT THE ANIMATION IS. A sequence of EQUILIBRIA, sampled at evenly spaced
values of the load parameter. Frame k is the state of the network at load
``lam_k``: every element locked, every shear stress below its static strength.
Scrubbing the slider is moving along the load axis, not along time -- there is
no time in this experiment and nothing in the animation propagates.

Sampling in LOAD rather than in event index is deliberate. Events are not
equally spaced in load (that is the point of event-driven stepping), so
indexing frames by event would stretch the quiet intervals and compress the
bursts, which is exactly backwards for seeing a cascade.

THE ENCODING IS THE SIBLING SCRIPT'S, unchanged: log10 into one byte, 254
levels, index 0 reserved for "has never slipped". Reused rather than reinvented
because the site's loader already understands it and because the quantisation
is the one step that can silently turn a picture into a plausible lie. log10
rather than linear because slip spans several decades across the network.

NO GATE COVERS THIS, so the script decodes what it wrote and compares the worst
round-trip error against an ARITHMETIC bound: 254 levels across ``d`` decades
put the levels ``d/253`` apart, so rounding moves log10 by at most half that
and the relative error is at most ``10**(d/506) - 1``. A payload exceeding it is
refused. That bound is derived, not borrowed -- an earlier version of the
sibling check used a "2.4 %" figure measured on a different dataset and the
first run looked like a failure at 3.17 %.

GEOMETRY GOES OUT AS BINARY, unlike ``fault.json``'s inline triangle lists.
That file holds 256 elements and the JSON is 26 kB; this one holds ten thousand
and the same encoding would be about 1.5 MB of text against 368 kB of float32.

THE STRESS VOLUME (--stress) IS COARSER THAN THE FAULTS, AND THAT IS NOT FIXABLE
BY SPENDING MORE. Evaluating the elastic field on a grid costs
``n_grid * n_elements`` kernel pairs; measured on this network at ~50 us/pair
and 12 processes, a 40^3 grid is about an hour and 48^3 is about two and a half.
The faults are 0.6 km across in a 40 km box, so no affordable grid resolves an
individual fault -- 40^3 is 1 km spacing. What the isosurface therefore shows is
the AGGREGATE stress structure of the network at the scale of the cascade's
upper levels (2.5 km and above), not the stress around any one fault. That is
worth drawing and it is not a picture of a stress concentration at a fault tip;
the page says so.

Observers are only placed strictly below z = 0, because the library refuses a
readout above the free surface.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cascade as C                                         # noqa: E402
import friction as Fr                                       # noqa: E402
import kernel as K                                          # noqa: E402
import measure as M                                         # noqa: E402
import network                                              # noqa: E402
import run as R                                             # noqa: E402

SCHEMA = 1
NODATA = 0
LEVELS = 254                     # levels 1..254; 0 means "never slipped"

#: Frames in the animation. 120 is about four seconds at 30 fps and keeps the
#: slip array near a megabyte at ten thousand elements.
N_FRAMES = 120

#: Load ceiling as a multiple of the first failure, with no interaction. 3x
#: takes the network well past the onset so the active set has structure to
#: measure, without running to saturation where every element has slipped and
#: the geometry question is empty.
LAM_MAX_FACTOR = 3.0

#: Refuse a payload whose measured round-trip error exceeds the arithmetic
#: bound by more than this. 1.15 is the sibling script's margin and exists
#: because the bound is on a single rounding and the measurement is a maximum
#: over millions of values.
ROUNDTRIP_MARGIN = 1.15

#: Cluster adjacency radius for the published planarity array, in units of the
#: element size. MATCHES ``run.py localize --reach`` so the page and the
#: measurement describe the same clusters. 1.5 h percolates on a dense cloud --
#: every active element lands in one component of ~10000 and the statistic
#: stops discriminating, reading "1 cluster" for the active set AND for every
#: null -- so the measurement runs at 0.7 and so does this.
REACH_OVER_H = 0.7


def encode_log_u8(v: np.ndarray) -> tuple[bytes, dict]:
    """Positive values -> uint8 on a log10 ramp; zeros -> ``NODATA``."""
    good = np.isfinite(v) & (v > 0.0)
    if not np.any(good):
        return (np.zeros(v.shape, np.uint8).tobytes(),
                {"encoding": "log10_u8", "log_min": 0.0, "log_max": 0.0,
                 "min": 0.0, "max": 0.0, "n_nonzero": 0})
    lo = float(np.log10(v[good].min()))
    hi = float(np.log10(v[good].max()))
    span = hi - lo if hi > lo else 1.0
    q = np.zeros(v.shape, np.uint8)
    t = (np.log10(np.where(good, v, 1.0)) - lo) / span
    q[good] = 1 + np.clip(np.round(t[good] * (LEVELS - 1)),
                          0, LEVELS - 1).astype(np.uint8)
    return q.tobytes(), {"encoding": "log10_u8", "log_min": lo, "log_max": hi,
                         "min": float(v[good].min()),
                         "max": float(v[good].max()),
                         "n_nonzero": int(good.sum())}


def decode_log_u8(buf: bytes, meta: dict) -> np.ndarray:
    q = np.frombuffer(buf, np.uint8).astype(np.float64)
    span = meta["log_max"] - meta["log_min"]
    span = span if span > 0.0 else 1.0
    out = np.full(q.shape, np.nan)
    ok = q > 0
    t = (q[ok] - 1.0) / (LEVELS - 1)
    out[ok] = 10.0 ** (meta["log_min"] + t * span)
    return out


def encode_linear_u8(v: np.ndarray, lo: float, hi: float) -> tuple[bytes,
                                                                   dict]:
    """Values on ``[lo, hi]`` -> uint8 on a LINEAR ramp, with no NODATA.

    THE LOG RAMP IS WRONG FOR TWO OF THE ARRAYS THIS PAYLOAD CARRIES, and
    forcing them through it would make the page lie twice over. Slip spans
    decades and is positive, so a log ramp with ``0 = never slipped`` is right
    for it. Cluster planarity lives on ``[0, 1]`` and its zero is a MEASURED
    zero -- a cluster with no planar character -- not an absence; and the
    active-against-null comparison is three CATEGORIES, where the arithmetic
    between codes has no meaning at all. Sent down the log path, both would
    render with a colour bar reading decades and with a legitimate zero drawn
    in the grey the viewer reserves for "this element never slipped".

    So the range is explicit rather than inferred from the data: a categorical
    array must land on exactly the codes the viewer switches on, and letting
    ``lo``/``hi`` float with the realized values would move them.
    """
    if not (hi > lo):
        raise ValueError(f"need hi > lo, got lo={lo!r} hi={hi!r}")
    t = (np.clip(np.asarray(v, float), lo, hi) - lo) / (hi - lo)
    q = np.clip(np.round(t * (LEVELS - 1)), 0, LEVELS - 1).astype(np.uint8)
    return q.tobytes(), {"encoding": "linear_u8", "lin_min": float(lo),
                         "lin_max": float(hi),
                         "min": float(np.nanmin(v)) if v.size else 0.0,
                         "max": float(np.nanmax(v)) if v.size else 0.0,
                         "n_nonzero": int(np.count_nonzero(v))}


def decode_linear_u8(buf: bytes, meta: dict) -> np.ndarray:
    q = np.frombuffer(buf, np.uint8).astype(np.float64)
    return meta["lin_min"] + (q / (LEVELS - 1)) * (meta["lin_max"]
                                                   - meta["lin_min"])


def encode_category_u8(v: np.ndarray, names: list[str]) -> tuple[bytes, dict]:
    """Small integer labels -> the SAME uint8, with the names carried along.

    A RAMP IS THE WRONG SHAPE FOR A LABEL, and the measurement says so rather
    than taste: three categories pushed through :func:`encode_linear_u8` on
    ``[0, 2]`` land on bytes 0, 126 and 253 and decode to 0, 0.996 and 2.0, so
    the viewer would have to recover a label by nearest match and the error
    would scale with ``LEVELS``. Identity is exact, needs no bound, and makes
    the manifest say what each code MEANS instead of leaving the page to
    hard-code three magic numbers.
    """
    q = np.asarray(v)
    if q.min() < 0 or q.max() >= len(names):
        raise ValueError(f"labels must index {names}, got "
                         f"{q.min()}..{q.max()}")
    return (q.astype(np.uint8).tobytes(),
            {"encoding": "category_u8", "categories": list(names),
             "min": int(q.min()), "max": int(q.max()),
             "counts": [int((q == i).sum()) for i in range(len(names))],
             "n_nonzero": int(np.count_nonzero(q))})


def roundtrip(v: np.ndarray, buf: bytes, meta: dict) -> tuple[float, float]:
    """``(worst measured relative error, arithmetic bound)``.

    Dispatches on the encoding, so a new ramp cannot be shipped without being
    held to the same check the log one is: the caller compares the measured
    error against ``ROUNDTRIP_MARGIN`` times the bound and refuses to write
    otherwise.
    """
    if meta.get("encoding") == "category_u8":
        got = np.frombuffer(buf, np.uint8).astype(np.int64)
        return float(np.abs(got - np.asarray(v, np.int64)).max()), 0.0
    if meta.get("encoding") == "linear_u8":
        got = decode_linear_u8(buf, meta)
        span = meta["lin_max"] - meta["lin_min"]
        err = float(np.abs(got - np.clip(np.asarray(v, float),
                                         meta["lin_min"],
                                         meta["lin_max"])).max() / span)
        return err, float(0.5 / (LEVELS - 1))
    got = decode_log_u8(buf, meta)
    good = np.isfinite(v) & (v > 0.0)
    if not np.any(good):
        return 0.0, 0.0
    err = float(np.abs(got[good] - v[good]).max()
                / max(float(np.abs(v[good]).max()), 1e-300))
    rel = float(np.abs((got[good] - v[good]) / v[good]).max())
    decades = meta["log_max"] - meta["log_min"]
    bound = float(10.0 ** (decades / (2.0 * (LEVELS - 1))) - 1.0)
    return max(err, rel), bound


def snapshots(net, fric, P, *, n_frames: int, lam_max: float,
              verbose: bool = True):
    """Cumulative slip at ``n_frames`` evenly spaced loads, and the per-frame
    increment.

    The history is run once and sampled, rather than restarted per frame: the
    events between two sampled loads are what the increment frame shows, so a
    frame is a difference of equilibria exactly as the statistics are.
    """
    n = net.n_elements
    d, g_s, g_d = C.coefficients(fric)
    state = C.State(s=np.zeros(n), q=np.zeros(n), lam=0.0)
    lam0 = float(np.nanmin(fric.first_failure_lambda()))
    grid = np.linspace(lam0, lam_max, n_frames)

    cum = np.zeros((n_frames, n))
    inc = np.zeros((n_frames, n))
    stats = []
    ev_total = 0
    mom_total = 0.0
    areas = net.areas()
    prev = np.zeros(n)
    for f, lam_target in enumerate(grid):
        n_ev = 0
        while True:
            lam_next, _ = C.next_failure(state, d, g_s)
            if not np.isfinite(lam_next) or lam_next > lam_target:
                break
            state.lam = lam_next
            before = state.s.copy()
            sweeps, ok = C.relax(state, P, d, g_s, g_d)
            if not ok:
                raise SystemExit(
                    f"runaway cascade at lam {lam_next:.6g}: the payload "
                    f"would show a frame that is not an equilibrium.")
            ds = state.s - before
            mom_total += float(30.0 * np.sum(areas * ds) * C.MOMENT_TO_NM)
            n_ev += 1
            ev_total += 1
        state.lam = float(lam_target)
        cum[f] = state.s
        inc[f] = state.s - prev
        prev = state.s.copy()
        stats.append({"lam": float(lam_target), "events": n_ev,
                      "events_cumulative": ev_total,
                      "moment_cumulative_Nm": mom_total,
                      "n_ever": int((state.s > 0.0).sum()),
                      "slipped_this_frame": int((inc[f] > 0.0).sum())})
        if verbose and (f % 20 == 0 or f == n_frames - 1):
            print(f"    frame {f:4d}  lam {lam_target:.5f}  "
                  f"events {n_ev:5d}  active {stats[-1]['n_ever']:6d}")
    return cum, inc, stats, grid


# ----------------------------------------------------------- stress volume --
_W: dict = {}


def _init_worker(tris, slip, mu, lam, eps) -> None:
    """Per-process state, set once. Module level and picklable: workers are
    SPAWNED, so a closure would not survive the trip."""
    _W.update(tris=tris, slip=slip, mu=mu, lam=lam, eps=eps)


def _stress_chunk(pts: np.ndarray) -> np.ndarray:
    """Maximum shear stress at ``pts`` from the stored slip.

    The stress matrix is built and contracted against the slip inside the
    worker and never returned: at 216 bytes per pair, the matrix for the whole
    grid would be 1.4 TB, so the contraction has to happen per chunk. This is
    the sibling exporter's observer-chunking pattern for the same reason.
    """
    import mhs as _mhs
    from mhs.parallel import pinned_blas
    with pinned_blas(1):
        S = _mhs.stress_matrix(pts, _W["tris"],
                               _mhs.Material(mu=_W["mu"], lam=_W["lam"]),
                               _W["eps"])
        sig = np.einsum("omnsk,sk->omn", S, _W["slip"])
    # Maximum shear = (s1 - s3)/2, an invariant, so the picture does not depend
    # on a reference plane orientation the way a Coulomb stress would.
    ev = np.linalg.eigvalsh(sig)
    return 0.5 * (ev[:, 2] - ev[:, 0])


def aggregate_faults(net, slip_scalar, fric):
    """Each fault as its OWN rectangle in two triangles, with uniform slip.

    MOMENT CONSERVING AND GEOMETRICALLY EXACT. The replacement rectangle is the
    fault's own -- same centre, azimuth, dip, length and width -- so there is no
    geometric error at all; the two triangles simply span it instead of twenty.
    The slip is set so that ``area_rect * s_agg == sum(area_i * s_i)``, which
    makes the total moment identical. What is lost is the slip DISTRIBUTION
    within a fault, and nothing else.

    MEASURED, AND NOT GOOD ENOUGH TO BE THE PRIMARY FIELD. Against the exact
    field, as a function of distance from the nearest fault in units of the
    median fault length L = 0.406 km:

        d/L     median     p90      max
        0.25     3.79 %   18.20 %  37.77 %
        0.50     3.55 %   13.66 %  51.31 %
        1.00     1.79 %    6.73 %  13.37 %
        2.00     0.63 %    3.97 %  15.95 %
        4.00     0.37 %    2.06 %  13.52 %
        8.00     0.09 %    0.43 %   4.20 %

    Sub-0.1 % through most of the volume, and tens of percent within about one
    fault length. An isosurface at a HIGH threshold selects exactly the
    near-fault points, so the approximation is worst where the picture is most
    interesting -- which is why the shipped volume is computed EXACTLY, on a
    coarser grid, and this function is kept only as a fast preview and as the
    thing that measured its own unsuitability.

    (A 10^3 grid alone would have passed it: median 0.00 %, p90 0.05 %,
    correlation 0.99972. At 4.4 km spacing no grid point is ever near a fault,
    so that test flattered the approximation and the distance-resolved one is
    the one that decided.)

    Why aggregation was tried at all: a stress readout parallelises at 1.45x,
    not 8.5x -- the baseline for that figure was already multi-threaded BLAS,
    and ``mhs.parallel`` correctly pins children to one thread, so twelve
    pinned workers barely beat one unpinned process. The README's 8.5x is for
    interaction_matrix ASSEMBLY, which is far more compute-dense per byte. That
    makes a 40^3 exact volume nine hours and a 24^3 one about two.
    """
    from mhs import tdcs
    n = net.n_elements
    local = np.zeros((n, 3))
    local[:, 0] = slip_scalar * fric.slip_dir[:, 0]
    local[:, 1] = slip_scalar * fric.slip_dir[:, 1]
    cart = tdcs.to_cartesian(net.tris, local)
    areas = net.areas()

    tris, slips = [], []
    for f in range(net.n_faults):
        sel = net.fault_id == f
        if not np.any(sel):
            continue
        rect = network.rect_patch(net.center[f], net.azimuth[f], net.dip[f],
                                  net.length[f], net.width[f],
                                  float(net.length[f]))   # n_s = n_d = 1
        a_rect = 0.5 * np.linalg.norm(
            np.cross(rect[:, 1] - rect[:, 0], rect[:, 2] - rect[:, 0]), axis=1)
        moment = (areas[sel, None] * cart[sel]).sum(axis=0)     # (3,) km^3
        s_agg = moment / a_rect.sum()
        tris.append(rect)
        slips.append(np.repeat(s_agg[None, :], rect.shape[0], axis=0))
    return (np.ascontiguousarray(np.concatenate(tris, axis=0)),
            np.ascontiguousarray(np.concatenate(slips, axis=0)))


def verify_aggregation(net, slip_scalar, fric, *, grid: int = 10,
                       workers: int | None = None) -> dict:
    """Exact field against the aggregated one, on a small grid. Measured, not
    assumed: the aggregation is the only approximation in this study."""
    exact, _ = stress_volume(net, slip_scalar, fric, grid=grid,
                             workers=workers, sources="elements")
    agg, _ = stress_volume(net, slip_scalar, fric, grid=grid,
                           workers=workers, sources="faults")
    rel = np.abs(agg - exact) / np.maximum(np.abs(exact), 1e-30)
    out = {"grid": grid, "median_rel": float(np.median(rel)),
           "p90_rel": float(np.percentile(rel, 90)),
           "max_rel": float(rel.max()),
           "corr": float(np.corrcoef(exact, agg)[0, 1])}
    print(f"  aggregation error on {grid}^3: median {out['median_rel']:.2%}, "
          f"p90 {out['p90_rel']:.2%}, max {out['max_rel']:.2%}, "
          f"correlation {out['corr']:.5f}")
    return out


def stress_volume(net, slip_scalar, fric, *, grid: int, workers: int | None,
                  chunk: int = 400, sources: str = "faults"):
    """Maximum shear stress on a ``grid**3`` lattice inside the box.

    Returns ``(values, meta)`` with ``values`` flat in x-fastest order, which
    is what the viewer's 3-D texture upload expects.
    """
    import concurrent.futures as cf
    import multiprocessing as mp
    from mhs import tdcs

    if sources == "faults":
        src_tris, cart = aggregate_faults(net, slip_scalar, fric)
    elif sources == "elements":
        n = net.n_elements
        local = np.zeros((n, 3))
        local[:, 0] = slip_scalar * fric.slip_dir[:, 0]
        local[:, 1] = slip_scalar * fric.slip_dir[:, 1]
        cart = tdcs.to_cartesian(net.tris, local)
        src_tris = net.tris
    else:
        raise ValueError(f"sources must be 'faults' or 'elements', "
                         f"got {sources!r}")
    n = src_tris.shape[0]

    half = net.meta["box_xy_km"] / 2.0
    z_top, z_bot = net.meta["z_top_km"], net.meta["z_bottom_km"]
    gx = np.linspace(-half, half, grid)
    gy = np.linspace(-half, half, grid)
    # strictly below the free surface: the library refuses z > 0, and a plane of
    # observers exactly ON a boundary is the half-value trap in any case
    gz = np.linspace(z_bot, min(z_top, -1e-3), grid)
    Z, Y, X = np.meshgrid(gz, gy, gx, indexing="ij")   # x fastest
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    w = workers or max(1, (mp.cpu_count() or 2) - 4)
    parts = [pts[i:i + chunk] for i in range(0, pts.shape[0], chunk)]
    print(f"  stress volume {grid}^3 = {pts.shape[0]} points x {n} "
          f"{sources} sources, {len(parts)} chunks on {w} workers", flush=True)
    out = np.empty(pts.shape[0])
    done = 0
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=w, mp_context=ctx,
                                initializer=_init_worker,
                                initargs=(src_tris, cart, R.MU, R.LAM,
                                          net.eps)) as ex:
        for i, got in enumerate(ex.map(_stress_chunk, parts, chunksize=1)):
            a = i * chunk
            out[a:a + got.size] = got
            done += got.size
            if i % 20 == 0 or done == pts.shape[0]:
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                print(f"    {done:7d}/{pts.shape[0]}  {el / 60:5.1f} min  "
                      f"eta {(pts.shape[0] - done) / max(rate, 1e-9) / 60:5.1f}"
                      f" min", flush=True)
    meta = {"grid": [grid, grid, grid],
            "origin": [float(gx[0]), float(gy[0]), float(gz[0])],
            "spacing": [float(gx[1] - gx[0]), float(gy[1] - gy[0]),
                        float(gz[1] - gz[0])],
            "n_points": int(pts.shape[0]), "order": "x fastest, then y, then z",
            "quantity": "max_shear_GPa", "sources": sources,
            "n_sources": int(n), "seconds": time.time() - t0,
            "us_per_pair": 1e6 * (time.time() - t0) / (pts.shape[0] * n),
            "resolves_individual_faults": False}
    return out, meta


def theta_mu_sweep(net, mat, lam: float, thetas, mus, *,
                   mean_stress_ratio: float, drop_ratio: float,
                   verbose: bool = True):
    """Static solutions over a grid of stress ORIENTATION and FRICTION.

    THE INTERACTION MATRIX IS ASSEMBLED ONCE AND NEVER AGAIN. ``K`` depends on
    geometry alone, so rotating the regional stress does not touch it -- what
    changes is each element's SLIP DIRECTION, and therefore the projected
    operator ``P``. One re-projection per orientation, which reads the 19 GiB
    matrix and contracts it, against forty minutes to re-assemble. That is the
    entire reason a two-dimensional sweep is affordable at this size.

    ORIENTATION IS THE PARAMETER A NETWORK CANNOT ABSORB. Friction can be
    rescaled, the load can be rescaled, but rotating the stress relative to a
    FIXED set of fault planes asks which populations a given tectonic regime
    selects -- which is exactly what critically-stressed-fracture analysis is
    used for, and the comparison against that screening is computed here for
    free at every grid point.

    Frames are stored row-major in ``(theta, mu)``, so frame
    ``i_theta * n_mu + i_mu``. Nesting in mu is verified SEPARATELY AT EVERY
    ORIENTATION, because it is a measured property of this operator and not a
    theorem -- ``A = -P`` is not a Z-matrix -- and the critical-friction array
    is meaningless without it.
    """
    import static as St                                        # noqa: PLC0415
    n = net.n_elements
    thetas = np.asarray(thetas, float)
    mus = np.asarray(mus, float)
    slip = np.zeros((thetas.size * mus.size, n))
    crit = np.full((thetas.size, n), -1, np.int64)
    stats, per_theta = [], []

    for it, th in enumerate(thetas):
        t0 = time.time()
        ref = Fr.build_friction(net, azimuth_deg=float(th),
                                mean_stress_ratio=mean_stress_ratio,
                                drop_ratio=drop_ratio)
        P = K.project(mat, ref.slip_dir, ref.slip_dir)
        ev = float(np.linalg.eigvalsh(-0.5 * (P + P.T)).min())
        if ev <= 0.0:
            raise SystemExit(
                f"sym(-P) is indefinite at azimuth {th:.1f} deg "
                f"(min eig {ev:.4g}). The LCP may have several solutions or "
                f"none, so these frames would not be THE answer.")
        t_proj = time.time() - t0

        warm = None
        base = it * mus.size
        for im, mu in enumerate(mus):
            fr = Fr.build_friction(net, mu_mean=float(mu),
                                   azimuth_deg=float(th),
                                   mean_stress_ratio=mean_stress_ratio,
                                   drop_ratio=drop_ratio)
            d, g_s, _ = C.coefficients(fr)
            sol = St.solve(P, d, g_s, lam, warm=warm)
            if not sol.converged:
                raise SystemExit(
                    f"no convergence at azimuth {th:.1f} deg, mu {mu:.4f} "
                    f"(residual {sol.residual:.3e}).")
            slip[base + im] = sol.delta
            warm = sol.delta.copy()
            scr = St.critically_stressed(fr, lam)
            inter = int((sol.active & scr).sum())
            union = int((sol.active | scr).sum())
            stats.append({
                "theta": float(th), "mu": float(mu),
                "n_active": sol.n_active, "n_screened": int(scr.sum()),
                "added": int((sol.active & ~scr).sum()),
                "removed": int((scr & ~sol.active).sum()),
                "jaccard": (inter / union) if union else 1.0,
                "locked_fraction": fr.locked_fraction(),
                "moment_Nm": float(R.MU * np.sum(net.areas() * sol.delta)
                                   * C.MOMENT_TO_NM),
                "sweeps": sol.sweeps})

        # NESTING IS MEASURED, NOT ASSUMED, AND NOT REQUIRED. LCP monotonicity
        # in mu would need A = -P to be a Z-matrix -- every coupling
        # promoting -- and the dislocation kernel has shadowing lobes, so an
        # element held locked at low mu by a neighbour's shadow may be freed
        # at high mu when that neighbour stops slipping. At azimuth 0 that
        # escape is never taken; at 15 deg it is, by one element. So the
        # viewer reads the active set from the PER-FRAME slip array, which is
        # exact either way, and this records how often the escape happens
        # together with how big the slips involved are -- a violator at the
        # solver's tolerance is noise, one at millimetres is physics.
        block = slip[base:base + mus.size] > 0.0
        vmask = np.zeros(n, bool)
        for f in range(1, mus.size):
            vmask |= block[f] & ~block[f - 1]
        vn = int(vmask.sum())
        vmax = (float(slip[base:base + mus.size][:, vmask].max()) if vn
                else 0.0)
        has = block.any(axis=0)
        crit[it] = np.where(has, block.shape[0] - 1 - block[::-1].argmax(0), -1)

        a0, a1 = stats[base], stats[base + mus.size - 1]
        per_theta.append({"theta": float(th), "min_eig_sym_negP": ev,
                          "project_s": t_proj, "n_mu_nonmonotone": vn,
                          "max_slip_of_nonmonotone_km": vmax})
        if verbose:
            print(f"    azimuth {th:6.1f} deg  SHmax {(135.0 - th) % 180:6.1f} "
                  f"  active {a0['n_active']:6d} -> {a1['n_active']:6d}"
                  f"  jaccard {a0['jaccard']:.3f}-{a1['jaccard']:.3f}"
                  f"  non-monotone {vn:3d}"
                  f"{'' if not vn else f' (max {vmax*1e3:.2e} m)'}"
                  f"  proj {t_proj:.0f}s  eig {ev:.2f}", flush=True)
    return slip, crit, stats, per_theta


def _principal(fric, lam: float, k: float) -> dict:
    """Principal values and axes of the TOTAL regional stress, compression
    positive, sorted so index 0 is the maximum compression.

    ``LOAD_TENSORS`` is written in the tension-positive convention that
    ``resolve_tensor`` flips, so the sign is applied here once and the axes
    come out meaning what a structural geologist expects: axis 0 is SHmax.
    """
    # ONLY THE DEVIATORIC PART FLIPS SIGN. ``resolve_tensor`` returns
    # ``sigma_n = -n.D.n``, so the compression-positive normal stress is
    # ``sigma_c + lam(k - n.D.n) = n.[(sigma_c + lam k) I - lam D].n``. The
    # confining and isotropic terms are ALREADY compression positive; negating
    # the whole tensor flips them too, which is how a first version of this
    # reported two negative principal stresses on a load that is compressive
    # everywhere.
    sig = ((fric.sigma_eff_0.mean() + lam * k) * np.eye(3)
           - lam * np.asarray(fric.meta["tensor"]))
    vals, vecs = np.linalg.eigh(sig)           # already compression positive
    o = np.argsort(-vals)                      # sigma_1 >= sigma_2 >= sigma_3
    return {"regime": fric.meta["regime"], "mean_stress_ratio": float(k),
            "lam": float(lam),
            "principal_GPa": [float(v) for v in vals[o]],
            "axes": [[float(c) for c in vecs[:, i]] for i in o],
            "note": "compression positive; axes[0] is the maximum "
                    "compression direction (SHmax), axes[2] the minimum"}


def mu_sweep(net, P, lam: float, mus, *, mean_stress_ratio: float,
             drop_ratio: float, verbose: bool = True):
    """Static solutions across a range of friction coefficients.

    ONE SOLVE PER FRAME, each the UNIQUE answer at that ``mu`` -- so unlike
    the loading history, a frame here depends on nothing that came before it
    and the animation is a family of exact solutions rather than a path. The
    sweep is only warm started, which by uniqueness can change the sweep count
    and never the answer.

    WHY MU IS THE BETTER SLIDER. The active set is NESTED in mu: measured over
    0.40 to 0.80, zero elements slip at a higher friction than at a lower one.
    So each element has a well defined CRITICAL FRICTION -- the largest mu at
    which it still slips -- which is one number per element and the natural
    thing to colour by. The loading parameter has no such invariant, because
    there the set grows by a path rather than shrinking by a threshold.

    That nesting is not guaranteed. LCP monotonicity needs ``A = -P`` to be a
    Z-matrix, every coupling promoting, and only 44 % of pairs here satisfy
    that -- the dislocation kernel has shadowing lobes. The theorem does not
    apply and the conclusion holds anyway, which is why it is measured.
    """
    import static as St                                        # noqa: PLC0415
    n = net.n_elements
    mus = np.asarray(mus, float)
    slip = np.zeros((mus.size, n))
    stats = []
    warm = None
    for f, mu in enumerate(mus):
        fr = Fr.build_friction(net, mu_mean=float(mu),
                               drop_ratio=drop_ratio,
                               mean_stress_ratio=mean_stress_ratio)
        d, g_s, _ = C.coefficients(fr)
        sol = St.solve(P, d, g_s, lam, warm=warm)
        if not sol.converged:
            raise SystemExit(
                f"the static solve did not converge at mu {mu:.4f} "
                f"(residual {sol.residual:.3e}). A frame that is not the "
                f"solution must not be published.")
        slip[f] = sol.delta
        warm = sol.delta.copy()
        screened = int(St.critically_stressed(fr, lam).sum())
        stats.append({"mu": float(mu), "n_active": sol.n_active,
                      "n_screened": screened, "sweeps": sol.sweeps,
                      "residual": sol.residual,
                      "moment_Nm": float(R.MU * np.sum(net.areas() * sol.delta)
                                         * C.MOMENT_TO_NM),
                      "locked_fraction": fr.locked_fraction()})
        if verbose and (f % 10 == 0 or f == mus.size - 1):
            print(f"    frame {f:4d}  mu {mu:.4f}  active {sol.n_active:6d}  "
                  f"screened {screened:6d}  sweeps {sol.sweeps:4d}")

    # NESTING, verified on what is about to be published rather than taken
    # from a separate experiment. If it fails, the critical-friction array
    # below is not well defined and must not be written.
    viol = 0
    for f in range(1, mus.size):
        viol += int(((slip[f] > 0) & ~(slip[f - 1] > 0)).sum())
    if viol:
        raise SystemExit(
            f"the active set is NOT nested in mu: {viol} elements slip at a "
            f"higher friction than at a lower one. 'critical friction' is "
            f"then not a single number per element and this payload would "
            f"misrepresent it.")

    # Critical friction: the largest mu at which an element still slips.
    # NODATA for elements that never slip at any mu in the range.
    active = slip > 0.0
    last = np.where(active.any(axis=0), active.shape[0] - 1
                    - active[::-1].argmax(axis=0), -1)
    return slip, last, stats, viol


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="patch", choices=sorted(R.CONFIGS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--drop-ratio", type=float, default=Fr.DROP_RATIO)
    ap.add_argument("--mean-stress-ratio", type=float, default=0.0,
                    help="isotropic part of the loading path; see "
                         "friction.py on why 0 bounds the usable load")
    ap.add_argument("--lam-factor", type=float, default=LAM_MAX_FACTOR,
                    help="load ceiling as a multiple of the first failure")
    ap.add_argument("--n-theta", type=int, default=12,
                    help="stress orientations over [0,180) for a grid sweep")
    ap.add_argument("--sweep", default="lam",
                    choices=("lam", "mu", "theta_mu"),
                    help="what the frames vary: the loading history (lam) or "
                         "the friction coefficient at fixed load (mu)")
    ap.add_argument("--mu-min", type=float, default=0.35)
    ap.add_argument("--mu-max", type=float, default=0.85)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None,
                    help="matrix cache directory; without it a 20k-element "
                         "config re-assembles for 40 minutes")
    ap.add_argument("--stress-sources", default="faults",
                    choices=("faults", "elements"))
    ap.add_argument("--verify-stress", type=int, default=0,
                    help="compare aggregated against exact on this grid first")
    ap.add_argument("--stress", type=int, default=0,
                    help="also evaluate a stress volume on this cube grid "
                         "(e.g. 40); 0 skips it")
    a = ap.parse_args()

    out = (pathlib.Path(__file__).resolve().parents[2] / "website" / "public"
           / "data" if a.out is None else pathlib.Path(a.out))

    print(f"[{a.config}] seed {a.seed}")
    net = R.build(a.config, a.seed)
    R.describe(net)
    mat, info = K.assemble(net, mu=R.MU, lam=R.LAM, workers=a.workers,
                           cache_dir=a.cache)
    fric = Fr.build_friction(net, drop_ratio=a.drop_ratio,
                             mean_stress_ratio=a.mean_stress_ratio)
    P = K.project(mat, fric.slip_dir, fric.slip_dir)
    lam0 = float(np.nanmin(fric.first_failure_lambda()))
    lam_max = a.lam_factor * lam0
    R.report_load(fric, lam0, lam_max)
    # REFUSED rather than warned, unlike the study runs. A published payload is
    # seen by people who will not read a log, and a load past the compressive
    # ceiling puts elements in effective tension where their strength is
    # negative and any shear fails them -- they would appear in the animation
    # as slipping faults for no mechanical reason.
    cap = fric.max_lambda_compressive()
    if lam_max > cap:
        raise SystemExit(
            f"refusing to publish: lam_max {lam_max:.5f} exceeds the "
            f"compressive ceiling {cap:.5f} ({cap / lam0:.2f}x first "
            f"failure). Pass --lam-factor {cap / lam0:.2f} or lower, or "
            f"--mean-stress-ratio 1 for a load with no ceiling.")

    if a.sweep == "theta_mu":
        thetas = np.linspace(0.0, 180.0, a.n_theta, endpoint=False)
        mus = np.linspace(a.mu_min, a.mu_max, a.frames)
        print(f"  GRID SWEEP at fixed lam {lam_max:.5f}: "
              f"{a.n_theta} orientations x {a.frames} frictions = "
              f"{a.n_theta * a.frames} static solves", flush=True)
        cum, crit2, stats, per_theta = theta_mu_sweep(
            net, mat, lam_max, thetas, mus,
            mean_stress_ratio=a.mean_stress_ratio, drop_ratio=a.drop_ratio)
        # No increment array for a grid: "the previous frame" is ambiguous on
        # a 2-D lattice, and shipping one would invite reading a step in
        # orientation as a change in friction.
        inc, crit, grid = None, None, mus
    elif a.sweep == "mu":
        mus = np.linspace(a.mu_min, a.mu_max, a.frames)
        print(f"  SWEEPING FRICTION at fixed lam {lam_max:.5f}: "
              f"mu {a.mu_min} to {a.mu_max} over {a.frames} static solves")
        cum, crit, stats, _ = mu_sweep(net, P, lam_max, mus,
                                       mean_stress_ratio=a.mean_stress_ratio,
                                       drop_ratio=a.drop_ratio)
        # The second array is the change between adjacent mu, which is what
        # "what did raising friction just switch off" looks like.
        inc = np.zeros_like(cum)
        inc[1:] = np.abs(np.diff(cum, axis=0))
        grid = mus
    else:
        cum, inc, stats, grid = snapshots(net, fric, P, n_frames=a.frames,
                                          lam_max=lam_max)
        crit = None
    n = net.n_elements

    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    worst_ratio = 0.0
    for name, arr in (("slip", cum), ("incr", inc)):
        if arr is None:
            continue                       # a grid sweep ships no increment
        buf, meta = encode_log_u8(arr.ravel())
        got, bound = roundtrip(arr.ravel(), buf, meta)
        ratio = got / bound if bound > 0 else 0.0
        worst_ratio = max(worst_ratio, ratio)
        print(f"  {name:5s} round trip: measured {got:.4%}  bound "
              f"{bound:.4%}  ratio {ratio:.3f}")
        if bound > 0.0 and got > ROUNDTRIP_MARGIN * bound:
            raise SystemExit(
                f"{name} quantisation error {got:.4%} exceeds the arithmetic "
                f"bound {bound:.4%} by more than {ROUNDTRIP_MARGIN}x. Do not "
                f"publish this payload.")
        path = out / f"fractal.{name}.bin"
        path.write_bytes(buf)
        meta.update({"bytes": len(buf), "frames": int(arr.shape[0]), "n": n,
                     "order": "frame-major: buf[f*n + i] is element i at "
                              "frame f",
                     "sha256": hashlib.sha256(buf).hexdigest()[:16],
                     "measured_rel_err": got, "bound_rel_err": bound})
        arrays[name] = meta

    # TWO ARRAYS THAT DESCRIBE THE FINAL STATE, not its history, so each is a
    # SINGLE frame of n bytes against slip's frames * n. The viewer reads each
    # array's own ``frames`` rather than the manifest's global one.
    #
    # They are here because the page's headline is a comparison, not a picture:
    # the active set is 97.4 % identical to the set that would fail with K = 0,
    # so showing slip alone shows mostly what the LOADING chose and lets a
    # reader credit it to interaction.
    final = cum[-1]
    act = final > 0.0
    cent = net.tris.mean(axis=1)
    if int(act.sum()) >= 3:
        # The no-interaction null, matched in cardinality: the elements the far
        # field alone would fail, scored exactly as `run.cmd_localize` does.
        score = -fric.first_failure_lambda()
        order = np.argsort(-np.where(np.isfinite(score), score, -np.inf))
        nul = np.zeros(n, bool)
        nul[order[:int(act.sum())]] = True
        code = np.where(act & nul, 0, np.where(act, 1, np.where(nul, 2, 0)))
        code = np.where(act | nul, code, 0).astype(np.uint8)
        NAMES = ["both or neither", "interaction added", "interaction removed"]
        buf, meta = encode_category_u8(code, NAMES)
        got, _ = roundtrip(code, buf, meta)
        if got != 0.0:
            raise SystemExit(f"category encoding is not exact: {got}")
        (out / "fractal.nulldiff.bin").write_bytes(buf)
        meta.update({"bytes": len(buf), "frames": 1, "n": n,
                     "order": "single frame: buf[i] is element i",
                     "sha256": hashlib.sha256(buf).hexdigest()[:16],
                     "measured_rel_err": 0.0, "bound_rel_err": 0.0,
                     "shared_fraction": float((act & nul).sum()
                                              / max(int(act.sum()), 1))})
        arrays["nulldiff"] = meta
        print(f"  nulldiff: {meta['counts']} for {NAMES}, "
              f"shared {meta['shared_fraction']:.1%}")

        # Planarity of the cluster each active element belongs to. Inactive
        # elements get 0, which here is a MEASURED zero and not a NODATA --
        # exactly why this cannot go down the log ramp.
        lab = M.connected_clusters(cent[act], REACH_OVER_H * net.h)
        pla = np.zeros(n)
        sub = np.zeros(int(act.sum()))
        for c_ in np.unique(lab):
            sel = lab == c_
            p_, _l = M.shape_indices(cent[act][sel])
            sub[sel] = 0.0 if not np.isfinite(p_) else p_
        pla[act] = sub
        buf, meta = encode_linear_u8(pla, 0.0, 1.0)
        got, bound = roundtrip(pla, buf, meta)
        print(f"  planarity round trip: measured {got:.4%}  bound "
              f"{bound:.4%}")
        if bound > 0.0 and got > ROUNDTRIP_MARGIN * bound:
            raise SystemExit(
                f"planarity quantisation error {got:.4%} exceeds the bound "
                f"{bound:.4%} by more than {ROUNDTRIP_MARGIN}x. Do not "
                f"publish this payload.")
        (out / "fractal.planarity.bin").write_bytes(buf)
        meta.update({"bytes": len(buf), "frames": 1, "n": n,
                     "order": "single frame: buf[i] is element i",
                     "sha256": hashlib.sha256(buf).hexdigest()[:16],
                     "measured_rel_err": got, "bound_rel_err": bound,
                     "reach_over_h": REACH_OVER_H,
                     "n_clusters": int(np.unique(lab).size)})
        arrays["planarity"] = meta

    geom = np.ascontiguousarray(net.tris, dtype=np.float32).tobytes()
    (out / "fractal.geom.bin").write_bytes(geom)

    # ``R.window_of`` rather than the cascade cell and ``R.BOX_XY`` directly:
    # a power-law network has no cascade cell and its box is not 40 km, so
    # reading those two fields here produced a window for a geometry that is
    # not the one being published. ``d_geom`` stays None when the window is
    # unusable, and the page then has no dimension to show -- correct, because
    # a 100 km domain holding a 38 km fault has no decade to fit a slope in.
    try:
        win = R.window_of(net)
        d_fit = M.dimension(net.tris, n_radii=10, **win)
        d_geom = {"known": net.meta.get("d_geom_known"),
                  "calibrated": M.calibrated(d_fit),
                  "window_km": [win["r_lo"], win["r_hi"]]}
    except ValueError:
        d_geom = None

    payload = {
        "schema": SCHEMA, "nodata": NODATA, "levels": LEVELS,
        "config": a.config, "unit": "km of slip",
        # DERIVED FROM THE DATA, never from the argument. A grid sweep ships
        # n_theta * n_mu frames while --frames names only the mu count, and
        # declaring the argument shipped 300 frames of slip labelled as 25 --
        # which the viewer refused, correctly, rather than reading a quarter
        # of the array.
        "n_elements": n, "n_faults": net.n_faults,
        "frames": int(cum.shape[0]),
        "lam": [float(v) for v in grid],
        # WHAT THE FRAMES VARY. The viewer labels its slider from this and
        # must not guess: a mu sweep is a family of exact static solutions
        # with a NESTED active set, a lam sweep is one path with a growing
        # one, and reading either as the other inverts the animation.
        "sweep": {"kind": a.sweep,
                  "label": ("friction" if a.sweep in ("mu", "theta_mu")
                            else "load"),
                  "symbol": ("\u03bc" if a.sweep in ("mu", "theta_mu")
                             else "\u03bb"),
                  "values": [float(v) for v in grid],
                  "lam_fixed": (float(lam_max)
                                if a.sweep in ("mu", "theta_mu") else None),
                  "nested": a.sweep in ("mu", "theta_mu"),
                  "grid": ({"n_theta": int(a.n_theta), "n_mu": int(a.frames),
                            "theta_deg": [float(v) for v in thetas],
                            "mu": [float(v) for v in mus],
                            "order": "frame = i_theta * n_mu + i_mu",
                            "shmax_azimuth_deg":
                                [float((135.0 - v) % 180.0) for v in thetas]}
                           if a.sweep == "theta_mu" else None)},
        "geometry": {"file": "fractal.geom.bin", "dtype": "float32",
                     "shape": [n, 3, 3], "bytes": len(geom),
                     "sha256": hashlib.sha256(geom).hexdigest()[:16]},
        "arrays": {k: {"file": f"fractal.{k}.bin", **v}
                   for k, v in arrays.items()},
        "fault_id": {"inline": True},
        # THE REGIONAL STRESS, as principal values and axes, so the viewer can
        # draw it without restating the load tensor in TypeScript -- a
        # convention written twice is a bug, and this one is already stated
        # once in friction.LOAD_TENSORS.
        "regional_stress": _principal(fric, lam_max, a.mean_stress_ratio),
        "frame_stats": stats,
        "d_geom": d_geom,
        "network": net.meta, "friction": fric.meta,
        "matrix": {k: info[k] for k in
                   ("key", "us_per_pair", "assemble_s", "eps_km", "mu_GPa",
                    "lam_GPa") if k in info},
        "roundtrip_worst_ratio_to_bound": worst_ratio,
    }
    # fault_id is small and the viewer needs it to outline whole faults.
    payload["fault_id"]["values"] = [int(v) for v in net.fault_id]
    if a.sweep == "theta_mu":
        # ONE CRITICAL-FRICTION MAP PER ORIENTATION. Because the active set is
        # nested in mu at every azimuth (verified above), this determines the
        # exact active set at ANY mu, not just the sampled ones -- 20771 bytes
        # per orientation against 20771 per (orientation, mu) frame.
        payload["critical_friction"] = {
            "inline": True, "never": -1, "per_theta": True,
            "frame_index": [[int(v) for v in row] for row in crit2],
            "meaning": "critical_friction[i_theta][i] is the largest mu INDEX "
                       "at which element i still slips at that orientation. "
                       "The active set is nested in mu at almost every "
                       "orientation but not quite all, so this is the LAST mu "
                       "that slips and not a guarantee that every lower mu "
                       "does; the viewer reads membership from the per-frame "
                       "slip array, which is exact regardless."}
        payload["per_theta"] = per_theta
    if crit is not None:
        # CRITICAL FRICTION, as a frame index per element so the viewer can
        # colour by it without a second decode. -1 means "never slips in this
        # range", which is categorically different from "slips at the lowest
        # mu" and is drawn grey rather than at the end of the ramp.
        payload["critical_friction"] = {
            "inline": True, "frame_index": [int(v) for v in crit],
            "never": -1,
            "meaning": "largest frame (hence mu) at which the element still "
                       "slips; the active set is nested so this is single "
                       "valued, which mu_sweep verifies before writing it"}
    (out / "fractal.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=str))

    if a.verify_stress:
        payload["stress_aggregation_error"] = verify_aggregation(
            net, cum[-1], fric, grid=a.verify_stress, workers=a.workers)
    if a.stress:
        vals, smeta = stress_volume(net, cum[-1], fric, grid=a.stress,
                                    workers=a.workers,
                                    sources=a.stress_sources)
        buf, enc = encode_log_u8(vals)
        got, bound = roundtrip(vals, buf, enc)
        print(f"  stress round trip: measured {got:.4%}  bound {bound:.4%}  "
              f"ratio {got / bound if bound else 0:.3f}")
        if bound > 0.0 and got > ROUNDTRIP_MARGIN * bound:
            raise SystemExit(
                f"stress quantisation {got:.4%} exceeds the bound "
                f"{bound:.4%}. Do not publish this payload.")
        (out / "fractal.stress.bin").write_bytes(buf)
        enc.update({"file": "fractal.stress.bin", "bytes": len(buf),
                    "sha256": hashlib.sha256(buf).hexdigest()[:16],
                    "measured_rel_err": got, "bound_rel_err": bound,
                    "at_frame": a.frames - 1, "at_lam": float(grid[-1]),
                    **smeta})
        payload["stress"] = enc
        (out / "fractal.json").write_text(
            json.dumps(payload, indent=1, sort_keys=True, default=str))
        print(f"  stress volume {len(buf) / 1024:.0f} kB, "
              f"{smeta['seconds'] / 60:.1f} min, "
              f"{smeta['us_per_pair']:.1f} us/pair")

    total = len(geom) + sum(v["bytes"] for v in arrays.values())
    print(f"  wrote fractal.json + 3 binaries, {total / 1024 ** 2:.2f} MiB "
          f"-> {out}")
    # The two sweeps carry different per-frame quantities, so the summary
    # reads whichever this payload actually has rather than assuming.
    last = stats[-1]
    if a.sweep == "theta_mu":
        import statistics as _st
        j = [v["jaccard"] for v in stats]
        ad = [v["added"] for v in stats]
        rm = [v["removed"] for v in stats]
        print(f"  grid: {len(stats)} solves over {a.n_theta} orientations "
              f"x {a.frames} frictions")
        print(f"  vs critically-stressed screening across the WHOLE grid: "
              f"Jaccard {min(j):.4f}-{max(j):.4f} (median {_st.median(j):.4f})")
        print(f"  interaction adds {min(ad)}-{max(ad)} elements and removes "
              f"{min(rm)}-{max(rm)}, over {n} total")
        act = [v["n_active"] for v in stats]
        print(f"  active ranges {min(act)}-{max(act)} "
              f"({min(act)/n:.1%}-{max(act)/n:.1%})")
        best = max(stats, key=lambda v: v["n_active"])
        worst = min(stats, key=lambda v: v["n_active"])
        print(f"  most active:  azimuth {best['theta']:.1f} deg, mu "
              f"{best['mu']:.3f} -> {best['n_active']}")
        print(f"  least active: azimuth {worst['theta']:.1f} deg, mu "
              f"{worst['mu']:.3f} -> {worst['n_active']}")
    elif a.sweep == "mu":
        never = sum(1 for v in payload["critical_friction"]["frame_index"]
                    if v < 0)
        print(f"  final: at mu {last['mu']:.3f}, {last['n_active']}/{n} slip "
              f"({last['n_active'] / n:.1%}); screening predicts "
              f"{last['n_screened']}; M0 {last['moment_Nm']:.3e} N m")
        print(f"  {never}/{n} ({never / n:.1%}) never slip anywhere in "
              f"mu {a.mu_min}-{a.mu_max}, so they have no critical friction")
    else:
        print(f"  final: {last['n_ever']}/{n} elements ever slipped "
              f"({last['n_ever'] / n:.1%}), "
              f"{last['events_cumulative']} events, "
              f"M0 {last['moment_cumulative_Nm']:.3e} N m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
