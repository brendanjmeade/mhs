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


def roundtrip(v: np.ndarray, buf: bytes, meta: dict) -> tuple[float, float]:
    """``(worst measured relative error, arithmetic bound)``."""
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="patch", choices=sorted(R.CONFIGS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--drop-ratio", type=float, default=Fr.DROP_RATIO)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
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
    mat, info = K.assemble(net, mu=R.MU, lam=R.LAM, workers=a.workers)
    fric = Fr.build_friction(net, drop_ratio=a.drop_ratio)
    P = K.project(mat, fric.slip_dir, fric.slip_dir)
    lam0 = float(np.nanmin(fric.first_failure_lambda()))
    lam_max = LAM_MAX_FACTOR * lam0
    print(f"  loading to lam {lam_max:.5f} = {LAM_MAX_FACTOR}x the first "
          f"failure ({lam0:.5f})")

    cum, inc, stats, grid = snapshots(net, fric, P, n_frames=a.frames,
                                      lam_max=lam_max)
    n = net.n_elements

    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    worst_ratio = 0.0
    for name, arr in (("slip", cum), ("incr", inc)):
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
        meta.update({"bytes": len(buf), "frames": a.frames, "n": n,
                     "order": "frame-major: buf[f*n + i] is element i at "
                              "frame f",
                     "sha256": hashlib.sha256(buf).hexdigest()[:16],
                     "measured_rel_err": got, "bound_rel_err": bound})
        arrays[name] = meta

    geom = np.ascontiguousarray(net.tris, dtype=np.float32).tobytes()
    (out / "fractal.geom.bin").write_bytes(geom)

    cell = net.meta["cascade_cell_km"]
    try:
        win = M.window_for(cell, R.BOX_XY)
        d_fit = M.dimension(net.tris, n_radii=10, **win)
        d_geom = {"known": net.meta["d_geom_known"],
                  "calibrated": M.calibrated(d_fit),
                  "window_km": [win["r_lo"], win["r_hi"]]}
    except ValueError:
        d_geom = None

    payload = {
        "schema": SCHEMA, "nodata": NODATA, "levels": LEVELS,
        "config": a.config, "unit": "km of slip",
        "n_elements": n, "n_faults": net.n_faults, "frames": a.frames,
        "lam": [float(v) for v in grid],
        "geometry": {"file": "fractal.geom.bin", "dtype": "float32",
                     "shape": [n, 3, 3], "bytes": len(geom),
                     "sha256": hashlib.sha256(geom).hexdigest()[:16]},
        "arrays": {k: {"file": f"fractal.{k}.bin", **v}
                   for k, v in arrays.items()},
        "fault_id": {"inline": True},
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
    print(f"  final: {stats[-1]['n_ever']}/{n} elements ever slipped "
          f"({stats[-1]['n_ever'] / n:.1%}), "
          f"{stats[-1]['events_cumulative']} events, "
          f"M0 {stats[-1]['moment_cumulative_Nm']:.3e} N m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
