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
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cascade as C                                         # noqa: E402
import friction as Fr                                       # noqa: E402
import kernel as K                                          # noqa: E402
import measure as M                                         # noqa: E402
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="patch", choices=sorted(R.CONFIGS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--drop-ratio", type=float, default=Fr.DROP_RATIO)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
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
