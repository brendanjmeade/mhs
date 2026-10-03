#!/usr/bin/env python
"""Run every gate and report. The authoritative runner.

THE PER-SUITE COUNT IS ASSERTED, because the discovery is a glob: pointed at the
wrong directory it finds nothing, every "all passed" check trivially holds, and
the exit code is 0. A green empty suite is the one failure a test runner must not
be able to report, so an unexpected count is itself a failure.

Each gate runs as a SUBPROCESS and its **exit code** is the verdict. The printed
``PASS:``/``FAIL:`` line is a cross-check, not the signal: a gate whose exit code
and printed line disagree is reported as MISMATCH rather than trusted. (Scraping
alone is not enough -- in the reference tree 24 of 43 gates once printed FAIL and
exited 0, and nothing could tell.)

The suites are split by COST, not by topic, because a suite is the unit
``--suite`` selects and CI needs "the gates that need no sympy oracle" to be a
directory rather than a filter:

    mhs     the shipped package, plus parity against the numpy-only clq oracle.
            numpy + numba + cutde. CI runs these on every push.
    parity  compares the shipped package against the SYMPY half of the vendored
            oracle. Each gate pays ~14 s to import it. Nightly.
    oracle  the vendored copies are themselves sound -- the derivation checks,
            ported from upstream. A sha256 proves an oracle did not CHANGE;
            these prove it was ever RIGHT.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

#: suite -> (directory, expected gate count). The count is a pin, not a hint.
SUITES = {
    "mhs": (HERE / "gates" / "mhs", 2),
    "parity": (HERE / "gates" / "parity", 1),
    "oracle": (HERE / "gates" / "oracle", 0),
}

#: ``<suite>/<stem>`` ids that take over 60 s; skipped by ``--fast``.
SLOW: set[str] = set()

DEFAULT_TIMEOUT = 1800


def discover(suite: str) -> list[pathlib.Path]:
    """Every ``verify_*.py`` in a suite directory.

    Support files are named with a leading underscore so this glob skips them.
    """
    directory, _ = SUITES[suite]
    return sorted(p for p in directory.glob("verify_*.py")
                  if p.name != "run_all.py")


def verdict(stdout: str) -> str | None:
    """The gate's printed verdict: the last line starting at COLUMN 0 with
    ``PASS`` or ``FAIL``.

    Column 0 matters in both directions. Per-clause lines are indented, so an
    indented ``PASS`` is a clause and not the verdict; and a gate that prints a
    mid-run verdict is summarised by its final one.
    """
    found = None
    for line in stdout.splitlines():
        if line.startswith("PASS"):
            found = "PASS"
        elif line.startswith("FAIL"):
            found = "FAIL"
    return found


def main(argv=None) -> bool:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-k", "--only", help="substring of <suite>/<gate>")
    ap.add_argument("--suite", choices=sorted(SUITES))
    ap.add_argument("--fast", action="store_true", help="skip the SLOW gates")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = ap.parse_args(argv)

    problems: list[str] = []
    selected: list[tuple[str, pathlib.Path]] = []
    for suite in sorted(SUITES):
        if args.suite and suite != args.suite:
            continue
        directory, expected = SUITES[suite]
        if not directory.is_dir():
            problems.append(f"{suite}: {directory} is not a directory")
            continue
        found = discover(suite)
        if len(found) != expected:
            problems.append(
                f"{suite}: {len(found)} gates in {directory.name}, "
                f"expected {expected} -- update SUITES in the same commit that "
                f"adds or removes a gate")
        for path in found:
            gid = f"{suite}/{path.stem}"
            if args.only and args.only not in gid:
                continue
            if args.fast and gid in SLOW:
                continue
            selected.append((gid, path))

    if args.list:
        for gid, _ in selected:
            print(gid)
        return not problems

    if not selected and not problems:
        problems.append("no gates selected -- a green empty run is not a pass")

    n_fail = 0
    for gid, path in selected:
        t0 = time.time()
        try:
            proc = subprocess.run([sys.executable, str(path)], cwd=ROOT,
                                  capture_output=True, text=True,
                                  timeout=args.timeout)
            rc, out = proc.returncode, proc.stdout
        except subprocess.TimeoutExpired:
            rc, out = 124, ""
        printed = verdict(out)
        dt = time.time() - t0
        if printed is None:
            tag = f"PROBLEM(no column-0 verdict/rc{rc})"
            n_fail += 1
        elif (printed == "PASS") != (rc == 0):
            tag = f"MISMATCH({printed}/rc{rc})"
            n_fail += 1
        else:
            tag = printed
            n_fail += rc != 0
        print(f"{tag:12s} {gid:48s} {dt:6.1f} s")
        if rc != 0 or printed != "PASS":
            tail = (out.strip().splitlines() or ["(no stdout)"])[-12:]
            for line in tail:
                print(f"             | {line}")
            if proc_err := getattr(locals().get("proc", None), "stderr", ""):
                for line in proc_err.strip().splitlines()[-6:]:
                    print(f"             ! {line}")

    for p in problems:
        print(f"PROBLEM      {p}")
    total = len(selected)
    ok = n_fail == 0 and not problems
    print()
    if ok:
        print(f"{total} / {total} gates PASS")
    else:
        print(f"{total - n_fail} / {total} gates pass, "
              f"{n_fail} failed, {len(problems)} problem(s)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
