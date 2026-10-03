#!/usr/bin/env python
"""Verify the vendored oracles are the files they claim to be, and unedited.

A parity gate compares the new implementation against a frozen reference. That
comparison means nothing unless three things hold, and this gate is each of them:

  [a] WHICH FILE each oracle name resolves to. Pinning only "something
      importable exists" is not enough: the upstream repo installs these same
      modules under their original names via an editable ``.pth``, so a copy
      vendored under the original name would resolve to whichever came first on
      ``sys.path``. The recorded failure upstream is exact -- three files
      hardcoded the other copy's prefix, resolved SUCCESSFULLY to it, and the
      parity clause compared a copy against itself and still printed PASS; the
      residual merely dropped from ~1e-12 to ~1e-16, which no tolerance rejects.
      Hence the rename to ``mhs_oracle.*``, and hence a path-tail check.

  [b] THE CONTENTS, by sha256. A frozen oracle is never improved, so a changed
      hash is either a mistake or a deliberate edit that must update this
      manifest in the SAME commit, with the commit message saying why the hash
      moved. That makes the edit visible in review instead of silent.

  [c] WHETHER THE VENDORING IS STILL FAITHFUL. Each record carries TWO hashes:
      the vendored file's, and the upstream file's as recorded when it was
      vendored. ``clq`` uses relative imports throughout, so it vendors
      byte-identically and the two agree for every file. A record where they
      DIFFER is a file that needed an edit to live under the new package name,
      and the manifest showing exactly that set is itself the audit. The gate
      cannot recompute the upstream hash -- the upstream tree need not exist on
      this machine -- which is the point: a value the gate could compute is a
      value that could drift with it.

Regenerate deliberately, never casually:

    python tests/gates/parity/verify_oracle_provenance.py --write

Run from anywhere:  python tests/gates/parity/verify_oracle_provenance.py
"""
from __future__ import annotations

import hashlib
import importlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MANIFEST = HERE / "oracle_manifest.json"

CHECKS: list[bool] = []

#: module name -> the repo-relative path tail its ``__file__`` must end with.
#: The tail pins WHICH COPY the name resolves to, not merely that it imports.
ORACLES = {
    "mhs_oracle.clq": "src/mhs_oracle/clq/__init__.py",
    "mhs_oracle.clq.api": "src/mhs_oracle/clq/api.py",
    "mhs_oracle.clq.defaults": "src/mhs_oracle/clq/defaults.py",
    "mhs_oracle.clq.frame": "src/mhs_oracle/clq/frame.py",
    "mhs_oracle.clq.kernels": "src/mhs_oracle/clq/kernels.py",
    "mhs_oracle.clq.moments": "src/mhs_oracle/clq/moments.py",
    "mhs_oracle.clq.pointwise": "src/mhs_oracle/clq/pointwise.py",
    "mhs_oracle.clq.primitives": "src/mhs_oracle/clq/primitives.py",
    "mhs_oracle.clq.quadrature": "src/mhs_oracle/clq/quadrature.py",
    "mhs_oracle.clq.shape": "src/mhs_oracle/clq/shape.py",
}

#: Where the vendored copies came from, for the record. Not checkable here --
#: the upstream tree need not exist on this machine -- which is why the upstream
#: hashes are stored rather than computed.
UPSTREAM = "moss-org src/clq/ @ ad0e992 (2026-10-03)"

#: Names that must be absent from an oracle module, because their presence would
#: mean the vendored copy had drifted into being the SHIPPED package. ``clq``'s
#: closed forms are numpy; if one ever grows a numba decorator it is no longer an
#: independent reference for the numba implementation being gated against it.
FORBIDDEN_IN_ORACLE = ("njit", "jit")


def check(label: str, ok: bool, note: str = "") -> bool:
    ok = bool(ok)
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {label:58s} {note}")
    return ok


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(name: str) -> tuple[pathlib.Path, str]:
    """Import ``name`` and return (resolved path, sha256).

    An import failure is a problem, not a skip: without the oracle the parity
    clauses that rest on it are unguarded, which is the state this whole gate
    exists to prevent.
    """
    mod = importlib.import_module(name)
    f = getattr(mod, "__file__", None)
    if f is None:
        raise ImportError(f"{name} has no __file__ (namespace package?)")
    path = pathlib.Path(f).resolve()
    return path, sha256(path)


def load() -> dict:
    if not MANIFEST.exists():
        return {}
    return json.loads(MANIFEST.read_text())


def write(records: dict) -> None:
    payload = {k: records[k] for k in sorted(records)}
    MANIFEST.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"  wrote {MANIFEST.relative_to(ROOT)} "
          f"({len(payload)} oracles pinned)")


def main(write_mode: bool = False) -> bool:
    print("=" * 76)
    print("Oracle provenance: which file, unedited, and still faithfully vendored")
    print("=" * 76)
    print(f"  upstream: {UPSTREAM}")

    pinned = load()
    live: dict[str, dict] = {}

    print("\n[a] RESOLUTION: each oracle name resolves inside src/mhs_oracle/")
    for name, tail in sorted(ORACLES.items()):
        try:
            path, digest = resolve(name)
        except ImportError as exc:
            check(f"a {name} imports", False, f"ImportError: {exc}")
            continue
        rel = path.as_posix()
        ok = rel.endswith(tail)
        check(f"a {name}", ok,
              tail if ok else f"resolved to {rel}, expected .../{tail}")
        live[name] = {"path": tail, "sha256": digest,
                      "sha256_upstream": pinned.get(name, {}).get(
                          "sha256_upstream", digest),
                      "upstream": pinned.get(name, {}).get("upstream", UPSTREAM)}

    if write_mode:
        print("\n[--write] regenerating the manifest from the live files")
        print("  NOTE: sha256_upstream is PRESERVED where already pinned and "
              "seeded from")
        print("        the live hash only for a new record. It is never "
              "recomputed, so a")
        print("        vendored edit cannot quietly redefine what 'faithful' "
              "means.")
        write(live)
        print("\nPASS: manifest written -- commit it with the reason the hash moved")
        return True

    if not pinned:
        check("b manifest exists", False,
              f"{MANIFEST.name} missing -- run with --write")
        print("-" * 76)
        print("FAIL: no manifest to check against")
        return False

    print("\n[b] CONTENTS: sha256 against the manifest")
    for name in sorted(ORACLES):
        if name not in live:
            continue
        if name not in pinned:
            check(f"b {name} is pinned", False,
                  "resolved but NOT in the manifest -- adding an oracle "
                  "without pinning it is red")
            continue
        want, got = pinned[name]["sha256"], live[name]["sha256"]
        check(f"b {name}", want == got,
              f"{got[:12]}" if want == got
              else f"CONTENTS CHANGED: pinned {want[:12]}, now {got[:12]}")
        if pinned[name]["path"] != live[name]["path"]:
            check(f"b {name} path pin", False,
                  f"pinned {pinned[name]['path']}, now {live[name]['path']}")

    stale = sorted(set(pinned) - set(ORACLES))
    check("b no stale manifest records", not stale,
          "" if not stale else f"pinned but no longer in ORACLES: {stale}")

    print("\n[c] FAITHFULNESS: vendored vs upstream hash")
    edited = [n for n in sorted(live)
              if pinned.get(n, {}).get("sha256_upstream") != live[n]["sha256"]]
    identical = [n for n in sorted(live) if n not in edited]
    print(f"       byte-identical to upstream: {len(identical)} of {len(live)}")
    for n in edited:
        print(f"       EDITED: {n} (upstream "
              f"{pinned[n]['sha256_upstream'][:12]}, vendored "
              f"{live[n]['sha256'][:12]})")
    # clq uses relative imports throughout, so nothing needed an edit. If that
    # ever changes, this clause must be updated in the same commit -- which is
    # the review moment it exists to create.
    check("c every clq file is byte-identical to upstream", not edited,
          f"{len(identical)}/{len(live)}"
          if not edited else f"{len(edited)} edited: {edited}")

    print("\n[d] THE ORACLE IS NOT THE PACKAGE")
    for name in sorted(ORACLES):
        if name not in live:
            continue
        path = ROOT / live[name]["path"]
        text = path.read_text()
        bad = [tok for tok in FORBIDDEN_IN_ORACLE if f"@{tok}" in text]
        if bad:
            check(f"d {name} carries no numba decorator", False,
                  f"found {bad} -- a numba'd oracle is no longer an "
                  f"independent reference for a numba implementation")
    check("d the oracle imports no part of mhs",
          not any("import mhs" in (ROOT / live[n]["path"]).read_text()
                  .replace("import mhs_oracle", "")
                  for n in live),
          "the reference must not read the thing it is the reference for")

    print("-" * 76)
    if all(CHECKS):
        print(f"PASS: oracles resolve, match their hashes and are faithful "
              f"({len(CHECKS)} checks)")
    else:
        print(f"FAIL: {sum(1 for c in CHECKS if not c)} of {len(CHECKS)} "
              f"checks failed")
    return all(CHECKS)


if __name__ == "__main__":
    sys.exit(0 if main(write_mode="--write" in sys.argv) else 1)
