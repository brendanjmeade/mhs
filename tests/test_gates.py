"""pytest front end over the same gates ``run_all.py`` spawns.

Two runners, asserting different things, and only running both keeps the
cross-check meaningful:

* ``run_all.py`` spawns each gate and asserts its **exit code**;
* this file imports each gate and asserts its **return value**.

Both then cross-check the printed column-0 verdict. A gate whose return value
said PASS while its output said FAIL would be the worst of both, and neither
runner alone can see it.

The per-suite gate COUNT is asserted here too, not only in ``run_all.py``,
because discovery is a glob: pointed at the wrong directory it finds nothing and
every "all passed" check trivially holds. A green empty suite is the one failure
a test runner must not be able to report.
"""
from __future__ import annotations

import importlib.util
import inspect
import pathlib
import sys

import pytest

import run_all as R

#: Gates that must run in their own interpreter, with the reason recorded here
#: so it travels with the exemption. Both assert on a PRISTINE module table:
#: imported into a shared session they would observe what every earlier gate
#: already pulled in, so they would pass or fail on gate ORDER rather than on the
#: thing they are checking.
SUBPROCESS_ONLY = {
    "mhs/verify_import_hygiene":
        "asserts that `import mhs` pulls in neither sympy, matplotlib nor "
        "mhs_oracle; in a shared interpreter another gate has already imported "
        "all three",
    "parity/verify_oracle_provenance":
        "asserts which file each oracle import resolves to, so it needs a "
        "module table no other gate has touched",
}


def _gates() -> list:
    out = []
    for suite in sorted(R.SUITES):
        for path in R.discover(suite):
            gid = f"{suite}/{path.stem}"
            marks = [pytest.mark.slow] if gid in R.SLOW else []
            out.append(pytest.param(suite, path, id=gid, marks=marks))
    return out


GATES = _gates()


def test_gate_count():
    """The pinned per-suite counts, and that discovery agrees with them.

    Asserted independently of ``run_all.py``'s own check so neither runner can
    be the only thing standing between a mis-pointed glob and a green report.
    """
    total = 0
    for suite, (directory, expected) in sorted(R.SUITES.items()):
        assert directory.is_dir(), f"{suite}: {directory} is not a directory"
        found = R.discover(suite)
        assert len(found) == expected, (
            f"{suite}: {len(found)} gates in {directory.name}, expected "
            f"{expected} -- update SUITES in the same commit that adds or "
            f"removes a gate")
        total += expected
    assert len(GATES) == total, f"{len(GATES)} collected, {total} pinned"
    assert total > 0, "a green empty suite is not a pass"


def _import_gate(suite: str, path: pathlib.Path):
    """Import one gate under a SUITE-QUALIFIED module name.

    ``verify_pde_residual`` is expected to exist in more than one suite. Under
    the bare stem the second import would return the first suite's module from
    ``sys.modules`` and the test would silently check the wrong file.
    """
    name = f"_gate_{suite}_{path.stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]
        raise
    return mod


@pytest.mark.parametrize("suite,path", GATES)
def test_gate(suite, path, capsys):
    gid = f"{suite}/{path.stem}"
    if gid in SUBPROCESS_ONLY:
        import subprocess
        proc = subprocess.run([sys.executable, str(path)],
                              cwd=R.ROOT, capture_output=True, text=True,
                              timeout=R.DEFAULT_TIMEOUT)
        printed = R.verdict(proc.stdout)
        assert printed is not None, (
            f"{gid} printed no column-0 verdict\n{proc.stdout[-2000:]}")
        assert proc.returncode == 0, (
            f"{gid} exited {proc.returncode} ({SUBPROCESS_ONLY[gid]})\n"
            f"{proc.stdout[-4000:]}")
        assert printed == "PASS", f"{gid} printed {printed}"
        return

    mod = _import_gate(suite, path)
    main = getattr(mod, "main", None)
    assert callable(main), f"{gid} has no callable main()"

    # Whether main takes argv is read off its signature rather than special-cased
    # by name, so a second gate needing arguments does not fail mysteriously.
    params = [p for p in inspect.signature(main).parameters.values()
              if p.default is inspect.Parameter.empty
              and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    returned = main([]) if len(params) == 1 else main()

    printed = R.verdict(capsys.readouterr().out)
    assert printed is not None, f"{gid} printed no column-0 verdict"
    # The return value is the authority; the printed line is the cross-check.
    assert bool(returned) == (printed == "PASS"), (
        f"{gid} disagrees with itself: returned {returned!r}, printed "
        f"{printed} -- one of the two is lying and neither runner alone sees it")
    assert returned, f"{gid} returned {returned!r}"
