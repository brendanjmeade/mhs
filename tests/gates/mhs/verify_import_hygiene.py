#!/usr/bin/env python
"""Verify that importing ``mhs`` stays cheap, and that the API surface is honest.

This is the one gate that can pass before any arithmetic exists, and it guards
the most expensive measured fact in the project: **importing the vendored
oracle's Mindlin kernels costs 14.25 s**, because they build and lambdify ~90
symbolic matrices at module scope. A library cannot do that. Nothing stops it
creeping back in except a check that fails when it does.

Four clauses:

  [a] DISJOINTNESS. After ``import mhs`` and nothing else, neither ``sympy`` nor
      ``matplotlib`` nor ``mhs_oracle`` is in ``sys.modules``. This is the
      structural replacement for the usual two-copies asymmetry check: mhs has
      one oracle copy, so nothing can converge with it, but the live hazard is
      that the SHIPPED path and the oracle come to share an implementation -- at
      which point a parity gate would compare ``new + oracle`` against
      ``quadrature + oracle``, silently stop testing the shared half, and stay
      green with a smaller residual. Import disjointness makes that unreachable
      rather than merely discouraged.

  [b] COST, measured RELATIVE to ``import numba`` in this same process. mhs's
      import runs every ``@njit`` decorator (compilation defers to first call),
      and numba's own import is ~1 s and drifts between releases, so an absolute
      budget would measure the machine and numba's release notes rather than
      mhs. The measurement PRINTS on every run, pass or fail, so drift is visible
      before it is fatal.

  [c] SURFACE. Every documented name exists, with the documented signature --
      pinned verbatim, so an API change cannot be merged without editing this
      gate. That is the point: a new keyword is cheap to add and expensive to
      discover, and `order` was added to seven entry points behind this clause.
      A matrix function that quietly lost its ``out=`` would otherwise be found
      by a caller rather than by a gate.

  [d] CONVENTIONS THAT ARE CHEAP TO CHECK AND EXPENSIVE TO GET WRONG: the
      validation layer refuses a source or observer above the free surface,
      refuses ``eps <= 0``, refuses ``nu`` passed where a ``Material`` belongs,
      and ``Material.from_mu_nu`` refuses ``nu >= 1/2`` rather than returning an
      infinity.

Run from anywhere:  python tests/gates/mhs/verify_import_hygiene.py
"""
from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap

CHECKS: list[bool] = []

# Measured on the reference tree: importing the sympy-backed Mindlin oracle.
ORACLE_IMPORT_COST_S = 14.25


def check(label: str, ok: bool, note: str = "") -> bool:
    ok = bool(ok)
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {label:58s} {note}")
    return ok


# --------------------------------------------------------------------- [a] ---
#: The threading variables `mhs.parallel` sets for its CHILDREN. Listed here
#: rather than imported, because the probe below must not import mhs before it
#: has recorded the environment.
_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")

_PROBE = textwrap.dedent(f"""
    import json, os, sys, time
    keys = {_THREAD_ENV!r}
    env_before = {{k: os.environ.get(k) for k in keys}}
    t0 = time.perf_counter(); import numba; t_numba = time.perf_counter() - t0
    t0 = time.perf_counter(); import mhs;   t_mhs   = time.perf_counter() - t0
    env_after = {{k: os.environ.get(k) for k in keys}}
    print("@@" + json.dumps({{
        "t_numba": t_numba,
        "t_mhs": t_mhs,
        "sympy": "sympy" in sys.modules,
        "matplotlib": "matplotlib" in sys.modules,
        "oracle": any(m == "mhs_oracle" or m.startswith("mhs_oracle.")
                      for m in sys.modules),
        "scipy": "scipy" in sys.modules,
        "thread_env_touched": [k for k in keys
                               if env_before[k] != env_after[k]],
        "parallel_imported": "mhs.parallel" in sys.modules,
    }}))
""")


def probe() -> dict:
    """Import mhs in a PRISTINE interpreter and report what it dragged in.

    A subprocess is not an optimisation here, it is the whole measurement: in a
    shared interpreter another gate has already imported sympy, matplotlib and
    the oracle, so the check would pass or fail on gate ORDER rather than on mhs.
    """
    import json
    proc = subprocess.run([sys.executable, "-c", _PROBE],
                          capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"probe interpreter failed (rc {proc.returncode})")
    line = next(l for l in proc.stdout.splitlines() if l.startswith("@@"))
    return json.loads(line[2:])


def a_disjointness(info: dict) -> None:
    print("\n[a] DISJOINTNESS: what `import mhs` pulls in")
    for mod in ("sympy", "matplotlib", "mhs_oracle"):
        key = "oracle" if mod == "mhs_oracle" else mod
        check(f"a {mod} absent after `import mhs`", not info[key],
              "absent" if not info[key] else "PRESENT -- the shipped path must "
                                             "not import it")
    # scipy is not a dependency either; absent is the expectation, not a rule.
    print(f"       (scipy {'present' if info['scipy'] else 'absent'}; "
          f"not a dependency)")


# --------------------------------------------------------------------- [b] ---
def a_threads(info: dict) -> None:
    """`import mhs` must not touch the BLAS threading environment.

    ``mhs.parallel`` pins those variables for its spawned CHILDREN and restores
    them, which is a different thing from a library setting them at import time
    -- that decides something belonging to the program using it, and the
    decision is invisible until something else in the process gets slower.
    Measured in the pristine probe, so a variable the developer's shell happens
    to export cannot mask it.
    """
    touched = info["thread_env_touched"]
    check("a `import mhs` touches no BLAS threading variable",
          not touched,
          f"changed {touched}" if touched else
          "mhs.parallel pins them for its children and restores them")
    check("a ... and mhs.parallel is importable without doing so",
          info["parallel_imported"],
          "it is imported by `import mhs`, so the clause above covers it")


def b_cost(info: dict) -> None:
    from mhs import defaults
    over = info["t_mhs"] - info["t_numba"]
    budget = defaults.IMPORT_BUDGET_OVER_NUMBA_S
    print("\n[b] COST: `import mhs` relative to `import numba`, same process")
    print(f"       import numba {info['t_numba']:6.3f} s | "
          f"import mhs {info['t_mhs']:6.3f} s | "
          f"mhs over numba {over:+6.3f} s | budget {budget:.3f} s")
    print(f"       for scale, the sympy oracle costs "
          f"{ORACLE_IMPORT_COST_S:.2f} s to import")
    check("b `import mhs` within budget over numba", over < budget,
          f"{over:+.3f} s < {budget:.3f} s")


# --------------------------------------------------------------------- [c] ---
EXPECTED = {
    "disp_matrix": ["obs_pts", "tris", "material", "eps", "order", "out"],
    "stress_matrix": ["obs_pts", "tris", "material", "eps",
                      "subtract_eigenstress", "parts", "order", "out"],
    "total_stress_matrix": ["obs_pts", "tris", "material", "eps", "order",
                            "out"],
    "eigenstress_matrix": ["obs_pts", "tris", "material", "eps", "order",
                           "out"],
    "elastic_strain_matrix": ["obs_pts", "tris", "material", "eps",
                              "subtract_eigenstress", "order", "out"],
    "traction_matrix": ["obs_pts", "obs_normals", "tris", "material", "eps",
                        "subtract_eigenstress", "order", "out"],
    "interaction_matrix": ["tris", "material", "eps", "receiver", "source",
                           "obs_tris", "obs_pts", "order", "shrink",
                           "subtract_eigenstress", "out"],
}


def c_surface() -> None:
    import mhs
    print("\n[c] SURFACE: the matrix entry points and their signatures")
    for name, params in EXPECTED.items():
        fn = getattr(mhs, name, None)
        if not check(f"c {name} exists and is callable", callable(fn)):
            continue
        got = list(inspect.signature(fn).parameters)
        check(f"c {name} signature", got == params,
              "" if got == params else f"{got} != {params}")
    for name in ("disp_matrix", "stress_matrix", "traction_matrix",
                 "interaction_matrix"):
        check(f"c {name} defaults to order=0 (P0)",
              inspect.signature(getattr(mhs, name)).parameters["order"]
              .default == 0,
              "P0 is the default, so n_dof == n_src and nothing written for "
              "elements changes when the argument is ignored")
    check("c stress default subtracts the eigenstress",
          inspect.signature(mhs.stress_matrix)
          .parameters["subtract_eigenstress"].default is True,
          "subtract_eigenstress=True")
    check("c there is NO bare strain_matrix",
          not hasattr(mhs, "strain_matrix"),
          "a cutde-style strain_matrix + strain_to_stress path would return "
          "the TOTAL stress")


# --------------------------------------------------------------------- [d] ---
def d_conventions() -> None:
    import numpy as np

    import mhs
    print("\n[d] CONVENTIONS the validation layer must refuse")
    mat = mhs.Material(mu=30.0, lam=45.0)
    tri = np.array([[[0.0, 0.0, -2.0], [1.0, 0.0, -2.0], [0.0, 1.0, -2.0]]])
    obs = np.array([[0.3, 0.2, -1.0]])

    def refuses(label, fn, exc=ValueError):
        try:
            fn()
        except exc:
            return check(label, True, f"{exc.__name__}")
        except NotImplementedError:
            return check(label, False, "accepted it and reached the kernel")
        except Exception as e:  # noqa: BLE001
            return check(label, False, f"raised {type(e).__name__}: {e}")
        return check(label, False, "returned without raising")

    refuses("d obs above the free surface refused",
            lambda: mhs.disp_matrix(np.array([[0.0, 0.0, +1.0]]), tri, mat, 0.1))
    refuses("d source above the free surface refused",
            lambda: mhs.disp_matrix(obs, tri + np.array([0.0, 0.0, 3.0]),
                                    mat, 0.1))
    refuses("d eps <= 0 refused", lambda: mhs.disp_matrix(obs, tri, mat, 0.0))
    refuses("d nu where a Material belongs refused",
            lambda: mhs.disp_matrix(obs, tri, 0.25, 0.1), TypeError)
    refuses("d wrong tris shape refused",
            lambda: mhs.disp_matrix(obs, tri[0], mat, 0.1))
    refuses("d from_mu_nu(nu >= 1/2) refused",
            lambda: mhs.Material.from_mu_nu(30.0, 0.5))

    # the (mu, lam) round trip, since from_mu_nu is the one adapter
    m = mhs.Material.from_mu_nu(30.0, 0.25)
    check("d from_mu_nu(30, 1/4) gives lam == mu", abs(m.lam - 30.0) < 1e-12,
          f"lam = {m.lam:.12g}")
    check("d nu round trip", abs(m.nu_if_you_must - 0.25) < 1e-15,
          f"nu = {m.nu_if_you_must:.15g}")

    # and the byte arithmetic, which the refusal message quotes
    plan = mhs.chunking.chunk_plan(10_000, 10_000, "stress")
    gib = plan.bytes_total / 1024**3
    check("d stress is 216 B per obs/source pair",
          mhs.chunking.pair_bytes("stress") == 216,
          f"{mhs.chunking.pair_bytes('stress')} B; "
          f"10k x 10k = {gib:.1f} GiB in {plan.n_chunks} chunks")
    check("d parts=True is twice that",
          mhs.chunking.pair_bytes("stress", parts=True) == 432, "432 B")
    # Enough SOURCES to pass the ceiling, not just enough obs: at one triangle
    # even 40k observers is 8.6 MB. The inputs stay small (2000 triangles is
    # 144 kB) precisely because the refusal must happen before any allocation.
    big_tris = np.repeat(tri, 2000, axis=0)
    big_obs = np.repeat(obs, 30_000, axis=0)
    want = mhs.chunking.chunk_plan(30_000, 2000, "stress").bytes_total
    refuses(f"d an over-ceiling request ({want / 1024**3:.0f} GiB) is refused",
            lambda: mhs.stress_matrix(big_obs, big_tris, mat, 0.1), MemoryError)


def main() -> bool:
    print("=" * 76)
    print("Import hygiene, API surface and the conventions the wrapper refuses")
    print("=" * 76)
    info = probe()
    a_disjointness(info)
    a_threads(info)
    b_cost(info)
    c_surface()
    d_conventions()
    print("-" * 76)
    if all(CHECKS):
        print(f"PASS: import stays cheap and the surface is honest "
              f"({len(CHECKS)} checks)")
    else:
        print(f"FAIL: {sum(1 for c in CHECKS if not c)} of {len(CHECKS)} "
              f"checks failed")
    return all(CHECKS)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
