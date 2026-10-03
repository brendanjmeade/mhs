# mhs

Mollified Mindlin half-space triangular-dislocation Green's functions, as
matrices. For a **homogeneous half space with no topography**.

```python
import numpy as np, mhs

mat  = mhs.Material(mu=30.0, lam=30.0)              # (mu, lam), never nu
G    = mhs.disp_matrix(obs, tris, mat, eps=0.1)     # (n_obs, 3, n_src, 3)
u    = np.einsum("oisk,sk->oi", G, slip)            # slip is CARTESIAN

S    = mhs.stress_matrix(obs, tris, mat, eps=0.1)   # (n_obs, 3, 3, n_src, 3)
                                                    # ELASTIC: eigenstress removed
```

## Why this exists

In this configuration **nothing is solved.** The free surface lives in the
Mindlin kernel, the half space is laterally and vertically infinite so there are
no sides or base, there is no material interface, and fault slip is prescribed
data. No linear system, no collocation, no free term, no boundary density, no box
truncation.

That matters because a boundary element method's one genuine weak spot is elastic
stress within about one element of a *solved* boundary — 10–45 % with
piecewise-constant density, and not improvable by shrinking the mollification
width, because the production value already sits at its optimum there. Putting
the free surface in the kernel removes that boundary entirely. Measured, against
the discretized alternative:

| | discretized free surface | this kernel |
|---|---|---|
| near-surface stress error | 10–45 % | kernel residual, ~1e-4 |
| improves as eps falls? | no (already optimal) | yes |
| far-field truncation | box-size dependent | none (infinite half space) |

And what it gives you that a classical triangular dislocation element does not: a
fault of **finite width** `eps`, and an **interpretable stress on and near the
fault**, because the eigenstress `C:eps*` that every mollified double layer
carries is subtracted in the readout.

## Conventions

- **Half space `z <= 0`**, free surface at `z = 0`, sources buried. The wrapper
  refuses an observer or a vertex above the surface.
- **`(mu, lam)`, never `nu`.** `1/(1 - 2 nu)` diverges toward incompressibility,
  and a `lam`/`mu` swap is invisible at `nu = 1/4`. `Material.from_mu_nu` is the
  one boundary adapter. See `src/mhs/materials.py`.
- **Cartesian slip.** `mhs.tdcs` builds the rotation to the per-triangle
  (strike, dip, tensile) frame; no matrix function takes it, because cutde's dip
  direction points *up* and restating that convention is how it gets restated
  wrongly.
- **Full `(3, 3)` tensors, not Voigt-6**, so no component ordering or
  factor-of-two convention is stated anywhere in this package.
- **`eps > 0`, scalar or one per source triangle.** There is no `"auto"`: that
  rule needs a mesh spacing, and these functions take a vertex array.

## Elastic, total, eigenstress

`stress_matrix` returns the **elastic** stress by default. The raw mollified
field diverges like `1/eps` on the source surface — the analytic on-fault peak is
`(3/4) mu s / eps` — and that divergence *is* the eigenstress the mollification
put there. Three entry points, and `total - eigenstress == elastic` is an
algebraic identity the gates assert at 1 ulp:

| | |
|---|---|
| `stress_matrix` | elastic; what you want |
| `total_stress_matrix` | raw mollified field, eigenstress included |
| `eigenstress_matrix` | the `C:eps*` term alone; zero at `eps = 0` |

There is deliberately **no** bare `strain_matrix`. cutde's idiom is
`strain_matrix` then `strain_to_stress`, and for a mollified kernel that path
silently returns the total stress — the single failure this package exists to
prevent. `elastic_strain_matrix` is the strain readout, named so the elastic
choice is visible.

## Memory

A stress matrix is 216 bytes per observation/source pair, so 10k × 10k is 20 GiB.
The ceiling **refuses** rather than trying, because a request that size does not
fail on most machines — it swaps, and looks like a hang. The refusal carries the
arithmetic. Escapes: `out=` (a memmap, a reused slab, or a float32 buffer to
halve it) and input slicing, with `mhs.chunking.chunk_plan()` reporting the
numbers without allocating.

## Install and test

```bash
pip install -e .[test,tde]
python tests/run_all.py --suite mhs      # the oracle-free suite
pytest -m "not slow" tests/test_gates.py
```

`tests/gates/` holds the gates: each is `main() -> bool`, prints `PASS:`/`FAIL:`
at column 0 as its last such line, and exits on it. `tests/run_all.py` is the
authoritative runner and asserts the **exit code**, cross-checking the printed
line — a gate whose two disagree is reported as a mismatch rather than trusted.
Per-suite gate counts are pinned, because a green empty suite is the one failure
a test runner must not be able to report.

`src/mhs_oracle/` holds vendored **frozen** reference implementations, pinned by
path and sha256. They are never improved and the shipped path never imports
them; a gate asserts that disjointness, because an implementation shared between
the new code and its oracle would make the parity check a tautology.

## Status

Early. The validation layer, the byte budget, the `out=` contract and the API
surface are complete and gated; the closed-form kernels are in progress. See the
plan for the order of work and an honest statement of which parts are engineering
and which are research.

## License

MIT.
