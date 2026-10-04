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

For **on-fault stress interaction** — the object Coulomb, rate-and-state and
cycle models want — ask for it directly rather than contracting a stress matrix
you cannot allocate:

```python
K = mhs.interaction_matrix(tris, mat, eps=0.1,          # (n, n, 3, 3)
                           receiver=("strike", "dip", "normal"),
                           source=("strike", "dip", "tensile"))

K = mhs.interaction_matrix(tris, mat, eps=0.1,          # (n, n, 1, 1), 0.75 GiB
                           receiver="strike", source="strike")  # at n = 10k

T = mhs.traction_matrix(obs, normals, tris, mat, eps=0.1)   # (n_obs, 3, n_src, 3)
```

Both contract **inside** the assembly loop, so the stress tensor of one source
is built and discarded rather than stored. Same arithmetic to 1e-15 (gated as an
identity against `stress_matrix`), same speed, and 3x to 27x less memory --
which is the difference between a request the ceiling allows and one it refuses.

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
- **Cartesian slip.** `mhs.tdcs` is the one statement of the per-triangle
  (strike, dip, tensile) frame -- cutde's convention, dip pointing *up* --
  with `to_cartesian` / `from_cartesian` to cross it. The matrix functions take
  Cartesian slip, because a convention restated in each caller is a convention
  restated wrongly in one of them. `interaction_matrix` reads its frames from
  there, and so does `verify_cutde_limit`, which is what anchors the convention
  externally instead of merely to a second copy of itself.
- **Full `(3, 3)` tensors, not Voigt-6**, so no component ordering or
  factor-of-two convention is stated anywhere in this package.
- **`order` in {0, 1, 2}** selects P0, P1 or P2 nodal slip. The source axis is
  slip DOFs, element-major with the node index fastest, so `n_dof = n_src * K`
  with `K = 1, 3, 6`. At the default `order=0` that is `n_src` and nothing
  changes. Uniform nodal slip reproduces the P0 answer exactly -- gated as an
  identity, which is also what catches a partial write into the wider buffer.
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

The better escape is usually to **ask for less**. Per obs/source pair:

| | bytes/pair | 10k x 10k |
|---|---|---|
| `stress_matrix` | 216 | 20.1 GiB |
| `disp_matrix`, `traction_matrix` | 72 | 6.7 GiB |
| `interaction_matrix`, 2 shear x 2 slip | 32 | 3.0 GiB |
| `interaction_matrix`, strike only | 8 | 0.75 GiB |

At `order > 0` the counts are per DOF rather than per element, so P1 is 9x and
P2 is 36x the pair count of P0.

Nothing about the calculation changes across those rows -- the component
selection is only how much of the result you keep.

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

Usable, with a known cost ceiling.

Working and gated: the full half-space kernel (closed-form direct and image
R-family, adaptive quadrature for the image Q-family), all seven matrix entry
points, the eigenstress split, and the cutde anchor. **On-fault stress is at
machine precision** (1e-15) at every realistic collocation point, including the
top element of a surface-breaking fault.

The honest limits:

- **The free-surface condition is satisfied to O(eps^2), not exactly.** Measured
  3.3e-3 at eps/h = 0.1 and 3.3e-5 at eps/h = 0.01, falling as eps^2 (order
  1.90-1.96 measured). This is a property of the mollified *kernel*, not of the
  integration, and it is the number to quote: an exact free surface via the
  potential-convolution route is foreclosed, because the Cortez blob has
  algebraic tails. `docs/derivation.md` has the proof and the measurements.
- **Cost.** About 1.1 ms per obs/source pair for stress, so 1k x 1k is minutes
  and 10k x 10k is tens of hours. The image R-family is 57% of it and has an
  unoptimised per-record inner loop; the Q-family was 89% until the same
  restructure took it to 3%.
- **P0 only through the public API.** The kernels support P1/P2 nodal density
  and the gates cover all three, but the assembly path fixes order 0.
- **No CI.** The gates are run by hand.

## License

MIT.
