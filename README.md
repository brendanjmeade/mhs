# mhs

Mollified Mindlin half-space triangular-dislocation Green's functions, as
matrices. For a **homogeneous half space with no topography**.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23242169.svg)](https://doi.org/10.5281/zenodo.23242169)

Project site, with both figures interactive:
**<https://brendanjmeade.github.io/mhs/>** (`website/`).

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

K = mhs.interaction_matrix(tris, mat, eps=0.1, order=2,  # (V+E, V+E, 3, 2)
                           layout="continuous",          # slip C0 across edges,
                           source=("strike", "dip"))     # collocated AT the nodes
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

- **Half space `z <= 0`**, free surface at `z = 0`. A source may reach the
  surface: one whose smeared slip reaches above `z = 0` is FOLDED back into
  the body (`mhs.kernels.fold`, below). The wrapper refuses an observer or a
  vertex above the surface.
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
- **`layout` in {`"discontinuous"`, `"continuous"`}** says what the source axis
  is. The default is the element-major axis above. `"continuous"` (P2 only)
  makes it the SHARED nodes of the mesh -- welded vertices, then edge
  midpoints, `n_dof = V + E`, one Cartesian slip vector each, the slip C0
  across every edge -- built by `mhs.shared_nodes(tris)`, which a network
  passes with `groups=fault_id` so that two faults meeting along a line keep
  separate nodes there. See **Continuous slip** below.

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

## Continuous slip

A P0 slip is a staircase, and every riser is a line dislocation. The elastic
traction read *between* the centroids therefore has a floor set by the
staircase and not by `eps`: on a penny crack at `h = a/8` the on-fault page
measures 8 % of the stress drop in the facet cores, the same for the mollified
and the classical kernel, and refinement does not move it, because the
centroid readout of a prescribed slip does not converge either (2.2 % at
`a/h = 6`, 2.5 % with `eps` halved).

`layout="continuous"` is quadratic slip that is continuous across every
element edge: one vector per shared node, collocated at the nodes themselves,
unshrunk. A kink is what is left at an edge, and a kink is admissible at a
mollified collocation point where a jump is not. Nothing in the kernels
changes -- a continuous field is a discontinuous one whose shared nodes agree,
so the assembler scatters each element's six P2 columns into the shared ones
and the element-major transient is never built. Measured on the penny crack
20 radii deep, dipping, rim exactly on the circle, nu = 0.30, locked rim
(`tests/gates/mhs/verify_shared_nodes.py`):

| a/h = 6, eps/h | readout of Eshelby's slip, median error | stress-drop solve, rms / peak slip | moment | tensile (Sneddon), rms / peak |
|---|---|---|---|---|
| **continuous P2**, 0.1 | 1.67 eps/a = 2.8 % | **1.6 %** | +0.8 % | **2.4 %** |
| continuous P2, 0.05 | 1.70 eps/a = 1.4 % | **1.1 %** | -1.9 % | 2.7 % |
| P0 centroids, 0.1 | 2.2 % | 6.4 % | +10 % | 4.3 % |
| P0 centroids, 0.05 | 2.5 % | 5.8 % | +9 % | 5.1 % |

The nodal readout error is `eps`'s -- it halves when `eps` halves, the same at
vertices and midpoints, the same at `a/h = 8` (1.68, 1.71) -- where the P0
one is the staircase's. Continuous P1 is refused: measured 15-22 % too stiff,
its kink sits at the collocation vertex with no node to carry the curvature.

Three things the layout does not do. `sym(-K)` loses positive definiteness
as `eps/h` falls and the mesh grows: +13 at `eps = 0.1 h`, -33 at `0.05 h`
on the gate's a/h = 6 disc, and on a production system (three rough faults at
a/h = 16-24, 7058 elements, 14,459 nodes) 37 negative modes out of 13,940 at
`0.1 h` -- 23 of them went with the 207 nodes within `2 eps` of the free
surface, where the smeared source straddles it (the fold removes those: the
kernel error, not the layout, was the cause), and 14 are broad body modes on
the faults themselves -- and at `0.2 h` three on one fault's near-surface
nodes plus a single long-wavelength mode at -0.7 % of the median, of the kind
the P0 operators carry on that system too. An LU does not care, a
complementarity solver must be told: projected Gauss-Seidel overflows on that
system at `0.1 h`, and an active-set Newton method thrashes there too. The
cure is the symmetric Galerkin form of the same kernel (test and trial both
P2), positive definite at every eps, which with the fold matches classical
P0 on a surface-breaking dipping fault to 1 % (Status). A crack front is still located to the node spacing,
so a Coulomb solve with a free front gains nothing without a front-conforming
mesh. And on a faceted
fault the nodal unknown is a VECTOR by construction -- a continuous magnitude
times a per-facet direction leaves a line dislocation at every crease with a
`1/eps` core at its irregular vertices, which the vector removes -- but a
log-type crease term remains; read the normal traction within a facet of a
crease at scales of two elements or more.

What it costs: the P2 kernels are ~110 us per observer-source pair serially
against ~40 at P0, and a disc has about twice as many nodes as elements, so
the matrix is some six times the P0 one. It parallelises by RECEIVER rows
(`mhs.parallel.by_rows`), not by source chunks, because a shared node's
column is written by every element touching it:

```python
sn = mhs.shared_nodes(tris, groups=fault_id)
call = functools.partial(mhs.interaction_matrix, tris, mat, 0.1, order=2,
                         layout="continuous", shared=sn,
                         receiver=("strike", "dip", "normal"),
                         source=("strike", "dip"))
K = mp.by_rows(call, sn.n, workers=12)             # (V+E, V+E, 3, 2), bitwise serial
u = mhs.disp_matrix(obs, tris, mat, 0.1, order=2,  # any entry point takes the
                    layout="continuous", shared=sn)  # nodal slip the same way
```

`sn.expand(slip)` is the element-major density the `order=2` entry points
consume, `sn.integrate(slip)` the moment (exact for a quadratic), `sn.front`
the boundary nodes inside the body a solve locks, `sn.trace` the ones on
`z = 0` it leaves free.

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

## Across cores

The matrix splits by source element -- each source owns a disjoint column
slice -- so it parallelises across **processes**, not threads:

```python
import functools, mhs, mhs.parallel as mp

call = functools.partial(mhs.interaction_matrix, material=mat, eps=0.1,
                         obs_tris=tris, receiver="strike", source="strike")
K = mp.by_source(call, tris, workers=12, source_axis=1, tris_kw="tris")
```

Measured on 800 elements (640k pairs), 16 cores, bitwise identical to serial at
every worker count:

| workers | s | us/pair | speedup | efficiency |
|---|---|---|---|---|
| 1 | 30.1 | 47.1 | 1.00x | 100% |
| 2 | 16.5 | 25.7 | 1.83x | 91% |
| 4 | 9.0 | 14.0 | 3.36x | 84% |
| 8 | 4.7 | 7.3 | 6.41x | 80% |
| 12 | 3.5 | 5.5 | **8.50x** | 71% |

So a 10k x 10k interaction matrix is about **9 minutes**, against ~1 h on one
core. Threads do not work here and the numbers are in `mhs/parallel.py`: 1.28x
at two, then *worse* -- the per-source body is many small numpy calls and the
GIL is held almost continuously.

Two things that module does deliberately. It pins BLAS to one thread **in the
children only** and restores the environment afterwards, because these matmuls
are small and BLAS's threads fight the pool for cores (1.83x with one BLAS
thread against 1.61x with sixteen); `import mhs` touches no threading variable,
which a gate asserts in a pristine subprocess. And the call site must be
guarded with `if __name__ == "__main__":`, because the workers are spawned and
re-import it -- without that the parent sees only `BrokenProcessPool`, so
`by_source` catches it and says what actually happened.

## CI

| workflow | when | what |
|---|---|---|
| `gates.yml` | **every push and PR, no path filter** | the `mhs` suite, under both runners, on ubuntu **and macOS**, py3.11 and py3.13 |
| `parity.yml` | nightly, or on demand | adds the `parity` and `oracle` suites, which need sympy |

No path filter, because a path-filtered test job reports success for a commit it
never tested. macOS is in the matrix for a specific reason: numba `parallel=True`
kernels must never be called from Python threads, and that crash is macOS-only,
so a Linux-only CI would never witness a violation of the rule that most
constrains `kernels/assemble.py`.

`CUTDE_USE_BACKEND=cpp` pins cutde's prebuilt C++ CPU backend. Unset, it probes
for CUDA and then OpenCL, which on a headless runner is a slow path to a
confusing `ImportError` rather than to the comparison the gate wants.

The `oracle` extra is nightly and not per push because importing the sympy half
of the vendored oracle costs a measured 14.25 s — it builds and lambdifies ~90
symbolic matrices at module scope. The nightly job runs those under pytest
first, so that import is paid once rather than once per subprocess.

`src/mhs_oracle/` holds vendored **frozen** reference implementations, pinned by
path and sha256. They are never improved and the shipped path never imports
them; a gate asserts that disjointness, because an implementation shared between
the new code and its oracle would make the parity check a tautology.

## Status

Usable, with a known cost ceiling.

Working and gated: the full half-space kernel (closed-form direct and image
R-family, adaptive quadrature for the image Q-family), the folded image of a
surface-reaching source, all seven matrix entry points in both layouts, the
continuous P2 layout against the penny crack, the eigenstress split, and the
cutde anchor. **On-fault stress is at machine precision** (1e-15) at every
realistic collocation point, including the top element of a surface-breaking
fault.

**A source that reaches the free surface is folded.** The lifted Mindlin
image is right to O(eps^2) for a BURIED source; continued above `z = 0` it
puts its image point and Boussinesq branch line inside the body, O(1/eps)
within eps of the trace. On a surface-breaking fault with continuous
quadratic slip that moved the whole fault's moment by 10-20 % and left the
collocation form's Coulomb solve divergent. The regularised problem must take
the smeared slip folded back into the body (the part above the surface
replaced by its mirror image: mass-preserving, exact in antiplane strain),
whose image is the mirror element's own full-space field plus the Boussinesq
field of the normal traction the symmetric pair leaves on `z = 0` -- zero for
antiplane strain, O(1) for dip slip. `mhs.kernels.fold` does this for every
source within 3 eps of the surface, at receivers within 10 eps of it (one
element height at eps = 0.1 h: deeper, the lift is as good, measured) and
within 60 eps of the source's crease. Gated by the seam -- for a buried
element the fold reproduces the lift to 9e-3 of the image at 12 eps and 6e-4
at 40 eps, where the mirror alone misses by 1.3 -- and by the free-surface
condition the lift fails by 43x within 10 eps of a trace. Measured on a
rough, surface-breaking, 33-degree dip-slip fault of 1730 elements at
eps = 0.1 h against classical P0 (cutde): the uniform-drop moment goes from
0.95 (collocation P2) and 0.90 (Galerkin P2) of classical to 1.013 for
both; the Coulomb solve from divergent (collocation) and 0.82 (Galerkin) to
1.02 and 1.01 with every depth bin within 3 %; the Galerkin form stays
positive definite. On a vertical strike-slip fault against the antiplane
solution: the trace row from 5.0x exact (collocation) and 3.3x (Galerkin) to
0.978 and 0.984 at every depth. The Boussinesq term is a surface quadrature,
about 0.2 s per straddling-source/band-receiver pair on one core, and
`fold=False` on every entry point keeps the lift for comparison.

The honest limits:

- **The free-surface condition is satisfied to O(eps^2), not exactly, for a
  buried source.** A property of the mollified *kernel*, not of the
  integration: an exact free surface via the potential-convolution route is
  foreclosed, because the Cortez blob has algebraic tails. `docs/derivation.md`
  has the proof. (A folded source satisfies it exactly, by construction: its
  surface load is cancelled, and the trade is a surface quadrature.)

  **It is the one free-surface number that can be stated without a reference
  solution**, and that is why it is the number to quote. Traction-free at
  `z = 0` is an exact condition the mollified problem must satisfy in its own
  right, so the residual needs no ground truth. There is no exact mollified
  half-space solution to difference against -- that is exactly what step 7
  foreclosed -- so comparing against an *unmollified* kernel (cutde) mixes this
  error with the mollification working as intended, and cannot separate them:
  both are O(eps^2). Measured on one geometry, the two were the same size.

  `verify_free_surface` measures it on a point source as
  `max|t_3k(z=0)| / max|sigma|` one unit below; run it and it prints the table.
  At nu = 0.25 that is 1.1e-2, 1.8e-3, 4.5e-4, 1.1e-4 for eps = 0.5, 0.2, 0.1,
  0.05, order 1.96 to 2.00.

  At fault scale, as a fraction of the PEAK in-plane surface stress -- a
  40 x 20 km fault dipping 60 deg, buried 5 km, 64 elements so h = 5 km, 1 m
  slip, nu = 0.25:

  | eps/h | 0.4 | 0.2 | 0.1 | 0.05 |
  |---|---|---|---|---|
  | worst on the free surface | 6.8% | 1.9% | **0.49%** | 0.12% |

  order 1.85, 1.96, 1.99. Two traps in reading that row. The worst value is a
  SUPREMUM and must be converged: sampling the surface every 5 km understates it
  by 29%, and it settles only below ~1 km. And a "typical" value quoted as a
  median over a box is a statement about the box -- it moves from 1.5e-4 to
  3.0e-9 as the box grows from +-30 km to +-240 km. Report it against distance
  from the surface trace instead; at eps/h = 0.1 the median is 1.7e-3 within
  5 km of the trace, 1.3e-4 at 10-20 km, 1.0e-5 at 20-40 km.

  **nu -> 1/2 is the bad case**, and structurally so: the image terms carrying
  the `(1-2nu)` prefactor are exactly the `1/Q` and `log Q` family whose
  mollification is the approximate part. The point-source residual at nu = 0.49
  is 2.0e-2 where nu = 0.25 gives 4.5e-4 -- forty times worse.
- **Cost is per pair, but the BATCH SIZE sets it.** Measured for `stress_matrix`
  against 256 source triangles, one call, JIT warm:

  | observers per call | 1 | 10 | 100 | 1000 | 4000 |
  |---|---|---|---|---|---|
  | us/pair, stress | 7620 | 1218 | 166 | 43.3 | 36.3 |
  | us/pair, displacement | 4077 | 754 | 103 | 23.4 | 17.3 |

  A factor of 200 across that row, and none of it is the kernel -- it is numpy
  call overhead amortised over the observer axis. So quote the batched figure,
  ~36 us/pair for stress, and pass observers in bulk. A per-pair cost measured
  at one observer per call describes a regime no production matrix occupies, and
  an earlier revision of this bullet quoted exactly that.
- **The on-fault interaction matrix is the fast path, not the stress matrix.**
  `stress_matrix` keeps a full `(3, 3)` tensor per pair and is the form to reach
  for last; `interaction_matrix` with the components you actually want is the
  one that scales. See `## Across cores` for the measured rate.

## License

MIT.
