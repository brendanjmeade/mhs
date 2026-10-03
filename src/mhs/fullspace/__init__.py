"""The full-space closed-form engine: moments, edge primitives, nodal kernels.

**Vendored, and shipped.** These six files are a byte-identical copy of the
upstream ``clq`` package's closed-form machinery -- the in-plane moment hierarchy
``M_n^(a,b) = int_T xi1^a xi2^b / R_eps^n dS``, the one-dimensional edge
antiderivatives it reduces to, and the nodal ``U``/``H``/``E`` kernels built from
them. ``mhs`` does not reimplement them, and the reason is specific rather than
lazy: ``primitives.py`` carries **four distinct numerical regimes** -- closed
form, small-``|u|`` series, ``rho = 0``, large-``|u|`` series -- each of which
exists because a naive difference loses ``(u/rho)^(m-1)`` digits somewhere. That
is the hard-won part, and rewriting it from scratch would risk losing precision
in one regime silently.

So the division of labour in this package is:

* ``mhs.fullspace`` -- the full-space Cortez-mollified closed forms. Vendored.
* ``mhs.kernels``   -- the half-space image correction, the eigenstress and the
  assembly. **The new work.**

**This is a copy, not an import.** A separate frozen copy lives under
``mhs_oracle.clq`` and the parity gate compares the two. The shipped path must
never import the oracle: if it did, a parity clause would compare
``new + oracle`` against ``reference + oracle``, silently stop testing the shared
half, and stay green with a smaller residual.

Be honest about what that gate proves **today**: the two copies are the same
bytes, so it proves they agree and nothing more. Its value is prospective -- it
is what will catch drift when these files are rewritten in numba, which is the
plan. Correctness here rests on the ported derivation gates under
``tests/gates/oracle/``, which check the mathematics rather than the agreement.

**Performance, measured.** numpy, batched over observers, costs ~15 us per
obs/source pair: 18 s for a 1000 x 1000 matrix, ~31 min for 10k x 10k. Usable
now, and the reason ``mhs.kernels.assemble`` is written so these calls can be
replaced by numba kernels one at a time behind the parity gate.

Upstream: moss-org ``src/clq/`` @ ad0e992 (2026-10-03). Pinned by sha256 in
``tests/gates/parity/oracle_manifest.json`` under the ``mhs_oracle.clq`` names;
the shipped copies are pinned separately, so a change to either side is visible.
"""
from . import defaults, frame, kernels, moments, primitives, shape

__all__ = ["defaults", "frame", "kernels", "moments", "primitives", "shape"]
