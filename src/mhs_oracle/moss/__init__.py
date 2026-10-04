"""Vendored frozen oracle: the mollified Mindlin half-space reference.

This is the comparison target for the new half-space work. It contains:

* ``mindlin_kernels`` -- the POINTWISE mollified Mindlin Green's function for a
  buried point force, via Papkovich-Neuber potentials following Apostol (2016).
  Its mollification is the literal lift ``r -> sqrt(r^2 + eps^2)`` on both the
  direct and the image distances, plus an explicit Cortez blob term on the
  direct part so that half satisfies ``L G = -delta phi_eps`` exactly. The image
  correction is therefore only ``O(eps^2)`` in both the PDE residual and the
  free-surface traction -- measured 7.0e-4 and 9.9e-5 relative at nu = 0.25,
  eps = 0.1. Replacing that with a Cortez-CONSISTENT image mollification is the
  new work; this copy is what it will be measured against.
* ``mindlin_triangle`` -- integration of that kernel over a flat triangle, by
  Gauss-Legendre quadrature. **This is the reference the closed form must
  match**, and the reason it is an oracle rather than a competitor: quadrature
  of a smooth integrand is a different computation from a closed form, so
  agreement between them is evidence rather than tautology.
* ``analytical_kernels``, ``analytical_batch``, ``mollified_elastic_kernels`` --
  the full-space closed forms and the eigenstress oracle this package's
  ``mhs.fullspace`` copy descends from, kept so the two can be compared.

**It is expensive.** ``mindlin_kernels`` builds and lambdifies about ninety
symbolic matrices at module scope: a measured **14.25 s** per import, and it
needs sympy; ``mollified_elastic_kernels`` imports matplotlib at module scope.
That is why this half sits behind the ``[oracle]`` extra and why the gates that
use it live in their own suite, run nightly rather than on every push. The
shipped package must never import it, and ``verify_import_hygiene`` asserts
that structurally.

**One edit, recorded here and in the manifest.** ``mindlin_triangle`` upstream
spells its intra-package imports absolutely inside a ``try``, with a FLAT
``from X import ...`` fallback on ``ImportError``. That fallback is a hazard
rather than a convenience: if the prefix is ever wrong the except branch fires,
the flat import succeeds whenever that directory happens to be on ``sys.path``,
and which copy you get depends on path order -- silently, and in a package whose
whole purpose is to be a distinguishable second copy. The vendored file uses
relative imports instead, which resolve within this package or raise. No
arithmetic is touched. It is the only file here whose vendored and upstream
hashes differ, and the manifest showing exactly that is the audit.

Upstream: moss-org ``src/moss_kernel/`` @ ad0e992 (2026-10-03).
"""
