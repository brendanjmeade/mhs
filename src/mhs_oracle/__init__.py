"""Vendored frozen oracles. Never improved, never imported by the shipped path.

Two independent copies of the same mathematics, compared entrywise by the parity
gates, is the only arrangement in which a parity check means anything. So this
package holds a frozen copy of the reference implementations and ``mhs`` holds the
new one, and neither may be edited to agree with the other.

**Why the rename.** These are copies of moss-org's ``src/moss_kernel/`` and
``src/clq/``, which that repo installs under those names via an editable ``.pth``
that puts its ``src`` on ``sys.path`` for every process on the development
machine. Vendoring under the original names would mean two packages contending
for one import name: one shadows the other silently, and the parity clause would
compare a copy against *itself* and still print PASS -- the residual merely drops
from ~1e-12 to ~1e-16, which no tolerance rejects. Worse, it would silently test
against whatever moss-org happens to be checked out at. Hence ``mhs_oracle.moss``
and ``mhs_oracle.clq``.

**The contract.** Every file here is pinned in
``tests/gates/parity/oracle_manifest.json`` by resolved path and by TWO hashes:
the vendored file's, and the upstream file's as recorded by moss-org. Four of the
moss files vendor byte-identically; the ones needing an import-prefix edit are
the only records where the two hashes differ, so the manifest showing exactly
that set is itself the audit. A deliberate edit updates the manifest in the same
commit, and the commit message says why the hash moved -- that makes the edit
visible in review instead of silent.

**Cost.** ``mhs_oracle.moss.mindlin_kernels`` builds and lambdifies ~90 symbolic
matrices at module scope: a measured **14.25 s** per import, and it needs sympy
and matplotlib. That is why it sits behind the ``[oracle]`` extra and why CI runs
those gates nightly rather than per push. ``mhs_oracle.clq`` is numpy-only and
costs nothing, so its parity gates run on every push.

Nothing in ``mhs`` may import anything here; ``verify_import_hygiene`` asserts it.
"""
