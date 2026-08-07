# Frozen acceptance bundle

Label: `final-acceptance-20260807`.

This directory is the durable software-level closure point for ROSClaw Know / How
final acceptance. `BUNDLE_MANIFEST.json` is the canonical machine-readable
manifest. It binds exact Git commits and trees, released package hashes, public
contract and migration fingerprints, the pinned real-source fixture, all final
reports, and the recorded live/self-bootstrap results.

`BUNDLE_MANIFEST.sha256` pins the manifest itself. The enclosing Git commit then
provides the final immutable identity for the complete closure bundle.

The source-snapshot fixture remains in the rosclaw-know repository at the exact
accepted commit rather than being duplicated here. Its repository, path, and
SHA-256 are fixed in the bundle manifest.

To verify the local report set, recompute SHA-256 for the paths under
`artifacts.reports` and compare each digest. To verify releases, compare the
wheel/sdist hashes with the corresponding PyPI release JSON.

This freeze is not an authorization to mutate knowledge automatically. It also
does not extend acceptance to physical-task utility, autonomous online refresh,
or multi-node production operation.
