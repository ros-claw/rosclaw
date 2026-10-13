# G2 normal integration and actual-main acceptance

PR #675 was normally squash merged at `d734fa44d9c0681e8b4e9ad3b5fa34bd7dc09745`, with its full tree equal to tested source `cf8d106387b3550b47dd677c2b5459d8a3b0b2bf`. Its three original source workflows and four original main workflows all passed, with original logs archived and independently verified. CI infrastructure #676 also passed all seven applicable original workflows.

The initial actual-main known World failed because its original physical sample was 692.311 ms old against the unchanged 500 ms maximum. This failed World and the later container-name collision remain preserved. The capture was repaired to read complete current original physics/ECM after imports and asset hashing, and perform one complete L1 review. SDK allocations were explicitly recorded as six CPUs. No source freshness or geometry threshold was relaxed.

Two new actual-main known/J100 Worlds then passed source binding MATCHED, L1 READY_FOR_SIM_TEST, independent raw Native/contact recomputation and pinned CLI artifact verification. Both remained passive, with zero goals and velocity publications. Their 253-file archive manifest SHA256 is `cdbf754b0068f7ac4d3fb7a9edf6b3e017dbd3e225bc81def059064aa5544eb1`.

G2 is closed. J100 is adapted held-out evidence, not zero-shot. G3 L2, G4 and G5 remain NOT_RUN, and v1_done=false. Local evidence and recovery files are operator-local artifacts, not assumed downloadable GitHub attachments.
