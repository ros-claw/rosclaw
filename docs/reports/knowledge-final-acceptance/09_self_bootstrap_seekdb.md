# Self-bootstrap: SeekDB native hybrid

Status: PASS.

The implementation was checked against OceanBase/SeekDB documentation for
full-text search, HNSW, hybrid search, `GET_SQL`, refresh/rebuild, and rerank
capability. It then ran on a real SeekDB 1.3 server.

An actual Know/How self-bootstrap replay produced:

- Reference Pack `reference_pack_c453434911bf813d1103af36`;
- opened evidence `self-ev-1`;
- pinned source version `document_version:seekdb-1.3`;
- How advice `advice_47688ea33894518ded0e2d97`;
- one cited recommendation, no abstention, no private reasoning.

The resulting git diff added native schema creation, BENG/IK/NGRAM/HNSW,
hybrid/RRF SQL and trace capture, filter propagation, bounded HNSW visibility
waits, capability reporting, and logical backup/restore. Live validation found
and fixed an asynchronous HNSW visibility race; full index rebuild is now used
after restore.

The code change is in rosclaw-know merge `3e15dfb`; native live tests passed.
