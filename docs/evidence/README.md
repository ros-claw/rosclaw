# Reviewed Evidence

This directory contains portable evidence summaries that support claims in
`src/rosclaw/product/status.yaml`.

Evidence is grouped by subsystem and retained only when it records a durable,
reviewed result. A summary must identify its scope, distinguish fixtures from
physical hardware, and avoid credentials, local machine paths, raw datasets,
or unbounded command output.

Test runners and hardware experiments write raw output to the ignored
`reports/` directory or to `ROSCLAW_HOME`. Promote a result here only after
reviewing and reducing it to the evidence needed for a product claim. Keep
separate reports when product status cites them independently; Git history is
the archive for superseded reports.
