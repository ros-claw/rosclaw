# Bounded original-file context across turns

The RSK A/B/C/E file operations return the same original corpus. Previously the
workbook retained structured evidence records and expanded record slices, but
`files.list` and `files.read` only occupied `recent_tool_result`. Another tool
replaced them. A file response larger than the 3,500-character recent-result
limit could disappear from the next model input entirely.

`EvidenceReadProgress` now also retains the successful original-file catalog,
declared staging field whitelist, file character ranges, and exact returned
slices from both `files.read` and `ros_engineer.read_application_input`.
File slices have a separate 12,000 UTF-8-byte context limit; the catalog
has a 3,500-byte limit. Whole entries are omitted with explicit counts instead
of silently truncating them. Existing serialized-record budgets are unchanged.

Ranges describe characters actually returned by the host, not parsed JSON,
current robot state, or a guarantee that every returned character is inline in
the model prompt. `file_slices_not_inline` and `catalog_omitted` disclose that
distinction. Original-file hashes come from an actual returned catalog or the
application reader's host response, with their origin recorded explicitly.
When neither exists, the digest remains unknown. Coverage from different
digests never merges; an application response must agree with any existing
catalog, and a returned whole file must match its original digest.
All retained values are copied, and rejected requests add no access record.

This adds no diagnosis, repair, active probe, model call, Memory retrieval or
execution authority. It preserves the existing workbook's distinction between
original observations and untrusted model notes. The registered source316 pilot
continues using its frozen code and inputs; its outcomes cannot be attributed
to this later change. Any model-efficiency improvement needs a new prospective
evaluation after the original pilot and NO-GO review.
