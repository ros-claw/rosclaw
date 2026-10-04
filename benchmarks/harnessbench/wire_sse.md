The metadata parser in `wire_sse.mjs` is for independent provider observation,
not for consuming the agent's response. Call `observeSseMetadata` with the body
of `response.clone()`. It handles LF, CRLF, CR, split UTF-8, multiline `data`
fields, and data fields without a space after the colon. It emits selected
start/end/error metadata and discards text, thinking, and tool deltas.

Keep request/body-hash binding in the caller. A response with no observed
metadata is **NOT_VERIFIED**, even when its HTTP status is 200. An EOF without
the event's blank separator is reported as incomplete, rather than certified
as a terminal event. The one-megacharacter limit bounds each event; the caller
must separately bound total time/bytes and own its observation lifecycle.
Reported model names are API metadata, not weight attestations.

An earlier experiment observer searched only for `\n\n`. An actual invocation
of that observer against a synthetic CRLF response missed both expected
metadata events. This reproduces a parser defect; it does not establish the
separator used by any historical live response. Keep the old experiment's
missing-metadata score unchanged and use the qualified parser for future
experiment packets.

Run the standalone checks with:

```sh
node --test benchmarks/harnessbench/test_wire_sse.mjs
```
