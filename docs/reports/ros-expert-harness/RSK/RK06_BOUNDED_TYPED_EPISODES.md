# RK06 bounded typed episode path

This separate B/C/E client connects the new RSK decision schema to the existing
read-only model client and an operator-injected `RskPhysicalACI`. It changes no
frozen G5 runner, result, arm meaning, Core, ROS probe or executor. It is not a
25-episode pilot launcher. New A/D orchestration and independent evaluation-case
registration remain separate requirements; no model study is claimed here.

Before the first provider call, the runner checks all four original pilot entry
attestations, their protocol/source binding, the allocated ordinal and arm, and
SHA-pinned original case/model/requirements/staging/task/episode/run settings.
A shared per-case Memory manifest is also pinned. E must match the actual query
policy and exclude the current original episode SHA and World run. Inputs and
entry are rechecked after provider latency, before host dispatch. Raw component
PASS reviews without the frozen protocol/source attestation cannot open entry.

`FrozenProviderV1` records the actual model, effort, client/settings code hashes
and executable hash. Actual mode accepts only the existing pinned client,
requires the exact clean harness source and imports from that worktree, and
replays usage and native-tool checks from original events. It creates no model
Runtime, ROS connection, driver, executor or permission. Any physical request is
still delegated to the existing SDK callback and MCP→rosclawd path. The caller
must supply the already constrained SDK factory; the runner cannot grant it
additional World, goal or motion authority.

Synthetic contract clients are explicitly separate: actual model calls stay
zero, results cannot become study success, and SDK/probe callbacks are forbidden
both at entry and after each client call. Tests use synthetic clients and gate
fixtures; their PASS results are not physical, model or pilot evidence.

An episode has a 300-second monotonic deadline, at most the protocol's 12 model
decisions and eight read-only investigation requests, and the existing ACI's
one active probe plus one canonical validation. Original SDK goal and safety
limits remain unchanged. No failed provider or World is automatically retried.
The first known-mechanism valid-proposal decision is measured separately from
independent physical success and from correct abstention.

The exclusive output directory preserves input prompts/schemas, original client
outcomes, original event SHA references, full host decisions/results, Memory
stages, verification records and a content-hashed result. Verified provider
usage, partial reported usage and unconfirmed client invocations are separate.
`physical_validation_phase=NOT_REQUESTED` prevents a model/context/budget failure
from being reported as a physical navigation failure. Causal Memory benefit
remains unestablished without an actual eligible paired study.
