# Source consumer contracts in the read-only Kernel

An expanded XML attribute can retain its namespace meaning while losing the
literal spelling expected by a particular consumer. The earlier fixed-SDK
experiment exposed this with friction `fdir1`: `ns0:expressed_in` and
`gz:expressed_in` refer to the same expanded source attribute, but the inspected
gz-physics 7.8.0 DART backend looks up the latter spelling.

The Kernel now computes this narrow diagnosis from original bytes, rather than
requiring an operator to read the XML and manually supply the verdict. It adds
one source-bound finding to a historical operational graph, creates the narrow
`SOURCE_CONSUMER_CONFLICT` hypothesis plus an unresolved-path alternative, and
proposes a bounded `SOURCE_CONSUMER_CONTRACT` read-only review. A compatible
source has no target issue; an unknown SDK/library or unresolved source facts
remain UNKNOWN.

## Offline interface

```bash
.venv/bin/python -m rosclaw.entrypoint ros source-consumer-review \
  --operational-graph original-graph.json \
  --urdf robot.urdf --world-sdf world.sdf \
  --consumer-contract declared-consumer-contract.json \
  --container-inspection original-container-inspection.json \
  --backend-observation original-backend-library-review.json \
  --parser-evidence original-parser-results.json \
  --robot-entity explicitly_selected_entity \
  --output new-review.json --json
```

Add `--propose-compatibility` to include concrete source-only `world.sdf` bytes,
their digest, the exact original-world rollback digest and verification duties.
This optional proposal reuses the existing pure adapter and is limited to the
independently inspected fixed image/library pair. Other declared contracts may
be diagnosed but cannot borrow this adaptation. Unresolved facts or conflicting
namespace bindings produce an explicit abstention without a repair file.

The artifact is not a launchable World or an installation request. Historical
run labels remain part of the bound original; fresh-run rebinding requires a
newly compiled proposal and new hashes, not edits spliced into this artifact.
Independent new-World identity, L1, stop/contact, strict-age and counterfactual
proof obligations remain unmet until actually performed.

Inputs are bounded regular files read without following their final symlink;
output creation is exclusive. The response contains the computed review,
derived operational graph, proposed diagnostic experiment and exact input file
digests. It opens no ROS transport, parser process, SDK container, Runtime or
device and changes no XML. A successful command can return an UNKNOWN finding;
`ok` reports completion of inspection, not physical or current-source acceptance.

`RosLiteralConsumerContractV1` declares a fixed image/library digest pair, the
literal attribute, the unique run-witness plugin selector and an exact parser
evidence reference/digest. Its content hash and the supplied parser original
are checked. The SDK image comes from the original container inspection and the
library digest from the original backend record. PASS-like labels are ignored.
These remain operator-declared historical inputs, not authentication of a
running consumer. Library/image drift cannot silently reuse the contract.

When the original graph has a run binding, its declared witness must occur
exactly once and have the same run ID. Other plugins may contain independent
historical run labels; they neither establish nor contradict that witness.
Source URDF digest contradictions, duplicate witnesses, DTDs, invalid XML and
live-observation graphs fail closed. Source directions, coefficients and loaded
frames must agree before diagnosing a spelling-only conflict.

## Evidence and compatibility

The finding is explicitly `STATIC_DECLARED_PARSER_CONTRACT_ONLY`. Its provenance
has UNKNOWN dynamic freshness. The dedicated predicate can use it as immutable
byte-contract evidence in sealed replay, but cannot treat it as a newer probe
measurement. Current L1, permissions, runtime consumption, physical stopping and
causal verification remain unestablished. The experiment stays
`PROPOSED_NOT_EXECUTED`; an existing native DDS provider is not extended or
authorized to run this new operation. The CLI performs the offline review;
fresh runtime observations and new physical validation remain separate duties.

This is a new read-only Core revision after main3021c6bd. It does not modify the
old frozen study's source, graph originals, model allocations, SDK callback,
World budgets, action policy, Runtime or driver. Default graph compilation and
diagnostic planning retain their existing outputs when no consumer finding is
attached. SIM preparation shares the unchanged pure friction-fact parser with
diagnosis; its existing XML adaptation algorithm is retained.

Acceptance includes synthetic negative cases, real CLI dispatch, unchanged
default graph/plan replay on 66 originals, and retrospective classification of
the four sealed ABI Worlds. Their intended results are VIOLATED / SATISFIED /
VIOLATED / SATISFIED. These are known-mechanism software regressions on historical
development evidence, not four new Worlds, blind generalization, autonomous new
root discovery, Memory benefit or a successful new model study. Earlier failed
gates, the source316 pilot's 25 allocations and 180 calls, and formal NO-GO stay
unchanged. Physical success remains bound to the original source/main Worlds.
