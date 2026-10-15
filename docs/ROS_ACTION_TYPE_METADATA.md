# ROS action type metadata — SOURCE only

## Contract and provenance

Only `RosGraphDiscovery.list_actions` and `_get_action_type` bodies change.
The existing transport, resolver, normalizer, compiler nonempty-type gate and
all four Docker smoke tests remain unchanged. No new dependencies or ROS Python
client are introduced. This is software-source evidence, not runtime adoption.

Pinned upstream rosbridge_suite source in `history_inputs`:

- ROS2 commit `ae4d7f8c08408cc6c7430425c2c0eebd3e225ed8`:
  `rosapi_msgs/srv/ActionType.srv` has request `string action`, response
  `string type`; `GetActionServers.srv` returns `string[] action_servers`.
  `rosapi_node` registers `~/action_servers` and `~/action_type`.
  The proxy reads the action graph and returns the first type for a matching
  action name, or an empty string. The client cannot detect upstream graph
  multiplicity once that proxy has reduced it to a single string.
- ROS1 commit `7369b80863bcc2fb1600b4d83a42b80a38c5d34d`:
  `GetActionServers.srv` also returns `string[] action_servers`.
  The node registers `/rosapi/action_servers`, but no ActionType service.
  Its proxy derives namespaces from cancel/feedback/goal/result/status topics.
  Names alone are not type evidence.

These are static version-specific references, NOT evidence of the installed
rosbridge version or service availability on any host. Live ROS2 compatibility
requires separately authorized observation of the installed version and actual
ActionType service, plus the unchanged deployment tests. Humble alone does not
guarantee that service exists.

## Behavior

ROS2 lookup uses the existing `_call_rosapi("action_type", {"action": name})`
and hence the resolved/configured service prefix and existing bounded
`call_service` timeout/retry/envelope behavior. No global endpoint, subscription,
rclpy import, direct DDS call, or topic-suffix inference is added.

Only scalar `pkg/action/Type` or short `pkg/Type` metadata with ASCII identifier
components is accepted. Outer whitespace is stripped; the existing action
normalizer produces `pkg/action/Type`. Wrong interface kinds, malformed strings,
arrays (even singleton arrays), dictionaries, missing values, failed transport
results and explicit failed service envelopes yield unknown (`""`). No type is
fabricated from an action name. Short-form acceptance is compatibility behavior,
not a claim that the pinned ROS2 proxy emits short-form metadata.

ROS1 and unknown-version lookup conservatively return unknown without probing a
nonexistent ActionType service. No ROS1 topic-based type derivation is claimed.

Listing uses official `action_servers` with the preexisting `actions` alias.
Order and risk classification are retained. Conflicting aliases or malformed
name arrays raise `ROS_ACTION_METADATA_INVALID` rather than hiding entries.
Every well-formed listed name survives unresolved metadata, with an empty type
and `ROS_ACTION_TYPE_UNRESOLVED` warning. The unchanged compiler then rejects it
with `ROS_INTERFACE_TYPE_INVALID`. Failed/unsupported listing retains the old
empty-list behavior; that is not proof the graph contains no actions.
Lookup has no added cache; later calls can observe changing metadata.
Goal/result/feedback schemas are not inferred by this patch.

## Validation and limits

The new offline tests use a strict scripted transport only, checking exact
service prefix/arguments, normalization, wire shapes, failure/ambiguity handling,
ROS1 unknown semantics, compiler rejection of unresolved entries, order, details,
and absence of stale caching. Existing graph/resolver/interface tests remain
protected. The authorized SOURCE_CHECK CHECK/RUN receipts and final source
hashes are recorded in `reports/evidence.json`; FORMAT is not a test receipt.

Production provider calls and actual ROS1/ROS2, Docker, DDS, SDK, physics,
engine and NN tests are **NOT_RUN** in this task. There is no live success,
dynamics simulation or physical evidence. The checker's import/socket audit is
bounded Python coverage, not an OS sandbox and not real-host zero-effects proof.
Independent semantic review and runtime validation remain necessary.
