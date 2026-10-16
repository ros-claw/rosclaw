# ROS request/profile/discovery repair — SOURCE only

## Scope and evidence

Status: **SOURCE_AUTHORED_NOT_RUNTIME_VALIDATED**. This delivery documents the current frozen implementation; it does not change implementation or tests. Its five source components are:

- `src/rosclaw/connectors/ros/transport/rosbridge.py`
- `src/rosclaw/connectors/ros/discovery/rosapi_resolver.py`
- `src/rosclaw/connectors/ros/discovery/graph.py`
- `tests/connectors/ros/test_request_profile_discovery_regression.py`
- this document.

The three production files and regression test are readonly in this phase. All older tests, compiler, transport interfaces and deployment configuration remain untouched. `reports/evidence.json` records the registered hashes; `delivery.json` references the five sources and report. Registration and schema validation establish artifact identity/shape, not runtime correctness.

The copied task reports Root independently obtained **107 old pure tests PASS** and **48 new pure tests PASS in separate runs**. These are externally reported results, not tests executed here, not a combined 155-test run, and not deployed ROS/Humble proof. No Bash, checker, test runner, provider, ROS, Docker, SDK, physics or NN execution occurs in this documentation phase. The original four Docker smoke tests have **not yet passed after the new source**; separate Root verification is pending.

## Request behavior and timeout boundary

`RosbridgeTransport.request` preserves an explicit nonempty request ID or generates one, then uses `time.monotonic()` to create one budget shared by receive loops and retries. Send/connect time consumes that budget when control returns. Idle receives (`ok=True, data=None`), unrelated IDs and service progress statuses do not reset it. Each receive gets the remaining time. A response observed at or after the deadline is not accepted. Retry count remains bounded by `max_retries + 1`, with the same request ID across attempts; retries do not replenish time.

**This is not an interruptible hard overall connection cap.** Blocking `send()`/`connect()` still use endpoint timeout boundaries rather than the request's remaining budget. Lock acquisition and underlying I/O are not forcibly interrupted. A short request budget can therefore be exceeded before the next deadline check. The scripted send-cost regression explicitly models a two-unit send against a one-unit budget; it does not measure wall-clock latency. Likewise, graph discovery's ROS2 0.5-second service-type/provider request settings do not prove a 0.5-second end-to-end bound or a bounded whole-graph scan. Subscription helpers are outside this request-budget change.

For matching service replies, error status, malformed status level, wrong operation, mismatched service, explicit `result:false`, non-boolean declared result and non-dict `values` yield failure. Missing `result` remains accepted for compatibility; an omitted service field defaults to the requested service. Informational/warning/none statuses are progress, not service success. A non-dict envelope fails; other IDs are ignored. Non-service requests retain status acknowledgements. This is envelope handling, not proof of service semantics or authorization. Transport failures can retry within the remaining budget; no exactly-once service execution guarantee is introduced.

## Profile resolution

The resolver probes `/rosapi` and `/rosapi_node` for `get_ros_version`, caching a successfully resolved profile unless forced. It requires an exact integer version 1 or 2 (booleans, strings, floats and other versions are rejected), compatible envelope fields, and a string distro whose nonempty stripped content is alphabetic. Version 2 selects `rosapi_msgs/srv`; version 1 selects `rosapi`. An explicit validated version is evidence distinct from fallback; explicit version 1 can still resolve even if the supplementary distro lookup is empty. Version 2 with no distro uses `unknown`.

Fallback runs only when both candidates return recognized explicit service-absence diagnostics. Generic timeout, permission failure, host-not-found wording or malformed responses do not establish absence. ROS1 fallback then queries `/rosapi/get_param` for `/rosdistro` and requires positive known ROS1 distro evidence. Raw or JSON-encoded strings are handled, including `/opt/ros/<distro>` forms. Empty, malformed, unknown or ROS2 distro values do not create an inferred ROS1 profile. Failure raises `RosApiDetectionError`; no fallback profile is cached. These validations do not authenticate an endpoint or prove the installed rosapi version.

## Action type metadata and unresolved deployment gap

For ROS2, lookup makes one `action_type` probe with `{"action": name}` using the profile prefix. A valid scalar `pkg/action/Type` or `pkg/Type` becomes canonical `pkg/action/Type`. ROS1 lookup returns unknown without inventing an ActionType service. Neither action namespace nor topic suffix alone supplies a type.

The fallback reads **already-cached generated service type metadata** at `<action>/_action/send_goal` and `<action>/_action/get_result`. It accepts exact `pkg/action/Type_SendGoal` and `pkg/action/Type_GetResult` forms and removes the respective generated suffix. It adds no discovery calls, subscriptions or hidden-service enumeration. Direct and cached candidates must agree: malformed present cached metadata, conflicting generated types, or disagreement with a direct type returns unknown. A single valid cached candidate can resolve when the direct probe fails. The cache is not freshness-validated by this lookup.

**Actual hidden `_action` cache availability is NOT_PROVEN.** The copied ROS2 upstream proxy defaults `get_services(..., include_hidden=False)` and `get_services_and_types(..., include_hidden=False)` to excluding hidden services; `get_service_type` also uses that non-hidden path. Discovery lists services before actions, but that order alone does not establish that hidden generated services or the expected generated type strings enter `_service_types`. The new regression hand-fills that private cache: it is unit parser/conflict coverage only, never evidence of actual Humble service compatibility or a deployed fix. Even explicit lookup may not expose hidden types through that upstream path. No implementation workaround is claimed here.

The pinned upstream sources in `history_inputs` describe version-specific contracts: both GetActionServers definitions return `action_servers`; the ROS2 ActionType request uses `action`, response `type`; the copied ROS1 node lacks ActionType. Static pins are not installed-version evidence, and Humble by itself does not guarantee ActionType availability. The older `ROS_ACTION_TYPE_METADATA.md` describes the prior narrower patch; use this document for current request/resolver/fallback limits.

Action listing accepts the official `action_servers` array and legacy `actions` alias; conflicting arrays or malformed names raise `ROS_ACTION_METADATA_INVALID`. Unresolved listed actions remain present with empty types and `ROS_ACTION_TYPE_UNRESOLVED`, allowing the unchanged compiler nonempty-type gate to reject rather than silently compile a partial action set. Failed listing still returns an empty list: that is not proof the graph has no actions. No action schemas or physical capabilities are inferred.

## Coverage and pending review

The frozen regression uses a scripted socket, synthetic monotonic clock and scripted service replies. It covers idle/unrelated frames, progress versus success, error/malformed envelopes, send cost, shared retry budget/ID/count, invalid version data, absence versus transport error, positive ROS1 fallback/cache, and cached action parser conflicts. It does not exercise a real websocket handshake, endpoint latency, live graph, hidden-service cache population, installed Humble services or actual manifest generation against deployment.

Root must separately review the source and independently verify the unchanged four deployment tests: stack reachability, doctor health, turtlesim topic discovery and manifest compilation. That review must capture the installed rosbridge/rosapi version, actual profile response and action metadata availability, including hidden-service visibility/type shapes when fallback is needed. A skip, synthetic cache, or offline PASS is not a deployment PASS. No runtime adoption, dynamics simulation, physical action, measured elapsed bound or deployed capability success is asserted. Stop here at SOURCE-only delivery pending Root Docker/runtime review.
