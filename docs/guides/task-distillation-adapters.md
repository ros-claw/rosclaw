# Guide: Task Distillation Adapters（数据库优化v3 §4）

## Why adapters exist

The generic distiller understands generic `failure_event` / verified-gesture shapes. Real tasks hide their failures in task-specific events — e.g. the RH56 RPS stress protocol resolves a round as `rps.stress.round.resolved {result: invalid}` with NO explicit failure event. Without an adapter, the most important failures never become Failure Memory.

## Framework

`src/rosclaw/memory/v2/adapters/`:

```python
class TaskDistillationAdapter(Protocol):
    task_ids: set[str]
    def extract_failures(context, events) -> list[MemoryItem]: ...
    def extract_body_patterns(context, events) -> list[MemoryItem]: ...
    def extract_skill_evidence(context, events) -> list[MemoryItem]: ...
    def build_episode_quality(context, events) -> dict: ...
```

`adapter_for(context, events)` selects by `task_id`, falling back to event-shape sniffing (`matches_events`). When an adapter matches, it OWNS failure + body semantics for that task; generic extractors still cover episode / intervention / skill.

## RH56 RPS adapter

Linkage priority (§4.2): `round_id` → time window (current `rps.gesture.executed` events carry no round_id, so windows from `round.started` → `round.resolved` do the real linking).

Failure coverage — every one of these becomes a Failure Memory:

1. invalid rounds: one item per failing gesture in the round (round-level, linked evidence);
2. gestures that failed verification inside rounds that still resolved VALID;
3. gestures executed BETWEEN rounds (e.g. the inter-round ready pose — no window at all, `round_id="between_rounds"`).

Episode quality (§5.1): `outcome` from task-declared thresholds (`success ≥ 0.98`, `partial ≥ 0.80`), with `total/verified/invalid/verified_rate/failure_distribution/first_degradation_round`.

Body patterns (§5.2): `observed_temperature_min/max`, delta, rise rate, `first_failure_temperature`, and `causal_status: observed_correlation | insufficient_data` — never a `thermal_limits` claim (§17.11).

Joint attribution is honest: sessions that never recorded the failing joint get `joint_name=None` and `joint_attribution=not_recorded_in_session`. Nothing is invented.

## Writing a new adapter

1. Subclass the protocol in `adapters/<task>.py` with `task_ids` and `matches_events`.
2. Register in `adapters/registry.py` (`_register_builtin`).
3. Reuse `MultilingualMemoryDocumentBuilder` for ZH/EN/CANONICAL documents (never hand-translate per memory).
4. Keep thresholds in the ADAPTER (task-declared), never in core (§5.1).
5. Tests: implicit failure extraction, episode quality bands, body observation semantics — plus a real-session re-distill check.
