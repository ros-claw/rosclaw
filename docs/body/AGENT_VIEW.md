# Agent View

The Agent View layer translates the Effective Body Model into forms that LLM
agents can consume quickly and safely.

## Why an Agent View?

The raw Effective Body Model contains every joint, frame, sensor, actuator, and
safety limit. An agent planning a task does not need all of that detail up
front. The Agent View provides:

- A concise natural-language summary
- A capability checklist with readiness
- A list of recently changed body state
- References back to the full model when detail is needed

## Components

| Component | File | Purpose |
|-----------|------|---------|
| Query engine | `src/rosclaw/body/query.py` | Answer natural-language questions about the body. |
| Summaries | `src/rosclaw/body/summaries.py` | Generate compact JSON summaries. |
| Renderer | `src/rosclaw/body/renderer.py` | Render `EMBODIMENT.md`. |
| Agent view | `src/rosclaw/body/agent_view.py` | High-level `AgentBodyView` dataclass. |

## AgentBodyView

`AgentBodyView` contains:

- `body_instance_id`
- `nickname`
- `effective_body_hash`
- `summary`: one-paragraph description
- `readiness`: overall status and per-category readiness
- `capabilities`: list with status and notes
- `recent_changes`: last N maintenance events
- `forbidden_capabilities`: list of blocked capabilities
- `references`: URI pointers to the full Effective Body Model and EMBODIMENT.md

## Rendering EMBODIMENT.md

`EmbodimentRenderer` produces `EMBODIMENT.md` from the Effective Body Model.
The renderer preserves a human-editable notes section delimited by:

```markdown
<!-- HUMAN-NOTES-START -->
... human notes are preserved across renders ...
<!-- HUMAN-NOTES-END -->
```

Generated sections include:

- Identity and e-URDF reference
- Quick capability summary
- Sensor inventory
- Actuator inventory
- Joint limits and safety
- Recent maintenance
- Skill compatibility snapshot

## Summaries

`BodySummaryBuilder` produces a JSON-serializable summary suitable for MCP
tool responses:

```json
{
  "body_instance_id": "g1-a",
  "effective_body_hash": "abc123...",
  "readiness": "ready",
  "capabilities": [
    {"id": "locomotion", "status": "compatible", "note": ""},
    {"id": "dual_arm_manipulation", "status": "blocked", "note": "right_arm_actuator_group unavailable"}
  ],
  "recent_changes": [...]
}
```

## Query engine

`BodyQueryEngine` answers questions like:

- "What sensors are available?"
- "Why is dual arm manipulation blocked?"
- "Has the right arm been calibrated recently?"

It uses the Effective Body Model and maintenance log; it does not guess beyond
the compiled state.

## Dashboard page

The dashboard `/body` page renders the Agent View as HTML for human operators.
The same content is available via `/api/body` for agents and integrations.

## Best practices for agents

1. Always read the Effective Body Model or Agent View before planning motion.
2. Treat `unknown` compatibility as `blocked`.
3. Check `recent_changes` when a previously working skill fails.
4. Use `rosclaw://body/current/effective` for the canonical body state.
5. Do not parse `body.yaml` directly.

## See also

- [BODY_RUNTIME_OVERVIEW.md](BODY_RUNTIME_OVERVIEW.md)
- [EMBODIMENT_FORMAT.md](EMBODIMENT_FORMAT.md)
- [MCP_BODY_TOOLS.md](MCP_BODY_TOOLS.md)
