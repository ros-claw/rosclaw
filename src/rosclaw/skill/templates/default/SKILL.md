# SKILL.md

## Skill ID

`{namespace}/{name}`

## Intent

{name}

## Preconditions

- robot_state available

## Effects

- task_completed == true

## Runtime Contract

- Input: robot_state
- Output: trace + runtime_events

## Safety Envelope

- sandbox_first

## Evidence

- See `evidence/reports/`
