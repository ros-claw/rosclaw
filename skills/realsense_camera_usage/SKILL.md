# SKILL.md

## Skill ID

`rosclaw/realsense_camera_usage`

## Intent

realsense_camera_usage

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
