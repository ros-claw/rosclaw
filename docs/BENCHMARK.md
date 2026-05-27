# ROSClaw v1.0 Performance Benchmark Report

> **Date**: 2026-05-28
> **Method**: Single-process synthetic benchmark on local machine

## EventBus Throughput
- Target: 10000 events
- Elapsed: 0.0452s
- Throughput: 221085.2 events/s
- Average latency: 0.0045 ms/event
- Received: 10000 / 10000
- Verdict: PASS (needs >= 10000 events/s)

## SeekDB Scale Performance
- Records inserted: 10000
- Total insert time: 0.023s
- Average insert: 0.0023 ms
- P99 insert: 0.0036 ms
- Query (success=True): 100 results in 3.81 ms
- Verdict: PASS

## SkillRegistry Scale Performance
- Skills registered: 1000
- Register time: 2.32 ms (0.0023 ms/skill)
- list_skills() [names]: 1000 in 0.038 ms
- list_skills(return_entries=True): 1000 in 0.007 ms
- list_skills(skill_type='programmed'): 500 in 0.032 ms
- Verdict: PASS

## FirewallValidator Trajectory Performance
- Trajectory waypoints: 100
- Validation time: 0.51 ms
- Is safe: True
- Violations: 0
- Warnings: 0
- Layers checked: [<ValidationLayer.EURDF_SOFT_LIMITS: 'eurdf_soft_limits'>, <ValidationLayer.SEMANTIC_SAFETY: 'semantic_safety'>]
- Verdict: PASS
