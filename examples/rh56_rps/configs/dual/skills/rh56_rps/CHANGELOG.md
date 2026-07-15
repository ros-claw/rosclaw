# Changelog

All notable changes to the `rh56_rps` ROSClaw skill package.

## 1.0.0 — 2026-07-14

- Ported the RH56 rock-paper-scissors demo to the `rosclaw.skill.v1` package schema.
- Added skill metadata, identity, task contract, and behavior-tree entrypoint.
- Declared compatibility for `inspire_rh56_right` and `inspire_rh56_left` robots.
- Added providers, safety, dojo, darwin eval, and lineage manifests.
- Wired explicit `body_id: rh56_rps_robot` and canonical `FAILED → failure` outcome mapping for SeekDB.
- Validated a 5-round full closed-loop session with `rosclaw practice verify --strict`.
