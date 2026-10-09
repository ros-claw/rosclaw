
## Explicit frozen-spawn localization prior source

An optional closed operator SIM declaration now initializes AMCL parameters from
the already frozen spawn and an explicit world-to-map rigid transform. It must
match the actual prepared World name and navigation map frame. The source
proposal never applies a live ground-truth correction, reports localization
unverified, and cannot grant action authority. The default keeps initialization
OFF. Controller, collision monitor, geometry and motion caps remain identical.
The joined workspace retains the declaration bytes and includes its hash in the
manifest. No third robot was selected or inspected to implement this feature.

Host validation: source prior/join26 passed; full ROS1818 passed/10 integration
deselected/one existing warning in32.13s; source mypy and scoped Ruff passed.
Actual SDK isolated `--network none` source preparation with the explicit prior
passed standard World SDF parsing, preserved control-resource parsing, RCL YAML
parsing and Nav2 Map IO. Source/helper/ELF/template bytes were captured before the
SDK call and rechecked. Attempt02 failed before SDK execution because appending
an executed source snapshot shifted positional template indexes. Named source
paths fixed the helper; attempt03 passed. Both attempts remain retained.

This is AMCL source initialization, not observed localization or Body admission.
The generic staged Graph/TF/cleaner bootstrap, guarded runtime launcher and all
held-out L0–L4 physical gates remain pending. v1_done=false.

## Prepared workspace handoff integrity

`read_prepared_generic_stack` captures and verifies every manifest-listed source
before a later launcher consumes it. It refuses missing or modified files,
absolute/traversing paths, file and parent-directory symlinks, nonregular or
oversized sources, changed file identities, incomplete original source
identities, and omitted/rebound map pixels. Reads use anchored directory file
descriptors and `O_NOFOLLOW`; captured bytes are returned to avoid a second
unchecked read. The closed manifest's digest is verified, and any manifest
claiming live admission or authorization is rejected. A rehashed self-consistent
manifest remains integrity evidence only; external approval and actual source
admission are still required.

Host full ROS regression:1845 passed/10 integration deselected/one existing
warning. Installed fixed generic SDK image source validation also passed with
network disabled, root filesystem read-only, UID1000 and no capabilities. The
helper captures its executed sources and parser identities before validation,
then rechecks them; it checks generated workspace integrity as well as actual
standard SDF, control-resource/RCL YAML, and Nav2 Map IO parsing. The robot/map
are explicit synthetic contract fixtures. No World, Node or action was started.
The staged live bootstrap/launcher and all held-out L0–L4 gates remain pending.
