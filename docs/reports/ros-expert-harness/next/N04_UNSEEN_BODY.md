
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
