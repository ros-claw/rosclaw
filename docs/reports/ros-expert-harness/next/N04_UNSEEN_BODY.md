
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


## Unactivated generic Nav2 SDK launch description

The generic launch source now consumes the sealed workspace inventory and
checks its original navigation report, parameter digest, launch-spec digest,
all ten standard executable roles, exact node identities and remappings.
Namespace and lifecycle targets are derived from the explicit source; there
is no third-body profile or fixed sensor/body frame. The SDK description
retains the full parameter file, including child costmap parameters, and
sets lifecycle autostart=false. Building this description creates unexecuted
process actions; it opens no DDS and admits no Body. A future runtime launcher
must recheck and mount the sealed source read-only before launching, then
complete independent Graph/TF/Body admission and guarded activation. Those
runtime stages and held-out L0–L4 gates remain pending.

## Source-derived lifecycle bootstrap checks

The existing owned lifecycle probe now accepts a bounded, distinct list of
explicit node identities. `--prepared-workspace` derives those identities from
the sealed navigation launch plan before importing or initializing ROS. Generic
namespaces are preserved in exact GetState service endpoints. The original
seven-node fixture remains the default; its snapshot cannot substitute for a
generic ten-node snapshot. Each required response must independently be fresh
and ACTIVE, regardless of a producer's claimed `ready` value. The probe performs
no lifecycle transition, admits no Body and grants no motion or stop proof.

Host validation: focused contracts25 passed; full ROS1892 passed/10 integration
deselected; Practice183 passed/9 skipped; required mypy121 files; scoped Ruff,
format, compileall and diff checks passed. With network disabled, read-only
root/source mounts, UID1000, no capabilities and isolated DDS domain83, actual
synthetic typed GetState services confirmed fresh ACTIVE acceptance and fresh
INACTIVE refusal for all ten source-derived names. This is synthetic DDS
evidence, not real Nav2 activation or physical evidence.

A separate installed-SDK fixture actually launched all ten declared Nav2 nodes
and their non-autostart lifecycle manager from the sealed synthetic workspace.
All ten actual GetState responses were UNCONFIGURED, and readiness remained
false. The fixture then shut down all owned processes cleanly. The first attempt
failed because the read-only container had no writable temporary directory;
the helper also incorrectly awaited an optional shutdown return. Both failures
are retained; a new attempt supplied a writable output temp directory and
handled the actual SDK shutdown API, then exited zero. No World, controller,
robot action or held-out asset was started. Executed helper/project/SDK sources
were hashed before calls and rechecked. Complete staged Graph/TF/Body admission,
guarded launch, generic feature freeze and held-out L0–L4 remain pending.

## Preserve declarations for the runtime handoff

The prepared workspace previously kept only a declaration digest, so later
runtime stages could not recover the original controller, navigation, contact
and cleaning-attachment inputs. Preparation now preserves their complete bytes
in `source-declarations.json`, includes that file in the sealed inventory, and
verifies its closed role set and original declaration digest on reopen. An
optional frozen localization declaration is retained there as well. Invented
roles, omitted declaration bytes, invalid types and rebindings against the
original digest are refused. Earlier prepared workspaces without these bytes
must be regenerated from their original inputs; retained historical evidence is
never rewritten. This integrity handoff does not grant operator approval or
Body/action admission.

Focused source contracts49 passed; full ROS1896 passed/10 integration
deselected. Scoped Ruff/format, compileall and diff checks passed. Core `src`
is unchanged from the preceding121-file mypy/183-test Practice checkpoint.
The fixed installed SDK reopened the new sealed synthetic workspace and built
eleven unexecuted launch actions, while refusing the old workspace missing
declaration bytes. Before calls, the helper captured109 imported project
sources and then rechecked them. No DDS/Node/World was started. The prior actual
unactivated Nav2 startup evidence remains attributed to its original source;
complete generic runtime admission and held-out physical gates remain pending.

## Source-derived inactive SIM bootstrap description

The owned bootstrap source now joins the frozen World/model, expanded robot
description, declared sensor/contact bridges, exact World clock, explicit joint
state topic, original spawn prior, controller-manager namespace and Nav2 plan.
Controller/navigation reports must correspond to the preserved declarations;
attachment/contact identities and controller parameter hashes are checked. The
source refuses another World clock, aliased topic roles, non-SIM declarations,
invalid seeds/heights or a resource mount other than `/evidence`.

The SDK description proposes World/spawn/state/bridge processes and the existing
non-autostart navigation nodes. It holds both controller spawners inactive until
successful model creation; failed creation requests owned launch shutdown. This
module constructs descriptions only and invokes no LaunchService. Execution
still requires an owned immutable deadline supervisor and a read-only source
mount, then fresh Graph/TF/map/sensor/independent-physics source admission and the
existing Native/MCP/rosclawd task path. No source plan grants Body or action
authority. The complete guarded runtime launcher remains pending.

Focused contracts13 passed; full ROS1909 passed/10 integration deselected.
Scoped Ruff/format, compileall and diff checks passed; Core src remains identical
to the preceding mypy121/Practice183 checkpoint. In the fixed installed SDK,
the helper captured114 sources before calls, verified15 installed executable
roles and built17 unexecuted top-level actions plus two held inactive controller
actions. Source/executable hashes stayed unchanged. No Node/DDS/World/action was
started. Generic feature freeze=false; no held-out asset selected or inspected.
