# Isaac ROS 5 intelligence and evidence boundary

The read-only `rosclaw ros performance` command now builds a sealed performance
graph from a ROS system snapshot. Node buffer backends and process boundaries
remain UNKNOWN without fresh sourced measurements. CPU/CUDA transitions are
inferences; measured copy events are reported separately. Missing, stale or
malformed traces do not become zero-copy evidence. A topology snapshot never
claims a verified before/after optimization.

Four source-backed Isaac ROS 5 options are exposed: cuVSLAM, nvblox, TensorRT
inference and CUDA buffer transport. Compatibility and runtime measurements are
separate. A working CPU navigation task does not automatically select or install
these options. The current Jazzy acceptance fixture is incompatible with the
official 5.0 ROS target and correctly reports BLOCKED.

Sources reviewed on 2026-10-04:

- [Official 5.0 platform requirements](https://nvidia-isaac-ros.github.io/v/release-5.0/getting_started/index.html): ROS 2 Lyrical; DGX Spark is a supported platform with its own software requirements.
- [CUDA buffer backend](https://nvidia-isaac-ros.github.io/concepts/rosidl_buffer/cuda_buffer_backend.html): native `rosidl::BufferBackend` storage and same-host compatible CUDA IPC transport.
- [NITROS migration](https://nvidia-isaac-ros.github.io/concepts/rosidl_buffer/nitros_migration.html): the 5.0 transport path must not be described using obsolete NITROS assumptions.

Live Isaac package execution, process transport/fallback/lifetime probes,
Nsight traces, measured before/after optimization and Isaac Sim acceptance are
NOT_RUN. GPU presence on the host does not prove a usable GPU in the ROS
container. These pending gates require a separate compatible disposable
container and measured workload; no host package replacement is part of this
implementation.
