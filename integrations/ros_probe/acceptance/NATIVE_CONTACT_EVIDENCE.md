# Native contact component evidence

Status: offline source contracts only; physical acceptance **NOT_RUN**.

The qualified dependency is Gazebo Sim **8.15.0**. Its native Contact system
resolves `sensor/contact/topic`, with a scoped default when that element is
absent. It does not create `SensorTopic`, and publishes contact messages only
when contacts exist. A silent ROS contact topic therefore cannot prove zero
contacts. The older v4 SDK inventory contract remains a historical offline
fixture; its required `SensorTopic` is not live native Contact admission.

Primary sources:

- https://github.com/gazebosim/gz-sim/blob/gz-sim8_8.15.0/src/systems/contact/Contact.cc
- https://github.com/gazebosim/gz-sim/blob/gz-sim8_8.15.0/src/systems/physics/Physics.cc

`rosclaw::PassiveContacts` is a separate plugin. It observes the actual World,
Body world pose from the physics-updated **Pose of a direct World child model**,
ContactSensor configuration, Collision entities and initialized
ContactSensorData in a const PostUpdate. Every actual Body collision must have
an unambiguous declared sensor. The packet retains actual contact entity IDs
and resolves names from actual same-world components. The plugin never creates
or changes ECM components, sensor configuration or actuator state.
The packet format is `rosclaw.gazebo_postupdate_contacts.v2`, with explicit
`body_pose_component=PHYSICS_UPDATED_DIRECT_WORLD_MODEL_POSE`. The model
WorldPose component is optional and not the source updated by native Physics;
it is ignored. Nested models or a missing actual model Pose are refused. The
earlier v1 SDK packets remain historical offline fixtures and are rejected by
the v2 decoder.

An empty initialized component is a measured **physics contact cache** snapshot,
not a fabricated ROS message. The cache may retain unchanged values. Packet
SIM time is the PostUpdate observation time, not an invented per-contact sample
time. Advancing packets and support contacts do not by themselves establish
that the physics backend remains healthy. Runtime admission must independently
qualify the pinned backend and its cache update behavior. That runtime gate is
not implemented here: `backend_health_admitted=false` remains explicit.

`native_contact_evidence.py` validates exact original UTF-8 packets, closed
schemas, frozen source identities, SDK version, complete inventory, source
sequence/iteration/time, normalized pose, and correspondence with a fresh
independent world pose (50 mm and 0.1 rad bounds, source ages below 300 ms).
Continuous declared support must show actual declared ground contact. Faults,
source loss, altered inventory or malformed packets latch rejection. Positive
contact counts are retained. The output is observation evidence and always
states `physical_acceptance=NOT_VERIFIED`.

`native_contact_observer.py` is an independent process with observation subscriptions.
It disables rosout, parameter/logger/type-description services and global
argument remapping, then retires the standard rclpy parameter metadata publisher
before observation. Standard constructor parameter metadata can be emitted
during initialization; no actuator publisher or motion service is ever created.
Its original pose CDR and original producer JSON bytes are retained with hashes
and sizes, including rejected inputs. It reopens supplied SDF/bridge/producer
bytes on preparation and closure. Source preparation is not runtime admission.
A closed-original-byte replay is implemented in
`closed_native_contact_evidence.py`: it reopens prepared sources, replays
installed official TFMessage CDR and original JSON, checks genesis/sequence/hash
chain and writer closure, and requires complete source brackets and no gaps.
Its 17 synthetic serialization contracts and 9 actual observer callback/resource cleanup contracts
retain original bytes and fault traces. Backend qualification and the replay
still need to be connected to final Native acceptance. No task or brush state is inferred here.

The N04 companion branch (maintained separately from N03) writes the nested native contact topic as well as the
explicit bridge topic. It preserves original source geometry and does not pick
or adapt any third robot asset. Frozen P0 source, drivers and image are unchanged.

The SDK contract constructs synthetic EntityComponentManager objects using the
installed official SDK, without a Gazebo server, actuator or publisher call.
Transport Node construction can initialize discovery; the SDK test container
has `--network none`. Those contracts are not actual simulation episodes.
The Python producer-file fixture uses synthetic bytes, not a loadable plugin.

Current offline checkpoint: 20 original native SDK component cases, 59 Python native contact cases, 33 existing independent-contact cases and 33 generic renderer cases. All are offline contracts. The installed SDK fixture validates direct-model Pose with an absent or misleading optional WorldPose, source immutability, and rejection of nested models. No actual backend or third robot has been admitted.
