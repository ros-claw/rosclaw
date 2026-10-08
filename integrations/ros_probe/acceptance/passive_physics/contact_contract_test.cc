// Official SDK component contracts; no Gazebo server or publisher call.
#include "passive_contacts.cc"
#include <fstream>
#include <sdf/parser.hh>

namespace rosclaw
{
struct PassiveContactsContractFixture
{
  gz::sim::EntityComponentManager ecm;
  PassiveContacts observer;
  gz::sim::UpdateInfo info;
  gz::sim::Entity world, body, bodyLink, collision, floorCollision, sensor;
  PassiveContactsContractFixture()
  {
    using namespace gz::sim::components;
    world = ecm.CreateEntity();
    ecm.CreateComponent(world, World()); ecm.CreateComponent(world, Name("fixture_world"));
    auto model = [&](const char *name)
    {
      auto entity = ecm.CreateEntity(); ecm.CreateComponent(entity, Model());
      ecm.CreateComponent(entity, Name(name)); ecm.CreateComponent(entity, ParentEntity(world));
      ecm.CreateComponent(entity, Pose(gz::math::Pose3d::Zero)); return entity;
    };
    body = model("anonymous_body");
    auto floor = model("support_plane");
    auto link = [&](auto parent, const char *name)
    {
      auto entity = ecm.CreateEntity(); ecm.CreateComponent(entity, Link());
      ecm.CreateComponent(entity, Name(name)); ecm.CreateComponent(entity, ParentEntity(parent)); return entity;
    };
    bodyLink = link(body, "actual_base");
    auto floorLink = link(floor, "floor_link");
    auto collider = [&](auto parent, const char *name)
    {
      auto entity = ecm.CreateEntity(); ecm.CreateComponent(entity, Collision());
      ecm.CreateComponent(entity, Name(name)); ecm.CreateComponent(entity, ParentEntity(parent)); return entity;
    };
    collision = collider(bodyLink, "actual_collision");
    floorCollision = collider(floorLink, "floor_collision");
    ecm.CreateComponent(collision, ContactSensorData());
    sensor = AddSensor(true);
    observer.world = world; observer.worldName = "fixture_world";
    observer.bodyName = "anonymous_body"; observer.runId = "synthetic_run";
    observer.bodyHash = "synthetic_body_hash"; observer.attachmentHash = "synthetic_attachment_hash";
    observer.producerId = "synthetic_actor";
    info.simTime = std::chrono::milliseconds(100); info.dt = std::chrono::milliseconds(1);
    info.iterations = 100; info.paused = false;
  }
  gz::sim::Entity AddSensor(bool nestedTopic)
  {
    using namespace gz::sim::components;
    sdf::Root root;
    const std::string topic = nestedTopic ? "<topic>/qualified/native_contact</topic>" : "";
    const auto errors = root.LoadSdfString("<sdf version='1.9'><model name='anonymous_body'><link name='actual_base'><sensor name='native_touch' type='contact'><topic>/ignored/generic_sensor_topic</topic><contact><collision>actual_collision</collision>" + topic + "</contact></sensor></link></model></sdf>");
    if (!errors.empty()) throw std::runtime_error("synthetic native contact SDF failed official parser");
    auto entity = ecm.CreateEntity();
    ecm.CreateComponent(entity, Sensor()); ecm.CreateComponent(entity, Name("native_touch"));
    ecm.CreateComponent(entity, ParentEntity(bodyLink));
    auto element = root.Model()->LinkByName("actual_base")->SensorByName("native_touch")->Element()->Clone();
    ecm.CreateComponent(entity, ContactSensor(element));
    return entity;
  }
  void GroundContact()
  {
    using namespace gz::sim::components;
    gz::msgs::Contacts data;
    auto pair = data.add_contact();
    pair->mutable_collision1()->set_id(collision);
    pair->mutable_collision2()->set_id(floorCollision);
    ecm.Component<ContactSensorData>(collision)->SetData(data, [](const auto &,const auto &){return false;});
  }
  std::string Packet() const { return observer.Packet(info, ecm, 0); }
};
}

int main(int argc, char **argv)
{
  std::ofstream file;
  if (argc == 2) file.open(argv[1]);
  std::ostream &out = file.is_open() ? file : std::cout;
  using F = rosclaw::PassiveContactsContractFixture;
  using namespace gz::sim::components;
  auto emit = [&](const char *name, F &f, bool expected)
  {
    const auto packet = f.Packet();
    const bool complete = packet.find("\"complete\":true") != std::string::npos;
    if (complete != expected) throw std::runtime_error(std::string("unexpected native SDK contact result ") + name + ": " + packet);
    out << "{\"case\":\"" << name << "\",\"expected_complete\":" << (expected ? "true" : "false") << ",\"packet\":" << packet << "}\n";
  };
  {
    F f;
    const auto element = f.ecm.Component<ContactSensor>(f.sensor)->Data();
    const auto before = element->ToString("");
    emit("native_empty_component_not_ros_silence", f, true);
    emit("native_repeated_read_is_immutable", f, true);
    if (element->ToString("") != before || f.ecm.Component<ContactSensorData>(f.collision)->Data().contact_size() != 0)
      throw std::runtime_error("native passive contact read changed SDK source");
    const auto packet = f.Packet();
    if (packet.find("/qualified/native_contact") == std::string::npos || packet.find("/ignored/generic_sensor_topic") != std::string::npos)
      throw std::runtime_error("native resolver used incorrect generic sensor/topic");
  }
  {
    F f; f.GroundContact(); emit("native_actual_contact_entity_names", f, true);
    const auto packet = f.Packet();
    if (packet.find("support_plane::floor_link::floor_collision") == std::string::npos)
      throw std::runtime_error("native contact IDs did not resolve actual ECS collision scope");
  }
  { F f; f.ecm.RemoveComponent<ContactSensor>(f.sensor); emit("native_missing_contact_sensor", f, false); }
  { F f; f.ecm.RemoveComponent<ContactSensorData>(f.collision); emit("native_missing_initialized_contact_data", f, false); }
  { F f; f.ecm.RemoveComponent<Sensor>(f.sensor); emit("native_missing_sensor_marker", f, false); }
  { F f; f.ecm.RemoveComponent<Name>(f.bodyLink); emit("native_missing_link_identity", f, false); }
  { F f; f.ecm.RemoveComponent<Pose>(f.body); emit("native_missing_body_model_pose", f, false); }
  { F f; f.info.paused = true; emit("native_paused_physics_unknown", f, false); }
  { F f; f.info.dt = std::chrono::milliseconds(0); emit("native_nonadvancing_step_unknown", f, false); }
  { F f; f.ecm.CreateComponent(f.sensor, SensorTopic("/wrong")); emit("native_contradictory_topic_component", f, false); }
  { F f; f.ecm.CreateComponent(f.sensor, SensorTopic("/qualified/native_contact")); emit("native_optional_topic_confirmation", f, true); }
  { F f; f.AddSensor(true); emit("native_duplicate_collision_mapping", f, false); }
  {
    F f; f.GroundContact(); f.ecm.RemoveComponent<Collision>(f.floorCollision);
    emit("native_contact_counterpart_missing", f, false);
  }
  {
    F f; gz::msgs::Contacts data; auto pair = data.add_contact();
    pair->mutable_collision1()->set_id(f.floorCollision); pair->mutable_collision2()->set_id(f.world);
    f.ecm.Component<ContactSensorData>(f.collision)->SetData(data, [](const auto &,const auto &){return false;});
    emit("native_contact_data_foreign_collision", f, false);
  }
  {
    F f; auto element = f.ecm.Component<ContactSensor>(f.sensor)->Data();
    element->GetElement("contact")->GetElement("collision")->Set<std::string>("foreign");
    emit("native_unresolved_source_collision", f, false);
  }
  {
    F f; f.ecm.RemoveComponent<ContactSensor>(f.sensor); f.ecm.RemoveComponent<Sensor>(f.sensor);
    f.ecm.RemoveComponent<ParentEntity>(f.sensor); f.ecm.RemoveComponent<Name>(f.sensor);
    f.AddSensor(false); emit("native_scoped_default_without_sensor_topic", f, true);
  }
  {
    F f; f.ecm.CreateComponent(f.body, WorldPose(gz::math::Pose3d(9, 8, 7, 0, 0, 0)));
    emit("native_optional_world_pose_is_not_physics_model_source", f, true);
    if (f.Packet().find("\"body_world_pose\":[0,0,0,1,0,0,0]") == std::string::npos)
      throw std::runtime_error("native model used optional stale WorldPose instead of physics Pose");
  }
  {
    F f; f.ecm.Component<Pose>(f.body)->SetData(gz::math::Pose3d(1, 2, 3, 0, 0, 0), [](const auto &,const auto &){return false;});
    emit("native_actual_model_pose_coordinates", f, true);
    if (f.Packet().find("\"body_world_pose\":[1,2,3,1,0,0,0]") == std::string::npos)
      throw std::runtime_error("native actual direct World model coordinates lost");
  }
  {
    F f; f.ecm.Component<ParentEntity>(f.body)->SetData(f.bodyLink, [](const auto &,const auto &){return false;});
    emit("native_nested_body_model_refused", f, false);
  }
  if (rosclaw::native_contacts::PublisherTopic("/rosclaw_sim/contact_components") != "/rosclaw_sim/contact_components" ||
      rosclaw::native_contacts::PublisherTopic("/rosclaw_sim/backend_probe_components") != "/rosclaw_sim/backend_probe_components")
    throw std::runtime_error("SIM observation publisher topic binding changed");
  for (const auto topic : {"/cmd_vel", "relative", "/rosclaw_sim/arbitrary"})
  {
    bool refused = false;
    try { rosclaw::native_contacts::PublisherTopic(topic); }
    catch (const std::runtime_error &) { refused = true; }
    if (!refused) throw std::runtime_error("native observer allowed non-observation publisher topic");
  }
  for (const auto delta : {1, 10, 49})
  {
    if (!rosclaw::native_contacts::PublishThisStep("/rosclaw_sim/contact_components", true,
        std::chrono::milliseconds(0), std::chrono::milliseconds(delta), false))
      throw std::runtime_error("robot evidence skipped an actual physics step");
    if (rosclaw::native_contacts::PublishThisStep("/rosclaw_sim/backend_probe_components", true,
        std::chrono::milliseconds(0), std::chrono::milliseconds(delta), false))
      throw std::runtime_error("instrument proof stream unexpectedly changed cadence");
  }
  if (!rosclaw::native_contacts::PublishThisStep("/rosclaw_sim/backend_probe_components", true,
      std::chrono::milliseconds(0), std::chrono::milliseconds(1), true))
    throw std::runtime_error("paused instrument source fault was hidden by throttle");
  {
    F f;
    auto obstacle = f.ecm.CreateEntity();
    f.ecm.CreateComponent(obstacle, Model()); f.ecm.CreateComponent(obstacle, Name("actual_obstacle"));
    f.ecm.CreateComponent(obstacle, ParentEntity(f.world));
    auto link = f.ecm.CreateEntity();
    f.ecm.CreateComponent(link, Link()); f.ecm.CreateComponent(link, Name("obstacle_link"));
    f.ecm.CreateComponent(link, ParentEntity(obstacle));
    auto collider = f.ecm.CreateEntity();
    f.ecm.CreateComponent(collider, Collision()); f.ecm.CreateComponent(collider, Name("obstacle_collision"));
    f.ecm.CreateComponent(collider, ParentEntity(link));
    for (std::uint64_t step = 0; step < 3; ++step)
    {
      f.GroundContact();
      if (step == 1)
      {
        auto data = f.ecm.Component<ContactSensorData>(f.collision)->Data();
        auto pair = data.add_contact();
        pair->mutable_collision1()->set_id(f.collision);
        pair->mutable_collision2()->set_id(collider);
        f.ecm.Component<ContactSensorData>(f.collision)->SetData(data, [](const auto &, const auto &){return false;});
      }
      f.info.simTime = std::chrono::milliseconds(100 + step);
      f.info.iterations = 100 + step;
      if (!rosclaw::native_contacts::PublishThisStep("/rosclaw_sim/contact_components", step != 0,
          std::chrono::milliseconds(99 + step), f.info.simTime, false))
        throw std::runtime_error("short transient contact physics step omitted");
      out << "{\"case\":\"native_all_step_transient_" << step
          << "\",\"expected_complete\":true,\"packet\":"
          << f.observer.Packet(f.info, f.ecm, step) << "}\n";
    }
  }
  return 0;
}
