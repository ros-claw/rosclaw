// Isolated SDK component fixtures; no Gazebo server, actuator or publisher call.
#include "passive_physics.cc"
#include <fstream>
#include <sdf/parser.hh>
namespace rosclaw
{
struct PassivePhysicsContractFixture
{
  gz::sim::EntityComponentManager ecm;
  PassivePhysics observer;
  gz::sim::UpdateInfo info;
  gz::sim::Entity world, body, obstacle, link, collision;
  explicit PassivePhysicsContractFixture()
  {
    using namespace gz::sim::components;
    world = ecm.CreateEntity();
    ecm.CreateComponent(world, World());
    ecm.CreateComponent(world, Name("fixture_world"));
    auto model = [&](const char *name)
    {
      auto id = ecm.CreateEntity();
      ecm.CreateComponent(id, Model());
      ecm.CreateComponent(id, Name(name));
      ecm.CreateComponent(id, ParentEntity(world));
      ecm.CreateComponent(id, Pose(gz::math::Pose3d::Zero));
      return id;
    };
    body = model("anonymous_body");
    obstacle = model("anonymous_blocker");
    link = ecm.CreateEntity();
    ecm.CreateComponent(link, ParentEntity(obstacle));
    ecm.CreateComponent(link, Pose(gz::math::Pose3d::Zero));
    collision = ecm.CreateEntity();
    ecm.CreateComponent(collision, Collision());
    ecm.CreateComponent(collision, ParentEntity(link));
    ecm.CreateComponent(collision, Pose(gz::math::Pose3d(0.1,0,0,0,0,0)));
    sdf::Geometry geometry;
    sdf::Sphere sphere;
    sphere.SetRadius(0.2);
    geometry.SetType(sdf::GeometryType::SPHERE);
    geometry.SetSphereShape(sphere);
    ecm.CreateComponent(collision, Geometry(geometry));
    observer.world = world;
    observer.worldName = "fixture_world";
    observer.runId = "synthetic_run";
    observer.bodyName = "anonymous_body";
    observer.bodyHash = "synthetic_body_hash";
    observer.attachmentHash = "synthetic_attachment_hash";
    observer.allowed = {"anonymous_body", "anonymous_blocker"};
    observer.obstacles = {"anonymous_blocker"};
    info.simTime = std::chrono::milliseconds(100);
    info.iterations = 10;
  }
  std::string Packet() const { return observer.Packet(info, ecm, 0); }
  void EnableBodyGeometry() { observer.includeBodyGeometry = true; }
  void RequireReference(const std::string &name = "actual_base") { observer.bodyReferenceLink = name; }
  void AddBodyGeometry(bool mesh = false)
  {
    using namespace gz::sim::components;
    observer.includeBodyGeometry = true;
    auto bodyLink = ecm.CreateEntity();
    ecm.CreateComponent(bodyLink, Link());
    ecm.CreateComponent(bodyLink, Name("actual_base"));
    ecm.CreateComponent(bodyLink, ParentEntity(body));
    ecm.CreateComponent(bodyLink, Pose(gz::math::Pose3d::Zero));
    auto bodyCollision = ecm.CreateEntity();
    ecm.CreateComponent(bodyCollision, Collision());
    ecm.CreateComponent(bodyCollision, Name("actual_collision"));
    ecm.CreateComponent(bodyCollision, ParentEntity(bodyLink));
    ecm.CreateComponent(bodyCollision, Pose(gz::math::Pose3d::Zero));
    sdf::Geometry geometry;
    if (mesh) geometry.SetType(sdf::GeometryType::MESH);
    else {
      sdf::Box box; box.SetSize(gz::math::Vector3d(0.4,0.3,0.2));
      geometry.SetType(sdf::GeometryType::BOX); geometry.SetBoxShape(box);
    }
    ecm.CreateComponent(bodyCollision, Geometry(geometry));
    auto wheelJoint = ecm.CreateEntity();
    ecm.CreateComponent(wheelJoint, Joint());
    ecm.CreateComponent(wheelJoint, ParentEntity(body));
  }
  void RequireContacts() { observer.includeBodyContacts = true; }
  gz::sim::Entity AddContact(const std::string &collisionName = "actual_collision",
      const std::string &topic = "/synthetic/body_contact")
  {
    using namespace gz::sim::components;
    gz::sim::Entity bodyLink = gz::sim::kNullEntity;
    ecm.Each<Link,Name>([&](auto entity, const auto *, const auto *name)
    {
      if (name->Data() == "actual_base") bodyLink = entity;
      return true;
    });
    if (bodyLink == gz::sim::kNullEntity) throw std::runtime_error("SDK fixture Body link absent");
    auto sensor = ecm.CreateEntity();
    ecm.CreateComponent(sensor, Sensor());
    ecm.CreateComponent(sensor, Name("actual_contact_sensor"));
    ecm.CreateComponent(sensor, ParentEntity(bodyLink));
    ecm.CreateComponent(sensor, SensorTopic(topic));
    auto document = std::make_shared<sdf::SDF>();
    sdf::init(document);
    const auto xml = "<sdf version='1.10'><model name='fixture'><link name='actual_base'>"
        "<sensor name='actual_contact_sensor' type='contact'><topic>" + topic +
        "</topic><contact><collision>" + collisionName +
        "</collision></contact></sensor></link></model></sdf>";
    if (!sdf::readString(xml, document)) throw std::runtime_error("actual SDK SDF parser fixture failed");
    auto element = document->Root()->GetElement("model")->GetElement("link")->GetElement("sensor");
    ecm.CreateComponent(sensor, ContactSensor(element));
    return sensor;
  }
  void MoveReference(const gz::math::Pose3d &pose)
  {
    using namespace gz::sim::components;
    ecm.Each<Link,Name>([&](auto entity, const auto *, const auto *name)
    {
      if (name->Data()=="actual_base")
        ecm.Component<Pose>(entity)->SetData(pose,[](const auto &,const auto &){return false;});
      return true;
    });
  }
};
}
int main()
{
  using rosclaw::PassivePhysicsContractFixture;
  using namespace gz::sim::components;
  std::ofstream out("contract-packets.jsonl");
  auto emit = [&](const char *test, PassivePhysicsContractFixture &f, bool expected)
  {
    const auto packet = f.Packet();
    const bool complete = packet.find("\"complete\":true") != std::string::npos;
    if (complete != expected) throw std::runtime_error(std::string("contract failed: ")+test);
    out << "{\"case\":" << rosclaw::Quote(test) << ",\"packet\":" << packet << "}\n";
  };
  {
    PassivePhysicsContractFixture f;
    emit("actual_collision_components", f, true);
    const auto before = f.ecm.Component<Pose>(f.collision)->Data();
    for (int i=0; i<3; ++i) emit("read_only_repeated", f, true);
    if (f.ecm.Component<Pose>(f.collision)->Data() != before)
      throw std::runtime_error("passive read changed collision pose");
  }
  {
    PassivePhysicsContractFixture f;
    auto extra=f.ecm.CreateEntity();
    f.ecm.CreateComponent(extra,Model()); f.ecm.CreateComponent(extra,Name("unregistered"));
    f.ecm.CreateComponent(extra,ParentEntity(f.world));
    emit("unexpected_model",f,false);
  }
  {
    PassivePhysicsContractFixture f;
    f.ecm.RemoveComponent<Geometry>(f.collision);
    emit("missing_collision_geometry",f,false);
  }
  {
    PassivePhysicsContractFixture f;
    sdf::Geometry unsupported; unsupported.SetType(sdf::GeometryType::MESH);
    f.ecm.Component<Geometry>(f.collision)->SetData(unsupported, [](const auto &, const auto &){return false;});
    emit("unsupported_mesh",f,false);
  }
  {
    PassivePhysicsContractFixture f;
    auto joint=f.ecm.CreateEntity(); f.ecm.CreateComponent(joint,Joint());
    f.ecm.CreateComponent(joint,ParentEntity(f.obstacle));
    emit("articulated_obstacle",f,false);
  }
  {
    PassivePhysicsContractFixture f;
    f.info.simTime=std::chrono::milliseconds(-1);
    emit("negative_time",f,false);
  }
  {
    PassivePhysicsContractFixture f;
    f.AddBodyGeometry();
    emit("v2_actual_body_collision_with_joint",f,true);
    const auto before=f.ecm.Component<Pose>(f.body)->Data();
    emit("v2_read_only_repeated",f,true);
    if (f.ecm.Component<Pose>(f.body)->Data()!=before)
      throw std::runtime_error("passive body component read changed pose");
  }
  {
    PassivePhysicsContractFixture f;
    f.EnableBodyGeometry();
    emit("v2_missing_body_geometry",f,false);
  }
  {
    PassivePhysicsContractFixture f;
    f.AddBodyGeometry(true);
    emit("v2_unsupported_body_mesh",f,false);
  }
  {
    PassivePhysicsContractFixture f;
    f.AddBodyGeometry(); f.RequireReference();
    emit("v3_actual_body_reference_identity",f,true);
    emit("v3_read_only_reference_repeated",f,true);
  }
  {
    PassivePhysicsContractFixture f;
    f.AddBodyGeometry(); f.RequireReference("missing_base");
    emit("v3_missing_reference_link",f,false);
  }
  {
    PassivePhysicsContractFixture f;
    f.AddBodyGeometry(); f.RequireReference();
    f.MoveReference(gz::math::Pose3d(0.01,0,0,0,0,0));
    emit("v3_shifted_reference_link",f,false);
  }
  {
    PassivePhysicsContractFixture f;
    f.AddBodyGeometry(); f.RequireReference();
    f.MoveReference(gz::math::Pose3d(0,0,0,0,0,0.01));
    emit("v3_rotated_reference_link",f,false);
  }
  {
    PassivePhysicsContractFixture f;
    f.AddBodyGeometry(); f.RequireReference(); f.RequireContacts();
    const auto sensor = f.AddContact();
    const auto before = f.ecm.Component<ContactSensor>(sensor)->Data()->ToString("");
    emit("v4_actual_contact_sensor_collision_mapping", f, true);
    emit("v4_read_only_contact_mapping_repeated", f, true);
    if (f.ecm.Component<ContactSensor>(sensor)->Data()->ToString("") != before)
      throw std::runtime_error("passive contact read changed SDK sensor configuration");
  }
  {
    PassivePhysicsContractFixture f;
    f.AddBodyGeometry(); f.RequireReference(); f.RequireContacts();
    emit("v4_missing_contact_sensor", f, false);
  }
  {
    PassivePhysicsContractFixture f;
    f.AddBodyGeometry(); f.RequireReference(); f.RequireContacts();
    auto sensor = f.AddContact(); f.ecm.RemoveComponent<SensorTopic>(sensor);
    emit("v4_missing_actual_sensor_topic", f, false);
  }
  {
    PassivePhysicsContractFixture f;
    f.AddBodyGeometry(); f.RequireReference(); f.RequireContacts();
    auto sensor = f.AddContact();
    f.ecm.Component<SensorTopic>(sensor)->SetData("/foreign", [](const auto &,const auto &){return false;});
    emit("v4_substituted_actual_sensor_topic", f, false);
  }
  {
    PassivePhysicsContractFixture f;
    f.AddBodyGeometry(); f.RequireReference(); f.RequireContacts(); f.AddContact("foreign_collision");
    emit("v4_missing_actual_collision_reference", f, false);
  }
  {
    PassivePhysicsContractFixture f;
    f.AddBodyGeometry(); f.RequireReference(); f.RequireContacts(); f.AddContact(); f.AddContact();
    emit("v4_duplicate_contact_coverage", f, false);
  }

}
