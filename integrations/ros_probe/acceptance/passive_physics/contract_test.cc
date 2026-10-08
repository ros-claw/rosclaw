// Isolated SDK component fixtures; no Gazebo server, actuator or publisher call.
#include "passive_physics.cc"
#include <fstream>
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
}
