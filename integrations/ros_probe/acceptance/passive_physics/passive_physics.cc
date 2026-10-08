// Read-only physical state/geometry producer. No actuator, service, ECM writes,
// task completion, localization estimate, or mission coverage calculation.
#include <gz/sim/System.hh>
#include <gz/sim/Conversions.hh>
#include <gz/msgs/geometry.pb.h>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Collision.hh>
#include <gz/sim/components/Geometry.hh>
#include <gz/sim/components/Joint.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/World.hh>
#include <gz/plugin/Register.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/stringmsg.pb.h>
#include <sdf/Geometry.hh>
#include <sdf/Box.hh>
#include <sdf/Sphere.hh>
#include <sdf/Cylinder.hh>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace rosclaw
{
std::string Quote(const std::string &_value)
{
  if (_value.size() > 256) throw std::runtime_error("entity name unbounded");
  std::ostringstream out;
  out << '"';
  for (unsigned char ch : _value)
  {
    if (ch == '"' || ch == '\\') out << '\\' << ch;
    else if (ch < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned>(ch) << std::dec;
    else out << ch;
  }
  out << '"';
  return out.str();
}

std::string PoseJson(const gz::math::Pose3d &_pose)
{
  const double values[] = {_pose.Pos().X(), _pose.Pos().Y(), _pose.Pos().Z(),
      _pose.Rot().W(), _pose.Rot().X(), _pose.Rot().Y(), _pose.Rot().Z()};
  std::ostringstream out;
  out << std::setprecision(17) << '[';
  for (std::size_t i = 0; i < 7; ++i)
  {
    if (!std::isfinite(values[i])) throw std::runtime_error("nonfinite physics pose");
    if (i) out << ',';
    out << values[i];
  }
  out << ']';
  return out.str();
}

class PassivePhysics : public gz::sim::System,
    public gz::sim::ISystemConfigure, public gz::sim::ISystemPostUpdate
{
  public: void Configure(const gz::sim::Entity &_entity,
      const std::shared_ptr<const sdf::Element> &_sdf,
      gz::sim::EntityComponentManager &_ecm, gz::sim::EventManager &) override
  {
    // Configure is part of the plugin ABI; no mutation of its ECM reference.
    const auto config = _sdf->Clone();
    this->world = _entity;
    if (!_ecm.Component<gz::sim::components::World>(_entity))
      throw std::runtime_error("passive observer must be attached to the world");
    const auto name = _ecm.Component<gz::sim::components::Name>(_entity);
    if (!name) throw std::runtime_error("world identity missing");
    this->worldName = name->Data();
    auto required = [&](const std::string &key)
    {
      if (!config->HasElement(key)) throw std::runtime_error("missing frozen observer identity");
      const auto value = config->Get<std::string>(key);
      if (value.empty() || value.size() > 256) throw std::runtime_error("invalid observer identity");
      return value;
    };
    this->runId = required("run_id");
    this->bodyName = required("body_model_name");
    this->bodyHash = required("body_snapshot_hash");
    this->attachmentHash = required("attachment_hash");
    this->allowed.insert(this->bodyName);
    for (const auto &key : {"static_model", "obstacle_model"})
    {
      if (!config->HasElement(key)) continue;
      for (auto e = config->GetElement(key); e; e = e->GetNextElement(key))
      {
        const auto value = e->Get<std::string>();
        if (value.empty() || !this->allowed.insert(value).second || this->allowed.size() > 64)
          throw std::runtime_error("ambiguous or unbounded scene model identity");
        if (std::string(key) == "obstacle_model") this->obstacles.insert(value);
      }
    }
    if (this->obstacles.empty() || this->obstacles.size() > 32)
      throw std::runtime_error("explicit bounded obstacle identities required");
    this->publisher = this->node.Advertise<gz::msgs::StringMsg>("/rosclaw_sim/physics_snapshot");
  }

  private: gz::math::Pose3d RelativePose(gz::sim::Entity entity,
      gz::sim::Entity model, const gz::sim::EntityComponentManager &ecm) const
  {
    gz::math::Pose3d result;
    for (std::size_t depth = 0; entity != model; ++depth)
    {
      if (depth > 64) throw std::runtime_error("collision parent chain unbounded");
      const auto pose = ecm.Component<gz::sim::components::Pose>(entity);
      const auto parent = ecm.Component<gz::sim::components::ParentEntity>(entity);
      if (!pose || !parent) throw std::runtime_error("collision model-relative pose missing");
      result = pose->Data() * result;
      entity = parent->Data();
    }
    return result;
  }

  private: bool Descendant(gz::sim::Entity entity, gz::sim::Entity model,
      const gz::sim::EntityComponentManager &ecm) const
  {
    for (std::size_t depth = 0; depth < 64; ++depth)
    {
      if (entity == model) return true;
      const auto parent = ecm.Component<gz::sim::components::ParentEntity>(entity);
      if (!parent) return false;
      entity = parent->Data();
    }
    throw std::runtime_error("entity parent chain unbounded");
  }

  private: std::string CollisionGeometry(gz::sim::Entity model,
      const gz::sim::EntityComponentManager &ecm) const
  {
    bool articulated = false;
    ecm.Each<gz::sim::components::Joint>([&](const gz::sim::Entity entity, const auto *)
    {
      if (this->Descendant(entity, model, ecm)) articulated = true;
      return true;
    });
    if (articulated) throw std::runtime_error("articulated obstacle unsupported");
    std::map<gz::sim::Entity, std::string> collisionRows;
    ecm.Each<gz::sim::components::Collision, gz::sim::components::Geometry>(
        [&](const gz::sim::Entity entity, const auto *, const auto *geometry)
    {
      if (!this->Descendant(entity, model, ecm)) return true;
      if (collisionRows.size() >= 256) throw std::runtime_error("collision count unbounded");
      const auto &g = geometry->Data();
      double radius = 0;
      std::ostringstream shape;
      shape << std::setprecision(17);
      if (g.Type() == sdf::GeometryType::BOX && g.BoxShape())
      {
        auto size = g.BoxShape()->Size();
        if (size.X() <= 0 || size.Y() <= 0 || size.Z() <= 0)
          throw std::runtime_error("nonpositive collision box");
        radius = size.Length() / 2;
        shape << "\"kind\":\"box\",\"size\":[" << size.X() << ',' << size.Y() << ',' << size.Z() << ']';
      }
      else if (g.Type() == sdf::GeometryType::SPHERE && g.SphereShape())
      {
        radius = g.SphereShape()->Radius();
        shape << "\"kind\":\"sphere\",\"radius\":" << radius;
      }
      else if (g.Type() == sdf::GeometryType::CYLINDER && g.CylinderShape())
      {
        const auto r = g.CylinderShape()->Radius(), length = g.CylinderShape()->Length();
        if (r <= 0 || length <= 0) throw std::runtime_error("nonpositive collision cylinder");
        radius = std::hypot(r, length / 2);
        shape << "\"kind\":\"cylinder\",\"radius\":" << r << ",\"length\":" << length;
      }
      else throw std::runtime_error("unsupported actual collision geometry");
      const auto relative = this->RelativePose(entity, model, ecm);
      radius += relative.Pos().Length();
      if (!std::isfinite(radius) || radius <= 0 || radius > 100)
        throw std::runtime_error("collision envelope invalid or unbounded");
      std::ostringstream row;
      row << std::setprecision(17) << "{\"entity_id\":" << entity << ',' << shape.str()
          << ",\"model_relative_pose\":" << PoseJson(relative)
          << ",\"enclosing_radius_m\":" << radius << '}';
      collisionRows.emplace(entity, row.str());
      return true;
    });
    if (collisionRows.empty()) throw std::runtime_error("actual obstacle collision missing");
    std::ostringstream out;
    out << '[';
    bool first = true;
    for (const auto &row : collisionRows) { if (!first) out << ','; first = false; out << row.second; }
    out << ']';
    return out.str();
  }

  public: std::string Packet(const gz::sim::UpdateInfo &info,
      const gz::sim::EntityComponentManager &ecm, std::uint64_t packetSequence) const
  {
    std::ostringstream packet;
    packet << std::setprecision(17) << "{\"schema_version\":\"rosclaw.gazebo_postupdate_observation.v1\","
        << "\"run_id\":" << Quote(this->runId) << ",\"body_snapshot_hash\":" << Quote(this->bodyHash)
        << ",\"attachment_hash\":" << Quote(this->attachmentHash)
        << ",\"world_name\":" << Quote(this->worldName)
        << ",\"sequence\":" << packetSequence << ",\"sim_time_sec\":"
        << std::chrono::duration<double>(info.simTime).count() << ",\"physics_iteration\":" << info.iterations
        << ",\"captured_at_unix_ns\":" << std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()
        << ",\"paused\":" << (info.paused ? "true" : "false")
        << ",\"source\":\"gazebo_ecm_postupdate\",\"evidence_domain\":\"GAZEBO_PHYSICS\"";
    try
    {
      if (info.simTime < std::chrono::steady_clock::duration::zero())
        throw std::runtime_error("negative physics time");
      std::map<std::string, gz::sim::Entity> models;
      ecm.Each<gz::sim::components::Model, gz::sim::components::Name, gz::sim::components::ParentEntity>(
          [&](gz::sim::Entity entity, const auto *, const auto *name, const auto *parent)
      {
        if (parent->Data() != this->world) return true;
        if (!models.emplace(name->Data(), entity).second || models.size() > 64)
          throw std::runtime_error("ambiguous or unbounded actual scene");
        return true;
      });
      std::set<std::string> observed;
      for (const auto &row : models) observed.insert(row.first);
      if (observed != this->allowed) throw std::runtime_error("loaded scene model set mismatch");
      packet << ",\"scene_models\":[";
      bool firstModel = true;
      for (const auto &model : models)
      {
        if (!firstModel) packet << ',';
        firstModel = false;
        packet << "{\"model_name\":" << Quote(model.first) << ",\"entity_id\":" << model.second << '}';
      }
      packet << ']';
      packet << ",\"body\":{\"model_name\":" << Quote(this->bodyName)
          << ",\"entity_id\":" << models.at(this->bodyName)
          << ",\"world_pose\":" << PoseJson(gz::sim::worldPose(models.at(this->bodyName), ecm)) << '}';
      packet << ",\"obstacles\":[";
      bool first = true;
      for (const auto &name : this->obstacles)
      {
        if (!first) packet << ',';
        first = false;
        const auto entity = models.at(name);
        packet << "{\"model_name\":" << Quote(name) << ",\"entity_id\":" << entity
            << ",\"world_pose\":" << PoseJson(gz::sim::worldPose(entity, ecm))
            << ",\"collision_geometry\":" << this->CollisionGeometry(entity, ecm) << '}';
      }
      packet << "],\"complete\":true}";
    }
    catch (const std::exception &error)
    {
      // A partially assembled object is never published. Fault has no free cells.
      packet.str(""); packet.clear();
      packet << "{\"schema_version\":\"rosclaw.gazebo_postupdate_observation.v1\",\"run_id\":"
          << Quote(this->runId) << ",\"sequence\":" << packetSequence
          << ",\"sim_time_sec\":" << std::setprecision(17) << std::chrono::duration<double>(info.simTime).count()
          << ",\"source\":\"gazebo_ecm_postupdate\",\"evidence_domain\":\"GAZEBO_PHYSICS\",\"complete\":false,\"fault\":"
          << Quote(error.what()) << '}';
    }
    return packet.str();
  }
  public: void PostUpdate(const gz::sim::UpdateInfo &info,
      const gz::sim::EntityComponentManager &ecm) override
  {
    if (this->published && info.simTime == this->lastTime) return;
    if (this->published && info.simTime > this->lastTime && info.simTime - this->lastTime < std::chrono::milliseconds(50)) return;
    this->published = true;
    this->lastTime = info.simTime;
    gz::msgs::StringMsg message;
    message.set_data(this->Packet(info, ecm, this->sequence++));
    this->publisher.Publish(message);
  }
  friend struct PassivePhysicsContractFixture;
  private: gz::sim::Entity world = gz::sim::kNullEntity;
  private: std::string runId, bodyName, bodyHash, attachmentHash, worldName;
  private: std::set<std::string> allowed, obstacles;
  private: gz::transport::Node node;
  private: gz::transport::Node::Publisher publisher;
  private: std::chrono::steady_clock::duration lastTime{};
  private: bool published = false;
  private: std::uint64_t sequence = 0;
};
}
GZ_ADD_PLUGIN(rosclaw::PassivePhysics, gz::sim::System,
    gz::sim::ISystemConfigure, gz::sim::ISystemPostUpdate)
GZ_ADD_PLUGIN_ALIAS(rosclaw::PassivePhysics, "rosclaw::PassivePhysics")
