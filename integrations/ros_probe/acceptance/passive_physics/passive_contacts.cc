// Separate, passive continuous ContactSensorData producer for qualified SIM.
#include "native_contacts.hh"
#include <gz/sim/System.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/plugin/Register.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/stringmsg.pb.h>
#include <chrono>
#include <cmath>
#include <cstdint>

namespace rosclaw
{
class PassiveContacts : public gz::sim::System,
    public gz::sim::ISystemConfigure, public gz::sim::ISystemPostUpdate
{
  public: void Configure(const gz::sim::Entity &entity,
      const std::shared_ptr<const sdf::Element> &config,
      gz::sim::EntityComponentManager &ecm,
      gz::sim::EventManager &) override
  {
    native_contacts::QualifiedVersion();
    const auto name = ecm.Component<gz::sim::components::Name>(entity);
    if (!ecm.Component<gz::sim::components::World>(entity) || !name)
      throw std::runtime_error("passive contacts must attach to the actual World");
    auto required = [&](const char *key)
    {
      if (!config->HasElement(key)) throw std::runtime_error(std::string("missing contact binding ") + key);
      const auto value = config->Get<std::string>(key);
      if (value.empty() || value.size() > 256) throw std::runtime_error("bounded frozen contact binding required");
      return value;
    };
    this->world = entity;
    this->worldName = required("world_name");
    this->bodyName = required("body_model_name");
    this->runId = required("run_id");
    this->bodyHash = required("body_snapshot_hash");
    this->attachmentHash = required("attachment_hash");
    this->producerId = required("producer_id");
    if (name->Data() != this->worldName)
      throw std::runtime_error("actual contact world name differs from declared source");
    this->publisher = this->node.Advertise<gz::msgs::StringMsg>("/rosclaw_sim/contact_components");
  }
  public: std::string Packet(const gz::sim::UpdateInfo &info,
      const gz::sim::EntityComponentManager &ecm, std::uint64_t sequence) const
  {
    using namespace gz::sim::components;
    std::ostringstream out;
    const auto simTime = std::chrono::duration<double>(info.simTime).count();
    const auto dt = std::chrono::duration<double>(info.dt).count();
    const auto captured = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    out << std::setprecision(17)
        << "{\"schema_version\":\"rosclaw.gazebo_postupdate_contacts.v2\""
        << ",\"source\":\"gazebo_ecm_contact_sensor_data\",\"evidence_domain\":\"GAZEBO_PHYSICS\""
        << ",\"sdk_version\":\"8.15.0\",\"component_semantics\":\"PHYSICS_UPDATE_CONTACT_CACHE\""
        << ",\"body_pose_component\":\"PHYSICS_UPDATED_DIRECT_WORLD_MODEL_POSE\""
        << ",\"run_id\":" << native_contacts::Quote(this->runId)
        << ",\"body_snapshot_hash\":" << native_contacts::Quote(this->bodyHash)
        << ",\"attachment_hash\":" << native_contacts::Quote(this->attachmentHash)
        << ",\"producer_id\":" << native_contacts::Quote(this->producerId)
        << ",\"world_name\":" << native_contacts::Quote(this->worldName)
        << ",\"body_model_name\":" << native_contacts::Quote(this->bodyName)
        << ",\"world_entity_id\":" << this->world
        << ",\"sequence\":" << sequence << ",\"iterations\":" << info.iterations
        << ",\"sim_time_sec\":" << simTime << ",\"physics_step_dt_sec\":" << dt
        << ",\"captured_at_unix_ns\":" << captured
        << ",\"paused\":" << (info.paused ? "true" : "false");
    try
    {
      if (!std::isfinite(simTime) || simTime < 0 || !std::isfinite(dt) || dt <= 0 || info.paused)
        throw std::runtime_error("contact evidence requires an advancing unpaused physics step");
      const auto worldName = ecm.Component<Name>(this->world);
      if (!ecm.Component<World>(this->world) || !worldName || worldName->Data() != this->worldName)
        throw std::runtime_error("actual contact World identity changed");
      auto body = gz::sim::kNullEntity;
      ecm.Each<Model,Name,ParentEntity>([&](auto entity, const auto *, const auto *name, const auto *parent)
      {
        if (parent->Data() != this->world || name->Data() != this->bodyName) return true;
        if (body != gz::sim::kNullEntity) throw std::runtime_error("ambiguous actual contact Body model");
        body = entity; return true;
      });
      if (body == gz::sim::kNullEntity) throw std::runtime_error("actual contact Body model missing");
      // Physics.cc updates Pose on a direct World child model. WorldPose is
      // optional, and native Physics only updates that component on links.
      const auto pose = ecm.Component<Pose>(body);
      if (!pose) throw std::runtime_error("actual physics-updated direct World Body Pose missing");
      const double values[] = {pose->Data().Pos().X(), pose->Data().Pos().Y(), pose->Data().Pos().Z(),
          pose->Data().Rot().W(), pose->Data().Rot().X(), pose->Data().Rot().Y(), pose->Data().Rot().Z()};
      const auto norm = values[3]*values[3] + values[4]*values[4] + values[5]*values[5] + values[6]*values[6];
      if (!std::isfinite(norm) || std::abs(norm - 1) > 1e-6)
        throw std::runtime_error("actual contact Body quaternion not normalized");
      std::ostringstream p; p << std::setprecision(17) << '[';
      for (std::size_t i = 0; i < 7; ++i)
      {
        if (!std::isfinite(values[i])) throw std::runtime_error("actual contact Body pose nonfinite");
        if (i) p << ',';
        p << values[i];
      }
      p << ']';
      const auto contacts = native_contacts::Read(body, this->world, ecm);
      out << ",\"body_model_entity_id\":" << body << ",\"body_world_pose\":" << p.str()
          << ",\"contact_sources\":" << contacts.inventory
          << ",\"collision_contacts\":" << contacts.observations << ",\"complete\":true}";
    }
    catch (const std::exception &error)
    {
      out << ",\"complete\":false,\"fault\":" << native_contacts::Quote(error.what()) << '}';
    }
    const auto packet = out.str();
    if (packet.size() > 2'000'000) throw std::runtime_error("native contact packet exceeds bounded source size");
    return packet;
  }
  public: void PostUpdate(const gz::sim::UpdateInfo &info,
      const gz::sim::EntityComponentManager &ecm) override
  {
    if (this->published && info.simTime >= this->lastTime && info.simTime - this->lastTime < std::chrono::milliseconds(50)) return;
    gz::msgs::StringMsg message;
    message.set_data(this->Packet(info, ecm, this->sequence++));
    this->publisher.Publish(message);
    this->lastTime = info.simTime; this->published = true;
  }
  friend struct PassiveContactsContractFixture;
  private: gz::sim::Entity world = gz::sim::kNullEntity;
  private: std::string worldName, bodyName, runId, bodyHash, attachmentHash, producerId;
  private: gz::transport::Node node;
  private: gz::transport::Node::Publisher publisher;
  private: std::chrono::steady_clock::duration lastTime{};
  private: bool published = false;
  private: std::uint64_t sequence = 0;
};
}
GZ_ADD_PLUGIN(rosclaw::PassiveContacts, gz::sim::System,
    gz::sim::ISystemConfigure, gz::sim::ISystemPostUpdate)
GZ_ADD_PLUGIN_ALIAS(rosclaw::PassiveContacts, "rosclaw::PassiveContacts")
