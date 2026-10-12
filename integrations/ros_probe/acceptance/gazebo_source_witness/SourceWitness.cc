// Read-only Gazebo 8 ECM observer. No publishers, services, or ECM mutations.
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/GpuLidar.hh>
#include <gz/sim/components/Lidar.hh>
#include <gz/sim/components/Imu.hh>
#include <gz/sim/components/ContactSensor.hh>
#include <gz/sim/components/Collision.hh>
#include <gz/sim/components/Geometry.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/Sensor.hh>
#include <gz/sim/components/World.hh>
#include <sdf/Element.hh>

namespace rosclaw {
using Json = nlohmann::json;
using namespace gz::sim;

class SourceWitness final : public System, public ISystemConfigure,
                            public ISystemPostUpdate, public ISystemReset {
 public:
  ~SourceWitness() override { if (fd >= 0) ::close(fd); }

  void Configure(const Entity &, const std::shared_ptr<const sdf::Element> &sdf,
                 EntityComponentManager &, EventManager &) override {
    if (!sdf->HasElement("output") || !sdf->HasElement("run_id"))
      throw std::runtime_error("explicit observer output and run_id required");
    run = sdf->Get<std::string>("run_id");
    const auto output = sdf->Get<std::string>("output");
    if (run.empty() || run.size() > 128 || output.empty())
      throw std::runtime_error("bounded observer configuration required");
    fd = ::open(output.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) throw std::runtime_error("fresh exclusive observer file required");
  }

  void Reset(const UpdateInfo &, EntityComponentManager &) override {
    ++resetEpoch;
    lastSample = std::chrono::steady_clock::time_point{};
  }

  void PostUpdate(const UpdateInfo &info, const EntityComponentManager &ecm) override {
    const auto now = std::chrono::steady_clock::now();
    if (fd < 0 || samples >= 600 || now - lastSample < std::chrono::milliseconds(100)) return;
    lastSample = now;
    Json record = {{"schema_version", "rosclaw.gazebo_ecm_source_sample.v1"},
                   {"observer_run_label", run}, {"sample", samples++},
                   {"sim_time_ns", info.simTime.count()}, {"iterations", info.iterations},
                   {"paused", info.paused}, {"reset_epoch", resetEpoch},
                   {"observer_monotonic_ns", std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count()},
                   {"worlds", Json::array()}, {"models", Json::array()},
                   {"collisions", Json::array()}, {"sensors", Json::array()},
                   {"execution_authority", "NONE"}, {"inventory_complete", true}};
    ecm.Each<components::World, components::Name>([&](const Entity &e,
        const components::World *, const components::Name *name) {
      if (record["worlds"].size() >= 4096) { record["inventory_complete"] = false; return false; }
      record["worlds"].push_back({{"entity", e}, {"name", name->Data()}}); return true;
    });
    ecm.Each<components::Model, components::Name>([&](const Entity &e,
        const components::Model *, const components::Name *name) {
      Json model = Identity(e, ecm); model["name"] = name->Data();
      model["source_file_path"] = Model(e).SourceFilePath(ecm);
      if (record["models"].size() >= 4096) { record["inventory_complete"] = false; return false; }
      record["models"].push_back(model); return true;
    });
    ecm.Each<components::Collision, components::Geometry>([&](const Entity &e,
        const components::Collision *, const components::Geometry *geometry) {
      Json collision = Identity(e, ecm);
      const auto element = geometry->Data().ToElement();
      collision["geometry_sdf"] = element ? Json(element->ToString("")) : Json(nullptr);
      if (record["collisions"].size() >= 4096) { record["inventory_complete"] = false; return false; }
      record["collisions"].push_back(collision); return true;
    });
    ecm.Each<components::Sensor>([&](const Entity &e, const components::Sensor *) {
      Json sensor = Identity(e, ecm);
      sensor["kind"] = ecm.Component<components::GpuLidar>(e) ? "gpu_lidar" :
                       ecm.Component<components::Lidar>(e) ? "lidar" :
                       ecm.Component<components::Imu>(e) ? "imu" :
                       ecm.Component<components::ContactSensor>(e) ? "contact" : "UNKNOWN";
      if (record["sensors"].size() >= 4096) { record["inventory_complete"] = false; return false; }
      record["sensors"].push_back(sensor); return true;
    });
    for (const auto &name : {"worlds", "models", "collisions", "sensors"}) {
      if (record[name].size() > 4096) record["inventory_complete"] = false;
    }
    const auto line = record.dump() + "\n";
    if (line.size() > 500000 || totalBytes + line.size() > 30000000) {
      ::close(fd); fd = -1; return;
    }
    totalBytes += line.size();
    size_t written = 0;
    while (written < line.size()) {
      const auto n = ::write(fd, line.data() + written, line.size() - written);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) { ::close(fd); fd = -1; return; }
      written += static_cast<size_t>(n);
    }
  }

 private:
  static Json Identity(Entity e, const EntityComponentManager &ecm) {
    Json out = {{"entity", e}, {"ancestry", Json::array()}, {"pose", nullptr}};
    if (const auto name = ecm.Component<components::Name>(e)) out["name"] = name->Data();
    if (const auto p = ecm.Component<components::Pose>(e)) {
      const auto &pose = p->Data();
      out["pose"] = {pose.X(), pose.Y(), pose.Z(), pose.Roll(), pose.Pitch(), pose.Yaw()};
    }
    const auto world = worldPose(e, ecm);
    out["world_pose"] = {world.X(), world.Y(), world.Z(), world.Roll(), world.Pitch(), world.Yaw()};
    Entity current = e;
    for (size_t depth = 0; depth < 64; ++depth) {
      const auto parent = ecm.Component<components::ParentEntity>(current);
      if (!parent || parent->Data() == kNullEntity) break;
      current = parent->Data();
      Json node = {{"entity", current}, {"model", ecm.Component<components::Model>(current) != nullptr},
                   {"world", ecm.Component<components::World>(current) != nullptr}};
      if (const auto name = ecm.Component<components::Name>(current)) node["name"] = name->Data();
      out["ancestry"].push_back(node);
    }
    return out;
  }
  int fd = -1;
  std::string run;
  uint64_t samples = 0, resetEpoch = 0;
  size_t totalBytes = 0;
  std::chrono::steady_clock::time_point lastSample{};
};
}
GZ_ADD_PLUGIN(rosclaw::SourceWitness, gz::sim::System,
              rosclaw::SourceWitness::ISystemConfigure,
              rosclaw::SourceWitness::ISystemPostUpdate,
              rosclaw::SourceWitness::ISystemReset)
