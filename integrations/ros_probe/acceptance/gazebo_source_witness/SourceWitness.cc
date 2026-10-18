// Read-only Gazebo 8 ECM observer. No publishers, services, or ECM mutations.
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <sys/random.h>
#include <openssl/evp.h>
#include <iomanip>
#include <sstream>
#include <nlohmann/json.hpp>
#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/GpuLidar.hh>
#include <gz/sim/components/Lidar.hh>
#include <gz/sim/components/Imu.hh>
#include <gz/sim/components/Link.hh>
#include <gz/sim/components/Joint.hh>
#include <gz/sim/components/JointType.hh>
#include <gz/sim/components/JointAxis.hh>
#include <gz/sim/components/JointPosition.hh>
#include <gz/sim/components/ParentLinkName.hh>
#include <gz/sim/components/ChildLinkName.hh>
#include <cmath>
#include <gz/sim/components/NavSat.hh>
#include <gz/sim/components/ContactSensor.hh>
#include <gz/sim/components/Collision.hh>
#include <gz/sim/components/Geometry.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/Sensor.hh>
#include <gz/sim/components/World.hh>
#include <gz/sim/components/Static.hh>
#include <sdf/Element.hh>

namespace rosclaw {
using Json = nlohmann::json;
using namespace gz::sim;

class SourceWitness final : public System, public ISystemConfigure,
                            public ISystemPostUpdate, public ISystemReset {
 public:
  ~SourceWitness() override {
    if (fd >= 0) ::close(fd);
    if (digest) EVP_MD_CTX_free(digest);
  }

  void Configure(const Entity &, const std::shared_ptr<const sdf::Element> &sdf,
                 EntityComponentManager &, EventManager &) override {
    if (!sdf->HasElement("output") || !sdf->HasElement("run_id"))
      throw std::runtime_error("explicit observer output and run_id required");
    run = sdf->Get<std::string>("run_id");
    observeJointKinematics = sdf->HasElement("observe_joint_kinematics") &&
                             sdf->Get<bool>("observe_joint_kinematics");
    output = sdf->Get<std::string>("output");
    if (run.empty() || run.size() > 128 || output.empty())
      throw std::runtime_error("bounded observer configuration required");
    fd = ::open(output.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) throw std::runtime_error("fresh exclusive observer file required");
    segmented = sdf->HasElement("segment_max_samples") || sdf->HasElement("segment_max_bytes") ||
                sdf->HasElement("max_segments");
    if (segmented) {
      if (!sdf->HasElement("segment_max_samples") || !sdf->HasElement("segment_max_bytes") ||
          !sdf->HasElement("max_segments"))
        throw std::runtime_error("complete explicit evidence budget required");
      segmentMaxSamples = sdf->Get<uint64_t>("segment_max_samples");
      segmentMaxBytes = sdf->Get<uint64_t>("segment_max_bytes");
      maxSegments = sdf->Get<uint64_t>("max_segments");
      if (!segmentMaxSamples || segmentMaxSamples > 600 || segmentMaxBytes < 500000 ||
          segmentMaxBytes > 30000000 || !maxSegments || maxSegments > 64 ||
          segmentMaxBytes * maxSegments > 120000000)
        throw std::runtime_error("bounded evidence budget required");
      unsigned char entropy[16];
      if (::getrandom(entropy, sizeof(entropy), 0) != sizeof(entropy))
        throw std::runtime_error("independent observer epoch unavailable");
      epoch = Hex(entropy, sizeof(entropy));
      digest = EVP_MD_CTX_new();
      if (!digest || EVP_DigestInit_ex(digest, EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("source evidence digest unavailable");
    }
  }

  void Reset(const UpdateInfo &, EntityComponentManager &) override {
    ++resetEpoch;
    lastSample = std::chrono::steady_clock::time_point{};
  }

  void PostUpdate(const UpdateInfo &info, const EntityComponentManager &ecm) override {
    const auto now = std::chrono::steady_clock::now();
    if (fd < 0 || (!segmented && samples >= 600) ||
        now - lastSample < std::chrono::milliseconds(100)) return;
    lastSample = now;
    Json record = {{"schema_version", "rosclaw.gazebo_ecm_source_sample.v1"},
                   {"observer_run_label", run}, {"sample", samples},
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
      const auto isStatic = ecm.Component<components::Static>(e);
      model["static"] = isStatic ? Json(isStatic->Data()) : Json(nullptr);
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
                       ecm.Component<components::NavSat>(e) ? "navsat" :
                       ecm.Component<components::ContactSensor>(e) ? "contact" : "UNKNOWN";
      if (record["sensors"].size() >= 4096) { record["inventory_complete"] = false; return false; }
      record["sensors"].push_back(sensor); return true;
    });
    if (observeJointKinematics) {
      Json kinematics = {{"schema_version", "rosclaw.ecm_joint_kinematics.v1"},
                         {"complete", true}, {"links", Json::array()},
                         {"joints", Json::array()}};
      ecm.Each<components::Link, components::Name>([&](const Entity &e,
          const components::Link *, const components::Name *) {
        if (kinematics["links"].size() >= 4096) {
          kinematics["complete"] = false; return false;
        }
        kinematics["links"].push_back(Identity(e, ecm)); return true;
      });
      ecm.Each<components::Joint, components::Name>([&](const Entity &e,
          const components::Joint *, const components::Name *) {
        if (kinematics["joints"].size() >= 4096) {
          kinematics["complete"] = false; return false;
        }
        Json joint = Identity(e, ecm);
        const auto type = ecm.Component<components::JointType>(e);
        const auto parent = ecm.Component<components::ParentLinkName>(e);
        const auto child = ecm.Component<components::ChildLinkName>(e);
        const auto axis = ecm.Component<components::JointAxis>(e);
        const auto position = ecm.Component<components::JointPosition>(e);
        std::string kind = "UNKNOWN";
        if (type) {
          switch (type->Data()) {
            case sdf::JointType::FIXED: kind = "fixed"; break;
            case sdf::JointType::REVOLUTE: kind = "revolute"; break;
            case sdf::JointType::PRISMATIC: kind = "prismatic"; break;
            default: break;
          }
        }
        joint["kind"] = kind;
        joint["parent"] = parent ? Json(parent->Data()) : Json(nullptr);
        joint["child"] = child ? Json(child->Data()) : Json(nullptr);
        joint["axis"] = nullptr; joint["axis_frame"] = "";
        joint["bounds"] = nullptr; joint["position"] = nullptr;
        if (!parent || !child || kind == "UNKNOWN") kinematics["complete"] = false;
        if (kind != "fixed") {
          if (!axis) kinematics["complete"] = false;
          else {
            const auto &a = axis->Data(); const auto xyz = a.Xyz();
            if (!std::isfinite(xyz.X()) || !std::isfinite(xyz.Y()) ||
                !std::isfinite(xyz.Z()) || !std::isfinite(a.Lower()) ||
                !std::isfinite(a.Upper())) kinematics["complete"] = false;
            joint["axis"] = {xyz.X(), xyz.Y(), xyz.Z()};
            joint["axis_frame"] = a.XyzExpressedIn();
            joint["bounds"] = {a.Lower(), a.Upper()};
          }
        }
        if (position) {
          joint["position"] = position->Data();
          for (const auto value : position->Data()) {
            if (!std::isfinite(value)) kinematics["complete"] = false;
          }
        }
        kinematics["joints"].push_back(joint); return true;
      });
      record["joint_kinematics"] = kinematics;
    }
    for (const auto &name : {"worlds", "models", "collisions", "sensors"}) {
      if (record[name].size() > 4096) record["inventory_complete"] = false;
    }
    if (segmented) StampSegment(record);
    auto line = record.dump() + "\n";
    if (segmented && (segmentSamples >= segmentMaxSamples ||
                      totalBytes + line.size() > segmentMaxBytes)) {
      if (!Rotate()) return;
      StampSegment(record);
      line = record.dump() + "\n";
    }
    if (line.size() > 500000 || totalBytes + line.size() >
        (segmented ? segmentMaxBytes : 30000000)) {
      ::close(fd); fd = -1; return;
    }
    size_t written = 0;
    while (written < line.size()) {
      const auto n = ::write(fd, line.data() + written, line.size() - written);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) { ::close(fd); fd = -1; return; }
      written += static_cast<size_t>(n);
    }
    if (segmented && EVP_DigestUpdate(digest, line.data(), line.size()) != 1) {
      ::close(fd); fd = -1; return;
    }
    totalBytes += line.size();
    ++samples; ++segmentSamples;
  }

 private:
  static std::string Hex(const unsigned char *bytes, size_t size) {
    std::ostringstream out;
    for (size_t i = 0; i < size; ++i)
      out << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(bytes[i]);
    return out.str();
  }
  void StampSegment(Json &record) const {
    record["source_stream"] = {
      {"schema_version", "rosclaw.ecm_segment.v1"}, {"observer_epoch", epoch},
      {"observer_pid", ::getpid()}, {"segment_index", segmentIndex},
      {"segment_first_sample", segmentFirstSample},
      {"previous_segment_sha256", previousDigest},
      {"segment_max_samples", segmentMaxSamples}, {"segment_max_bytes", segmentMaxBytes},
      {"max_segments", maxSegments}, {"sample_interval_ms", 100}};
  }
  bool Rotate() {
    // Finalize before opening the next exclusive segment. Capacity exhaustion
    // closes the stream permanently; freshness watchdogs must fail closed.
    unsigned char bytes[EVP_MAX_MD_SIZE]; unsigned int size = 0;
    const bool ok = EVP_DigestFinal_ex(digest, bytes, &size) == 1 && size == 32;
    const bool synced = ::fdatasync(fd) == 0;
    ::close(fd); fd = -1;
    if (!ok || !synced) return false;
    previousDigest = Hex(bytes, size);
    if (++segmentIndex >= maxSegments) return false;
    std::ostringstream suffix;
    suffix << ".segment-" << std::setw(6) << std::setfill('0') << segmentIndex << ".jsonl";
    fd = ::open((output + suffix.str()).c_str(),
                O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) return false;
    if (EVP_DigestInit_ex(digest, EVP_sha256(), nullptr) != 1) {
      ::close(fd); fd = -1; return false;
    }
    totalBytes = 0; segmentSamples = 0; segmentFirstSample = samples;
    return true;
  }
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
  std::string output, epoch, previousDigest;
  bool segmented = false;
  bool observeJointKinematics = false;
  EVP_MD_CTX *digest = nullptr;
  uint64_t segmentIndex = 0, segmentSamples = 0, segmentFirstSample = 0;
  uint64_t segmentMaxSamples = 600, segmentMaxBytes = 30000000, maxSegments = 1;
  uint64_t samples = 0, resetEpoch = 0;
  size_t totalBytes = 0;
  std::chrono::steady_clock::time_point lastSample{};
};
}
GZ_ADD_PLUGIN(rosclaw::SourceWitness, gz::sim::System,
              rosclaw::SourceWitness::ISystemConfigure,
              rosclaw::SourceWitness::ISystemPostUpdate,
              rosclaw::SourceWitness::ISystemReset)
