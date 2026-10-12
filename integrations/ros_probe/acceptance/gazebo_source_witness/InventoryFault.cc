// Disposable SDK-only fault actor, separate from the read-only SourceWitness.
// Never loaded by default. A unique run and independently selected model/entity
// must match an exclusive operator trigger before a sensor component is removed.
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/GpuLidar.hh>
#include <gz/sim/components/Lidar.hh>
#include <sdf/Element.hh>

namespace rosclaw {
class SourceInventoryFault final : public gz::sim::System,
    public gz::sim::ISystemConfigure, public gz::sim::ISystemPreUpdate {
 public:
  void Configure(const gz::sim::Entity &, const std::shared_ptr<const sdf::Element> &sdf,
                 gz::sim::EntityComponentManager &, gz::sim::EventManager &) override {
    if (!sdf->HasElement("run_id") || !sdf->HasElement("robot_entity"))
      throw std::runtime_error("explicit disposable operator fault configuration required");
    run = sdf->Get<std::string>("run_id"); robot = sdf->Get<std::string>("robot_entity");
    if (run.size() != 32 || run.find_first_not_of("0123456789abcdef") != std::string::npos ||
        robot.empty() || robot.size() > 128)
      throw std::runtime_error("bounded disposable run and robot required");
  }
  void PreUpdate(const gz::sim::UpdateInfo &, gz::sim::EntityComponentManager &ecm) override {
    if (finished) return;
    const int fd = ::open("/evidence/source-inventory-fault-trigger.json", O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) return;
    finished = true;
    char bytes[4097]; const auto n = ::read(fd, bytes, sizeof(bytes)); ::close(fd);
    if (n <= 0 || n > 4096) throw std::runtime_error("bounded complete operator trigger required");
    const auto trigger = nlohmann::json::parse(bytes, bytes + n);
    if (trigger.size() != 4 || trigger.at("run_id") != run || trigger.at("mode") != "sensor_kind_drift")
      throw std::runtime_error("fault trigger run/mode mismatch");
    const auto model = trigger.at("model_entity").get<gz::sim::Entity>();
    const auto sensor = trigger.at("sensor_entity").get<gz::sim::Entity>();
    const auto name = ecm.Component<gz::sim::components::Name>(model);
    if (!name || name->Data() != robot || !ecm.Component<gz::sim::components::Model>(model))
      throw std::runtime_error("fault model entity mismatch");
    auto ancestor = sensor; bool belongs = false;
    for (size_t i = 0; i < 64; ++i) {
      const auto parent = ecm.Component<gz::sim::components::ParentEntity>(ancestor);
      if (!parent) break;
      ancestor = parent->Data(); if (ancestor == model) { belongs = true; break; }
    }
    if (!belongs) throw std::runtime_error("fault sensor is not in the pinned model");
    if (ecm.Component<gz::sim::components::GpuLidar>(sensor))
      ecm.RemoveComponent<gz::sim::components::GpuLidar>(sensor);
    else if (ecm.Component<gz::sim::components::Lidar>(sensor))
      ecm.RemoveComponent<gz::sim::components::Lidar>(sensor);
    else throw std::runtime_error("fault sensor has no original lidar kind component");
  }
 private:
  bool finished = false;
  std::string run, robot;
};
}
GZ_ADD_PLUGIN(rosclaw::SourceInventoryFault, gz::sim::System,
              rosclaw::SourceInventoryFault::ISystemConfigure,
              rosclaw::SourceInventoryFault::ISystemPreUpdate)
