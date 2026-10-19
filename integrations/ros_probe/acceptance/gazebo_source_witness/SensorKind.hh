// SDK-only read-only component metadata. No image, ROS, or execution authority.
#ifndef ROSCLAW_SOURCE_WITNESS_SENSOR_KIND_HH_
#define ROSCLAW_SOURCE_WITNESS_SENSOR_KIND_HH_
#include <string>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/components/Camera.hh>
#include <gz/sim/components/DepthCamera.hh>
#include <gz/sim/components/RgbdCamera.hh>

namespace rosclaw {
template <typename Component>
std::string CameraComponentKind(const gz::sim::Entity &entity,
                               const gz::sim::EntityComponentManager &ecm,
                               const char *expected) {
  const auto *component = ecm.Component<Component>(entity);
  if (!component) return {};
  const std::string kind = component->Data().TypeStr();
  return kind == expected ? kind : "UNKNOWN";
}

inline std::string CameraKind(const gz::sim::Entity &entity,
                              const gz::sim::EntityComponentManager &ecm) {
  std::string result;
  for (const auto &kind : {
         CameraComponentKind<gz::sim::components::Camera>(entity, ecm, "camera"),
         CameraComponentKind<gz::sim::components::DepthCamera>(entity, ecm, "depth_camera"),
         CameraComponentKind<gz::sim::components::RgbdCamera>(entity, ecm, "rgbd_camera")}) {
    if (kind.empty()) continue;
    if (!result.empty()) return "UNKNOWN";
    result = kind;
  }
  return result.empty() ? "UNKNOWN" : result;
}
}  // namespace rosclaw
#endif
