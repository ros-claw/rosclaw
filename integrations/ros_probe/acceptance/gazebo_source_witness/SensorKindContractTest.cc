// Synthetic stock ECM component contracts; zero World/ROS/driver creation.
#include <cstdlib>
#include <iostream>
#include <sdf/Sensor.hh>
#include "SensorKind.hh"

template <typename Component>
bool Check(sdf::SensorType type, const char *expected) {
  gz::sim::EntityComponentManager ecm;
  const auto entity = ecm.CreateEntity();
  sdf::Sensor sensor;
  sensor.SetType(type);
  sensor.SetName("gpu_lidar_name_is_not_type_evidence");
  ecm.CreateComponent(entity, Component(sensor));
  const auto actual = rosclaw::CameraKind(entity, ecm);
  if (actual == expected) return true;
  std::cerr << "expected " << expected << " received " << actual << '\n';
  return false;
}

int main() {
  using namespace gz::sim;
  const bool typed =
    Check<components::Camera>(sdf::SensorType::CAMERA, "camera") &&
    Check<components::DepthCamera>(sdf::SensorType::DEPTH_CAMERA, "depth_camera") &&
    Check<components::RgbdCamera>(sdf::SensorType::RGBD_CAMERA, "rgbd_camera") &&
    Check<components::Camera>(sdf::SensorType::DEPTH_CAMERA, "UNKNOWN") &&
    Check<components::DepthCamera>(sdf::SensorType::RGBD_CAMERA, "UNKNOWN") &&
    Check<components::RgbdCamera>(sdf::SensorType::CAMERA, "UNKNOWN");
  EntityComponentManager ecm;
  const auto absent = ecm.CreateEntity();
  const bool missing = rosclaw::CameraKind(absent, ecm) == "UNKNOWN";
  const auto ambiguous = ecm.CreateEntity();
  sdf::Sensor color, depth;
  color.SetType(sdf::SensorType::CAMERA);
  depth.SetType(sdf::SensorType::DEPTH_CAMERA);
  ecm.CreateComponent(ambiguous, components::Camera(color));
  ecm.CreateComponent(ambiguous, components::DepthCamera(depth));
  const bool conflict = rosclaw::CameraKind(ambiguous, ecm) == "UNKNOWN";
  if (!typed || !missing || !conflict) return EXIT_FAILURE;
  std::cout << "PASS_8_SYNTHETIC_NATIVE_ECM_CAMERA_CONTRACTS_ZERO_WORLD_ROS_DRIVER\n";
  return EXIT_SUCCESS;
}
