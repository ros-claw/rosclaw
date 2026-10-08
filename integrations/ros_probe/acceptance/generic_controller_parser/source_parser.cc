// Source parser APIs only: no Context, Node, World, hardware/provider or transport.
#include <hardware_interface/component_parser.hpp>
#include <rcl_yaml_param_parser/parser.h>
#include <rcutils/allocator.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

std::string Read(const char *path)
{
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("original source missing");
  std::string value((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  if (value.empty() || value.size() > 2000000) throw std::runtime_error("bounded original source required");
  return value;
}

const rcl_variant_t & Parameter(const rcl_node_params_t & node, const std::string & name)
{
  for (size_t i = 0; i < node.num_params; ++i)
    if (name == node.parameter_names[i]) return node.parameter_values[i];
  throw std::runtime_error("declared controller parameter missing: " + name);
}

int main(int argc, char **argv)
{
  try {
    if (argc == 3 && std::string(argv[1]) == "--yaml-source-only") {
      Read(argv[2]);
      auto raw = rcl_yaml_node_struct_init(rcutils_get_default_allocator());
      if (!raw) throw std::runtime_error("source parser allocation failed");
      std::unique_ptr<rcl_params_t, decltype(&rcl_yaml_node_struct_fini)> params(raw, rcl_yaml_node_struct_fini);
      if (!rcl_parse_yaml_file(argv[2], params.get()) || params->num_nodes == 0)
        throw std::runtime_error("installed ROS parameter parser refused source");
      std::cout << "{\"status\":\"PASS_ACTUAL_ROS_PARAMETER_SOURCE_PARSER\",\"nodes_parsed\":"
        << params->num_nodes << ",\"Node_started\":false,\"hardware_loaded\":false,\"physical_acceptance\":\"NOT_RUN\"}\n";
      return 0;
    }
    if (argc != 5) throw std::runtime_error("URDF, YAML, drive FQN, expected joint CSV required");
    const auto source = Read(argv[1]);
    Read(argv[2]);
    std::set<std::string> expected;
    std::istringstream csv(argv[4]);
    std::string name;
    while (std::getline(csv, name, ','))
      if (name.empty() || !expected.insert(name).second) throw std::runtime_error("joint source aliases");
    if (expected.size() < 2 || expected.size() > 16) throw std::runtime_error("bounded source joints required");
    const auto resources = hardware_interface::parse_control_resources_from_urdf(source);
    if (resources.size() != 1 || resources[0].hardware_plugin_name != "gz_ros2_control/GazeboSimSystem")
      throw std::runtime_error("exact single declared SDK SIM source interface required");
    std::set<std::string> actual;
    for (const auto & joint : resources[0].joints) {
      if (!actual.insert(joint.name).second || joint.command_interfaces.size() != 1 ||
          joint.command_interfaces[0].name != "velocity") throw std::runtime_error("source command interface differs");
      std::set<std::string> state;
      for (const auto & value : joint.state_interfaces) state.insert(value.name);
      if (state != std::set<std::string>{"position", "velocity"}) throw std::runtime_error("source state interfaces differ");
    }
    if (actual != expected) throw std::runtime_error("actual SDK joint inventory differs from source");
    auto raw = rcl_yaml_node_struct_init(rcutils_get_default_allocator());
    if (!raw) throw std::runtime_error("source parser allocation failed");
    std::unique_ptr<rcl_params_t, decltype(&rcl_yaml_node_struct_fini)> params(raw, rcl_yaml_node_struct_fini);
    if (!rcl_parse_yaml_file(argv[2], params.get())) throw std::runtime_error("installed ROS parameter parser refused source");
    const rcl_node_params_t *drive = nullptr;
    for (size_t i = 0; i < params->num_nodes; ++i)
      if (std::string(argv[3]) == params->node_names[i]) {
        if (drive) throw std::runtime_error("parsed controller node aliases");
        drive = &params->params[i];
      }
    if (!drive) throw std::runtime_error("exact declared fully-qualified controller node lost");
    std::set<std::string> yaml_joints;
    for (const auto side : {"left_wheel_names", "right_wheel_names"}) {
      const auto &value = Parameter(*drive, side);
      if (!value.string_array_value) throw std::runtime_error("actual wheel parameter type differs");
      for (size_t i=0; i<value.string_array_value->size; ++i)
        if (!yaml_joints.insert(value.string_array_value->data[i]).second) throw std::runtime_error("wheel aliases");
    }
    if (yaml_joints != expected) throw std::runtime_error("actual YAML/URDF joint names differ");
    const auto &timeout = Parameter(*drive, "cmd_vel_timeout");
    const auto &prefix = Parameter(*drive, "tf_frame_prefix_enable");
    if (!timeout.double_value || *timeout.double_value != 0.2 || !prefix.bool_value || *prefix.bool_value)
      throw std::runtime_error("explicit deadman/frame policy differs after SDK parse");
    for (const auto key : {"wheel_radius", "wheel_separation", "linear.x.max_velocity", "angular.z.max_velocity"}) {
      const auto &value = Parameter(*drive, key);
      if (!value.double_value || !std::isfinite(*value.double_value) || *value.double_value <= 0)
        throw std::runtime_error("finite positive parsed source dimension/limit required");
    }
    std::cout << "{\"status\":\"PASS_ACTUAL_CONTROL_RESOURCE_AND_ROS_PARAMETER_SOURCE_PARSERS\","
      << "\"declared_control_joints\":" << actual.size() << ",\"nodes_parsed\":" << params->num_nodes
      << ",\"Node_started\":false,\"hardware_loaded\":false,\"physical_acceptance\":\"NOT_RUN\"}\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
