// Actual installed parser APIs only. No Node, Hardware, World or transport.
#include <hardware_interface/component_parser.hpp>
#include <ros_gz_bridge/bridge_config.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>

std::string Read(const char *path)
{
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("original parser input missing");
  std::string raw((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (raw.empty() || raw.size() > 2000000) throw std::runtime_error("bounded parser input required");
  return raw;
}

auto Key(const ros_gz_bridge::BridgeConfig &row)
{
  const auto p = row.PublisherQoS().get_rmw_qos_profile();
  const auto s = row.SubscriberQoS().get_rmw_qos_profile();
  return std::make_tuple(row.ros_topic_name, row.gz_topic_name, row.ros_type_name,
      row.gz_type_name, static_cast<int>(row.direction), row.frame_id, row.is_lazy,
      row.publisher_queue_size, row.subscriber_queue_size, p.depth, p.reliability,
      p.durability, s.depth, s.reliability, s.durability);
}

int main(int argc, char **argv)
{
  try
  {
    if (argc != 6) throw std::runtime_error("five explicit source inputs required");
    const auto hardware = hardware_interface::parse_control_resources_from_urdf(Read(argv[1]));
    if (hardware.size() != 1 || hardware[0].joints.size() != 2)
      throw std::runtime_error("known declared differential drive control source required");
    std::set<std::string> joints;
    for (const auto &joint : hardware[0].joints)
    {
      joints.insert(joint.name);
      if (joint.command_interfaces.size() != 1 || joint.command_interfaces[0].name != "velocity")
        throw std::runtime_error("actual velocity command interface source changed");
      std::set<std::string> states;
      for (const auto &state : joint.state_interfaces) states.insert(state.name);
      if (states != std::set<std::string>{"position", "velocity"})
        throw std::runtime_error("actual position/velocity state interfaces source changed");
    }
    if (joints.size() != 2) throw std::runtime_error("control source joint names alias");
    auto original = ros_gz_bridge::readFromYamlString(Read(argv[2]));
    const auto truth = ros_gz_bridge::readFromYamlString(Read(argv[3]));
    original.insert(original.end(), truth.begin(), truth.end());
    const auto robot = ros_gz_bridge::readFromYamlString(Read(argv[4]));
    const auto world = ros_gz_bridge::readFromYamlString(Read(argv[5]));
    if (original.empty() || truth.size() != 1 || robot.size() != original.size() + 1 ||
        world.size() != robot.size() + 3)
      throw std::runtime_error("installed bridge parser lost a declared source role");
    for (const auto &row : original)
    {
      if (std::count_if(robot.begin(), robot.end(), [&](const auto &r) {return Key(r) == Key(row);}) != 1)
        throw std::runtime_error("installed bridge parser correspondence differs after normalization");
    }
    std::set<std::string> ros, gz;
    for (const auto &row : world)
    {
      if (row.direction != ros_gz_bridge::BridgeDirection::GZ_TO_ROS ||
          !row.service_name.empty() || !ros.insert(row.ros_topic_name).second ||
          !gz.insert(row.gz_topic_name).second)
        throw std::runtime_error("assembled bridge has control, service or duplicate source roles");
    }
    std::cout << "{\"status\":\"PASS_INSTALLED_CONTROL_AND_BRIDGE_SOURCE_PARSERS\","
      << "\"declared_control_joints\":" << joints.size()
      << ",\"original_bridge_roles\":" << original.size()
      << ",\"assembled_bridge_roles\":" << world.size()
      << ",\"node_or_world_started\":false,\"hardware_or_plugin_loaded\":false,"
      << "\"physical_acceptance\":\"NOT_RUN\"}\n";
    return 0;
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
