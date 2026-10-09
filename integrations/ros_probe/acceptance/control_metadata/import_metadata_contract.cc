// Isolated SDK mock import regression. No lifecycle activation or robot I/O.
#include <iostream>
#include <memory>
#include <string>
#include "hardware_interface/component_parser.hpp"
#include "hardware_interface/resource_manager.hpp"
#include "mock_components/generic_system.hpp"
#include "rclcpp/clock.hpp"
#include "rclcpp/logging.hpp"

int main(int argc, char ** argv)
{
  const std::string urdf = R"(
<robot name="metadata_sdk_mock_only">
  <link name="mock_base"/><link name="mock_rotor"/>
  <joint name="mock_joint" type="continuous">
    <parent link="mock_base"/><child link="mock_rotor"/><axis xyz="0 1 0"/>
  </joint>
  <ros2_control name="ImportedMock" type="system">
    <hardware><plugin>mock_components/GenericSystem</plugin></hardware>
    <joint name="mock_joint">
      <command_interface name="velocity"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
  </ros2_control>
</robot>)";
  auto infos = hardware_interface::parse_control_resources_from_urdf(urdf);
  auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
  hardware_interface::ResourceManager resources(clock, rclcpp::get_logger("metadata_sdk_mock"));
  resources.import_component(std::make_unique<mock_components::GenericSystem>(), infos.at(0));
  const auto & row = resources.get_components_status().at("ImportedMock");
  const bool complete = row.name == "ImportedMock" && row.type == "system" &&
    row.plugin_name == "mock_components/GenericSystem";
  std::cout << "{\"name\":\"" << row.name << "\",\"type\":\"" << row.type
            << "\",\"plugin_name\":\"" << row.plugin_name << "\",\"complete\":"
            << (complete ? "true" : "false") << "}" << std::endl;
  if (argc != 2) {return 2;}
  const std::string expected = argv[1];
  if (expected == "complete") {return complete ? 0 : 1;}
  if (expected == "missing") {return !complete && row.type.empty() && row.plugin_name.empty() ? 0 : 1;}
  return 2;
}
