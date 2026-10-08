// Decode original map files using installed Nav2 Map IO; no rclcpp init/Node.
#include <nav2_map_server/map_io.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::runtime_error("one original map YAML path required");
    nav_msgs::msg::OccupancyGrid map;
    const auto status = nav2_map_server::loadMapFromYaml(argv[1], map);
    if (status != nav2_map_server::LOAD_MAP_SUCCESS)
      throw std::runtime_error("installed Nav2 source map decoder rejected input");
    const uint64_t count = uint64_t(map.info.width) * map.info.height;
    if (!count || count > 1048576 || map.data.size() != count ||
        !std::isfinite(map.info.resolution) || map.info.resolution <= 0)
      throw std::runtime_error("bounded complete SDK-decoded map required");
    const auto &p = map.info.origin.position;
    const auto &q = map.info.origin.orientation;
    std::cout << std::setprecision(17)
      << "{\"Node_started\":false,\"physical_acceptance\":\"NOT_RUN\","
      << "\"width\":" << map.info.width << ",\"height\":" << map.info.height
      << ",\"resolution_float32\":" << map.info.resolution
      << ",\"origin\":[" << p.x << ',' << p.y << ',' << p.z << ','
      << q.x << ',' << q.y << ',' << q.z << ',' << q.w << "],\"data\":[";
    for (size_t i = 0; i < map.data.size(); ++i) {
      if (map.data[i] < -1 || map.data[i] > 100)
        throw std::runtime_error("invalid SDK occupancy value");
      if (i) std::cout << ',';
      std::cout << int(map.data[i]);
    }
    std::cout << "]}" << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
