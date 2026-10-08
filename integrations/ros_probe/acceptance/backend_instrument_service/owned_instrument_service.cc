// Owned SIM instrument only. Source validation mode constructs no Node/World.
#include <gz/transport/Node.hh>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/boolean.pb.h>
#include <google/protobuf/text_format.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

std::string Hex(const std::string &raw)
{
  const char *digits = "0123456789abcdef";
  std::string result;
  for (const unsigned char c : raw)
  {
    result += digits[c >> 4]; result += digits[c & 15];
  }
  return result;
}

std::string Unhex(const std::string &raw)
{
  if (raw.empty() || raw.size() > 8192 || raw.size() % 2 != 0 ||
      !std::regex_match(raw, std::regex("[0-9a-f]+")))
    throw std::runtime_error("bounded exact original protobuf hex required");
  std::string result;
  for (std::size_t i = 0; i < raw.size(); i += 2)
    result += static_cast<char>(std::stoul(raw.substr(i, 2), nullptr, 16));
  return result;
}

double Number(const char *input)
{
  std::size_t end;
  const std::string source(input);
  const double value = std::stod(source, &end);
  if (end != source.size() || !std::isfinite(value) || std::abs(value) > 100)
    throw std::runtime_error("finite frozen instrument target required");
  return value;
}

int main(int argc, char **argv)
{
  try
  {
    if (argc != 9) throw std::runtime_error("eight explicit frozen instrument arguments required");
    const std::string world(argv[1]), instrument(argv[2]), robot(argv[3]);
    const std::regex name("[A-Za-z_][A-Za-z0-9_]{0,63}");
    if (!std::regex_match(world, name) || !std::regex_match(instrument, name) ||
        !std::regex_match(robot, name) || instrument == robot)
      throw std::runtime_error("disjoint exact owned world/instrument/robot roles required");
    const double x = Number(argv[4]), y = Number(argv[5]), z = Number(argv[6]);
    if (z < 2 || z > 20) throw std::runtime_error("bounded declared instrument lift height required");
    const std::string partition(argv[7]), mode(argv[8]);
    if (!std::regex_match(partition, std::regex("rosclaw_backend_[a-f0-9]{32}")) ||
        (mode != "--source-validate-only" && mode != "--owned-runtime"))
      throw std::runtime_error("explicit owned GZ partition and execution mode required");
    const bool runtime = mode == "--owned-runtime";
    std::unique_ptr<gz::transport::Node> node;
    const std::string service = "/world/" + world + "/set_pose";
    if (runtime)
    {
      const char *actual = std::getenv("GZ_PARTITION");
      if (!actual || partition != actual)
        throw std::runtime_error("actual instrument process partition differs");
      node = std::make_unique<gz::transport::Node>();
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (true)
      {
        std::vector<std::string> services;
        node->ServiceList(services);
        bool found = false;
        for (const auto &item : services) if (item == service) found = true;
        if (found) break;
        if (std::chrono::steady_clock::now() >= deadline)
          throw std::runtime_error("owned instrument scene service discovery deadline expired");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    std::cout << "{\"status\":\"READY_INSTRUMENT_SOURCE_PROCESS\",\"transport_node_started\":"
      << (runtime ? "true" : "false") << ",\"authorization\":false}\n" << std::flush;
    std::size_t count = 0;
    while (true)
    {
      std::string line;
      char c;
      while (std::cin.get(c) && c != '\n')
      {
        if (line.size() >= 16385) throw std::runtime_error("instrument input line exceeds bound");
        line += c;
      }
      if (line.empty() && std::cin.eof()) break;
      if (std::cin.eof() || ++count > 500)
        throw std::runtime_error("complete bounded instrument transaction line required");
      const auto split = line.find(' ');
      const std::string requestRaw = Unhex(line.substr(0, split));
      const std::string responseHex = split == std::string::npos ? "-" : line.substr(split + 1);
      gz::msgs::Pose request;
      if (!google::protobuf::TextFormat::ParseFromString(requestRaw, &request) ||
          request.name() != instrument || request.id() != 0 ||
          request.position().x() != x || request.position().y() != y || request.position().z() != z ||
          request.orientation().w() != 1 || request.orientation().x() != 0 ||
          request.orientation().y() != 0 || request.orientation().z() != 0 || request.has_header())
        throw std::runtime_error("original request differs from frozen instrument-only target");
      gz::msgs::Boolean response;
      const std::string requestWire = request.SerializeAsString();
      std::string responseWire;
      bool returned = false, result = false;
      if (runtime)
      {
        if (responseHex != "-") throw std::runtime_error("runtime cannot inject a service reply");
        returned = node->RequestRaw(service, requestWire, request.GetTypeName(),
            response.GetTypeName(), 100u, responseWire, result);
        if (responseWire.size() > 4096 || !response.ParseFromString(responseWire))
          throw std::runtime_error("original owned instrument service reply malformed or oversized");
      }
      else if (responseHex != "-")
      {
        responseWire = Unhex(responseHex);
        if (!response.ParseFromString(responseWire))
          throw std::runtime_error("original bounded SDK Boolean wire response malformed");
      }
      const auto completed = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count();
      std::cout << "{\"request_text_hex\":\"" << Hex(requestRaw)
        << "\",\"request_protobuf_hex\":\"" << Hex(requestWire)
        << "\",\"response_protobuf_hex\":\"" << Hex(responseWire)
        << "\",\"response_text_hex\":\"" << Hex(response.DebugString())
        << "\",\"transport_returned\":" << (returned ? "true" : "false")
        << ",\"service_result\":" << (result ? "true" : "false")
        << ",\"response_data\":" << (response.data() ? "true" : "false")
        << ",\"acknowledged_at_unix_ns\":" << completed
        << ",\"service_executed\":" << (runtime ? "true" : "false")
        << ",\"authorization\":false}\n" << std::flush;
    }
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
