// Optional ROS-host observer. No ROSClaw Runtime, driver or actuator interface.
// RMW MessageInfo attributes actual received messages; discovery cannot replace it.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

namespace {
using Steady = std::chrono::steady_clock;
std::string quote(const std::string &value) {
  std::string out = "\"";
  for (const unsigned char c : value) {
    if (c == '\\' || c == '"') { out += '\\'; out += c; }
    else if (c < 32 || c >= 127) { throw std::invalid_argument("bounded ASCII ROS identity required"); }
    else { out += c; }
  }
  return out + '"';
}
std::string utc() {
  const auto now = std::chrono::system_clock::now();
  const auto raw = std::chrono::system_clock::to_time_t(now);
  std::tm result{}; gmtime_r(&raw, &result);
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
  std::ostringstream out; out << std::put_time(&result, "%Y-%m-%dT%H:%M:%S") << '.' << std::setw(3) << std::setfill('0') << ms << 'Z';
  return out.str();
}
std::int64_t ns(const Steady::time_point &value) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(value.time_since_epoch()).count();
}
std::string gid(const rmw_gid_t &value) {
  if (std::all_of(std::begin(value.data), std::end(value.data), [](auto v) { return v == 0; })) { return {}; }
  std::ostringstream out; out << std::hex << std::setfill('0');
  for (const auto v : value.data) { out << std::setw(2) << static_cast<unsigned int>(v); }
  return out.str();
}
struct Edge {
  std::string topic, parent, child, authority, received_at;
  std::int32_t stamp_sec{}; std::uint32_t stamp_nanosec{};
  Steady::time_point received;
};
using Key = std::tuple<std::string, std::string, std::string, std::string>;
class Observer final : public rclcpp::Node {
 public:
  Observer() : Node("rosclaw_tf_attribution_probe") {
    window_ms_ = declare_parameter<int>("window_ms", 500);
    if (window_ms_ < 100 || window_ms_ > 5000) { throw std::invalid_argument("window_ms requires 100..5000"); }
    topics_ = declare_parameter<std::vector<std::string>>("topics", std::vector<std::string>{});
    if (topics_.size() > 64) { throw std::invalid_argument("at most 64 TF topics"); }
    for (const auto &topic : topics_) { if (topic.empty() || topic[0] != '/' || topic.size() > 512) { throw std::invalid_argument("fully qualified bounded TF topic required"); } quote(topic); }
    publisher_ = create_publisher<std_msgs::msg::String>("/rosclaw_probe/tf_attribution", rclcpp::QoS(1).transient_local());
    timer_ = create_wall_timer(std::chrono::milliseconds(100), [this]() { refresh(); });
  }
 private:
  void observe(const std::string &topic, const tf2_msgs::msg::TFMessage &msg, const rclcpp::MessageInfo &info) {
    const auto identity = gid(info.get_rmw_message_info().publisher_gid);
    const auto now = Steady::now();
    if (msg.transforms.size() > 1024) { ++invalid_; return; }
    for (const auto &tf : msg.transforms) {
      const auto &parent = tf.header.frame_id; const auto &child = tf.child_frame_id;
      try { if (parent.empty() || child.empty() || parent.size() > 512 || child.size() > 512 || tf.header.stamp.nanosec >= 1000000000) { throw std::invalid_argument("bounded TF identity and stamp required"); } quote(parent); quote(child); }
      catch (const std::invalid_argument &) { ++invalid_; continue; }
      const Key key{topic, parent, child, identity};
      if (edges_.find(key) == edges_.end() && edges_.size() >= 1024) {
        const auto oldest = std::min_element(edges_.begin(), edges_.end(), [](const auto &a, const auto &b) { return a.second.received < b.second.received; });
        edges_.erase(oldest); ++dropped_;
      }
      edges_[key] = Edge{topic, parent, child, identity, utc(), tf.header.stamp.sec, tf.header.stamp.nanosec, now};
      ++callbacks_;
    }
  }
  void refresh() {
    bool discovery_ok = true;
    try {
      for (const auto &entry : get_topic_names_and_types()) {
        if (entry.second.size() != 1 || entry.second.front() != "tf2_msgs/msg/TFMessage" || subscriptions_.count(entry.first) || (!topics_.empty() && std::find(topics_.begin(), topics_.end(), entry.first) == topics_.end())) { continue; }
        if (subscriptions_.size() >= 64 || entry.first.size() > 512) { ++dropped_; discovery_ok = false; continue; }
        quote(entry.first);
        auto callback = [this, topic=entry.first](tf2_msgs::msg::TFMessage::ConstSharedPtr message, const rclcpp::MessageInfo &info) { observe(topic, *message, info); };
        auto stream = create_subscription<tf2_msgs::msg::TFMessage>(entry.first, rclcpp::SensorDataQoS(), callback);
        auto latched = create_subscription<tf2_msgs::msg::TFMessage>(entry.first, rclcpp::QoS(100).transient_local().reliable(), callback);
        subscriptions_.emplace(entry.first, std::make_pair(stream, latched));
      }
    } catch (const std::exception &) { discovery_ok = false; ++discovery_errors_; }
    const auto now = Steady::now();
    for (auto it=edges_.begin(); it!=edges_.end();) {
      if (std::chrono::duration_cast<std::chrono::milliseconds>(now-it->second.received).count() > window_ms_) { it=edges_.erase(it); } else { ++it; }
    }
    std::ostringstream out;
    out << "{\"schema_version\":\"rosclaw.tf_attribution_window.v1\",\"captured_at\":" << quote(utc()) << ",\"received_steady_ns\":" << ns(now)
        << ",\"window_ms\":" << window_ms_ << ",\"max_edges\":1024,\"dropped_records\":" << dropped_ << ",\"invalid_records\":" << invalid_
        << ",\"discovery_errors\":" << discovery_errors_ << ",\"discovery_current\":" << (discovery_ok ? "true" : "false")
        << ",\"received_transform_callbacks\":" << callbacks_ << ",\"completeness\":\"NOT_ESTABLISHED\",\"execution_authority\":\"NONE\",\"records\":[";
    bool first = true;
    for (const auto &entry : edges_) {
      const auto &edge = entry.second; if (!first) { out << ','; } first = false;
      out << "{\"topic\":" << quote(edge.topic) << ",\"parent\":" << quote(edge.parent) << ",\"child\":" << quote(edge.child)
          << ",\"publisher_gid\":" << (edge.authority.empty() ? "null" : quote(edge.authority)) << ",\"received_at\":" << quote(edge.received_at)
          << ",\"received_steady_ns\":" << ns(edge.received) << ",\"stamp_sec\":" << edge.stamp_sec << ",\"stamp_nanosec\":" << edge.stamp_nanosec << '}';
    }
    out << "]}"; std_msgs::msg::String message; message.data = out.str(); publisher_->publish(message);
  }
  int window_ms_{}; std::vector<std::string> topics_;
  std::map<Key, Edge> edges_;
  using Subscription = rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr;
  std::map<std::string, std::pair<Subscription, Subscription>> subscriptions_;
  std::uint64_t callbacks_{}, dropped_{}, invalid_{}, discovery_errors_{};
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};
}  // namespace
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try { rclcpp::spin(std::make_shared<Observer>()); }
  catch (const std::exception &error) { std::fprintf(stderr, "TF attribution observer failed: %s\n", error.what()); rclcpp::shutdown(); return 1; }
  rclcpp::shutdown(); return 0;
}
