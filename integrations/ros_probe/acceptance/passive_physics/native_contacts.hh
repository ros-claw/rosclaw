// Read-only contact data and topic binding for the qualified Gazebo 8.15.0 SDK.
// No component creation, sensor reconfiguration, service, or actuator operation.
#pragma once
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/config.hh>
#include <gz/transport/TopicUtils.hh>
#include <gz/sim/components/Collision.hh>
#include <gz/sim/components/ContactSensor.hh>
#include <gz/sim/components/ContactSensorData.hh>
#include <gz/sim/components/Link.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/Sensor.hh>
#include <gz/sim/components/World.hh>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rosclaw::native_contacts
{
inline void QualifiedVersion()
{
  if (std::string(GZ_SIM_VERSION_FULL) != "8.15.0")
    throw std::runtime_error("contact component semantics require qualified Gazebo 8.15.0");
}
inline std::string Quote(const std::string &value)
{
  if (value.size() > 1024) throw std::runtime_error("bounded native contact source text required");
  std::ostringstream out; out << '"';
  for (unsigned char ch : value)
  {
    if (ch == '"' || ch == '\\') out << '\\' << ch;
    else if (ch < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                         << static_cast<unsigned>(ch) << std::dec;
    else out << ch;
  }
  out << '"'; return out.str();
}
inline std::string CollisionName(gz::sim::Entity collision,
    gz::sim::Entity world, const gz::sim::EntityComponentManager &ecm)
{
  using namespace gz::sim::components;
  const auto name = ecm.Component<Name>(collision);
  const auto parent = ecm.Component<ParentEntity>(collision);
  if (!ecm.Component<Collision>(collision) || !name || !parent)
    throw std::runtime_error("actual contact collision identity missing");
  const auto link = parent->Data();
  const auto linkName = ecm.Component<Name>(link);
  const auto linkParent = ecm.Component<ParentEntity>(link);
  if (!ecm.Component<Link>(link) || !linkName || !linkParent)
    throw std::runtime_error("actual contact collision link missing");
  const auto model = linkParent->Data();
  const auto modelName = ecm.Component<Name>(model);
  const auto modelParent = ecm.Component<ParentEntity>(model);
  if (!ecm.Component<Model>(model) || !modelName || !modelParent || modelParent->Data() != world)
    throw std::runtime_error("contact collision outside declared direct world model");
  for (const auto &value : {name->Data(), linkName->Data(), modelName->Data()})
    if (value.empty() || value.size() > 256 || value.find("::") != std::string::npos)
      throw std::runtime_error("actual contact collision scope ambiguous");
  return modelName->Data() + "::" + linkName->Data() + "::" + name->Data();
}
inline std::string Topic(gz::sim::Entity entity,
    const gz::sim::components::ContactSensor &sensor,
    const gz::sim::EntityComponentManager &ecm)
{
  QualifiedVersion();
  const auto element = sensor.Data();
  if (!element || !element->HasElement("contact"))
    throw std::runtime_error("actual native contact sensor configuration missing");
  const auto contact = element->GetElement("contact");
  // Contact.cc reads the nested contact/topic, not the generic sensor/topic.
  const auto requested = contact->Get<std::string>("topic", "__default_topic__").first;
  const auto raw = requested == "__default_topic__" ?
      gz::sim::scopedName(entity, ecm, "/") + "/contact" : requested;
  // Use the transport SDK qualification of a native Node with its default
  // empty namespace, including the leading slash of the scoped default.
  std::string qualified, partition, resolved;
  if (!gz::transport::TopicUtils::FullyQualifiedName("", "", raw, qualified) ||
      !gz::transport::TopicUtils::DecomposeFullyQualifiedTopic(qualified, partition, resolved))
    throw std::runtime_error("invalid native transport contact topic");
  if (resolved.empty() || resolved.size() > 1024 || resolved[0] != '/')
    throw std::runtime_error("bounded actual native contact topic required");
  // Contact.cc does not create SensorTopic. If supplied by another system,
  // its value must still agree with the exact qualified native resolver.
  const auto declared = ecm.Component<gz::sim::components::SensorTopic>(entity);
  if (declared && declared->Data() != resolved)
    throw std::runtime_error("actual native contact topic components disagree");
  return resolved;
}
inline std::string PublisherTopic(const std::string &requested)
{
  // Two simulator-owned observation streams only; no actuator endpoint.
  if (requested != "/rosclaw_sim/contact_components" &&
      requested != "/rosclaw_sim/backend_probe_components")
    throw std::runtime_error("declared simulator-owned contact observation topic required");
  return requested;
}
struct Sources
{
  std::string inventory;
  std::string observations;
};
inline Sources Read(gz::sim::Entity body, gz::sim::Entity world,
    const gz::sim::EntityComponentManager &ecm)
{
  QualifiedVersion();
  using namespace gz::sim::components;
  if (!ecm.Component<Model>(body) || !ecm.Component<World>(world))
    throw std::runtime_error("actual native contact Body/world required");
  std::set<gz::sim::Entity> collisions, covered;
  std::map<gz::sim::Entity, std::string> inventory, observations;
  std::set<std::string> topics;
  ecm.Each<Collision,ParentEntity>([&](auto entity, const auto *, const auto *parent)
  {
    const auto linkParent = ecm.Component<ParentEntity>(parent->Data());
    if (linkParent && linkParent->Data() == body) collisions.insert(entity);
    return true;
  });
  if (collisions.empty() || collisions.size() > 256)
    throw std::runtime_error("bounded actual Body contact collision set required");
  std::size_t totalContacts = 0;
  ecm.Each<ContactSensor>([&](auto entity, const auto *sensor)
  {
    const auto parent = ecm.Component<ParentEntity>(entity);
    if (!parent) throw std::runtime_error("native contact sensor parent missing");
    const auto link = parent->Data();
    const auto linkParent = ecm.Component<ParentEntity>(link);
    if (!linkParent || linkParent->Data() != body) return true;
    const auto name = ecm.Component<Name>(entity);
    const auto linkName = ecm.Component<Name>(link);
    if (!ecm.Component<Sensor>(entity) || !ecm.Component<Link>(link) || !name || !linkName ||
        name->Data().empty() || linkName->Data().empty())
      throw std::runtime_error("actual native contact sensor/link identity missing");
    const auto topic = Topic(entity, *sensor, ecm);
    if (!topics.insert(topic).second)
      throw std::runtime_error("duplicate actual native contact sensor topic");
    const auto contact = sensor->Data()->GetElement("contact");
    if (!contact->HasElement("collision"))
      throw std::runtime_error("actual native contact collision reference missing");
    std::vector<gz::sim::Entity> references;
    for (auto element = contact->GetElement("collision"); element;
         element = element->GetNextElement("collision"))
    {
      const auto wanted = element->template Get<std::string>();
      gz::sim::Entity matched = gz::sim::kNullEntity;
      ecm.Each<Collision,Name,ParentEntity>([&](auto candidate, const auto *,
          const auto *candidateName, const auto *candidateParent)
      {
        if (candidateParent->Data() != link || candidateName->Data() != wanted) return true;
        if (matched != gz::sim::kNullEntity)
          throw std::runtime_error("ambiguous actual native contact collision name");
        matched = candidate; return true;
      });
      if (matched == gz::sim::kNullEntity || !collisions.count(matched) || !covered.insert(matched).second)
        throw std::runtime_error("missing or duplicate native contact collision coverage");
      references.push_back(matched);
      const auto data = ecm.Component<ContactSensorData>(matched);
      if (!data) throw std::runtime_error("actual initialized physics ContactSensorData missing");
      std::ostringstream samples; samples << '[';
      bool first = true;
      for (const auto &pair : data->Data().contact())
      {
        if (++totalContacts > 4096)
          throw std::runtime_error("actual native contact pair count unbounded");
        const auto a = pair.collision1().id(), b = pair.collision2().id();
        if ((a != matched && b != matched) || a == b)
          throw std::runtime_error("native contact data does not belong to mapped collision");
        const auto aName = CollisionName(a, world, ecm), bName = CollisionName(b, world, ecm);
        if (!first) samples << ',';
        first = false;
        samples << "{\"collision1_entity_id\":" << a << ",\"collision2_entity_id\":" << b
            << ",\"collision1_name\":" << Quote(aName) << ",\"collision2_name\":" << Quote(bName) << '}';
      }
      samples << ']';
      std::ostringstream row;
      row << "{\"collision_entity_id\":" << matched << ",\"collision_name\":"
          << Quote(CollisionName(matched, world, ecm)) << ",\"contacts\":" << samples.str() << '}';
      observations.emplace(matched, row.str());
    }
    if (references.empty() || references.size() > 256)
      throw std::runtime_error("bounded native sensor collision references required");
    std::ostringstream row;
    row << "{\"sensor_entity_id\":" << entity << ",\"sensor_name\":" << Quote(name->Data())
        << ",\"link_entity_id\":" << link << ",\"link_name\":" << Quote(linkName->Data())
        << ",\"gz_topic\":" << Quote(topic) << ",\"collision_entity_ids\":[";
    for (std::size_t i = 0; i < references.size(); ++i) { if (i) row << ','; row << references[i]; }
    row << "]}"; inventory.emplace(entity, row.str());
    return true;
  });
  if (covered != collisions || observations.size() != collisions.size())
    throw std::runtime_error("native contact sensors do not cover every Body collision");
  auto array = [](const auto &rows)
  {
    std::ostringstream out; out << '['; bool first = true;
    for (const auto &row : rows) { if (!first) out << ','; first = false; out << row.second; }
    out << ']'; return out.str();
  };
  return {array(inventory), array(observations)};
}
}  // namespace rosclaw::native_contacts
