cmake_minimum_required(VERSION 3.16)
project(rosclaw_gazebo_source_witness LANGUAGES CXX)
# ROS vendor prefixes can also export OR-Tools protobuf; Gazebo 8's installed
# message headers require the system Protobuf module rather than that config.
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG OFF)
find_package(Protobuf REQUIRED MODULE)
find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
add_library(rosclaw_source_witness SHARED SourceWitness.cc)
target_compile_features(rosclaw_source_witness PRIVATE cxx_std_17)
target_compile_options(rosclaw_source_witness PRIVATE -Wall -Wextra -Werror)
target_link_libraries(rosclaw_source_witness PRIVATE gz-sim8::gz-sim8 gz-plugin2::register OpenSSL::Crypto)
# Separate disposable fault plugin: never loaded by production/read-only World.
add_library(rosclaw_source_inventory_fault SHARED InventoryFault.cc)
target_compile_features(rosclaw_source_inventory_fault PRIVATE cxx_std_17)
target_compile_options(rosclaw_source_inventory_fault PRIVATE -Wall -Wextra -Werror)
target_link_libraries(rosclaw_source_inventory_fault PRIVATE gz-sim8::gz-sim8 gz-plugin2::register)

# Opt-in SDK component tests allocate ECM objects only, never a server or World.
option(ROSCLAW_SOURCE_WITNESS_TESTS "Build native metadata contract tests" OFF)
if(ROSCLAW_SOURCE_WITNESS_TESTS)
  enable_testing()
  add_executable(rosclaw_sensor_kind_contracts SensorKindContractTest.cc)
  target_compile_features(rosclaw_sensor_kind_contracts PRIVATE cxx_std_17)
  target_compile_options(rosclaw_sensor_kind_contracts PRIVATE -Wall -Wextra -Werror)
  target_link_libraries(rosclaw_sensor_kind_contracts PRIVATE gz-sim8::gz-sim8)
  add_test(NAME native_sensor_kind_contracts COMMAND rosclaw_sensor_kind_contracts)
endif()
