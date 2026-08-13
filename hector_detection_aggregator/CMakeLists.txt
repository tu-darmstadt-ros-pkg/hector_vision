cmake_minimum_required(VERSION 3.8)
project(hector_detection_aggregator)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(hector_ros2_utils REQUIRED)
find_package(image_transport REQUIRED)
# find_package(camera_info_manager REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

add_library(detection_aggregator_component SHARED src/detection_aggregator.cpp)

target_include_directories(
  detection_aggregator_component
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(detection_aggregator_component PUBLIC
  OpenCV
  rclcpp
  rclcpp_components
  hector_ros2_utils
  cv_bridge
  # camera_info_manager
  image_transport
  std_msgs
  sensor_msgs
  vision_msgs
)

rclcpp_components_register_node(
  detection_aggregator_component PLUGIN
  "hector_detection_aggregator::DetectionAggregator" EXECUTABLE
  detection_aggregator
)
ament_export_targets(export_detection_aggregator_component)
install(
  TARGETS detection_aggregator_component
  EXPORT export_detection_aggregator_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY config include launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
