cmake_minimum_required(VERSION 3.8)
project(hector_tag_detection)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(hector_ros2_utils REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(apriltag REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)

include_directories(include)

add_library(tag_detection_component SHARED src/tag_detection.cpp)

target_include_directories(
  tag_detection_component
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(tag_detection_component PUBLIC ${hector_ros2_utils_INCLUDE_DIRS})
target_link_libraries(tag_detection_component PUBLIC
  ${hector_ros2_utils_TARGETS}
  ${hector_ros2_utils_LIBRARIES}
  ${sensor_msgs_TARGETS}
  ${std_msgs_TARGETS}
  ${vision_msgs_TARGETS}
  apriltag::apriltag
  cv_bridge::cv_bridge
  image_transport::image_transport
  rclcpp::rclcpp
  rclcpp_components::component
  zbar
)

rclcpp_components_register_node(
  tag_detection_component PLUGIN
  "hector_tag_detection::TagDetection" EXECUTABLE
  tag_detection
)
ament_export_targets(export_tag_detection_component)
install(
  TARGETS tag_detection_component
  EXPORT export_tag_detection_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY config include
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
