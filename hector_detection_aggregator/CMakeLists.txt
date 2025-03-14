cmake_minimum_required(VERSION 3.5)
project(hector_detection_aggregator)

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(image_transport REQUIRED)
find_package(camera_info_manager REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

add_executable(detection_aggregator src/detection_aggregator.cpp
  include/hector_detection_aggregator/detection_aggregator.hpp)
ament_target_dependencies(detection_aggregator PUBLIC
  OpenCV
  rclcpp
  cv_bridge
  camera_info_manager
  image_transport
  std_msgs
  sensor_msgs
  vision_msgs
)

install(TARGETS detection_aggregator
  DESTINATION lib/${PROJECT_NAME}
)
install(
  DIRECTORY launch include
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
