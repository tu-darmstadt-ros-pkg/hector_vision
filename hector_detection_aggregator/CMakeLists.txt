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
find_package(cv_bridge REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(hector_perception_msgs REQUIRED)
find_package(generate_parameter_library REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})

generate_parameter_library(hector_detection_aggregator_params
                           parameters/detection_aggregator_params.yaml)

add_library(detection_aggregator SHARED src/detection_aggregator.cpp)

target_include_directories(
  detection_aggregator
  PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
          $<INSTALL_INTERFACE:include/hector_detection_aggregator>)

target_link_libraries(detection_aggregator PUBLIC  cv_bridge::cv_bridge ${OpenCV_LIBS} time_sync
                      hector_detection_aggregator_params)

add_library(time_sync SHARED src/time_sync.cpp)
target_link_libraries(time_sync PUBLIC hector_detection_aggregator_params)

add_executable(detection_aggregator_node src/detection_aggregator_node.cpp)
target_link_libraries(detection_aggregator_node detection_aggregator)


ament_target_dependencies(detection_aggregator PUBLIC
  OpenCV
  rclcpp
  rclcpp_components
  hector_ros2_utils
  cv_bridge
  image_transport
  std_msgs
  sensor_msgs
  hector_perception_msgs
)

ament_target_dependencies(time_sync PUBLIC 
hector_ros2_utils
image_transport
sensor_msgs
hector_perception_msgs
)

install(
  TARGETS detection_aggregator_node detection_aggregator time_sync
  EXPORT export_hector_detection_aggregator
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config include launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
