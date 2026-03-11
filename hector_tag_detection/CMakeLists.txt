cmake_minimum_required(VERSION 3.5)
project(hector_tag_detection)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(apriltag REQUIRED)
find_package(vision_msgs REQUIRED)

include_directories(include)

<<<<<<< Updated upstream
add_executable(tag_detection src/tag_detection.cpp)
=======
add_executable(tag_detection src/tag_detection.cpp
  include/hector_tag_detection/tag_detection.hpp
)

>>>>>>> Stashed changes
ament_target_dependencies(tag_detection
  rclcpp
  tf2
  tf2_ros
  cv_bridge
  image_transport
  apriltag
  vision_msgs
)
target_link_libraries(tag_detection zbar apriltag)

install(TARGETS tag_detection
  DESTINATION lib/${PROJECT_NAME}
)
install(
  DIRECTORY config include
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
