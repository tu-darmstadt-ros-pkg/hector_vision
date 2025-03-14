# Catkin User Guide: http://www.ros.org/doc/groovy/api/catkin/html/user_guide/user_guide.html
# Catkin CMake Standard: http://www.ros.org/doc/groovy/api/catkin/html/user_guide/standards.html
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

add_executable(tag_detection src/tag_detection.cpp
  include/hector_tag_detection/tag_detection.hpp)
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
  DIRECTORY include
  DESTINATION share/${PROJECT_NAME}/
)

if (BUILD_TESTING)
  add_executable(camera_dummy_node src/camera_dummy_node.cpp)
  ament_target_dependencies(camera_dummy_node
    rclcpp
    cv_bridge
    image_transport
  )
  install(TARGETS camera_dummy_node
    DESTINATION lib/${PROJECT_NAME}
  )
  install(
    DIRECTORY test
    DESTINATION share/${PROJECT_NAME}/
  )
endif ()

ament_package()
