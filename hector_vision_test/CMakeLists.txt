# Catkin User Guide: http://www.ros.org/doc/groovy/api/catkin/html/user_guide/user_guide.html
# Catkin CMake Standard: http://www.ros.org/doc/groovy/api/catkin/html/user_guide/standards.html
cmake_minimum_required(VERSION 3.5)
project(hector_vision_test)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(ament_index_cpp REQUIRED)

include_directories(include)

add_executable(camera_dummy_node src/camera_dummy_node.cpp)
ament_target_dependencies(camera_dummy_node
  rclcpp
  cv_bridge
  image_transport
  ament_index_cpp
)
install(TARGETS camera_dummy_node
  DESTINATION lib/${PROJECT_NAME}
)
install(
  DIRECTORY include images launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
