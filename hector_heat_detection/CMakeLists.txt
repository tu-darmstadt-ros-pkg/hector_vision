# Catkin User Guide: http://www.ros.org/doc/groovy/api/catkin/html/user_guide/user_guide.html
# Catkin CMake Standard: http://www.ros.org/doc/groovy/api/catkin/html/user_guide/standards.html
cmake_minimum_required(VERSION 2.8.3)
project(hector_heat_detection)
# Load catkin and all dependencies required for this package
find_package(catkin REQUIRED COMPONENTS roscpp rospy image_transport camera_info_manager hector_worldmodel_msgs sensor_msgs cv_bridge std_msgs image_transport dynamic_reconfigure nodelet)

find_package(OpenCV 3 REQUIRED)
#find_package(OpenCV REQUIRED)
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

generate_dynamic_reconfigure_options(cfg/HeatDetection.cfg)

add_library(heat_detection src/heat_detection.cpp)
add_dependencies(heat_detection ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(heat_detection
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(heat_detection_node src/heat_detection_node.cpp)
target_link_libraries(heat_detection_node heat_detection)

add_library(heat_detection_nodelet src/heat_detection_nodelet.cpp)
target_link_libraries(heat_detection_nodelet heat_detection)

## Generate added messages and services with any dependencies listed here
#generate_messages(
    #TODO DEPENDENCIES geometry_msgs std_msgs
#)

# catkin_package parameters: http://ros.org/doc/groovy/api/catkin/html/dev_guide/generated_cmake_api.html#catkin-package
# TODO: fill in what other packages will need to use this package
#catkin_package(
#    DEPENDS roscpp rospy image_transport camera_info_manager hector_worldmodel_msgs sensor_msgs cv_bridge std_msgs image_transport dynamic_reconfigure
#    CATKIN_DEPENDS # TODO
#    INCLUDE_DIRS # TODO include
#    LIBRARIES # TODO
#)

install(DIRECTORY cfg launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
