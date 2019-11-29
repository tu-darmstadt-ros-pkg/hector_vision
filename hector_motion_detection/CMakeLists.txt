cmake_minimum_required(VERSION 2.8.3)
project(hector_motion_detection)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

# Load catkin and all dependencies required for this package
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  image_transport
  camera_info_manager
  hector_worldmodel_msgs
  sensor_msgs
  cv_bridge
  std_msgs
  dynamic_reconfigure
  geometry_msgs
  hector_perception_msgs
  nodelet
)

find_package(OpenCV REQUIRED)

include_directories(include ${catkin_INCLUDE_DIRS})

generate_dynamic_reconfigure_options(cfg/MotionDetection.cfg)

#common commands for building c++ executables and libraries
add_library(${PROJECT_NAME}
  include/${PROJECT_NAME}/motion_detection.h
  src/motion_detection.cpp
  src/motion_detection_nodelet.cpp)
add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBS})


add_executable(motion_detection_node src/motion_detection_node.cpp)
add_dependencies(motion_detection_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(motion_detection_node ${PROJECT_NAME})

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS
      roscpp
      rospy
      image_transport
      camera_info_manager
      hector_worldmodel_msgs
      sensor_msgs
      cv_bridge
      std_msgs
      dynamic_reconfigure
      geometry_msgs
      hector_perception_msgs
      nodelet
)

install(DIRECTORY cfg config launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
