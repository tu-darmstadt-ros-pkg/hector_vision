cmake_minimum_required(VERSION 3.0.2)
project(hector_detection_aggregator)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  image_transport
  camera_info_manager
  hector_worldmodel_msgs
  hector_perception_msgs
  sensor_msgs
  cv_bridge
  std_msgs
  dynamic_reconfigure
  nodelet
)
generate_dynamic_reconfigure_options(config/HectorDetectionAggregator.cfg)

catkin_package(
   INCLUDE_DIRS include
   LIBRARIES ${PROJECT_NAME}
   CATKIN_DEPENDS
     roscpp
     image_transport
     camera_info_manager
     hector_worldmodel_msgs
     hector_perception_msgs
     sensor_msgs
     cv_bridge
     std_msgs
     dynamic_reconfigure
#   DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

## Declare a cpp library
 add_library(${PROJECT_NAME}
   src/detection_aggregator.cpp
   src/detection_aggregator_nodelet.cpp
   include/${PROJECT_NAME}/detection_aggregator.h
 )
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})


## Declare a cpp executable
add_executable(${PROJECT_NAME}_node src/detection_aggregator_node.cpp)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME})


install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY config launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(FILES nodelet_plugins.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
