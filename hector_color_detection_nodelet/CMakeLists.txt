cmake_minimum_required(VERSION 3.0.2)
project(hector_color_detection_nodelet)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  roscpp
  sensor_msgs
  pcl_conversions
  cv_bridge
  image_transport
  message_filters
  tf_conversions
  image_geometry
  nodelet
  dynamic_reconfigure
  hector_worldmodel_msgs
  hector_nav_msgs
)

#add dynamic reconfigure api
generate_dynamic_reconfigure_options(
  cfg/ColorDetection.cfg
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package(
#  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS pcl_ros roscpp message_filters pcl_conversions geometry_msgs nodelet visualization_msgs cv_bridge hector_worldmodel_msgs image_transport tf_conversions image_geometry
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(include ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

## Declare a cpp library
add_library(${PROJECT_NAME}
  src/hector_color_detection_nodelet.cpp
  include/${PROJECT_NAME}/hector_color_detection_nodelet.h
)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg ${catkin_EXPORTED_TARGETS})

#############
## Install ##
#############

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(FILES nodelet_plugins.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
