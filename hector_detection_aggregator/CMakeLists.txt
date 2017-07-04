cmake_minimum_required(VERSION 2.8.3)
project(hector_detection_aggregator)

if (NOT MSVC)
  # Enable the C++11 standard
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)
endif()

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy image_transport camera_info_manager hector_worldmodel_msgs hector_perception_msgs sensor_msgs cv_bridge std_msgs image_transport dynamic_reconfigure)
generate_dynamic_reconfigure_options(config/HectorDetectionAggregatorConfig.cfg)


catkin_package(
   INCLUDE_DIRS include
   LIBRARIES ${PROJECT_NAME}
   CATKIN_DEPENDS roscpp rospy image_transport camera_info_manager hector_worldmodel_msgs hector_perception_msgs sensor_msgs cv_bridge std_msgs image_transport dynamic_reconfigure
   DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

## Declare a cpp library
 add_library(${PROJECT_NAME}
   src/detection_aggregator.cpp
   src/detection_aggregator_node.cpp
   include/${PROJECT_NAME}/detection_aggregator.h
 )

add_dependencies(hector_detection_aggregator ${PROJECT_NAME}_gencfg ${catkin_EXPORTED_TARGETS})

## Declare a cpp executable
add_executable(hector_detection_aggregator_node src/detection_aggregator_node.cpp)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(hector_tutorial_node hector_tutorial_generate_messages_cpp)

## Specify libraries to link a library or executable target against
 target_link_libraries(hector_detection_aggregator_node
   hector_detection_aggregator
   ${catkin_LIBRARIES}
)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${boost_LIBRARIES})


#common commands for building c++ executables and libraries
#add_executable(detection_aggregator src/detection_aggregator.cpp)
#target_link_libraries(detection_aggregator ${catkin_LIBRARIES})

