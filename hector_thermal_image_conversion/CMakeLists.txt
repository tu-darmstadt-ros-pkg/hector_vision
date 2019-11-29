cmake_minimum_required(VERSION 2.8.3)
project(hector_thermal_image_conversion)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_geometry
  image_transport
  pcl_ros
  roscpp
  sensor_msgs
  tf
  tf_conversions
  eigen_conversions
  dynamic_reconfigure
)

find_package(OpenCV REQUIRED)

generate_dynamic_reconfigure_options(
  cfg/ThermalImageConversion.cfg
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES hotspot_proc
#  CATKIN_DEPENDS cv_bridge geometry_msgs image_geometry image_transport pcl_ros roscpp sensor_msgs tf
#  DEPENDS system_lib
)

add_library(heat_image_translator src/heat_image_translator.cpp)
#add_dependencies(heat_image_translator ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(heat_image_translator ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(heat_image_translator_node src/heat_image_translator_node.cpp)
target_link_libraries(heat_image_translator_node heat_image_translator)
add_dependencies(heat_image_translator_node ${PROJECT_NAME}_gencfg)

add_library(heat_image_translator_nodelet src/heat_image_translator_nodelet.cpp)
target_link_libraries(heat_image_translator_nodelet heat_image_translator )
add_dependencies(heat_image_translator_nodelet ${PROJECT_NAME}_gencfg)

add_library(heat_image_colorizer src/heat_image_colorizer.cpp include/hector_thermal_image_conversion/iron_bow_color_mapping.h)
#add_dependencies(heat_image_translator ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(heat_image_colorizer ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(heat_image_colorizer_node src/heat_image_colorizer_node.cpp)
target_link_libraries(heat_image_colorizer_node heat_image_colorizer)

add_library(heat_image_colorizer_nodelet src/heat_image_colorizer_nodelet.cpp)
target_link_libraries(heat_image_colorizer_nodelet heat_image_colorizer)

#add_dependencies(hotspot_detector_node ${${PROJECT_NAME}_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
install(DIRECTORY include/${PROJECT_NAME}/ DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION} FILES_MATCHING PATTERN "*.h")
install(TARGETS heat_image_translator heat_image_translator_node heat_image_translator_nodelet
                heat_image_colorizer heat_image_colorizer_node heat_image_colorizer_nodelet
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(FILES nodelet_plugins.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
