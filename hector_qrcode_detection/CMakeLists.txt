# Catkin User Guide: http://www.ros.org/doc/groovy/api/catkin/html/user_guide/user_guide.html
# Catkin CMake Standard: http://www.ros.org/doc/groovy/api/catkin/html/user_guide/standards.html
cmake_minimum_required(VERSION 2.8.3)
project(hector_qrcode_detection)

find_package(catkin REQUIRED COMPONENTS roscpp cv_bridge image_transport hector_worldmodel_msgs hector_perception_msgs cv_bridge tf nodelet)

include_directories(include ${catkin_INCLUDE_DIRS})

add_library(qrcode_detection src/qrcode_detection.cpp)
target_link_libraries(qrcode_detection zbar ${catkin_LIBRARIES})

add_library(qrcode_detection_nodelet src/nodelet.cpp)
target_link_libraries(qrcode_detection_nodelet qrcode_detection)

add_executable(qrcode_detection_node src/node.cpp)
target_link_libraries(qrcode_detection_node qrcode_detection)

catkin_package(
    DEPENDS
    CATKIN_DEPENDS  roscpp cv_bridge image_transport hector_worldmodel_msgs hector_perception_msgs cv_bridge tf nodelet
    INCLUDE_DIRS include
    LIBRARIES qrcode_detection qrcode_detection_nodelet
)

install(TARGETS qrcode_detection qrcode_detection_nodelet qrcode_detection_node
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(FILES nodelet.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
