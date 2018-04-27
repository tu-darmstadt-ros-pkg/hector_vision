cmake_minimum_required(VERSION 2.8.3)
project(hector_qrcode_detection)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  image_transport
  hector_worldmodel_msgs
  hector_perception_msgs
  cv_bridge
  tf
  nodelet)

catkin_package(
    CATKIN_DEPENDS  roscpp cv_bridge image_transport hector_worldmodel_msgs hector_perception_msgs cv_bridge tf nodelet
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
)

include_directories(include ${catkin_INCLUDE_DIRS})

add_library(${PROJECT_NAME}
  include/${PROJECT_NAME}/qrcode_detection.h
  src/qrcode_detection.cpp
  src/nodelet.cpp
)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} zbar)

add_executable(${PROJECT_NAME}_node src/node.cpp)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME})


