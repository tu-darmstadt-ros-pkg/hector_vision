cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE Debug)

rosbuild_init()

find_package(cmake_modules)

find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)
find_package(PCL REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS} ${Eigen_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

## Specify additional locations of header files
set(HEADERS
  include/${PROJECT_NAME}/soft_obstacle_detection.h
)

set(SOURCES
  src/soft_obstacle_detection.cpp
)

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

#common commands for building c++ executables and libraries
rosbuild_add_library(${PROJECT_NAME}  ${SOURCES} ${HEADERS})
#target_link_libraries(${PROJECT_NAME} another_library)

#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
#rosbuild_add_executable(example examples/example.cpp)
#target_link_libraries(example ${PROJECT_NAME})

#add dynamic reconfigure api
rosbuild_find_ros_package(dynamic_reconfigure)
include(${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake)
gencfg()

rosbuild_add_executable(soft_obstacle_detection src/soft_obstacle_detection.cpp)

target_link_libraries(soft_obstacle_detection ${PROJECT_NAME} ${OpenCV_LIBRARIES} ${Eigen_LIBRARIES} ${PCL_LIBRARIES})
