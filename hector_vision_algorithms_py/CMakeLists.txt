cmake_minimum_required(VERSION 2.8.3)
project(hector_vision_algorithms_py)

find_package(catkin_simple REQUIRED COMPONENTS hector_vision_algorithms)

catkin_simple()

# Set up the python exports.
SET(PY_PROJECT_NAME ${PROJECT_NAME})
SET(PY_PACKAGE_DIR python/hector_vision)

add_python_export_library(${PY_PROJECT_NAME} ${PY_PACKAGE_DIR}
                          src/module.cpp np_opencv_converter/np_opencv_converter.cpp np_opencv_converter/utils/conversion.cpp
                          )

cs_install()
cs_export()