cmake_minimum_required(VERSION 3.20)
project(gvins_part)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Debug")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Iconv REQUIRED)
find_package(VTK REQUIRED)

add_subdirectory(camera_models)
add_subdirectory(gnss_comm)

include_directories("include")
include_directories("gnss_comm/include")
include_directories("camera_models/include")

file(GLOB_RECURSE source_files ${PROJECT_SOURCE_DIR}/src/*.cpp
    src/interface/imu.cpp)

add_library(gvins_lib ${source_files})
target_link_libraries(gvins_lib camera_models gnss_comm_lib ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(feature_tracker main.cpp)
target_link_libraries(feature_tracker gvins_lib)