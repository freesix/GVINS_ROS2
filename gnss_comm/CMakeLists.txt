cmake_minimum_required(VERSION 3.5)
project(gnss_comm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3 REQUIRED)
find_package(Glog REQUIRED)

include_directories("include")


file(GLOB_RECURSE source_files ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_library(${PROJECT_NAME}_lib ${source_files})
target_include_directories(${PROJECT_NAME}_lib PUBLIC ${EIGEN3_INCLUDE_DIR} ${GLOG_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME}_lib ${GLOG_LIBRARIES})

