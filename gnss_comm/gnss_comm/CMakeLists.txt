cmake_minimum_required(VERSION 3.5)
project(gnss_comm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(rclpy REQUIRED)
find_package(gnss_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_cmake REQUIRED)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3 REQUIRED)
find_package(Glog REQUIRED)

set(INCLUDE_DIRS include ${rclpy_INCLUDE_DIRS} ${gnss_interfaces_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS} ${ament_cmake_INCLUDE_DIRS})
include_directories(${INCLUDE_DIRS})

set(LIBRARY_DIRS ${rclpy_LIBRARY_DIRS} ${gnss_interfaces_LIBRARY_DIRS}
  ${rclcpp_LIBRARY_DIRS} ${ament_cmake_LIBRARY_DIRS})

link_directories(${LIBRARY_DIRS})

set(LIBS ${rclpy_LIBRARIES} ${gnss_interfaces_LIBRARIES} ${rclcpp_LIBRARIES}
  ${ament_cmake_LIBRARIES} ${GLOG_LIBRARIES})

file(GLOB_RECURSE source_files ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_library(${PROJECT_NAME} ${source_files})
target_include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIR} ${GLOG_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} ${LIBS})

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
  FILES_MATCHING
  PATTERN "*.hpp")

install(TARGETS ${PROJECT_NAME}
  ARCHIVE
  DESTINATION lib
  LIBRARY
  DESTINATION lib
  RUNTIME
  DESTINATION lib/${PROJECT_NAME})

ament_export_dependencies(rclpy)
ament_export_dependencies(gnss_interfaces)
ament_export_dependencies(rclcpp)
ament_export_dependencies(ament_cmake)
ament_export_include_directories(${INCLUDE_DIRS})
ament_export_libraries(${PROJECT_NAME} ${LIBS})

ament_package()
