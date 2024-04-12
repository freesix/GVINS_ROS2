cmake_minimum_required(VERSION 3.20)
project(gvins_feature_tracker)

set(CMAKE_BUILD_TYPE "Debug")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(camera_models REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(VTK REQUIRED)
find_package(Ceres REQUIRED)
find_package(Iconv REQUIRED)


# include camera_models
include_directories("../camera_models/include")
include_directories("include")

add_library(feature_tracker_lib
  src/feature_tracker.cpp 
  src/parameters.cpp 
)
target_link_libraries(feature_tracker_lib
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
)
ament_target_dependencies(feature_tracker_lib rclcpp camera_models geometry_msgs std_msgs sensor_msgs cv_bridge message_filters)

add_executable(feature_tracker_node
  feature_tracker_node.cpp
)
ament_target_dependencies(feature_tracker_node rclcpp camera_models geometry_msgs std_msgs sensor_msgs cv_bridge message_filters)
target_link_libraries(feature_tracker_node feature_tracker_lib)

# install executables/libs
install(TARGETS
  feature_tracker_node
  feature_tracker_lib
  DESTINATION lib/${PROJECT_NAME}
)

#includ launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  ../config 
  DESTINATION share/${PROJECT_NAME}
)



if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
