cmake_minimum_required(VERSION 3.20)
project(gvins)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcpputils REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(gnss_comm REQUIRED)
find_package(gnss_interfaces REQUIRED)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Glog REQUIRED)
find_package(VTK REQUIRED)
find_package(Iconv)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${CERES_INCLUDE_DIRS})
# gnns_comm includes
include_directories("../gnss_comm/include")

rosidl_generate_interfaces(${PROJECT_NAME}_interface
    "msg/LocalSensorExternalTrigger.msg"
    DEPENDENCIES std_msgs)


add_library(${PROJECT_NAME}_lib
    src/estimator_node.cpp
    src/parameters.cpp
    src/estimator.cpp
    src/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projection_factor.cpp
    src/factor/projection_td_factor.cpp
    src/factor/marginalization_factor.cpp
    src/factor/gnss_psr_dopp_factor.cpp
    src/factor/gnss_dt_ddt_factor.cpp
    src/factor/gnss_dt_anchor_factor.cpp
    src/factor/gnss_ddt_smooth_factor.cpp
    src/factor/pos_vel_factor.cpp
    src/factor/pose_anchor_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    src/initial/gnss_vi_initializer.cpp
)
ament_target_dependencies(${PROJECT_NAME}_lib rclcpp rcpputils std_msgs sensor_msgs rosidl_default_generators
    geometry_msgs nav_msgs tf2 tf2_ros cv_bridge visualization_msgs gnss_comm gnss_interfaces)
rosidl_get_typesupport_target(cpp_typesupport ${PROJECT_NAME}_interface "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_lib ${OpenCV_LIBS} ${CERES_LIBRARIES} ${EIGEN3_LIBS} "${cpp_typesupport}")

add_executable(estimator_node src/estimator_node.cpp)
ament_target_dependencies(${PROJECT_NAME}_lib rclcpp rcpputils std_msgs sensor_msgs rosidl_default_generators
    geometry_msgs nav_msgs tf2 tf2_ros cv_bridge visualization_msgs gnss_comm)
rosidl_get_typesupport_target(cpp_typesupport ${PROJECT_NAME}_interface "rosidl_typesupport_cpp")
target_link_libraries(estimator_node ${PROJECT_NAME}_lib "${cpp_typesupport}")


ament_export_include_directories(include)


install(TARGETS 
    estimator_node
    DESTINATION lib/${PROJECT_NAME}
)


install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  ../config 
  DESTINATION share/${PROJECT_NAME}
)

ament_package()