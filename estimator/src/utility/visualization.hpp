#pragma once

#include <fstream>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/transform_datatypes.h>
#include "CameraPoseVisualization.hpp"
#include <eigen3/Eigen/Dense>
#include <gnss_comm/gnss_ros.hpp>

#include "../estimator.hpp"
#include "../parameters.hpp"

extern rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry;
extern rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path, pub_pose;
extern rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_key_poses;
extern nav_msgs::msg::Path path;
extern int IMAGE_ROW, IMAGE_COL;
// extern ros::Publisher pub_odometry;
// extern ros::Publisher pub_path, pub_pose;
// extern ros::Publisher pub_cloud, pub_map;
// extern ros::Publisher pub_key_poses;
// extern ros::Publisher pub_ref_pose, pub_cur_pose;
// extern ros::Publisher pub_key;
// extern nav_msgs::Path path;
// extern ros::Publisher pub_pose_graph;
// extern int IMAGE_ROW, IMAGE_COL;

void registerPub(rclcpp::Node::SharedPtr n);

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, 
    const Eigen::Vector3d &V, const std_msgs::msg::Header &header);

void printStatistics(const Estimator &estimator, double t);

void pubOdometry(const Estimator &estimator, const std_msgs::msg::Header &header);

void pubGnssResult(const Estimator &estimator, const std_msgs::msg::Header &header);

void pubInitialGuess(const Estimator &estimator, const std_msgs::msg::Header &header);

void pubKeyPoses(const Estimator &estimator, const std_msgs::msg::Header &header);

void pubCameraPose(const Estimator &estimator, const std_msgs::msg::Header &header);

void pubPointCloud(const Estimator &estimator, const std_msgs::msg::Header &header);

void pubTF(const Estimator &estimator, const std_msgs::msg::Header &header);

void pubKeyframe(const Estimator &estimator);