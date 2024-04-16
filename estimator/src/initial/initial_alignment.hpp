#pragma once 
#include <rclcpp/rclcpp.hpp>
#include "../factor/imu_factor.hpp"
#include "../utility/utility.hpp"
#include "../feature_manager.hpp"
/**
 * @brief 图像帧队形，保存了当前帧的特征点，当前帧位姿R和t，时间戳，预积分，是否关键帧
*/
class ImageFrame{
public:
    ImageFrame(){};
    ImageFrame(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>
        &_points, double _t) : t(t), is_key_frame(false){
        points = _points;    
    }
    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> points;
    double t;
    Eigen::Matrix3d R;
    Eigen::Vector3d T;
    IntegrationBase *per_integration;
    bool is_key_frame;
};

bool VisualIMUAlignment(std::map<double, ImageFrame> &all_image_frame, 
    Eigen::Vector3d* Bgs, Eigen::Vector3d &g, Eigen::VectorXd &d);