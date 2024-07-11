#pragma once

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <rclcpp/rclcpp.hpp>

#include "../parameters.hpp"

class InitialEXRotation
{
public:
    InitialEXRotation();

    bool CalibrationExRotation(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres,
            Eigen::Quaterniond delta_q_imu, Eigen::Matrix3d &calib_ric_result);
    
private:
    Eigen::Matrix3d solveRelativeR(const std::vector<std::pair<Eigen::Vector3d, 
                                Eigen::Vector3d>> &corres);
    /**
     * @brief 检查三角化的结果是否正确，判断R和t的合理性
     * @param f 前一帧图像上的特征点
     * @param r 后一帧图像上的特征点
     * @param R 两帧图像之间的旋转矩阵
     * @param t 两帧图像之间的平移向量
    */
    double testTriangulation(const std::vector<cv::Point2f> &f,
                             const std::vector<cv::Point2f> &r,
                             cv::Mat_<double> R, cv::Mat_<double> t);

    void decomposeE(cv::Mat E, cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                    cv::Mat_<double> &t1, cv::Mat_<double> &t2);

    int frame_count;

    std::vector<Eigen::Matrix3d> Rc;
    std::vector<Eigen::Matrix3d> Rimu;
    std::vector<Eigen::Matrix3d> Rc_g;
    Eigen::Matrix3d ric;
};