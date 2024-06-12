#pragma once 

#include <rclcpp/rclcpp.hpp>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.hpp"
#include "../utility/tic_toc.h"
#include "../parameters.hpp"
/**
 * @brief 引入时间差的视觉投影测量因子
 * 残差维度为2，特征点在不同相机位姿下的投影误差
 * 机体在i帧相对世界坐标系的位姿，维度为7
 * 机体在j帧相对世界坐标系的位姿，维度为7
 * 特征点的逆深度，维度为1
 * 时间差，维度为1
*/
class ProjectionTdFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1, 1>
{
public:
    ProjectionTdFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j,
        const Eigen::Vector2d &_velocity_i, const Eigen::Vector2d &_velocity_j,
        const double _td_i, const double _td_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals,
        double **jacobians) const;
    void check(double **parameters);

    Eigen::Vector3d pts_i, pts_j; // 像素点在不同相机位姿(不同帧)下的投影坐标
    Eigen::Vector3d velocity_i, velocity_j; // 像素点在不同相机位姿(不同帧)下的速度
    double td_i, td_j; // 时间差
    Eigen::Matrix<double, 2, 3> tangent_base;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
};