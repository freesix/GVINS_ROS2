#pragma once 

#include <rclcpp/rclcpp.hpp>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.hpp"
#include "../utility/tic_toc.h"
#include "../parameters.hpp"
/**
 * @brief 视觉投影测量因子
 * 残差维度为2，特征点在不同相机位姿下的投影误差
 * 机体在i处相对世界坐标系的位姿，维度为7
 * 机体在j处相对世界坐标系的位姿，维度为7
 * 相机到imu的平移和旋转，维度为7
 * 特征点的逆深度，维度为1
*/
class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1>
{
public:
    ProjectionFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d & _pts_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    Eigen::Vector3d pts_i, pts_j; // 特征点在不同相机位姿下的投影坐标
    Eigen::Matrix<double, 2, 3> tangent_base; // 单位球误差相关，未用
    static Eigen::Matrix2d sqrt_info;
    static double sum_t; 
};