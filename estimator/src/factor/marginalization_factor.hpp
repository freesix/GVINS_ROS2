#pragma once

#include <rclcpp/rclcpp.hpp>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.hpp"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{   // 将不同损失函数_cost_function和优化变量_parameter_blocks组合起来添加到marginlization_info中
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, 
                            std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), 
        parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function; // 核函数
    std::vector<double *> parameter_blocks;
    std::vector<int> drop_set; // 被边缘化的变量

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};
// 边缘化类
class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info); // 添加残差块信息
    // 计算每个残差对应的雅可比，并更新parameter_block_data
    // 得到每次IMU、视觉观测(cost_function)对应的参数块，雅可比矩阵、残差
    void preMarginalize();
    // 多线程进行marg，执行了Shur补操作，pose为所有变量维度，m为需要marg掉的变量，n为保留的变量
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors; // 所有观测项
    int m, n;
    // 这三个unordered_map的key表示地址，value表示优化变量的长度
    std::unordered_map<long, int> parameter_block_size; //global size 需要marg掉的变量大小<优化变量内存地址，优化变量长度>
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; //local size<优化变量内存地址，在矩阵中的id>
    std::unordered_map<long, double *> parameter_block_data; //<优化变量内存地址，数据指针>
    // 边缘化后留下来的变量
    std::vector<int> keep_block_size; //global size保留下来的变量长度
    std::vector<int> keep_block_idx;  //local size id
    std::vector<double *> keep_block_data; // 数据

    Eigen::MatrixXd linearized_jacobians; // 边缘化后从信息矩阵中恢复出来的雅可比矩阵
    Eigen::VectorXd linearized_residuals; // 边缘化后的残差
    const double eps = 1e-8;

};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
