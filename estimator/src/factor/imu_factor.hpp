#pragma once 
#include <rcpputils/asserts.hpp>
#include <eigen3/Eigen/Dense>
#include "../utility/utility.hpp"
#include "../parameters.hpp"
#include "integration_base.hpp"
#include <ceres/ceres.h>
/**
 * @brief imu 因子
 * 残差维度15
 * 机体在i帧的位姿，维度为7
 * 机体在i帧的速度和imu偏置，维度为9
 * 机体在j帧的位姿，维度为7
 * 机体在j帧的速度和imu偏置，维度为9
*/
class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9>{
public:
    IMUFactor() = delete;
    IMUFactor(IntegrationBase* _pre_integration) : pre_integration(_pre_integration){}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians)const{
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]); // 机体在i帧的世界坐标系下的位置
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]); // 机体在i帧的世界坐标系下的姿态
        Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]); // 机体在i帧的速度
        Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]); // imu加速度偏置
        Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]); // imu角速度偏置

        Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]); // 机体在j帧的世界坐标系下的位置
        Eigen::Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]); // 机体在j帧的世界坐标系下的姿态
        Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]); // 机体在j帧的速度
        Eigen::Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]); 
        Eigen::Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]);

#if 0 
        if((Bai - pre_integration->linearized_ba).norm() > 0.10 || 
            Baj - pre_integration->linearized_bg).norm() > 0.01){
            pre_integration->repropagate(Bai, Bgi);
        }
#endif
        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals); // 残差
        residual = pre_integration->evaluate(Pi, Qi, Vi, Bai, Bgi,
                                            Pj, Qj, Vj, Baj, Bgj);
        // 协方差矩阵，做LLT分解，仅对正定矩阵有效
        // $A=L * L^T$
        Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>
                                    (pre_integration->covariance.inverse()).matrixL().transpose();
        residual = sqrt_info * residual; // 马氏距离加权后得到的残差

        if(jacobians){
            double sum_dt = pre_integration->sum_dt;
            Eigen::Matrix3d dp_dba = pre_integration->jacobian.template block<3, 3>(O_P, O_BA);
            Eigen::Matrix3d dp_dbg = pre_integration->jacobian.template block<3, 3>(O_P, O_BG);

            Eigen::Matrix3d dq_dbg = pre_integration->jacobian.template block<3, 3>(O_R, O_BG);

            Eigen::Matrix3d dv_dba = pre_integration->jacobian.template block<3, 3>(O_V, O_BA);
            Eigen::Matrix3d dv_dbg = pre_integration->jacobian.template block<3, 3>(O_V, O_BG);

            if(pre_integration->jacobian.maxCoeff() > 1e8 || pre_integration->jacobian.minCoeff() < -1e8){
                RCUTILS_LOG_WARN("numerical unstable in preintegration");
            }

            if(jacobians[0]){
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();

                jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
                jacobian_pose_i.block<3, 3>(O_P, O_R) = 
                    Utility::skewSymmetric(Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));

#if 0 
                jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Qj.inverse() * Qi).toRotationMatrix();
#else
                Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * 
                    Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Utility::Qleft(Qj.inverse() * Qi) * 
                    Utility::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();
#endif

                jacobian_pose_i.block<3, 3>(O_V, O_R) = Utility::skewSymmetric(Qi.inverse() * 
                    (G * sum_dt + Vj - Vi));

                jacobian_pose_i = sqrt_info * jacobian_pose_i;

                if(jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8){
                    RCUTILS_LOG_WARN("numerical unstable in preintegration");
                }
            }
            if(jacobians[1]){
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
                jacobian_speedbias_i.setZero();
                jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
                jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
                jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;
#if 0
                jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -dq_dbg;
#else   
                jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi *
                    pre_integration->delta_q).bottomRightCorner<3, 3>() * dq_dbg;
#endif

                jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();
                jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
                jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;

                jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity();
                jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity();

                jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;
            
            }
            if(jacobians[2]){
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);
                jacobian_pose_j.setZero();
                jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();

#if 0
                jacobian_pose_j,block<3, 3>(O_R, O_R) = Eigen::Matrix3d::Identity();
#else   
                Eigen::Quaterniond corrected_delta_q = 
                    pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                jacobian_pose_j.block<3, 3>(O_R, O_R) = 
                    Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();   
#endif      
                jacobian_pose_j = sqrt_info * jacobian_pose_j;
            }

            if(jacobians[3]){
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[3]);
                jacobian_speedbias_j.setZero();

                jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();
                jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();
                jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();
                jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;
            }
        }
        return true;
    }
    
     
    IntegrationBase* pre_integration;
    
};