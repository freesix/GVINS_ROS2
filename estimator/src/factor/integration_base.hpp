#pragma once 

#include "../utility/utility.hpp"
#include "../parameters.hpp"
#include <ceres/ceres.h>

class IntegrationBase{
public:
    IntegrationBase() = delete;
    /**
     * @brief 构造函数
     * @param _acc_0 上一次测量加速度
     * @param _gyr_0 上一次测量角速度
     * @param _linearized_ba 上一次测量时刻的加速度计偏置
     * @param _linearized_bg 上一次测量时刻的陀螺仪偏置
    */
    IntegrationBase(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                    const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
        : acc_0{_acc_0}, gyr_0{_gyr_0}, linearized_acc{_acc_0}, linearized_gyr{_gyr_0},
          linearized_ba{_linearized_ba}, linearized_bg{_linearized_bg},
          jacobian{Eigen::Matrix<double, 15, 15>::Identity()}, // 预积分量对状态的雅可比矩阵
          covariance{Eigen::Matrix<double, 15, 15>::Zero()}, // 协方差矩阵，初始为0
          sum_dt{0.0}, 
          delta_p{Eigen::Vector3d::Zero()},  // 初始化位移变化量
          delta_q{Eigen::Quaterniond::Identity()}, // 初始化旋转变化量
          delta_v{Eigen::Vector3d::Zero()} // 初始化速度变化量
    {   
        // ΔX_{k+1} = F * ΔX_k + G * nk，其中nk为噪声，维度为18*1 (状态方程)
        // ΔX_{k+1}的协方差矩阵为P_{k+1} = F * P_k * F^T + G * V * G^T，因此噪声的协方差矩阵为18*18
        noise =  Eigen::Matrix<double, 18, 18>::Zero();
        noise.block<3, 3>(0, 0) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity(); // k时刻加速度计噪声
        noise.block<3, 3>(3, 3) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity(); // k陀螺仪噪声
        noise.block<3, 3>(6, 6) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity(); // k+1时刻加速度计噪声
        noise.block<3, 3>(9, 9) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity(); // k+1时刻陀螺仪噪声
        noise.block<3, 3>(12, 12) = (ACC_W * ACC_W) * Eigen::Matrix3d::Identity(); // 加速度计零偏噪声
        noise.block<3, 3>(15, 15) = (GYR_W * GYR_W) * Eigen::Matrix3d::Identity(); // 陀螺仪零偏噪声
    }

    void push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr){
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        propagate(dt, acc, gyr);
    }
    /**
     * @brief 重新计算积分
     * @details 论文将预积分量的偏置作为优化变量，当偏置在优化后发生变化，
     *          需要重新计算预积分量
    */
    void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
    {
        sum_dt = 0.0;
        acc_0 = linearized_acc; // 重新积分，需要重置上一时刻的加速度计测量值
        gyr_0 = linearized_gyr; // 重新积分，需要重置上一时刻的陀螺仪测量值
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        linearized_ba = _linearized_ba; // 优化后的加速度计偏置
        linearized_bg = _linearized_bg; // 优化后的陀螺仪偏置
        jacobian.setIdentity();
        covariance.setZero();
        for (int i = 0; i < static_cast<int>(dt_buf.size()); i++) // 对每个imu数据重新积分
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
    }
    /**
     * @brief 中值积分法递推Jacobian和Covariance
     * @param _acc_0 上一次测量加速度
     * @param _gyr_0 上一次测量角速度
     * @param _acc_1 本次测量加速度
     * @param _gyr_1 本次测量角速度
     * @param delta_p 位置变化量
     * @param delta_q 姿态变化量
     * @param delta_v 速度变化量
     * @param linearized_ba 加速度计偏置
     * @param linearized_bg 陀螺仪偏置
     * @param result_delta_p 更新后的位置变化量
     * @param result_delta_q 更新后的姿态变化量
     * @param result_delta_v 更新后的速度变化量
     * @param result_linearized_ba 更新后的加速度计偏置
     * @param result_linearized_bg 更新后的陀螺仪偏置
     * @param update_jacobian 是否更新Jacobian
    */
    void midPointIntegration(double _dt, 
                            const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                            const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                            Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                            Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian)
    {
        //ROS_INFO("midpoint integration");
        Eigen::Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba); 
        Eigen::Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
        result_delta_q = delta_q * Eigen::Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
        Eigen::Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
        result_delta_v = delta_v + un_acc * _dt;
        result_linearized_ba = linearized_ba; // 保持偏差不变，假定偏置固定，后续在优化中会改变并重新积分
        result_linearized_bg = linearized_bg;         

        if(update_jacobian)
        {
            Eigen::Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
            Eigen::Vector3d a_0_x = _acc_0 - linearized_ba;
            Eigen::Vector3d a_1_x = _acc_1 - linearized_ba;
            Eigen::Matrix3d R_w_x, R_a_0_x, R_a_1_x;

            R_w_x<<0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
            R_a_0_x<<0, -a_0_x(2), a_0_x(1),
                a_0_x(2), 0, -a_0_x(0),
                -a_0_x(1), a_0_x(0), 0;
            R_a_1_x<<0, -a_1_x(2), a_1_x(1),
                a_1_x(2), 0, -a_1_x(0),
                -a_1_x(1), a_1_x(0), 0;

            Eigen::MatrixXd F = Eigen::MatrixXd::Zero(15, 15);
            F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt + 
                                  -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
            F.block<3, 3>(0, 6) = Eigen::MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
            F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
            F.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() - R_w_x * _dt;
            F.block<3, 3>(3, 12) = -1.0 * Eigen::MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt + 
                                  -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * _dt) * _dt;
            F.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
            F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
            F.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();
            //cout<<"A"<<endl<<A<<endl;

            Eigen::MatrixXd V = Eigen::MatrixXd::Zero(15,18);
            V.block<3, 3>(0, 0) =  0.25 * delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 3) =  0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * _dt * 0.5 * _dt;
            V.block<3, 3>(0, 6) =  0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 9) =  V.block<3, 3>(0, 3);
            V.block<3, 3>(3, 3) =  0.5 * Eigen::MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(3, 9) =  0.5 * Eigen::MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(6, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 3) =  0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * 0.5 * _dt;
            V.block<3, 3>(6, 6) =  0.5 * result_delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 9) =  V.block<3, 3>(6, 3);
            V.block<3, 3>(9, 12) = Eigen::MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(12, 15) = Eigen::MatrixXd::Identity(3,3) * _dt;

            //step_jacobian = F;
            //step_V = V;
            jacobian = F * jacobian;
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();
        }

    }
    /**
     * @brief IMU预积分的传播过程
     * @details 积分计算两个关键帧之间IMU测量值的预积分量(变化量)
     * @param _dt 时间间隔
     * @param _acc_1 当前时刻的加速度计测量值
     * @param _gyr_1 当前时刻的陀螺仪测量值
    */
    void propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1)
    {
        dt = _dt;
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;
        Eigen::Vector3d result_delta_p;
        Eigen::Quaterniond result_delta_q;
        Eigen::Vector3d result_delta_v;
        Eigen::Vector3d result_linearized_ba;
        Eigen::Vector3d result_linearized_bg;

        midPointIntegration(_dt, acc_0, gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 1);

        //checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
        //                    linearized_ba, linearized_bg);
        delta_p = result_delta_p; // 更新中值积分后的位置变化量
        delta_q = result_delta_q; // 姿态变化量
        delta_v = result_delta_v; // 速度变换量
        linearized_ba = result_linearized_ba; // 加速度计偏置
        linearized_bg = result_linearized_bg; // 陀螺仪偏置
        delta_q.normalize(); // 归一化
        sum_dt += dt;  // 累计时间加上当前积分的两个时刻j和j+1的时间间隔
        acc_0 = acc_1; // 更新acc_0
        gyr_0 = gyr_1; // 更新gyr_0
     
    }


    Eigen::Matrix<double, 15, 1> evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                                          const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj)
    {
        Eigen::Matrix<double, 15, 1> residuals;

        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        Eigen::Vector3d dba = Bai - linearized_ba;
        Eigen::Vector3d dbg = Bgi - linearized_bg;

        Eigen::Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
        Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;
        // imu 预积分后误差
        residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;
    }

        
    // 中值积分需要用到前后两个时刻的IMU数据
    double dt; // 前后两个时刻的时间差
    Eigen::Vector3d acc_0, gyr_0; // 前一帧IMU数据中的加速度计测量值和陀螺仪测量值
    Eigen::Vector3d acc_1, gyr_1; // 后一帧

    const Eigen::Vector3d linearized_acc, linearized_gyr; // 这一段预积分初始时刻的imu测量值，作为常量一直保存
    Eigen::Vector3d linearized_ba, linearized_bg; // 这一段预积分对应的加速度计和陀螺仪偏置

    Eigen::Matrix<double, 15, 15> jacobian, covariance;
    Eigen::Matrix<double, 15, 15> step_jacobian;
    Eigen::Matrix<double, 15, 18> step_V;
    Eigen::Matrix<double, 18, 18> noise;

    double sum_dt; // 这一段预积分的总时间间隔(多个积分间隔dt的累加)
    Eigen::Vector3d delta_p;  // 当前时刻的本体坐标系下位置
    Eigen::Quaterniond delta_q; // 当前时刻本体坐标系下旋转
    Eigen::Vector3d delta_v; // 当前时刻本体坐标系下速度
    // 该段预积分所使用的imu数据的缓存
    // 当bias变换时，需要使用这些数据进重新预积分
    std::vector<double> dt_buf;  // 预积分要使用的imu时间间隔
    std::vector<Eigen::Vector3d> acc_buf; // 这一段预积分要使用的imu加速度计测量值
    std::vector<Eigen::Vector3d> gyr_buf; // 预积分要使用的imu陀螺仪测量值
};