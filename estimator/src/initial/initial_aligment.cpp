#include "initial_aligment.hpp"
/**
 * @brief 更新得到新的陀螺仪漂移，并用新漂移重新积分
 * @details 用相邻两帧之间的相机旋转、IMU积分旋转，构建最小二乘问题，ldlt求解
*/
void solveGyroscopeBias(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d* Bgs){
    Eigen::Matrix3d A;
    Eigen::Vector3d b;
    Eigen::Vector3d delta_bg;
    A.setZero();
    b.setZero();
    std::map<double, ImageFrame>::iterator frame_i;
    std::map<double, ImageFrame>::iterator frame_j;
    // 遍历滑窗内所有图像帧
    for(frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++){
        frame_j = next(frame_i); // 下一帧
        Eigen::MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        Eigen::VectorXd tmp_b(3);
        tmp_b.setZero();
        // R_ij = (R^c0_bk)^-1 * (R^c0_bk+1) 转换为四元数 q_ij = (q^c0_bk)^-1 * (q^c0_bk+1)
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R); // 纯视觉里程计求解的i到j的旋转
        // tmp_A = J_j_bw 雅可比，没有算上偏置的预积分雅可比
        tmp_A = frame_j->second.per_integration->jacobian.template block<3, 3>(O_R, O_BG);
        //tmp_b = 2 * (r^bk_bk+1)^-1 * (q^c0_bk)^-1 * (q^c0_bk+1)
        //      = 2 * (r^bk_bk+1)^-1 * q_ij
        tmp_b = 2 * (frame_j->second.per_integration->delta_q.inverse() * q_ij).vec();
        //tmp_A * delta_bg = tmp_b
        A += tmp_A.transpose() * tmp_A; // 累计雅可比
        b += tmp_A.transpose() * tmp_b; // 累计误差
    }
    delta_bg = A.ldlt().solve(b); 
    RCLCPP_WARN_STREAM(rclcpp::get_logger("rclcpp"), "gyroscope bias initial calibration " << delta_bg.transpose());
    // 更新偏置(假设窗口内Bgs[i]相同)
    for(int i=0; i<=WINDOW_SIZE; i++){
        Bgs[i] += delta_bg;
    }
    // 更新偏置后重新计算积分
    for(frame_i = all_image_frame.begin(); next(frame_j) != all_image_frame.end(); frame_i++){
        frame_j = next(frame_i);
        frame_j->second.per_integration->repropagate(Eigen::Vector3d::Zero(), Bgs[0]);
    }
}
/**
 * @brief 计算重力矢量的切线空间
 * @details 以重力矢量为z轴，构建一个正交基
*/
Eigen::MatrixXd TangentBasis(Eigen::Vector3d &g0){
    Eigen::Vector3d b, c;
    Eigen::Vector3d a = g0.normalized(); // 重力矢量归一化(单位向量)
    Eigen::Vector3d tmp(0, 0, 1);
    if(a == tmp){ // 重力矢量与z轴平行，改变tmp的方向为x轴，避免奇点
        tmp << 1, 0, 0;
    }
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    Eigen::MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}
/**
 * @brief 修正重力矢量(用半球约束来参数化重力)
*/
void RefineGravity(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d &g,
    Eigen::VectorXd &x){

    Eigen::Vector3d g0 = g.normalized() * G.norm();  // 方向\times模长 
    Eigen::Vector3d lx, ly; // 切空间的两个基

    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1; // 初始待估计状态量(每帧速度、重力、尺度因子)，这里精细化重力方向是单位向量，所以只有两个自由度

    Eigen::MatrixXd A{n_state, n_state};
    A.setZero();
    Eigen::VectorXd b{n_state};
    b.setZero();

    std::map<double, ImageFrame>::iterator frame_i;
    std::map<double, ImageFrame>::iterator frame_j;
    for(int k=0; k<4; k++){
        Eigen::MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0); // 求g0的切线空间，构成一个正交基
        int i = 0; 
        for(frame_i = all_image_frame.begin(); next(frame_j) != all_image_frame.end(); frame_i++, i++){
            frame_j = next(frame_i);
            Eigen::MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            Eigen::VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.per_integration->sum_dt;

            tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Eigen::Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
            tmp_b.block<3, 1>(0, 0) = frame_j->second.per_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Eigen::Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.per_integration->delta_v - frame_i->second.R.transpose() * dt * Eigen::Matrix3d::Identity() * g0;        

            Eigen::Matrix<double, 6, 6> cov_inv = Eigen::Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            Eigen::MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            Eigen::VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }

        A = A * 1000.0;
        b = b * 1000.0;
        x = A.ldlt().solve(b);
        Eigen::VectorXd dg = x.segment<2>(n_state - 3);
        g0 = (g0 + lxly * dg).normalized() * G.norm();
        //double s = x(n_state - 1);
    }
    g = g0;
}

/**
 * @brief 初始化速度、重力和尺度因子
*/
bool LinearAlignment(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d &g, Eigen::VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1; // 初始待估计状态量(每帧速度、重力、尺度因子)

    Eigen::MatrixXd A{n_state, n_state};
    A.setZero();
    Eigen::VectorXd b{n_state};
    b.setZero();

    std::map<double, ImageFrame>::iterator frame_i;
    std::map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    // 遍历滑窗中第一帧到倒数第二帧
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        Eigen::MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        Eigen::VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.per_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Eigen::Matrix3d::Identity();
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
        tmp_b.block<3, 1>(0, 0) = frame_j->second.per_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Eigen::Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.per_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Eigen::Matrix<double, 6, 6> cov_inv = Eigen::Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        Eigen::MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        Eigen::VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    double s = x(n_state - 1) / 100.0;
    RCUTILS_LOG_DEBUG("estimated scale: %f", s);
    g = x.segment<3>(n_state - 4);
    RCLCPP_DEBUG_STREAM(rclcpp::get_logger("rclcpp"), " result g " << g.norm() << " " << g.transpose());
    if(fabs(g.norm() - G.norm()) > 1.0 || s < 0) // 判断重力矢量是否合理，尺度因子是否合理
    {
        return false;
    }

    RefineGravity(all_image_frame, g, x);
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    RCLCPP_DEBUG_STREAM(rclcpp::get_logger("rclcpp"), " refine     " << g.norm() << " " << g.transpose());
    if(s < 0.0){
        return false;  
    } 
    else{
        return true;
    }
}
/**
 * @brief 视觉SFM结果和IMU预积分结果对齐
*/
bool VisualIMUAlignment(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d* Bgs, 
    Eigen::Vector3d &g, Eigen::VectorXd &x)
{   
    solveGyroscopeBias(all_image_frame, Bgs);

    if(LinearAlignment(all_image_frame, g, x)){
        return true;
    }
    else{ 
        return false;
    }
}