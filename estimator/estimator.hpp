#pragma once 

#include <typeinfo>
#include "parameters.hpp"
#include "feature_manager.hpp"
#include "utility/utility.hpp"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.hpp"
#include "initial/initial_sfm.hpp"
#include "initial/initial_ex_rotation.hpp"
#include "initial/gnss_vi_initializer.hpp"
#include "initial/initial_alignment.hpp"
#include <std_msgs/msg/header.hpp>

#include <ceres/ceres.h>
#include "factor/imu_factor.hpp"
#include "factor/pose_local_parameterization.hpp"
#include "factor/projection_factor.hpp"
#include "factor/projection_td_factor.hpp"
#include "factor/marginalization_factor.hpp"
#include "factor/gnss_dt_ddt_factor.hpp"
#include "factor/gnss_psr_dopp_factor.hpp"
#include "factor/gnss_dt_anchor_factor.hpp"
#include "factor/gnss_ddt_smooth_factor.hpp"
#include "factor/pos_vel_factor.hpp"
#include "factor/pose_anchor_factor.hpp"

#include <opencv2/opencv.hpp>
#include <gnns_comm/gnss_utility.hpp>
#include <gnns_comm/gnss_ros.hpp>
#include <gnns_comm/gnss_spp.hpp>

using namespace gnns_comm::msg;

class Estimator{
public:
    Estimator();

    void setParameter();

        // interface
    void processIMU(double t, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity);
    void processGNSS(const std::vector<ObsPtr> &gnss_mea);
    void inputEphem(EphemBasePtr ephem_ptr);
    void inputIonoParams(double ts, const std::vector<double> &iono_params);
    void inputGNSSTimeDiff(const double t_diff);

    void processImage(const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::msg::Header &header);

    // internal
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    // GNSS related
    bool GNSSVIAlign();

    void updateGNSSStatistics();

    bool relativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l);
    void slideWindow();
    void solveOdometry();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();

    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Eigen::Vector3d g;
    // MatrixXd Ap[2];
    // VectorXd bp[2];

    Eigen::Matrix3d ric[NUM_OF_CAM]; // 从相机到imu的旋转
    Eigen::Vector3d tic[NUM_OF_CAM]; // 从相机到imu的平移
    
    Eigen::Vector3d Ps[(WINDOW_SIZE + 1)]; // 滑动窗口中各帧在世界坐标系下的位置
    Eigen::Vector3d Vs[(WINDOW_SIZE + 1)]; // 滑动窗口中各帧在世界坐标系下的速度
    Eigen::Matrix3d Rs[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Bas[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Bgs[(WINDOW_SIZE + 1)];
    double td; // camera数据和imu数据的时间戳偏移

    Eigen::Matrix3d back_R0, last_R, last_R0;
    Eigen::Vector3d back_P0, last_P, last_P0;
    std_msgs::msg::Header Headers[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)]; // 滑动窗口每帧图像对应一个IntergrationBase对象
    Eigen::Vector3d acc_0, gyr_0; // 最近一次接收到的imu数据
    // 滑动窗口内每一帧对应预积分用到的imu数据
    std::vector<double> dt_buf[(WINDOW_SIZE + 1)];
    std::vector<Eigen::Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    std::vector<Eigen::Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    // GNSS related
    bool gnss_ready;
    Eigen::Vector3d anc_ecef;
    Eigen::Matrix3d R_ecef_enu;
    double yaw_enu_local;
    std::vector<ObsPtr> gnss_meas_buf[(WINDOW_SIZE+1)];
    std::vector<EphemBasePtr> gnss_ephem_buf[(WINDOW_SIZE+1)];
    std::vector<double> latest_gnss_iono_params;
    std::map<uint32_t, std::vector<EphemBasePtr>> sat2ephem;
    std::map<uint32_t, std::map<double, size_t>> sat2time_index;
    std::map<uint32_t, uint32_t> sat_track_status;
    double para_anc_ecef[3];
    double para_yaw_enu_local[1];
    double para_rcv_dt[(WINDOW_SIZE+1)*4];
    double para_rcv_ddt[WINDOW_SIZE+1];
    // GNSS statistics
    double diff_t_gnss_local;
    Eigen::Matrix3d R_enu_local;
    Eigen::Vector3d ecef_pos, enu_pos, enu_vel, enu_ypr;

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    FeatureManager f_manager; // 用于管理滑动窗口内对应特征点数据
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    std::vector<Eigen::Vector3d> point_cloud;
    std::vector<Eigen::Vector3d> margin_cloud;
    std::vector<Eigen::Vector3d> key_poses;
    double initial_timestamp; // VIO完成初始化操作对应的图像帧(不代表初始化成功)
    // 用于ceres优化的参数块
    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE]; // 窗口内帧+新帧的位姿
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS]; 
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    double para_Td[1][1];

    MarginalizationInfo *last_marginalization_info;
    std::vector<double *> last_marginalization_parameter_blocks;

    std::map<double, ImageFrame> all_image_frame; // 滑动窗口中所有图像帧
    IntegrationBase *tmp_pre_integration; // 用于在创建ImageFrame对象时，把指针赋给imageframe.per_integration

    bool first_optimization;

};
