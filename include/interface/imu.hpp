#pragma once 
#include <Eigen/Eigen>
#include <queue>
#include <mutex>
#include <iostream>
#include <fstream>
#include <glog/logging.h>
#include "utility/utility.hpp"

namespace Data{
    struct ImuTimeMsg{
        double timestamp;
        Eigen::Vector3d linear_acceleration;
        Eigen::Vector3d angular_velocity;
    };

    double latest_time;
    double last_imu_t = -1;
    bool init_imu = 1;

    Eigen::Vector3d tmp_P;
    Eigen::Quaterniond tmp_Q;
    Eigen::Vector3d tmp_V;
    Eigen::Vector3d tmp_Ba;
    Eigen::Vector3d tmp_Bg;
    Eigen::Vector3d acc_0;
    Eigen::Vector3d gyr_0;


    void readImu(const std::string& imu_path);

} // namespace Data

