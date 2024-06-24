#include "interface/imu.hpp"

namespace Data{

void predict(const ImuTimeMsg &imu_msg)
{
    double t = imu_msg.timestamp;
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg.linear_acceleration[0];
    double dy = imu_msg.linear_acceleration[1];
    double dz = imu_msg.linear_acceleration[2];
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg.angular_velocity[0];
    double ry = imu_msg.angular_velocity[1];
    double rz = imu_msg.angular_velocity[2];
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator_ptr->g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator_ptr->g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}



} // namespace Data
