#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logger.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <gnss_comm/gnss_ros.hpp>
#include <gnss_comm/gnss_utility.hpp>
#include <estimator_interfaces/msg/local_sensor_external_trigger.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>

#include "estimator.hpp"
#include "parameters.hpp"
#include "utility/visualization.hpp"

using namespace gnss_comm;

#define MAX_GNSS_CAMERA_DELAY 0.05

std::unique_ptr<Estimator> estimator_ptr;

std::condition_variable con; // 状态变量
double current_time = -1;
std::queue<std::shared_ptr<sensor_msgs::msg::Imu>> imu_buf;
std::queue<std::shared_ptr<sensor_msgs::msg::PointCloud>> feature_buf;
std::queue<std::vector<ObsPtr>> gnss_meas_buf;
std::queue<std::shared_ptr<sensor_msgs::msg::PointCloud>> relo_buf;
int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = -1;

std::mutex m_time;
double next_pulse_time;
bool next_pulse_time_valid;
double time_diff_gnss_local;
bool time_diff_valid;
double latest_gnss_time;
double tmp_last_feature_time;
uint64_t feature_msg_counter;
int skip_parameter;

void predict(const std::shared_ptr<sensor_msgs::msg::Imu> &imu_msg)
{
    double t = stamp2Sec(imu_msg->header.stamp);
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
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

void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator_ptr->Ps[WINDOW_SIZE];
    tmp_Q = estimator_ptr->Rs[WINDOW_SIZE];
    tmp_V = estimator_ptr->Vs[WINDOW_SIZE];
    tmp_Ba = estimator_ptr->Bas[WINDOW_SIZE];
    tmp_Bg = estimator_ptr->Bgs[WINDOW_SIZE];
    acc_0 = estimator_ptr->acc_0;
    gyr_0 = estimator_ptr->gyr_0;

    std::queue<std::shared_ptr<sensor_msgs::msg::Imu>> tmp_imu_buf = imu_buf;
    for (std::shared_ptr<sensor_msgs::msg::Imu> tmp_imu_msg; !tmp_imu_buf.empty(); 
        tmp_imu_buf.pop()){
        
        predict(tmp_imu_buf.front());
    }
}
/**
 * @brief 根据时间戳检查传感器输入数据的合法性
*/
bool getMeasurements(std::vector<std::shared_ptr<sensor_msgs::msg::Imu>> &imu_msg, 
    std::shared_ptr<sensor_msgs::msg::PointCloud> &img_msg, std::vector<ObsPtr> &gnss_msg)
{   
    // 当imu、feature、gnss数据有一个为空直接返回false
    if (imu_buf.empty() || feature_buf.empty() || (GNSS_ENABLE && gnss_meas_buf.empty()))
        return false;
    // 将imu和图像的时间戳尽量对齐，front_feature_ts指feature_buf中的第一帧图像时间
    double front_feature_ts = stamp2Sec(feature_buf.front()->header.stamp);

    if (!stamp2Sec(imu_buf.back()->header.stamp) > front_feature_ts)
    {
        //ROS_WARN("wait for imu, only should happen at the beginning");
        sum_of_wait++;
        return false;
    }
    // feature缓存不为空，且imu时间大于feature时间，要丢弃部分图像帧数据
    double front_imu_ts =stamp2Sec(imu_buf.front()->header.stamp);
    while (!feature_buf.empty() && front_imu_ts > front_feature_ts)
    {
        RCUTILS_LOG_WARN("throw img, only should happen at the beginning");
        feature_buf.pop();
        front_feature_ts = stamp2Sec(feature_buf.front()->header.stamp);
    }
    // 大致将gnss数据和图像数据对齐，实现三者之间的对齐
    if (GNSS_ENABLE)
    {
        front_feature_ts += time_diff_gnss_local; // gnss时间和local本地时间偏差
        double front_gnss_ts = time2sec(gnss_meas_buf.front()[0]->time);
        // gnss不为空，但时间小于图像帧，需要丢弃一部分
        while (!gnss_meas_buf.empty() && front_gnss_ts < front_feature_ts-MAX_GNSS_CAMERA_DELAY) 
        {
            RCUTILS_LOG_WARN("throw gnss, only should happen at the beginning");
            gnss_meas_buf.pop();
            if (gnss_meas_buf.empty()) return false;
            front_gnss_ts = time2sec(gnss_meas_buf.front()[0]->time);
        }
        if (gnss_meas_buf.empty())
        {
            RCUTILS_LOG_WARN("wait for gnss...");
            return false;
        }
        else if (abs(front_gnss_ts-front_feature_ts) < MAX_GNSS_CAMERA_DELAY)
        {
            gnss_msg = gnss_meas_buf.front();
            gnss_meas_buf.pop();
        }
    }

    img_msg = feature_buf.front();
    feature_buf.pop();
    // 将早于当前帧图像晚于前一帧图像之间的imu和晚于当前图像的第一帧imu加入imu_msg和当前图像对齐(多帧imu对一帧图像)
    while (stamp2Sec(imu_buf.front()->header.stamp) < stamp2Sec(img_msg->header.stamp) + estimator_ptr->td)
    {
        imu_msg.emplace_back(imu_buf.front());
        imu_buf.pop();
    }
    imu_msg.emplace_back(imu_buf.front());
    if (imu_msg.empty())
        RCUTILS_LOG_WARN("no imu between two image");
    return true;
}

void imu_callback(const sensor_msgs::msg::Imu::SharedPtr imu_msg)
{
    if (stamp2Sec(imu_msg->header.stamp) <= last_imu_t)
    {
        RCUTILS_LOG_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = stamp2Sec(imu_msg->header.stamp);

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one(); // 唤醒getMeasurements()

    last_imu_t = stamp2Sec(imu_msg->header.stamp);

    {
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);
        std_msgs::msg::Header header = imu_msg->header;
        header.frame_id = "world";
        // 初始化完成，处于滑动窗口非线性优化状态，如果不处于则不发布里程计信息
        if (estimator_ptr->solver_flag == Estimator::SolverFlag::NON_LINEAR) 
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header); // 发布频率很高，和imu数据同频
    }
}

void gnss_ephem_callback(const gnss_interfaces::msg::GnssEphemMsg::SharedPtr ephem_msg)
{
    EphemPtr ephem = msg2ephem(ephem_msg);
    estimator_ptr->inputEphem(ephem);
}

void gnss_glo_ephem_callback(const gnss_interfaces::msg::GnssGloEphemMsg::SharedPtr glo_ephem_msg)
{
    GloEphemPtr glo_ephem = msg2glo_ephem(glo_ephem_msg);
    estimator_ptr->inputEphem(glo_ephem);
}
/**
 * @brief 电离层参数订阅
 * @details 考虑电离层和对流层对gnss传播的影响，后面会加上卫星仰角参数进行建模，因为仰角小
 * 的卫星在电离层中传播的时间较长，对定位影响大
*/
void gnss_iono_params_callback(const gnss_interfaces::msg::StampedFloat64Array::SharedPtr iono_msg)
{
    double ts = stamp2Sec(iono_msg->header.stamp);
    std::vector<double> iono_params;
    std::copy(iono_msg->data.begin(), iono_msg->data.end(), std::back_inserter(iono_params));
    assert(iono_params.size() == 8);
    estimator_ptr->inputIonoParams(ts, iono_params);
}

void gnss_meas_callback(const gnss_interfaces::msg::GnssMeasMsg::SharedPtr meas_msg)
{
    std::vector<ObsPtr> gnss_meas = msg2meas(meas_msg);

    latest_gnss_time = time2sec(gnss_meas[0]->time);

    // cerr << "gnss ts is " << std::setprecision(20) << time2sec(gnss_meas[0]->time) << endl;
    if (!time_diff_valid)   return;

    m_buf.lock();
    gnss_meas_buf.push(std::move(gnss_meas));
    m_buf.unlock();
    con.notify_one();
}

void feature_callback(const sensor_msgs::msg::PointCloud::SharedPtr feature_msg)
{   
    RCUTILS_LOG_DEBUG("I coming feature_callback");
    ++ feature_msg_counter;

    if (skip_parameter < 0 && time_diff_valid)
    {
        const double this_feature_ts = stamp2Sec(feature_msg->header.stamp)+time_diff_gnss_local;
        if (latest_gnss_time > 0 && tmp_last_feature_time > 0)
        {
            if (abs(this_feature_ts - latest_gnss_time) > abs(tmp_last_feature_time - latest_gnss_time))
                skip_parameter = feature_msg_counter%2;       // skip this frame and afterwards
            else
                skip_parameter = 1 - (feature_msg_counter%2);   // skip next frame and afterwards
        }
        // cerr << "feature counter is " << feature_msg_counter << ", skip parameter is " << int(skip_parameter) << endl;
        tmp_last_feature_time = this_feature_ts;
    }

    if (skip_parameter >= 0 && int(feature_msg_counter%2) != skip_parameter)
    {
        m_buf.lock();
        feature_buf.push(feature_msg);
        m_buf.unlock();
        con.notify_one();
    }
}
/**
 * @brief 获得local和gnss的时间差
 * @details trigger_msg记录的是相机被gnss触发的时间，可以理解为图像命名的时间，和gnss时间
 * 有差别，因为硬件存在延迟等。因此后面对这个时间进行矫正(当然这是在作者那个硬件系统中)
*/
void local_trigger_info_callback(const estimator_interfaces::msg::LocalSensorExternalTrigger::SharedPtr trigger_msg)
{
    std::lock_guard<std::mutex> lg(m_time);

    if (next_pulse_time_valid)
    {
        time_diff_gnss_local = next_pulse_time - stamp2Sec(trigger_msg->header.stamp);
        estimator_ptr->inputGNSSTimeDiff(time_diff_gnss_local);
        if (!time_diff_valid)       // just get calibrated
            std::cout << "time difference between GNSS and VI-Sensor got calibrated: "
                << std::setprecision(15) << time_diff_gnss_local << " s\n";
        time_diff_valid = true;
    }
}

void gnss_tp_info_callback(const gnss_interfaces::msg::GnssTimePulseInfoMsg::SharedPtr tp_msg)
{
    gtime_t tp_time = gpst2time(tp_msg->time.week, tp_msg->time.tow);
    if (tp_msg->utc_based || tp_msg->time_sys == SYS_GLO)
        tp_time = utc2gpst(tp_time);
    else if (tp_msg->time_sys == SYS_GAL)
        tp_time = gst2time(tp_msg->time.week, tp_msg->time.tow);
    else if (tp_msg->time_sys == SYS_BDS)
        tp_time = bdt2time(tp_msg->time.week, tp_msg->time.tow);
    else if (tp_msg->time_sys == SYS_NONE)
    {
        std::cerr << "Unknown time system in GNSSTimePulseInfoMsg.\n";
        return;
    }
    double gnss_ts = time2sec(tp_time);

    std::lock_guard<std::mutex> lg(m_time);
    next_pulse_time = gnss_ts;
    next_pulse_time_valid = true;
}

void restart_callback(const std_msgs::msg::Bool::SharedPtr restart_msg)
{
    if (restart_msg->data == true)
    {
        RCUTILS_LOG_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator_ptr->clearState();
        estimator_ptr->setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}
/**
 * @brief 这是measurement线程的线程函数，用于处理后端部分，包括imu预积分、耦合初始化和local BA
*/
void process()
{
    while (true)
    {   
        std::vector<std::pair<std::vector<std::shared_ptr<sensor_msgs::msg::Imu>>,
            std::shared_ptr<sensor_msgs::msg::PointCloud>>>  measurements;
        std::vector<std::shared_ptr<sensor_msgs::msg::Imu>> imu_msg;
        std::shared_ptr<sensor_msgs::msg::PointCloud> img_msg;
        std::vector<ObsPtr> gnss_msg;

        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
                    return getMeasurements(imu_msg, img_msg, gnss_msg);
                 });
        lk.unlock();
        m_estimator.lock();
        double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
        for (auto &imu_data : imu_msg)
        {
            double t = stamp2Sec(imu_data->header.stamp);
            double img_t = stamp2Sec(img_msg->header.stamp) + estimator_ptr->td;
            if (t <= img_t)
            { 
                if (current_time < 0)
                    current_time = t;
                double dt = t - current_time;
                assert(dt >= 0);
                current_time = t;
                dx = imu_data->linear_acceleration.x;
                dy = imu_data->linear_acceleration.y;
                dz = imu_data->linear_acceleration.z;
                rx = imu_data->angular_velocity.x;
                ry = imu_data->angular_velocity.y;
                rz = imu_data->angular_velocity.z;
                estimator_ptr->processIMU(dt, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
                //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

            }
            else
            {
                double dt_1 = img_t - current_time;
                double dt_2 = t - img_t;
                current_time = img_t;
                assert(dt_1 >= 0);
                assert(dt_2 >= 0);
                assert(dt_1 + dt_2 > 0);
                double w1 = dt_2 / (dt_1 + dt_2);
                double w2 = dt_1 / (dt_1 + dt_2);
                dx = w1 * dx + w2 * imu_data->linear_acceleration.x;
                dy = w1 * dy + w2 * imu_data->linear_acceleration.y;
                dz = w1 * dz + w2 * imu_data->linear_acceleration.z;
                rx = w1 * rx + w2 * imu_data->angular_velocity.x;
                ry = w1 * ry + w2 * imu_data->angular_velocity.y;
                rz = w1 * rz + w2 * imu_data->angular_velocity.z;
                estimator_ptr->processIMU(dt_1, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
                //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
            }
        }

        if (GNSS_ENABLE && !gnss_msg.empty())
            estimator_ptr->processGNSS(gnss_msg);

        RCUTILS_LOG_DEBUG("processing vision data with stamp %f \n", stamp2Sec(img_msg->header.stamp));

        TicToc t_s;
        std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> image;
        for (unsigned int i = 0; i < img_msg->points.size(); i++)
        {
            int v = img_msg->channels[0].values[i] + 0.5;
            int feature_id = v / NUM_OF_CAM;
            int camera_id = v % NUM_OF_CAM;
            double x = img_msg->points[i].x;
            double y = img_msg->points[i].y;
            double z = img_msg->points[i].z;
            double p_u = img_msg->channels[1].values[i];
            double p_v = img_msg->channels[2].values[i];
            double velocity_x = img_msg->channels[3].values[i];
            double velocity_y = img_msg->channels[4].values[i];
            assert(z == 1);
            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
        }
        estimator_ptr->processImage(image, img_msg->header);

        double whole_t = t_s.toc();
        printStatistics(*estimator_ptr, whole_t);
        std_msgs::msg::Header header = img_msg->header;
        header.frame_id = "world";

        pubOdometry(*estimator_ptr, header);
        pubKeyPoses(*estimator_ptr, header);
        pubCameraPose(*estimator_ptr, header);
        pubPointCloud(*estimator_ptr, header);
        pubTF(*estimator_ptr, header);
        pubKeyframe(*estimator_ptr);
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator_ptr->solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char** argv){
    rclcpp::init(argc, argv);
    auto n = rclcpp::Node::make_shared("gvins");

    readParameters(n);
    estimator_ptr.reset(new Estimator());
    estimator_ptr->setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    RCUTILS_LOG_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    registerPub(n);

    next_pulse_time_valid = false;
    time_diff_valid = false;
    latest_gnss_time = -1;
    tmp_last_feature_time = -1;
    feature_msg_counter = 0;

    if(GNSS_ENABLE){
        skip_parameter = -1;
    }
    else{
        skip_parameter = 0;
    }
    auto sub_imu = n->create_subscription<sensor_msgs::msg::Imu>(IMU_TOPIC, 
        rclcpp::QoS(rclcpp::KeepLast(2000)), imu_callback);
    
    auto sub_feature = n->create_subscription<sensor_msgs::msg::PointCloud>("feature",
        rclcpp::QoS(rclcpp::KeepLast(2000)), feature_callback);

    auto sub_restart = n->create_subscription<std_msgs::msg::Bool>("restart",
        rclcpp::QoS(rclcpp::KeepLast(2000)), restart_callback);
    rclcpp::Subscription<gnss_interfaces::msg::GnssEphemMsg>::SharedPtr sub_ephem;
    rclcpp::Subscription<gnss_interfaces::msg::GnssGloEphemMsg>::SharedPtr sub_glo_ephem;
    rclcpp::Subscription<gnss_interfaces::msg::GnssMeasMsg>::SharedPtr sub_gnss_meas;
    rclcpp::Subscription<gnss_interfaces::msg::StampedFloat64Array>::SharedPtr sub_gnss_iono_params;
    rclcpp::Subscription<gnss_interfaces::msg::GnssTimePulseInfoMsg>::SharedPtr sub_gnss_time_pluse_info;
    rclcpp::Subscription<estimator_interfaces::msg::LocalSensorExternalTrigger>::SharedPtr sub_local_trigger_info;
    
    if(GNSS_ENABLE){
        sub_ephem = n->create_subscription<gnss_interfaces::msg::GnssEphemMsg>(GNSS_EPHEM_TOPIC, rclcpp::QoS(rclcpp::KeepLast(100)), gnss_ephem_callback);
        sub_glo_ephem = n->create_subscription<gnss_interfaces::msg::GnssGloEphemMsg>(GNSS_GLO_EPHEM_TOPIC, rclcpp::QoS(rclcpp::KeepLast(100)),
            gnss_glo_ephem_callback);
        sub_gnss_meas = n->create_subscription<gnss_interfaces::msg::GnssMeasMsg>(GNSS_MEAS_TOPIC, rclcpp::QoS(rclcpp::KeepLast(100)), 
            gnss_meas_callback);
        sub_gnss_iono_params = n->create_subscription<gnss_interfaces::msg::StampedFloat64Array>(GNSS_IONO_PARAMS_TOPIC,
            rclcpp::QoS(rclcpp::KeepLast(100)), gnss_iono_params_callback);
        
        if(GNSS_LOCAL_ONLINE_SYNC){    
            sub_gnss_time_pluse_info = n->create_subscription<gnss_interfaces::msg::GnssTimePulseInfoMsg>(GNSS_TP_INFO_TOPIC,
                rclcpp::QoS(rclcpp::KeepLast(100)), gnss_tp_info_callback);
                
            sub_local_trigger_info = n->create_subscription<estimator_interfaces::msg::LocalSensorExternalTrigger>(
                LOCAL_TRIGGER_INFO_TOPIC, rclcpp::QoS(rclcpp::KeepLast(100)), local_trigger_info_callback);
        }
        else{ 
            time_diff_gnss_local = GNSS_LOCAL_TIME_DIFF;
            estimator_ptr->inputGNSSTimeDiff(time_diff_gnss_local);
            time_diff_valid = true;
        } 
    }
    std::thread measurement_process{process};
    rclcpp::spin(n);
    rclcpp::shutdown();

    return 0;
}