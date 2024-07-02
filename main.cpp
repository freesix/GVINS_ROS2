#include <thread>
#include <queue>
#include <condition_variable>
#include <chrono>
#include "feature_tracker/feature_tracker.hpp"
#include "gnss_constant.hpp"
#include "interface/img.hpp"
#include "interface/imu.hpp"
#include "estimator.hpp"

#define MAX_GNSS_CAMERA_DELAY 0.05

// std::queue<std::shared_ptr<Data::ImgMsg>> img_buf; // 图像缓存
std::queue<Data::FeatureTimeMsg> feature_buf; // 图像特征点缓存
std::queue<Data::ImuTimeMsg> imu_buf; // imu数据缓存
std::queue<std::vector<gnss_comm::ObsPtr>> gnss_meas_buf; // gnss数据缓存
int sum_of_wait = 0;

std::unique_ptr<Estimator> estimator_ptr;

std::condition_variable con; // 条件变量
double current_time = -1;
std::mutex m_buf; 
std::mutex f_buf; // 锁住特征点缓存
std::mutex i_buf; // 锁住imu数据缓存
std::mutex m_state;
std::mutex m_estimator;
// img
FeatureTracker trackerData[NUM_OF_CAM];
bool RESTART_FLAG = false;
bool first_image_flag = true;
double first_image_time;
double last_image_time = 0;
int pub_count = 1;
bool init_pub = false;
// feature
uint64_t feature_msg_counter;
int skip_parameter;
// gnss
double time_diff_gnss_local; // pps触发时间和vi传感器实际被触发时间之间的差
bool time_diff_valid; // 为false则对于收到的gnss数据不会存储
double latest_gnss_time;
double tmp_last_feature_time;
bool next_pulse_time_valid; // 如果进入pps触发的回调函数，这个为true

using namespace Data;
        
/**
 * @brief 根据时间戳检查传感器输入数据的合法性
*/
bool getMeasurements(std::vector<ImuTimeMsg> &imu_msg, FeatureTimeMsg &img_msg, 
        std::vector<gnss_comm::ObsPtr> &gnss_msg)
{   
    // 当imu、feature、gnss数据有一个为空直接返回false
    if (imu_buf.empty() || feature_buf.empty() || (GNSS_ENABLE && gnss_meas_buf.empty()))
        return false;

    LOG(INFO) << "imu_buf size: %d", imu_buf.size();
    // 将imu和图像的时间戳尽量对齐，front_feature_ts指feature_buf中的第一帧图像时间
    double front_feature_ts = feature_buf.front().timestamp;

    if (!imu_buf.back().timestamp > front_feature_ts)
    {
        //ROS_WARN("wait for imu, only should happen at the beginning");
        sum_of_wait++;
        return false;
    }
    // feature缓存不为空，且imu时间大于feature时间，要丢弃部分图像帧数据
    double front_imu_ts =imu_buf.front().timestamp;
    while (!feature_buf.empty() && front_imu_ts > front_feature_ts)
    {
        LOG(WARNING) << "throw img, only should happen at the beginning";
        feature_buf.pop();
        front_feature_ts = feature_buf.front().timestamp;
    }
    // 大致将gnss数据和图像数据对齐，实现三者之间的对齐
    if (GNSS_ENABLE)
    {
        front_feature_ts += time_diff_gnss_local; // gnss时间和local本地时间偏差
        double front_gnss_ts = time2sec(gnss_meas_buf.front()[0]->time);
        // gnss不为空，但时间小于图像帧，需要丢弃一部分
        while (!gnss_meas_buf.empty() && front_gnss_ts < front_feature_ts-MAX_GNSS_CAMERA_DELAY) 
        {
            LOG(WARNING) << "throw gnss, only should happen at the beginning";
            gnss_meas_buf.pop();
            if (gnss_meas_buf.empty()) return false;
            front_gnss_ts = time2sec(gnss_meas_buf.front()[0]->time);
        }
        if (gnss_meas_buf.empty())
        {
            LOG(WARNING) << "wait for gnss...";
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
    while (imu_buf.front().timestamp < img_msg.timestamp + estimator_ptr->td)
    {
        imu_msg.emplace_back(imu_buf.front());
        imu_buf.pop();
    }
    imu_msg.emplace_back(imu_buf.front());
    if (imu_msg.empty())
        LOG(WARNING) << "no imu between two image";
    return true;
}
/**
 * @brief imu数据传播过程
 * @details 通过imu数据进行预积分，得到当前时刻的位姿和速度，作为后续里程计的输出
*/
void predict(ImuTimeMsg &imu_msg)
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

// 优化后，更新imu积分结果作为里程计输出
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

    std::queue<ImuTimeMsg> tmp_imu_buf = imu_buf;
    for (ImuTimeMsg tmp_imu_msg; !tmp_imu_buf.empty(); 
        tmp_imu_buf.pop()){
        
        predict(tmp_imu_buf.front());
    }
}

void FeatureCallback(const FeatureTimeMsg feature_msg)
{   
    LOG(INFO) << "I coming feature_callback";
    ++ feature_msg_counter;

    if (skip_parameter < 0 && time_diff_valid)
    {
        const double this_feature_ts = feature_msg.timestamp+time_diff_gnss_local;
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


void ImuCallback(const std::string& imu_path){
    std::ifstream imu_file(imu_path);
    if(!imu_file.is_open()){
        LOG(ERROR) << "Open imu file failed: " << imu_path;
        return;
    }
    std::string line;
    while(std::getline(imu_file, line)){
        std::istringstream iss(line);
        ImuTimeMsg imu_data;
        iss >> imu_data.timestamp;
        for(int i=0; i<3; i++){
            iss >> imu_data.linear_acceleration(i);
        }
        for(int i=0; i<3; i++){
            iss >> imu_data.angular_velocity(i);
        }
        
        std::cout<<std::fixed<<"imu_data.timestamp: "<<imu_data.timestamp<<" last_imu_t: "<<last_imu_t<<std::endl;
        // 检查imu数据时间是否合法
        assert(imu_data.timestamp > last_imu_t);
        if(imu_data.timestamp <= last_imu_t){
            std::cout<<std::fixed<<"imu_data.timestamp: "<<imu_data.timestamp<<" last_imu_t: "<<last_imu_t<<std::endl;
            LOG(ERROR) << "imu data disordered!";
            continue;
        }
        last_imu_t = imu_data.timestamp;
        m_buf.lock();
        imu_buf.push(imu_data);
        m_buf.unlock();
        con.notify_one();

        last_imu_t = imu_data.timestamp;

        {
            std::lock_guard<std::mutex> lg(m_state);
            //TODO
            predict(imu_data);
            // 此处预测完后，发布tmp_P, tmp_Q, tmp_V, timestamp来做可视化
            std::cout<<"tmp_P: "<<tmp_P.transpose()<<std::endl;
            std::cout<<"tmp_Q: "<<tmp_Q.coeffs().transpose()<<std::endl;
            std::cout<<"tmp_V: "<<tmp_V.transpose()<<std::endl; 
        }
        line.clear();
        
    }
    imu_file.close();

}

void imgCallback(const std::vector<std::string>& imgs_path){
    for(int i=0; i<imgs_path.size(); i++){
        ImgMsg img_msg = readImg(imgs_path[i]);
        if(first_image_flag) // 如果为第一张图像
        {
            first_image_flag = false;
            first_image_time = img_msg.timestamp;
            last_image_time = img_msg.timestamp;
            continue;
        }
        // 判断图片是否连续
        if(img_msg.timestamp < last_image_time){ // 此条件是读取文件中图像时，有时会将小时间读在后面
            continue;
        }
        if(img_msg.timestamp - last_image_time > 1.0){
            LOG(ERROR) << "image discontinue! reset the feature tracker";
            std::cout<<std::fixed<<std::setprecision(9) << "img_msg.timestamp: " 
                << img_msg.timestamp << " last_image_time: " << last_image_time;
            first_image_flag = true;
            last_image_time = 0;
            pub_count = 1; 
            // RESTART_FLAG = true; // 重启标志
            continue; 
        }
        last_image_time = img_msg.timestamp;

        // 频率控制
        if(round(1.0 * pub_count / (img_msg.timestamp - first_image_time)) <= FREQ){
            PUB_THIS_FRAME = true;
            // 重置频率控制
            if(abs(1.0 * pub_count / (img_msg.timestamp - first_image_flag) - FREQ) < 0.01 * FREQ){
                first_image_time = img_msg.timestamp;
                pub_count = 0;    
            }
        }
        else{
            PUB_THIS_FRAME = false;
        }

        cv::Mat show_img = img_msg.img;
        TicToc t_r;
        for(int i=0; i<NUM_OF_CAM; i++){
            LOG(INFO) << "feature tracker, processing camera: " << i;
            if(i !=1 || !STEREO_TRACK){
                trackerData[i].readImage(img_msg.img.rowRange(ROW * i, ROW * (i+1)), img_msg.timestamp);
            }
            else{
                if(EQUALIZE){
                    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                    clahe->apply(img_msg.img.rowRange(ROW * i, ROW * (i+1)), trackerData[i].cur_img);
                }
                else{
                    trackerData[i].cur_img = img_msg.img.rowRange(ROW * i, ROW * (i+1));    
                }
            }
#if SHOW_UNDISTORTION
            trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
        }

        for(unsigned int i=0; ; i++){
            bool completed = false;
            for(int j=0; j<NUM_OF_CAM; j++){
                if(i != 1 || !STEREO_TRACK){
                    completed = trackerData[i].updateID(i);
                }
            }
            if(!completed) break;
        }

        if(PUB_THIS_FRAME){
            pub_count++;
            FeatureTimeMsg feature_points;
            
            feature_points.timestamp = img_msg.timestamp;
            for(int i=0; i<NUM_OF_CAM; i++){
                auto& un_pts = trackerData[i].cur_un_pts;
                auto& cur_pts = trackerData[i].cur_pts;
                auto& ids = trackerData[i].ids;
                auto& pts_velocity = trackerData[i].pts_velocity;
                for(unsigned int j=0; j<ids.size(); j++){
                    int p_id = ids[j];
                    Eigen::Vector3d p;
                    p << un_pts[j].x, un_pts[j].y, 1;

                    feature_points.points.push_back(p);
                    feature_points.id_of_point.push_back(p_id * NUM_OF_CAM + i);
                    feature_points.u_of_point.push_back(cur_pts[j].x);
                    feature_points.v_of_point.push_back(cur_pts[j].y);
                    feature_points.velocity_x_of_point.push_back(pts_velocity[j].x);
                    feature_points.velocity_y_of_point.push_back(pts_velocity[j].y);
                }
            }
            LOG(INFO)<< "publish feature points: " << feature_points.timestamp;
            if(!init_pub){
                init_pub = true;
            }
            else{
                FeatureCallback(feature_points);
                /* f_buf.lock();
                feature_buf.push(feature_points);                
                f_buf.unlock(); */
            }
            if(SHOW_TRACK){  // 是否做特征点可视化
                cv::Mat show_img;
                cv::cvtColor(img_msg.img, show_img, cv::COLOR_GRAY2BGR);
                for(int i=0; i<NUM_OF_CAM; i++){
                    for(unsigned int j=0; j<trackerData[i].ids.size(); j++){
                        cv::circle(show_img, trackerData[i].cur_pts[j], 2, cv::Scalar(0, 0, 255), 2);
                    }
                }
                cv::imshow("show_img", show_img);
                cv::waitKey(1);                
            }
        }
    }
}

void process(){
    while(true){
        std::vector<ImuTimeMsg> imu_msg; 
        FeatureTimeMsg img_msg;
        std::vector<gnss_comm::ObsPtr> gnss_msg;

        // unique_lock对象lk独占所有权的方式管理mutex对象m_buf的上锁和解锁
        std::unique_lock<std::mutex> lk(m_buf);

        con.wait(lk, [&]{
            return getMeasurements(imu_msg, img_msg, gnss_msg);
        });
        m_estimator.lock(); // 后端估计上锁
        double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
        // 遍历该组imu_msg中的各帧imu数据进行预积分
        for(auto &imu_data : imu_msg){
            double t = imu_data.timestamp;
            double img_t = img_msg.timestamp + estimator_ptr->td; // 图像帧时间戳，补偿一个通过优化得到的时间偏移
            if(t <= img_t){ // imu时间小于等于图像时间
                if(current_time < 0){ 
                    current_time = t;
                }
                double dt = t - current_time;
                assert(dt >= 0);
                current_time = t;
                dx = imu_data.linear_acceleration[0];
                dy = imu_data.linear_acceleration[1];
                dz = imu_data.linear_acceleration[2];
                rx = imu_data.angular_velocity[0];
                ry = imu_data.angular_velocity[1];
                rz = imu_data.angular_velocity[2];
                estimator_ptr->processIMU(dt, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
            }
            else{ // 对于改组最后一帧imu数据时间大于图像时间，做个简单的线性插值进行预积分
                double dt_1 = img_t - current_time;
                double dt_2 = t - img_t;
                current_time = img_t;
                assert(dt_1 >= 0);
                assert(dt_2 >= 0);
                assert(dt_1 + dt_2 > 0);
                double w1 = dt_2 / (dt_1 + dt_2);
                double w2 = dt_1 / (dt_1 + dt_2);
                dx = w1 * dx + w2 * imu_data.linear_acceleration[0];
                dy = w1 * dy + w2 * imu_data.linear_acceleration[1];
                dz = w1 * dz + w2 * imu_data.linear_acceleration[2];
                rx = w1 * rx + w2 * imu_data.angular_velocity[0];
                ry = w1 * ry + w2 * imu_data.angular_velocity[1];
                rz = w1 * rz + w2 * imu_data.angular_velocity[2];
                estimator_ptr->processIMU(dt_1, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
            }
        }
        // 处理观测数据和星历数据，放入estimator中
        // TODO
        // if (GNSS_ENABLE && !gnss_msg.empty())
            // estimator_ptr->processGNSS(gnss_msg);

        // 处理前端特征跟踪得到的特征点和里程计结果送入后端
        TicToc t_s;
        std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> image;
        for(unsigned int i=0; i<img_msg.points.size(); i++){
            int v = img_msg.id_of_point[i] + 0.5;
            int feature_id = v / NUM_OF_CAM; // 特征点id
            int camera_id = v % NUM_OF_CAM;  // 特征点所在相机
            double x = img_msg.points[i][0];
            double y = img_msg.points[i][1];
            double z = img_msg.points[i][2];
            double p_u = img_msg.u_of_point[i];
            double p_v = img_msg.v_of_point[i];
            double velocity_x = img_msg.velocity_x_of_point[i];
            double velocity_y = img_msg.velocity_y_of_point[i];
            assert(z == 1);
            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
        }
        estimator_ptr->processImage(image, img_msg.timestamp);

        double whole_t = t_s.toc();
        //TODO 打印信息，输出，可视化

        m_estimator.unlock(); // 后端估计解锁
        m_buf.lock();
        m_state.lock();
        if(estimator_ptr->solver_flag == Estimator::SolverFlag::NON_LINEAR){
            update();
        }
        m_state.unlock();
        m_buf.unlock();
    }
}


int main(int argc, char** argv){
    if(argc !=2){
        LOG(ERROR) << "usage: ./${exe} path_to_config_path";
        return -1;
    }
    google::InitGoogleLogging(argv[0]); // 初始化glog日志系统
    FLAGS_minloglevel = google::INFO; // 设置日志级别

    readParameters(argv[1]); // 读取参数
    estimator_ptr.reset(new Estimator); // 初始化估计器
    estimator_ptr->setParameter(); // 设置参数
#ifdef EIGEN_DONT_PARALLELIZE
    LOG(INFO) << "EIGEN_DONT_PARALLELIZE";
#endif
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

    for(int i=0; i<NUM_OF_CAM; i++){
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]); // 读取相机参数
    }

    std::vector<std::string> imgs_path = Data::getImgsPath(IMAGE_FILE);
    // Data::imgCallback(imgs_path, feature_buf, m_buf);
    std::thread img_callback(imgCallback, imgs_path);
    std::thread imu_callback(ImuCallback, IMU_FILE);
    
    std::thread measurement_process{process};



    if(imu_callback.joinable()){
        imu_callback.join();
    }
    if(img_callback.joinable()){
        img_callback.join();
    }
    /* if(measurement_process.joinable()){
        measurement_process.join();
    } */
    return 0;
}

