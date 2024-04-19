#include "estimator.hpp"

Estimator::Estimator() : f_manager(Rs){
    RCUTILS_LOG_INFO("init beings");
    for(int i=0; i<WINDOW_SIZE+1; i++){
        pre_integrations[i] = nullptr; 
    }
    clearState(); // 初始化清零
}

void Estimator::setParameter(){
    for(int i=0; i<NUM_OF_CAM; i++){
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Eigen::Matrix2d::Identity(); // 协方差矩阵
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Eigen::Matrix2d::Identity();
    td =TD;
}

void Estimator::clearState(){
    for(int i=0; i<WINDOW_SIZE+1; i++){
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bgs[i].setZero();
        Bas[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if(pre_integrations[i] != nullptr){
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for(int i=0; i<NUM_OF_CAM; i++){
        tic[i] = Eigen::Vector3d::Zero();
        ric[i] = Eigen::Matrix3d::Identity();
    }

    for(auto &it : all_image_frame){
        if(it.second.per_integration != nullptr){
            delete it.second.per_integration;
            it.second.per_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false;
    sum_of_back = 0; 
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;

    gnss_ready = false;
    anc_ecef.setZero();
    R_ecef_enu.setIdentity();
    para_yaw_enu_local[0] = 0;
    yaw_enu_local = 0;
    sat2ephem.clear();
    sat2time_index.clear();
    sat_track_status.clear();
    latest_gnss_iono_params.clear();
    std::copy(GNSS_IONO_DEFAULT_PARAMS.begin(), GNSS_IONO_DEFAULT_PARAMS.end(),
        std::back_inserter(latest_gnss_iono_params));
    diff_t_gnss_local = 0;

    first_optimization = true;

    if(tmp_pre_integration != nullptr){
        delete tmp_pre_integration;
    }
    if(last_marginalization_info != nullptr){
        delete last_marginalization_info;
    }

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
}

void Estimator::processIMU(double dt, const Eigen::Vector3d &linear_acceleration,
    const Eigen::Vector3d &angular_velocity){

    if(!first_imu){
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity; 
    }  
    
    if(!pre_integrations[frame_count]){
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count],
                                                            Bgs[frame_count]}; 
    }
    if(frame_count !=0){ // 在初始化时，第一帧图像特征点没有对应的预积分
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity); // 预积分
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity); // 指针，在创建ImageFrame对象时赋给imageframe.pre_integration
        // 滑动窗口中每一帧图像对应的预积分用到的IMU数据存在三个缓存中
        dt_buf[frame_count].push_back(dt);  // imu数据对应的时间间隔
        linear_acceleration_buf[frame_count].push_back(linear_acceleration); // 加速度测量值
        angular_velocity_buf[frame_count].push_back(angular_velocity); // 陀螺仪测量值
        // 根据预积分结果计算Rs、Ps、Vs
        int j = frame_count;
        Eigen::Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Eigen::Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    } 
    // 更新
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity; 
}

void Estimator::processImage(const std::map<int, std::vector<std::pair<int, 
    Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::msg::Header &header){
    RCUTILS_LOG_DEBUG("new image coming -----------------------------------");
    RCUTILS_LOG_DEBUG("Adding feature points %lu", image.size());
    if(f_manager.addFeatureCheckParallax(frame_count, image, td)){
        marginalization_flag = MARGIN_OLD;
    } 
    else{
        marginalization_flag = MARGIN_SECOND_NEW;
    }

    RCUTILS_LOG_DEBUG("this frame is ------------%s", marginalization_flag ? "reject" : "accept");
    RCUTILS_LOG_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    RCUTILS_LOG_DEBUG("Solving %d", frame_count);
    RCUTILS_LOG_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    ImageFrame imageframe(image, stamp2Sec(header.stamp));
    imageframe.per_integration = tmp_pre_integration;
    all_image_frame.insert(std::make_pair(stamp2Sec(header.stamp), imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    if(ESTIMATE_EXTRINSIC == 2){
        RCUTILS_LOG_INFO("calibrating extrinsic param, rotation movement is needed");
        if(frame_count != 0){
            std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres = 
            f_manager.getCorresponding(frame_count-1, frame_count);
            Eigen::Matrix3d calib_ric;
            if(initial_ex_rotation.CalibrationExRotation(corres, 
                pre_integrations[frame_count]->delta_q, calib_ric));{

                RCUTILS_LOG_WARN("initial extrinsic rotation calib success");
                RCLCPP_WARN_STREAM(rclcpp::get_logger("estimator"), "initial extrinsic rotation: "<<
                    std::endl<<calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    if(solver_flag == INITIAL){
        if(frame_count == WINDOW_SIZE){
            bool result = false;
            if(ESTIMATE_EXTRINSIC != 2 && (stamp2Sec(header.stamp)-initial_timestamp) > 0.1){
                result = initialStructure();
                initial_timestamp = stamp2Sec(header.stamp);
            }
            if(result){
                solver_flag = NON_LINEAR;
                solveOdometry();
                slideWindow();
                f_manager.removeFailures();
                RCUTILS_LOG_INFO("Initialization finish!");
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
            }
            else{
                slideWindow();
            }
        }
        else{
            frame_count++;
        }
    }
    else{
        TicToc t_solve;
        solveOdometry();
        RCUTILS_LOG_DEBUG("solver cost: %f ms", t_solve.toc());  

        if(failureDetection()){
            RCUTILS_LOG_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            RCUTILS_LOG_WARN("system reboot!");
            return;
        } 

        TicToc t_margin;
        slideWindow();
        f_manager.removeFailures();
        RCUTILS_LOG_DEBUG("marginalization costs: %fms", t_margin.toc());
        key_poses.clear();               
    }
}
