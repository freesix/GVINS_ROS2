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

/**
 * @brief 处理接受的图像特征点数据，在线标定和初始化VIO，滑动窗口满了才开始初始化，初始化成
 * 功紧接着进行耦合优化
 * @param image 输入为一帧图像的特征点，多目为多帧
 * @param header 时间戳
*/
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

    if(solver_flag == INITIAL){ // 需要进行初始化
        if(frame_count == WINDOW_SIZE){ // 滑动窗口塞满了才进行初始化
            bool result = false;
            if(ESTIMATE_EXTRINSIC != 2 && (stamp2Sec(header.stamp)-initial_timestamp) > 0.1){ // 在上一次初始化失败后至少0.1秒才进行下一次初始化
                result = initialStructure();
                initial_timestamp = stamp2Sec(header.stamp);
            }
            if(result){ // 初始化成功
                solver_flag = NON_LINEAR;
                solveOdometry(); // 紧耦合优化
                slideWindow(); // 滑动窗口
                f_manager.removeFailures(); // 去除不在窗内的特征点
                RCUTILS_LOG_INFO("Initialization finish!");
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
            }
            else{
                slideWindow(); // 初始化不成功，对窗口进行滑动
            }
        }
        else{
            frame_count++; // 滑动窗口没满，接着塞
        }
    }
    else{ // 已经初始化，进行正常的VIO紧耦合
        TicToc t_solve;
        solveOdometry(); // 优化
        RCUTILS_LOG_DEBUG("solver cost: %f ms", t_solve.toc());  // 优化的时间
        // 失效检测，如果失效则重启VINS系统
        if(failureDetection()){
            RCUTILS_LOG_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            RCUTILS_LOG_WARN("system reboot!");
            return;
        } 
        // 对窗口进行滑动
        TicToc t_margin;
        slideWindow();
        f_manager.removeFailures();
        RCUTILS_LOG_DEBUG("marginalization costs: %fms", t_margin.toc());
        key_poses.clear();  
        for(int i=0; i<=WINDOW_SIZE; i++){
            key_poses.push_back(Ps[i]);
        }             

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}

/**
 * @brief 星历信息的输入
*/
void Estimator::inputEphem(EphemBasePtr ephem_ptr){
    double toe = time2sec(ephem_ptr->toe);
    // if a new ephemeris comes
    if(sat2time_index.count(ephem_ptr->sat) == 0 || sat2time_index.at(ephem_ptr->sat).count(toe) == 0){
        sat2ephem[ephem_ptr->sat].emplace_back(ephem_ptr);
        sat2time_index[ephem_ptr->sat].emplace(toe, sat2ephem.at(ephem_ptr->sat).size()-1);
    }
}

void Estimator::inputIonoParams(double ts, const std::vector<double> &iono_params){
    if(iono_params.size() != 8) return ;

    // update ionospahere parameters
    latest_gnss_iono_params.clear();
    std::copy(iono_params.begin(), iono_params.end(), std::back_inserter(latest_gnss_iono_params));
}

void Estimator::inputGNSSTimeDiff(const double t_diff){
    diff_t_gnss_local = t_diff;
}

/**
 * @brief 根据一定的条件，获取可用的GNSS观测值数据
*/
void Estimator::processGNSS(const std::vector<ObsPtr> &gnss_meas){
    std::vector<ObsPtr> valid_meas; // 有用的观测数据
    std::vector<EphemBasePtr> valid_ephems; // 有用的星历数据
    // 遍历
    for(auto obs : gnss_meas){
        // filter according to system
        uint32_t sys = satsys(obs->sat, NULL);
        if(sys != SYS_GPS && sys != SYS_GLO && sys != SYS_GAL && sys != SYS_BDS){
            continue;
        }
        // if not got cooresponding ephemeris yet
        if(sat2ephem.count(obs->sat) == 0){ // 确定观测到的卫星数量
            continue;
        }
        // 如果观测数据中频率值不为空，选择L1频率，并确定相应的观测值和星历信息
        if(obs->freqs.empty()){ // no valid signal measurement
            continue;  
        }
        int freq_idx = -1;
        L1_freq(obs, &freq_idx); // 获取导航系统的L1频率
        if(freq_idx < 0){ // no L1 observation
            continue;
        }
        // 根据星历参数时间to3，判断当前星历参数是否有效
        // 一般认为当前的GPS时间在星历参考时间前后2小时内(7200秒)有效
        double obs_time = time2sec(obs->time);
        std::map<double, size_t> time2indx = sat2time_index.at(obs->sat);
        double ephem_time = EPH_VALID_SECONDS;
        size_t ephem_index = -1;
        for(auto ti : time2indx){
            if(std::abs(ti.first-obs_time) < ephem_time){
                ephem_time = std::abs(ti.first-obs_time);
                ephem_index = ti.second;
            }
        }
        if(ephem_time >= EPH_VALID_SECONDS){ // 超过7200
            std::cerr << "ephemeris not valid anymore\n";
            continue;
        }
        const EphemBasePtr &best_ephem = sat2ephem.at(obs->sat).at(ephem_index);

        // filter by tracking status
        // 根据卫星的跟踪状态，伪距、多普勒测量的标准差，判断数据的合法性
        LOG_IF(FATAL, freq_idx < 0) << "No L1 observation found.\n";
        // pst_std:伪距标准差
        // dopp_std:多普勒标准差
        // 如果伪距和多普勒测量中任一数据的标准差超过了规定数值，则认为当前值不行，将当前测量值下的卫星被跟踪次数清零
        if(obs->psr_std[freq_idx] > GNSS_PSR_STD_THRES || obs->dopp_std[freq_idx] > GNSS_DOPP_STD_THRES){
            sat_track_status[obs->sat] = 0;
            continue;
        }
        else{
            if(sat_track_status.count(obs->sat) == 0){
                sat_track_status[obs->sat] = 0;
            }
            ++ sat_track_status[obs->sat]; // 跟踪次数
        }
        if(sat_track_status[obs->sat] < GNSS_TRACK_NUM_THRES){
            continue;
        }
        // filter by elevation angle
        // 根据观测卫星的仰角
        if(gnss_ready){  // 当GNSSVIAlign()对齐后，gnss_ready为true
            Eigen::Vector3d sat_ecef;
            if(sys == SYS_GLO){
                sat_ecef = geph2pos(obs->time, std::dynamic_pointer_cast<GloEphem>(best_ephem), NULL);
            }
            else{
                sat_ecef = eph2pos(obs->time, std::dynamic_pointer_cast<Ephem>(best_ephem), NULL);
            }
            double azel[2] = {0, M_PI/2.0};
            sat_azel(ecef_pos, sat_ecef, azel); // 通过enu坐标系下，接收机的位置和卫星位置的向量差，求出仰角
            if(azel[1] < GNSS_ELEVATION_THRES*M_PI/180.0){ // GNSS_ELEVATION_THRES=30，要求仰角不能小于30度
                continue;
            }
        }
        valid_meas.push_back(obs);
        valid_ephems.push_back(best_ephem);
    }
    // 将好的卫星观测值和星历数据放入全局变量
    gnss_meas_buf[frame_count] = valid_meas;
    gnss_ephem_buf[frame_count] = valid_ephems;
}

/**
 * @brief VINS系统初始化
 * 1、确保IMU有足够的excitation
 * 2、检查当前帧(滑动窗口中的最新帧)与滑动窗口中所有图像帧之间的特征点匹配关系
 *    选择跟当前帧中有足够多数量的特征点(30个)被跟踪，且由足够视差(20 pixels)的某一帧，利用五点法恢复相对旋转和平移量
 *    如果找不到，则在滑动窗口中保留当前帧，然后等待新的图像帧
 * 3、sfm.construct 全局SFM 恢复滑动窗口中所有帧的位姿，以及特征点三角化
 * 4、利用pnp恢复其它帧
 * 5、视觉SFM结果和IMU预积分结果对齐
 * 6、给滑动窗口中要优化的变量一个合理的初始值以便进行非线性优化
*/
bool Estimator::initialStructure(){
    TicToc t_sfm;
    // check imu observation
    // 1、通过重力variance确保IMU有足够的激发
    {
        std::map<double, ImageFrame>::iterator frame_it;
        Eigen::Vector3d sum_g;
        for(frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++){ // 从第二个元素开始
            double dt = frame_it->second.per_integration->sum_dt;
            Eigen::Vector3d tmp_g = frame_it->second.per_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Eigen::Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0; 
        for(frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++){
            double dt = frame_it->second.per_integration->sum_dt;
            Eigen::Vector3d tmp_g = frame_it->second.per_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        if(var < 0.25){
            RCUTILS_LOG_INFO("IMU excitation not enough!");
        }
    } 
    // global sfm
    Eigen::Quaterniond Q[frame_count + 1]; // 滑动窗口中每一帧的姿态
    Eigen::Vector3d T[frame_count + 1]; // 滑动窗口中每一帧的位置
    std::map<int, Eigen::Vector3d> sfm_tracked_points;

    std::vector<SFMFeature> sfm_f;  // 用于视觉初始化的图像特征数据
    for(auto &it_per_id : f_manager.feature){
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false; // 该特征点的初始化状态为，未被三角化
        tmp_feature.id = it_per_id.feature_id;
        for(auto &it_per_frame : it_per_id.feature_per_frame){
            imu_j ++; // 观测到该特征点的图像帧号
            Eigen::Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(std::make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }
    Eigen::Matrix3d relative_R;
    Eigen::Vector3d relative_T;
    int l;
    if(!relativePose(relative_R, relative_T, l)){
        RCUTILS_LOG_INFO("Not enough features or parallax; Move divece around");
        return false;
    }
    // 初始化滑动窗口中全部初始帧的相机位姿和特征点的空间3d位置
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l, relative_R, relative_T, sfm_f, sfm_tracked_points)){
        RCUTILS_LOG_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    // solve pnp for all frame
    // 由于初始化并不是一次就能成功，因此图像帧数量可能会超过滑动窗口的大小
    // 所以在视觉初始化的最后，要求出滑动窗口为帧的位姿
    // 最后把世界坐标系从帧1的相机坐标系，转换到帧1的IMU坐标系
    // 对于非滑动窗口的所有帧，提供一个初始的R、T，然后solve pnp求解pose
    std::map<double, ImageFrame>::iterator frame_it;
    std::map<int, Eigen::Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    for(int i=0; frame_it != all_image_frame.end(); frame_count++){
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == stamp2Sec(Headers[i].stamp)){ // all_image_frame与滑动窗口中对应的帧
            frame_it->second.is_key_frame = true;  // 滑动窗口所有帧都是关键帧
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose(); // 根据各帧相机坐标系的姿态和外参，得到imu坐标系下的位姿
            frame_it->second.T = T[i];
            i++;
            continue;  
        }
        if((frame_it->first) > stamp2Sec(Headers[i].stamp)){
            i++;
        }
        // 为滑动窗口外的帧提供一个初始位姿
        Eigen::Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Eigen::Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false; // 初始化位于窗口外的帧为非关键帧
        std::vector<cv::Point3f> pts_3_vector; // 用于pnp解算的3D点
        std::vector<cv::Point2f> pts_2_vector; // 用于pnp结算的2D点
        for (auto &id_pts : frame_it->second.points) // 对于该帧中的特征点
        {
            int feature_id = id_pts.first;  // 特征点id
            for (auto &i_p : id_pts.second) // 由于可能有多个相机，所以需要遍历
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end()) // 不是尾部迭代器，说明在sfm_tracked_points中找到了
                {   // 记录该特征点的3D位置
                    Eigen::Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    // 记录该id在该帧图像中的2D位置
                    Eigen::Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6) // 匹配到的3d点少于6个，则初始化失败
        {
            std::cout << "pts_3_vector size " << pts_3_vector.size() << std::endl;
            RCUTILS_LOG_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1)) // pnp 求解失败
        {
            RCUTILS_LOG_DEBUG("solve pnp fail!");
            return false;
        }
        // 求解成功
        cv::Rodrigues(rvec, r);
        Eigen::MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        Eigen::MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose(); // 根据各帧相机坐标的姿态和外参，得到各帧在imu坐标系的姿态
        frame_it->second.T = T_pnp;
    }
    // camera和imu对齐
    if (!visualInitialAlign())
    {
        RCUTILS_LOG_WARN("misalign visual structure with IMU");
        return false;
    }
    return true;
}

/**
* @brief 视觉SFM的结果和IMU预积分结果对齐
*/
bool Estimator::visualInitialAlign(){
    TicToc t_g;
    Eigen::VectorXd x;
    // solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x); // SFM和IMU预积分结果对齐
    if(!result){
        RCUTILS_LOG_DEBUG("solve g failed!");
        return false;
    }
    // change state
    // 滑动窗口中各帧在世界坐标系下的旋转和平移
    for(int i=0; i<=frame_count; i++){
        Eigen::Matrix3d Ri = all_image_frame[stamp2Sec(Headers[i].stamp)].R;
        Eigen::Vector3d Pi = all_image_frame[stamp2Sec(Headers[i].stamp)].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[stamp2Sec(Headers[i].stamp)].is_key_frame = true; // 滑动窗口中所有初始帧都是关键帧
    }

    Eigen::VectorXd dep = f_manager.getDepthVector();
    for(int i=0; i<dep.size(); i++){
        dep[i] = -1;
    }
    f_manager.clearDepth(dep);

    // triangulate on cam pose, no tic
    Eigen::Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i=0; i<NUM_OF_CAM; i++){
        TIC_TMP[i].setZero();
    }
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);
    for(int i=0; i<=WINDOW_SIZE; i++){
        pre_integrations[i]->repropagate(Eigen::Vector3d::Zero(), Bgs[i]);
    }
    for(int i=frame_count; i>=0; i--){
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);  
    }
    int kv = -1;
    std::map<double, ImageFrame>::iterator frame_i;
    for(frame_i=all_image_frame.begin(); frame_i!=all_image_frame.end(); frame_i++){
        if(frame_i->second.is_key_frame){
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv*3);
        }
    }
    for(auto &it_per_id : f_manager.feature){
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if(!(it_per_id.used_num>=2 && it_per_id.start_frame<WINDOW_SIZE-2)){
            continue;
        }
        it_per_id.estimated_depth *= s;
    }

    Eigen::Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;

    Eigen::Matrix3d rot_diff = R0;
    for(int i=0; i<=frame_count; i++){
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];  
    }

    RCLCPP_DEBUG_STREAM(rclcpp::get_logger("estimator"), "g0   "<<g.transpose());
    RCLCPP_DEBUG_STREAM(rclcpp::get_logger("estimator"), "my R0  "<< Utility::R2ypr(Rs[0]).transpose());

    return true;
}

/**
 * @brief GNSS和VIO进行联合优化，在VIO初始化完成后
*/
bool Estimator::GNSSVIAlign(){
    if(solver_flag == INITIAL){ // visual-inertial no initialzed
        return false;
    }
    if(gnss_ready){ // GNSS-VI already initialized
        return true;
    }
    // GNSS的观测数据为空或者数量小于10，退出
    for(uint32_t i=0; i<(WINDOW_SIZE+1); ++i){
        if(gnss_meas_buf[i].empty() || gnss_meas_buf[i].size()<10){
            return false;
        }
    }

    // check horizontal velocity excitation
    // 检测水平方向的速度激励
    Eigen::Vector2d avg_hor_vel(0.0, 0.0);
    for(uint32_t i=0; i<(WINDOW_SIZE+1); ++i){
        avg_hor_vel += Vs[i].head<2>().cwiseAbs();
    }
    avg_hor_vel /= (WINDOW_SIZE+1);
    if(avg_hor_vel.norm() < 0.3){
        std::cerr<<"velocity excitation not enough for GNSS-VI aligment.\n";
        return false;
    }
    // 获取当前窗口内的GNSS观测数据和星历数据
    std::vector<std::vector<ObsPtr>> curr_gnss_meas_buf;
    std::vector<std::vector<EphemBasePtr>> curr_gnss_ephem_buf;
    for(uint32_t i=0; i<(WINDOW_SIZE+1); ++i){
        curr_gnss_meas_buf.push_back(gnss_meas_buf[i]);
        curr_gnss_ephem_buf.push_back(gnss_ephem_buf[i]);
    }

    GNSSVIInitializer gnss_vi_initializer(curr_gnss_meas_buf, curr_gnss_ephem_buf,
        latest_gnss_iono_params);

    // 1、get a rough global location
    // 得到接收机在ecef下的粗糙伪距位置和导航系统的时钟差
    Eigen::Matrix<double, 7, 1> rough_xyzt;
    rough_xyzt.setZero();
    if(!gnss_vi_initializer.coarse_localization(rough_xyzt)){
        std::cerr<<"Fail to obtain a coarse location.\n";
        return false;
    }

    // 2、perform yaw alignment
    // local world frame和enu的对齐，即求解yaw
    std::vector<Eigen::Vector3d> local_vs; // local world frame钟body的速度
    for(uint32_t i=0; i<(WINDOW_SIZE+1); ++i){
        local_vs.push_back(Vs[i]);
    }
    Eigen::Vector3d rough_anchor_ecef = rough_xyzt.head<3>(); // 粗糙位置
    double aligned_yaw = 0; // 待矫正的yaw角
    double aligned_rcv_ddt = 0; // 接收机时钟钟差变化率
    if(!gnss_vi_initializer.yaw_alignment(local_vs, rough_anchor_ecef, aligned_yaw, aligned_rcv_ddt)){
        std::cerr<<"Fail to align ENU and local frames.\n";
        return false;
    }

    // 3、perform anchor refinement
    // 锚点位置优化
    std::vector<Eigen::Vector3d> local_ps;
    for(uint32_t i=0; i<(WINDOW_SIZE+1); ++i){
        local_ps.push_back(Ps[i]);
    }
    Eigen::Matrix<double, 7, 1> refined_xyzt;
    refined_xyzt.setZero();
    if(!gnss_vi_initializer.anchor_refinement(local_ps, aligned_yaw,
        aligned_rcv_ddt, rough_xyzt, refined_xyzt)){
        std::cerr<<"Fail to refine anchor point.\n";
        return false;
    }

    // restore GNSS states
    uint32_t one_observed_sys = static_cast<uint32_t>(-1);
    for (uint32_t k = 0; k < 4; ++k)
    {
        if (rough_xyzt(k+3) != 0)
        {
            one_observed_sys = k;
            break;
        }
    }
    for (uint32_t i = 0; i < (WINDOW_SIZE+1); ++i)
    {
        para_rcv_ddt[i] = aligned_rcv_ddt;
        for (uint32_t k = 0; k < 4; ++k)
        {
            if (rough_xyzt(k+3) == 0)
                para_rcv_dt[i*4+k] = refined_xyzt(3+one_observed_sys) + aligned_rcv_ddt * i;
            else
                para_rcv_dt[i*4+k] = refined_xyzt(3+k) + aligned_rcv_ddt * i;
        }
    }
    anc_ecef = refined_xyzt.head<3>();
    R_ecef_enu = ecef2rotation(anc_ecef);

    yaw_enu_local = aligned_yaw;

    return true;
}

void Estimator::updateGNSSStatistics()
{
    R_enu_local = Eigen::AngleAxisd(yaw_enu_local, Eigen::Vector3d::UnitZ());
    enu_pos = R_enu_local * Ps[WINDOW_SIZE];
    enu_vel = R_enu_local * Vs[WINDOW_SIZE];
    enu_ypr = Utility::R2ypr(R_enu_local*Rs[WINDOW_SIZE]);
    ecef_pos = anc_ecef + R_ecef_enu * enu_pos;
}

/**
 * @brief 在滑动窗口中，寻找与最新帧有足够多数量的特征点对应关系和视差的帧，然后用5点法恢复相对位姿
*/
bool Estimator::relativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l){
    // find previous frame which contians enough correspondance and parallex with newest frame
    for(int i=0; i<WINDOW_SIZE; i++){
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if(corres.size() > 20){
            double sum_parallax = 0;
            double average_parallax;
            for(int j=0; j<int(corres.size()); j++){
                Eigen::Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Eigen::Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallex = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallex;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax*460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T)){
                l = i;
                RCUTILS_LOG_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }    
    return false;
}
/**
 * @brief 紧耦合优化
*/
void Estimator::solveOdometry(){
    if(frame_count < WINDOW_SIZE){
        return;
    }
    if(solver_flag == NON_LINEAR){
        TicToc t_tri;
        f_manager.triangulate(Ps, tic, ric);
        RCUTILS_LOG_DEBUG("triangulation costs %f", t_tri.toc());
        optimization();
        if(GNSS_ENABLE){
            if(!gnss_ready){
                gnss_ready = GNSSVIAlign();
            }
            if(gnss_ready){
                updateGNSSStatistics();
            }
        }
    }
}

/**
 * @brief ceres中变量必须用数组类型，因此此处将用于边缘化操作的变量转为数组类型
 * 由此可得参与优化和边缘化的变量有(相机位姿6维)(相机速度、加速度偏置、角速度偏置9维)
 * (相机IMU外参6维)(特征点深度1维)(时间偏差1维)(GNSS地心地固坐标位置3维)
*/
void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Eigen::Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Eigen::Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    Eigen::VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
    
    para_yaw_enu_local[0] = yaw_enu_local;
    for (uint32_t k = 0; k < 3; ++k)
        para_anc_ecef[k] = anc_ecef(k);
}

void Estimator::double2vector()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = Eigen::Quaterniond(para_Pose[i][6], para_Pose[i][3], 
                            para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        
        Ps[i] = Eigen::Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);

        Vs[i] = Eigen::Vector3d(para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2]);

        Bas[i] = Eigen::Vector3d(para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5]);

        Bgs[i] = Eigen::Vector3d(para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Eigen::Vector3d(para_Ex_Pose[i][0], para_Ex_Pose[i][1], para_Ex_Pose[i][2]);
        ric[i] = Eigen::Quaterniond(para_Ex_Pose[i][6], para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4], para_Ex_Pose[i][5]).normalized().toRotationMatrix();
    }

    Eigen::VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];
    
    if (gnss_ready)
    {
        yaw_enu_local = para_yaw_enu_local[0];
        for (uint32_t k = 0; k < 3; ++k)
            anc_ecef(k) = para_anc_ecef[k];
        R_ecef_enu = ecef2rotation(anc_ecef);
    }
}

bool Estimator::failureDetection()
{
    if (f_manager.last_track_num < 2)
    {
        RCUTILS_LOG_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        RCUTILS_LOG_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        RCUTILS_LOG_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Eigen::Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        RCUTILS_LOG_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        RCUTILS_LOG_INFO(" big z translation");
        return true; 
    }
    Eigen::Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Eigen::Matrix3d delta_R = tmp_R.transpose() * last_R;
    Eigen::Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        RCUTILS_LOG_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

/**
 * @brief 基于滑动窗口的紧耦合非线性优化
 * 1、添加要优化的变量
*/
void Estimator::optimization(){
    ceres::Problem problem;
    ceres::LossFunction *loss_function; // 核函数
    loss_function = new ceres::CauchyLoss(1.0); // 当预测偏差小于δ时用平方误差，大于时用线性误差
    for(int i=0; i<WINDOW_SIZE+1; i++){
        // 对于四元数或旋转矩阵需要定义加法，让其在ceres迭代更新时候用此类加法
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        // AddParameterBlock 向该问题添加具有适当大小和参数化的参数块
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    for(int i=0; i<NUM_OF_CAM; i++){
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization); // 外参
        if(!ESTIMATE_EXTRINSIC){
            RCUTILS_LOG_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else{
            RCUTILS_LOG_DEBUG("estimate extinsic param");
        }
    }
    if(ESTIMATE_TD){
        problem.AddParameterBlock(para_Td[0], 1); // 时间
    }
    if(gnss_ready){
        problem.AddParameterBlock(para_yaw_enu_local, 1); // 偏航角
        // 计算水平面平均速度
        Eigen::Vector2d avg_hor_vel(0.0, 0.0);
        for(uint32_t i=0; i<=WINDOW_SIZE; ++i){
            avg_hor_vel += Vs[i].head<2>().cwiseAbs();
        }
        avg_hor_vel /= (WINDOW_SIZE+1);
        if(avg_hor_vel.norm() < 0.3){ // 水平面平均速度激励是否足够
            // std::cerr<<"velocity excitation not enough, fix yaw angle.\n";
            problem.SetParameterBlockConstant(para_yaw_enu_local);
        }
        for(uint32_t i=0; i<=WINDOW_SIZE; ++i){
            if(gnss_meas_buf[i].size() < 10){ // gnss观测数据是否足够
                problem.SetParameterBlockConstant(para_yaw_enu_local); // 固定参数不被优化
            }
        }

        problem.AddParameterBlock(para_anc_ecef, 3); // anchor point 位置

        for(uint32_t i=0; i<=WINDOW_SIZE; ++i){
            for(uint32_t k=0; k<4; ++k){
                problem.AddParameterBlock(para_rcv_dt+i*4+k, 1); // 钟差
            }
            problem.AddParameterBlock(para_rcv_ddt+i, 1); // 钟差变化率
        }
    }

    TicToc t_whole, t_prepare;
    vector2double();

    if(first_optimization){
        std::vector<double> anchor_value;
        for(uint32_t k=0; k<7; ++k){
            anchor_value.push_back(para_Pose[0][k]);
        }
        PoseAnchorFactor *pose_anchor_factor = new PoseAnchorFactor(anchor_value);
        problem.AddResidualBlock(pose_anchor_factor, NULL, para_Pose[0]);
        first_optimization = false;
    }

    // ----------- 在问题中添加约束，构造残差函数 ------------- //
    if(last_marginalization_info){
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(
                last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL, last_marginalization_parameter_blocks);
    }
    for(int i=0; i<WINDOW_SIZE; i++){
        int j = i+1;
        if(pre_integrations[j]->sum_dt > 10.0){
            continue;
        }
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i],
            para_Pose[j], para_SpeedBias[j]);
    }

    if(gnss_ready){
        for(int i=0; i<=WINDOW_SIZE; ++i){ // gnss约束和imu约束一样，帧与帧之间形成约束
            const std::vector<ObsPtr> &curr_obs = gnss_meas_buf[i]; // gnss观测数据，前端processGNSS函数处理后，存储的vaild data
            const std::vector<EphemBasePtr> &curr_ephem = gnss_ephem_buf[i];

            for(uint32_t j=0; j<curr_obs.size(); ++j){
                const uint32_t sys = satsys(curr_obs[j]->sat, NULL);
                const uint32_t sys_idx = gnss_comm::sys2idx.at(sys);

                int lower_idx = -1;
                const double obs_local_ts = time2sec(curr_obs[j]->time) - diff_t_gnss_local;
                if(stamp2Sec(Headers[i].stamp) > obs_local_ts){
                    lower_idx = (i==0 ? 0 : i-1);
                }
                else{
                    lower_idx = (i==WINDOW_SIZE ? WINDOW_SIZE-1 : i); 
                }
                const double lower_ts = stamp2Sec(Headers[lower_idx].stamp);
                const double upper_ts = stamp2Sec(Headers[lower_idx+1].stamp);

                const double ts_ratio = (upper_ts-obs_local_ts) / (upper_ts-lower_ts);
                GnssPsrDoppFactor * gnss_factor = new GnssPsrDoppFactor(curr_obs[j],
                    curr_ephem[j], latest_gnss_iono_params, ts_ratio);
                problem.AddResidualBlock(gnss_factor, NULL, para_Pose[lower_idx],
                    para_SpeedBias[lower_idx], para_Pose[lower_idx+1], 
                    para_SpeedBias[lower_idx+1], para_rcv_dt+i*4+sys_idx, para_rcv_ddt+i,
                    para_yaw_enu_local, para_anc_ecef);
            }
        }

        // build realationship between rcv_dt and rcv_ddt
        for(size_t k=0; k<4; ++k){
            for(uint32_t i=0; i<WINDOW_SIZE; ++i){
                const double gnss_dt = stamp2Sec(Headers[i+1].stamp) - stamp2Sec(Headers[i].stamp);
                DtDdtFactor *dt_ddt_factor = new DtDdtFactor(gnss_dt);
                problem.AddResidualBlock(dt_ddt_factor, NULL, para_rcv_ddt+i*4+k,
                    para_rcv_dt+(i+1)*4+k, para_rcv_ddt+i, para_rcv_ddt+i+1);
            }
        }

        // add rcv_ddt smooth factor
        for(int i=0; i<WINDOW_SIZE; ++i){
            DdtSmoothFactor *ddt_smooth_factor = new DdtSmoothFactor(GNSS_DDT_WEIGHT);
            problem.AddResidualBlock(ddt_smooth_factor, NULL, para_rcv_ddt+i, para_rcv_ddt+i+1);
        }
    }

    int f_m_cnt = 0; // 每个特征点，观测到它得相机计数
    int feature_index = -1;
    for(auto &it_per_id : f_manager.feature){
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if(!(it_per_id.used_num>=2 && it_per_id.start_frame<WINDOW_SIZE-2)){
            continue;
        }
        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i+1;
        Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for(auto &it_per_frame : it_per_id.feature_per_frame){
            imu_j++;
            if(imu_i == imu_j){
                continue;
            }
            Eigen::Vector3d pts_j = it_per_frame.point;
            if(ESTIMATE_TD){
                  ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, 
                        it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                        it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], 
                    para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], 
                    para_Td[0]); 
            }
            else
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
    }

    RCUTILS_LOG_DEBUG("visual measurement count: %d", f_m_cnt);
    RCUTILS_LOG_DEBUG("prepare for cers: %f", t_prepare.toc());

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    // options.use_explicit_schur_complement = true;
    // options.minimizer_progress_to_stdout = true;
    options.use_nonmonotonic_steps = true;
    if(marginalization_flag = MARGIN_OLD){
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    }
    else{
        options.max_solver_time_in_seconds = SOLVER_TIME;
    }
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    RCUTILS_LOG_DEBUG("Iterations: %d", static_cast<int>(summary.iterations.size()));
    RCUTILS_LOG_DEBUG("solver costs: %f", t_solver.toc());

    while(para_yaw_enu_local[0] > M_PI){
        para_yaw_enu_local[0] -= 2.0*M_PI;
    }
    while(para_yaw_enu_local[0] < -M_PI){
        para_yaw_enu_local[0] += 2.0*M_PI;
    }

    double2vector();
    // ------------- marginalization ------------- //
    TicToc t_whole_marginalization;
    if(marginalization_flag == MARGIN_OLD){
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();
        if(last_marginalization_info){
            std::vector<int> drop_set;
            for(int i=0; i<static_cast<int>(last_marginalization_parameter_blocks.size()); i++){
                if(last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                   last_marginalization_parameter_blocks[i] == para_SpeedBias[0]){
                    drop_set.push_back(i);   
                }
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(
                last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                marginalization_factor, NULL, last_marginalization_parameter_blocks, drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }
        else{
            std::vector<double> anchor_value;
            for(uint32_t k=0; k<7; ++k){
                anchor_value.push_back(para_Pose[0][k]);
            }
            PoseAnchorFactor *pose_anchor_factor = new PoseAnchorFactor(anchor_value);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(pose_anchor_factor,
                NULL, std::vector<double *>{para_Pose[0]}, std::vector<int>{0});
            marginalization_info->addResidualBlockInfo(residual_block_info); 
        }
        {
            if(pre_integrations[1]->sum_dt < 10.0){
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                    std::vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], 
                    para_SpeedBias[1]}, std::vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        if(gnss_ready){
            for(uint32_t j=0; j<gnss_meas_buf[0].size(); ++j){
                const uint32_t sys = satsys(gnss_meas_buf[0][j]->sat, NULL);
                const uint32_t sys_idx = gnss_comm::sys2idx.at(sys);

                const double obs_local_ts = time2sec(gnss_meas_buf[0][j]->time) - diff_t_gnss_local;
                const double lower_ts = stamp2Sec(Headers[0].stamp);
                const double upper_ts = stamp2Sec(Headers[1].stamp);
                const double ts_ratio = (upper_ts-obs_local_ts) / (upper_ts-lower_ts);

                GnssPsrDoppFactor *gnss_factor = new GnssPsrDoppFactor(gnss_meas_buf[0][j],
                    gnss_ephem_buf[0][j], latest_gnss_iono_params, ts_ratio);
                ResidualBlockInfo *psr_dopp_residual_block_info = new ResidualBlockInfo(gnss_factor,
                    NULL, std::vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1],
                    para_yaw_enu_local, para_anc_ecef}, std::vector<int>{0, 1, 4, 5});
                marginalization_info->addResidualBlockInfo(psr_dopp_residual_block_info);
            }

            const double gnss_dt = stamp2Sec(Headers[1].stamp) - stamp2Sec(Headers[0].stamp);
            for (size_t k = 0; k < 4; ++k)
            {
                DtDdtFactor *dt_ddt_factor = new DtDdtFactor(gnss_dt);
                ResidualBlockInfo *dt_ddt_residual_block_info = new ResidualBlockInfo(dt_ddt_factor, 
                    NULL, std::vector<double *>{para_rcv_dt+k, para_rcv_dt+4+k, 
                    para_rcv_ddt, para_rcv_ddt+1}, std::vector<int>{0, 2});
                marginalization_info->addResidualBlockInfo(dt_ddt_residual_block_info);
            }

            // margin rcv_ddt smooth factor
            DdtSmoothFactor *ddt_smooth_factor = new DdtSmoothFactor(GNSS_DDT_WEIGHT);
            ResidualBlockInfo *ddt_smooth_residual_block_info = new ResidualBlockInfo(ddt_smooth_factor, 
                NULL, std::vector<double *>{para_rcv_ddt, para_rcv_ddt+1}, 
                std::vector<int>{0});
            marginalization_info->addResidualBlockInfo(ddt_smooth_residual_block_info);
        }

        
        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Eigen::Vector3d pts_j = it_per_frame.point;
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, 
                            it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                            it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, 
                            loss_function, std::vector<double *>{para_Pose[imu_i], para_Pose[imu_j], 
                                para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                            std::vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, 
                            loss_function, std::vector<double *>{para_Pose[imu_i], para_Pose[imu_j], 
                                para_Ex_Pose[0], para_Feature[feature_index]},
                            std::vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        RCUTILS_LOG_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

        TicToc t_margin;
        marginalization_info->marginalize();
        RCUTILS_LOG_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double*> addr_shift;
        for(int i=1; i<=WINDOW_SIZE; i++){
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i-1];

            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
            for (uint32_t k = 0; k < 4; ++k)
                addr_shift[reinterpret_cast<long>(para_rcv_dt+i*4+k)] = para_rcv_dt+(i-1)*4+k;
            addr_shift[reinterpret_cast<long>(para_rcv_ddt+i)] = para_rcv_ddt+i-1;
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        addr_shift[reinterpret_cast<long>(para_yaw_enu_local)] = para_yaw_enu_local;
        addr_shift[reinterpret_cast<long>(para_anc_ecef)] = para_anc_ecef;
        std::vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if(last_marginalization_info){
            delete last_marginalization_info;
        }
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
    }
    else{
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(
                last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                std::vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    assert(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            RCUTILS_LOG_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            RCUTILS_LOG_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            RCUTILS_LOG_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            RCUTILS_LOG_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                    for (uint32_t k = 0; k < 4; ++k)
                        addr_shift[reinterpret_cast<long>(para_rcv_dt+i*4+k)] = para_rcv_dt+(i-1)*4+k;
                    addr_shift[reinterpret_cast<long>(para_rcv_ddt+i)] = para_rcv_ddt+i-1;
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                    for (uint32_t k = 0; k < 4; ++k)
                        addr_shift[reinterpret_cast<long>(para_rcv_dt+i*4+k)] = para_rcv_dt+i*4+k;
                    addr_shift[reinterpret_cast<long>(para_rcv_ddt+i)] = para_rcv_ddt+i;
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }
            addr_shift[reinterpret_cast<long>(para_yaw_enu_local)] = para_yaw_enu_local;
            addr_shift[reinterpret_cast<long>(para_anc_ecef)] = para_anc_ecef;
            std::vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    RCUTILS_LOG_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());
    
    RCUTILS_LOG_DEBUG("whole time for ceres: %f", t_whole.toc());
}

void Estimator::slideWindow(){
    TicToc t_margin;
    if(marginalization_flag == MARGIN_OLD){
        double t_0 = stamp2Sec(Headers[0].stamp);
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if(frame_count == WINDOW_SIZE){
            for(int i=0; i<WINDOW_SIZE; i++){
                Rs[i].swap(Rs[i + 1]);
                std::swap(pre_integrations[i], pre_integrations[i+1]);
                dt_buf[i].swap(dt_buf[i+1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i+1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i+1]);

                Headers[i] = Headers[i+1];
                Ps[i].swap(Ps[i+1]);
                Vs[i].swap(Vs[i+1]);
                Bgs[i].swap(Bgs[i+1]);
                Bas[i].swap(Bas[i+1]);

                // gnss related
                gnss_meas_buf[i].swap(gnss_meas_buf[i+1]);
                gnss_ephem_buf[i].swap(gnss_ephem_buf[i+1]);
                for(uint32_t k=0; k<4; ++k){
                    para_rcv_dt[i*4+k] = para_rcv_dt[(i+1)*4+k];
                }
                para_rcv_ddt[i] = para_rcv_ddt[i+1];
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE-1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE-1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE-1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE-1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE-1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE-1];

            gnss_meas_buf[WINDOW_SIZE].clear();
            gnss_ephem_buf[WINDOW_SIZE].clear();

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE],
                Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            if(true || solver_flag == INITIAL){
                std::map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.per_integration;
                it_0->second.per_integration=nullptr;

                for(std::map<double, ImageFrame>::iterator it = all_image_frame.begin();
                    it != it_0; ++it){
                    if(it->second.per_integration){
                        delete it->second.per_integration;
                    }   
                    it->second.per_integration = NULL; 
                }
                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0); 
            }

            slideWindowOld();
        }
    }
    else{
        if(frame_count == WINDOW_SIZE){
            for(unsigned int i=0; i<dt_buf[frame_count].size(); i++){
                double tmp_dt = dt_buf[frame_count][i];
                Eigen::Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Eigen::Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count-1]->push_back(tmp_dt, tmp_linear_acceleration, 
                    tmp_angular_velocity);

                dt_buf[frame_count-1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count-1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count-1].push_back(tmp_angular_velocity);
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            // GNSS related
            gnss_meas_buf[frame_count-1] = gnss_meas_buf[frame_count];
            gnss_ephem_buf[frame_count-1] = gnss_ephem_buf[frame_count];
            for (uint32_t k = 0; k < 4; ++k)
                para_rcv_dt[(frame_count-1)*4+k] = para_rcv_dt[frame_count*4+k];
            para_rcv_ddt[frame_count-1] = para_rcv_ddt[frame_count];
            gnss_meas_buf[frame_count].clear();
            gnss_ephem_buf[frame_count].clear();

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew();            
        
        } 
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Eigen::Matrix3d R0, R1;
        Eigen::Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}
