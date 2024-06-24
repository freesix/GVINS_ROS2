#include "initial/initial_sfm.hpp"

GlobalSFM::GlobalSFM(){}

/**
 * @brief 三角化两帧图像某个对应特征点的深度
*/
void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                        Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d){

    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();                        
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);

    Eigen::Vector4d triangulated_point;
    triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3); 
}

/**
 * @brief PNP方法得到第1帧到第i帧的R_intial和P_initial
*/
bool GlobalSFM::solveFrameByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, 
                                int i, std::vector<SFMFeature> &sfm_f){
    std::vector<cv::Point2f> pts_2_vector;
    std::vector<cv::Point3f> pts_3_vector;
    for(int j=0; j<feature_num; j++){
        if(sfm_f[j].state != true){
            continue;
        }
        Eigen::Vector2d point2d;
        for(int k=0; k<(int)sfm_f[j].observation.size(); k++){
            if(sfm_f[j].observation[k].first == i){ // 找到第i帧的sfm_f
                Eigen::Vector2d img_pts = sfm_f[j].observation[k].second; // 图像特征点坐标
                cv::Point2f pts_2(img_pts(0), img_pts(1)); // 转化为opencv点
                pts_2_vector.push_back(pts_2);
                cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
                pts_3_vector.push_back(pts_3); // 对应三角化后的3d点坐标
                break;   
            }
        }
    }
    if(int(pts_2_vector.size()) < 15){
        printf("unstable features tracking, please slowly move you device to get more features\n");
        if(int(pts_2_vector.size()) < 10){
            return false;
        }
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec); // 旋转矩阵转为旋转向量
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    bool pnp_succ;
    // 特征点的3d坐标、特征点的图像坐标、相机内参、畸变参数、相机的旋转向量、相机的平移向量，求解方法 
    pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
    if(!pnp_succ){
        return false;
    }
    cv::Rodrigues(rvec, r);
    Eigen::MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    Eigen::MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);
    R_initial = R_pnp;
    P_initial = T_pnp;
    return true;
}

/**
 * @brief 三角化frame0和frame1之间所有对应点
*/
void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, int frame1, 
            Eigen::Matrix<double, 3, 4> &Pose1, std::vector<SFMFeature> &sfm_f){

    assert(frame0 != frame1);
    for(int j=0; j<feature_num; j++){
        if(sfm_f[j].state == true){
            continue;
        }
        bool has_0 = false, has_1 = false;
        Eigen::Vector2d point0;
        Eigen::Vector2d point1;
        for(int k=0; k<(int)sfm_f[j].observation.size(); k++){
            if(sfm_f[j].observation[k].first == frame0){
                point0 = sfm_f[j].observation[k].second;
                has_0 = true;
            }
            if(sfm_f[j].observation[k].first == frame1){
                point1 = sfm_f[j].observation[k].second;
                has_1 = true;
            }
        }

        if(has_0 && has_1){
            Eigen::Vector3d point_3d;
            triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
            sfm_f[j].state = true;
            sfm_f[j].position[0] = point_3d[0];
            sfm_f[j].position[1] = point_3d[1];
            sfm_f[j].position[2] = point_3d[2];
        }
    }
}

/**
 * @brief 纯视觉sfm, 求解窗口中的所有图像帧的位姿和特征点坐标
 * @param frame_num 窗口总帧数(frame_cout + 1)
 * @param[out] q 窗口内图像帧的旋转四元数(相对于第一帧)
 * @param[out] T 窗口内图像帧的平移向量T(相对于第一帧)
 * @param l 第l帧 
 * @param relative_R 当前帧到第一帧的旋转矩阵 
 * @param relative_T 当前帧到第一帧的平移向量
 * @param sfm_f 所有特征点
 * @param sfm_tracked_points 所有在sfm中三角化的特征点ID和坐标
 * @return bool 求解成功
*/
bool GlobalSFM::construct(int frame_num, Eigen::Quaterniond* q, Eigen::Vector3d* T, int l,
		const Eigen::Matrix3d relative_R, const Eigen::Vector3d relative_T,
	    std::vector<SFMFeature> &sfm_f, std::map<int, Eigen::Vector3d> &sfm_tracked_points){

    feature_num = sfm_f.size();
    // 假设第一帧为原点，根据当前帧到第一帧的relative_R, relative_T，得到当前帧的位姿
    // 清零要返回的q和T
    q[l].w() = 1;
    q[l].x() = 0;
    q[l].y() = 0;
    q[l].z() = 0;
    T[l].setZero();
    q[frame_num-1] = q[l] * Eigen::Quaterniond(relative_R); // 获取fram_num-1帧相对l的旋转
    T[frame_num-1] = relative_T; // 获取fram_num-1帧相对l的平移，因为l帧被设为0，所以直接等于

    // rotate to cam frame
    Eigen::Matrix3d c_Rotation[frame_num];
    Eigen::Vector3d c_Translation[frame_num];
    Eigen::Quaterniond c_Quat[frame_num];    
    double c_rotation[frame_num][4];
    double c_translation[frame_num][3];
    Eigen::Matrix<double, 3, 4> Pose[frame_num];
    // 获取第l帧的camera 位姿
    c_Quat[l] = q[l].inverse();
    c_Rotation[l] = c_Quat[l].toRotationMatrix();
    c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
    Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
    Pose[l].block<3, 1>(0, 3) = c_Translation[l];
    // 获取第fram_num-1帧(最新加入窗口)相对于l帧的位姿
    c_Quat[frame_num-1] = q[frame_num-1].inverse();
    c_Rotation[frame_num-1] = c_Quat[frame_num-1].toRotationMatrix();
    c_Translation[frame_num-1] = -1 * (c_Rotation[frame_num-1] * T[frame_num-1]);
    Pose[frame_num-1].block<3, 3>(0, 0) = c_Rotation[frame_num-1];
    Pose[frame_num-1].block<3, 1>(0, 3) = c_Translation[frame_num-1];

    // 1、三角化第l帧和第frame_num-1帧之间的特征点
    // 2、pnp求解从第l+1帧开始的每一帧到l帧的变换矩阵R_initial和P_initial保存在Pose中
    // 并与当前帧进行三角化
    for(int i=l; i<frame_num-1; i++){
        if(i > l){
            Eigen::Matrix3d R_initial = c_Rotation[i-1];
            Eigen::Vector3d P_initial = c_Translation[i-1];
            if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f)){
                return false;
            }
            c_Rotation[i] = R_initial;
            c_Translation[i] = P_initial;
            c_Quat[i] = c_Rotation[i];
            Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
            Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        }
        // triangulate point based on the solve pnp result
        // 根据pnp得到的相对位姿三角化两帧之间的特征点
        triangulateTwoFrames(i, Pose[i], frame_num-1, Pose[frame_num-1], sfm_f);
    }
    // 3、三角化l --- l+1 l+2 ... frame_num-2
    for(int i=l+1; i<frame_num-1; i++){
        triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
    }
    // 4、PNP求解从第l-1帧到第0帧的每一帧与第一帧之间的变换矩阵，并进行三角化
    for(int i=l-1; i>=0; i--){
        // solve pnp
        Eigen::Matrix3d R_initial = c_Rotation[i+1];
        Eigen::Vector3d P_initial = c_Translation[i+1];
        if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f)){
            return false;
        }
        c_Rotation[i] = R_initial;
        c_Translation[i] = P_initial;
        c_Quat[i] = c_Rotation[i];
        Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
        Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        // 三角化
        triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
    }
    // 5、三角化其它未恢复的特征点，由此得到滑动窗口中所有图像帧的位姿及特征点的3d坐标
    for(int j=0; j<feature_num; j++){
        if(sfm_f[j].state == true){
            continue;
        }
        if((int)sfm_f[j].observation.size() >= 2){
            Eigen::Vector2d point0, point1;
            int frame_0 = sfm_f[j].observation[0].first;
            point0 = sfm_f[j].observation[0].second;
            int frame_1 = sfm_f[j].observation.back().first;
            point1 = sfm_f[j].observation.back().second;
            Eigen::Vector3d point_3d;
            triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
            sfm_f[j].state = true;
            sfm_f[j].position[0] = point_3d[0];
            sfm_f[j].position[1] = point_3d[1];
            sfm_f[j].position[2] = point_3d[2];
        }
    }

    // full BA
    ceres::Problem problem;
    ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
    for(int i=0; i<frame_num; i++){
        c_translation[i][0] = c_Translation[i].x();
        c_translation[i][1] = c_Translation[i].y();
        c_translation[i][2] = c_Translation[i].z();
        c_rotation[i][0] = c_Quat[i].w();
        c_rotation[i][1] = c_Quat[i].x();
        c_rotation[i][2] = c_Quat[i].y();
        c_rotation[i][3] = c_Quat[i].z();
        problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
        problem.AddParameterBlock(c_translation[i], 3);
        if(i == l){
            problem.SetParameterBlockConstant(c_rotation[i]);
        }
        if(i == l || i == frame_num-1){
            problem.SetParameterBlockConstant(c_translation[i]);
        }
    }

    for(int i=0; i<feature_num; i++){
        if(sfm_f[i].state != true){
            continue;
        }
        for(int j=0; j<int(sfm_f[i].observation.size()); j++){
            int l = sfm_f[i].observation[j].first;
            ceres::CostFunction* cost_function = ReprojectionError3D::Create(
                sfm_f[i].observation[j].second.x(), sfm_f[i].observation[j].second.y());

            problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l],
                                    sfm_f[i].position);
        }
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 0.2;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if(summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03){
        // std::cout<<"vision only BA converge"<<std::endl;
    }
    else{
        // std::cout<<"vision only BA not converge"<<std::endl;
        return false;
    }
    for(int i=0; i<frame_num; i++){
        q[i].w() = c_rotation[i][0];
        q[i].x() = c_rotation[i][1];
        q[i].y() = c_rotation[i][2];
        q[i].z() = c_rotation[i][3];
        q[i] = q[i].inverse();
    }
    for(int i=0; i<frame_num; i++){
        T[i] = -1 * (q[i] * Eigen::Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
    }
    for(int i=0; i<(int)sfm_f.size(); i++){
        if(sfm_f[i].state){
            sfm_tracked_points[sfm_f[i].id] = Eigen::Vector3d(sfm_f[i].position[0], sfm_f[i].position[1],
                                                            sfm_f[i].position[2]);
        }
    }
    return true;
}