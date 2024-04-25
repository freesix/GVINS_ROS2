#include "visualization.hpp"

rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odometry, pub_latest_odometry;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path;
rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_point_cloud, pub_margin_cloud;
rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_key_poses;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_camera_pose;
rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_camera_pose_visual;
nav_msgs::msg::Path path;

rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_keyframe_pose;
rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_keyframe_point;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_extrinsic;

rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr pub_gnss_lla;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_enu_path, pub_rtk_enu_path;
nav_msgs::msg::Path enu_path, rtk_enu_path;
rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr pub_anc_lla;
rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_enu_pose;
// rclcpp::Publisher pub_sat_info;
// rclcpp::Publisher pub_yaw_enu_local;

CameraPoseVisualization cameraposevisual(0, 1, 0, 1);
CameraPoseVisualization keyframebasevisual(0.0, 0.0, 1.0, 1.0);
static double sum_of_path = 0;
static Eigen::Vector3d last_path(0.0, 0.0, 0.0);

void registerPub(rclcpp::Node::SharedPtr n)
{
    pub_path = n->create_publisher<nav_msgs::msg::Path>("path", 1000);
    pub_odometry = n->create_publisher<nav_msgs::msg::Odometry>("odometry", 1000);
    pub_point_cloud = n->create_publisher<sensor_msgs::msg::PointCloud>("point_cloud", 1000);
    pub_margin_cloud = n->create_publisher<sensor_msgs::msg::PointCloud>("history_cloud", 1000);
    pub_key_poses = n->create_publisher<visualization_msgs::msg::Marker>("key_poses", 1000);
    pub_camera_pose = n->create_publisher<nav_msgs::msg::Odometry>("camera_pose", 1000);
    pub_camera_pose_visual = n->create_publisher<visualization_msgs::msg::MarkerArray>("camera_pose_visual", 1000);
    pub_keyframe_pose = n->create_publisher<nav_msgs::msg::Odometry>("keyframe_pose", 1000);
    pub_keyframe_point = n->create_publisher<sensor_msgs::msg::PointCloud>("keyframe_point", 1000);
    pub_extrinsic = n->create_publisher<nav_msgs::msg::Odometry>("extrinsic", 1000);
    pub_gnss_lla = n->create_publisher<sensor_msgs::msg::NavSatFix>("gnss_fused_lla", 1000);
    pub_enu_path = n->create_publisher<nav_msgs::msg::Path>("gnss_enu_path", 1000);
    pub_anc_lla = n->create_publisher<sensor_msgs::msg::NavSatFix>("gnss_anchor_lla", 1000);
    pub_enu_pose = n->create_publisher<geometry_msgs::msg::PoseStamped>("enu_pose", 1000);

    cameraposevisual.setScale(1);
    cameraposevisual.setLineWidth(0.05);
    keyframebasevisual.setScale(0.1);
    keyframebasevisual.setLineWidth(0.01);
}

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, 
    const Eigen::Vector3d &V, const std_msgs::msg::Header &header)
{
    Eigen::Quaterniond quadrotor_Q = Q ;

    nav_msgs::msg::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = quadrotor_Q.x();
    odometry.pose.pose.orientation.y = quadrotor_Q.y();
    odometry.pose.pose.orientation.z = quadrotor_Q.z();
    odometry.pose.pose.orientation.w = quadrotor_Q.w();
    odometry.twist.twist.linear.x = V.x();
    odometry.twist.twist.linear.y = V.y();
    odometry.twist.twist.linear.z = V.z();
    pub_latest_odometry->publish(odometry);
}

void printStatistics(const Estimator &estimator, double t)
{
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    printf("ts: %f\n", stamp2Sec(estimator.Headers[WINDOW_SIZE].stamp));
    printf("position: %f, %f, %f\n", estimator.Ps[WINDOW_SIZE].x(), estimator.Ps[WINDOW_SIZE].y(), estimator.Ps[WINDOW_SIZE].z());
    // printf("body acc bias: %f, %f, %f\n", estimator.Bas[WINDOW_SIZE].x(), estimator.Bas[WINDOW_SIZE].y(), 
    //     estimator.Bas[WINDOW_SIZE].z());
    // Eigen::Vector3d Bas_w = estimator.Rs[WINDOW_SIZE] * estimator.Bas[WINDOW_SIZE];
    // printf("world acc bias: %f, %f, %f\n", Bas_w.x(), Bas_w.y(), Bas_w.z());
    RCLCPP_DEBUG_STREAM(rclcpp::get_logger("visualization"), "position: " << estimator.Ps[WINDOW_SIZE].transpose());
    RCLCPP_DEBUG_STREAM(rclcpp::get_logger("visualization"), "orientation: " << estimator.Vs[WINDOW_SIZE].transpose());
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        //ROS_DEBUG("calibration result for camera %d", i);
        RCLCPP_DEBUG_STREAM(rclcpp::get_logger("visualization"), "extirnsic tic: " << estimator.tic[i].transpose());
        RCLCPP_DEBUG_STREAM(rclcpp::get_logger("visualization"), "extrinsic ric: " << Utility::R2ypr(estimator.ric[i]).transpose());
        if (ESTIMATE_EXTRINSIC)
        {
            cv::FileStorage fs(EX_CALIB_RESULT_PATH, cv::FileStorage::WRITE);
            Eigen::Matrix3d eigen_R;
            Eigen::Vector3d eigen_T;
            eigen_R = estimator.ric[i];
            eigen_T = estimator.tic[i];
            cv::Mat cv_R, cv_T;
            cv::eigen2cv(eigen_R, cv_R);
            cv::eigen2cv(eigen_T, cv_T);
            fs << "extrinsicRotation" << cv_R << "extrinsicTranslation" << cv_T;
            fs.release();
        }
    }

    static double sum_of_time = 0;
    static int sum_of_calculation = 0;
    sum_of_time += t;
    sum_of_calculation++;
    RCUTILS_LOG_DEBUG("vo solver costs: %f ms", t);
    RCUTILS_LOG_DEBUG("average of time %f ms", sum_of_time / sum_of_calculation);

    sum_of_path += (estimator.Ps[WINDOW_SIZE] - last_path).norm();
    last_path = estimator.Ps[WINDOW_SIZE];
    RCUTILS_LOG_DEBUG("sum of path %f", sum_of_path);
    if (ESTIMATE_TD)
        RCUTILS_LOG_INFO("td %f", estimator.td);
}

void pubOdometry(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        nav_msgs::msg::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.child_frame_id = "world";
        Eigen::Quaterniond tmp_Q;
        tmp_Q = Eigen::Quaterniond(estimator.Rs[WINDOW_SIZE]);
        odometry.pose.pose.position.x = estimator.Ps[WINDOW_SIZE].x();
        odometry.pose.pose.position.y = estimator.Ps[WINDOW_SIZE].y();
        odometry.pose.pose.position.z = estimator.Ps[WINDOW_SIZE].z();
        odometry.pose.pose.orientation.x = tmp_Q.x();
        odometry.pose.pose.orientation.y = tmp_Q.y();
        odometry.pose.pose.orientation.z = tmp_Q.z();
        odometry.pose.pose.orientation.w = tmp_Q.w();
        odometry.twist.twist.linear.x = estimator.Vs[WINDOW_SIZE].x();
        odometry.twist.twist.linear.y = estimator.Vs[WINDOW_SIZE].y();
        odometry.twist.twist.linear.z = estimator.Vs[WINDOW_SIZE].z();
        pub_odometry->publish(odometry);

        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose = odometry.pose.pose;
        path.header = header;
        path.header.frame_id = "world";
        path.poses.push_back(pose_stamped);
        pub_path->publish(path);

        // write result to file
        std::ofstream foutC(VINS_RESULT_PATH, std::ios::app);
        foutC.setf(std::ios::fixed, std::ios::floatfield);
        foutC.precision(0);
        foutC << stamp2Sec(header.stamp) * 1e9 << ",";
        foutC.precision(5);
        foutC << estimator.Ps[WINDOW_SIZE].x() << ","
              << estimator.Ps[WINDOW_SIZE].y() << ","
              << estimator.Ps[WINDOW_SIZE].z() << ","
              << tmp_Q.w() << ","
              << tmp_Q.x() << ","
              << tmp_Q.y() << ","
              << tmp_Q.z() << ","
              << estimator.Vs[WINDOW_SIZE].x() << ","
              << estimator.Vs[WINDOW_SIZE].y() << ","
              << estimator.Vs[WINDOW_SIZE].z() << "," << std::endl;
        foutC.close();

        pubGnssResult(estimator, header);
    }
}

void pubGnssResult(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    if (!estimator.gnss_ready)      return;
    // publish GNSS LLA
    const double gnss_ts = stamp2Sec(estimator.Headers[WINDOW_SIZE].stamp) + 
        estimator.diff_t_gnss_local;
    Eigen::Vector3d lla_pos = ecef2geo(estimator.ecef_pos);
    printf("global time: %f\n", gnss_ts);
    printf("latitude longitude altitude: %f, %f, %f\n", lla_pos.x(), lla_pos.y(), lla_pos.z());
    sensor_msgs::msg::NavSatFix gnss_lla_msg;
    gnss_lla_msg.header.stamp = rclcpp::Time(gnss_ts); // ！！这里不一定对
    gnss_lla_msg.header.frame_id = "geodetic";
    gnss_lla_msg.latitude = lla_pos.x();
    gnss_lla_msg.longitude = lla_pos.y();
    gnss_lla_msg.altitude = lla_pos.z();
    pub_gnss_lla->publish(gnss_lla_msg);

    // publish anchor LLA
    const Eigen::Vector3d anc_lla = ecef2geo(estimator.anc_ecef);
    sensor_msgs::msg::NavSatFix anc_lla_msg;
    anc_lla_msg.header = gnss_lla_msg.header;
    anc_lla_msg.latitude = anc_lla.x();
    anc_lla_msg.longitude = anc_lla.y();
    anc_lla_msg.altitude = anc_lla.z();
    pub_anc_lla->publish(anc_lla_msg);

    // publish ENU pose and path
    geometry_msgs::msg::PoseStamped enu_pose_msg;
    // camera-front orientation
    Eigen::Matrix3d R_s_c;
    R_s_c <<  0,  0,  1,
             -1,  0,  0,
              0, -1,  0;
    Eigen::Matrix3d R_w_sensor = estimator.Rs[WINDOW_SIZE] * estimator.ric[0] * R_s_c.transpose();
    Eigen::Quaterniond enu_ori(estimator.R_enu_local * R_w_sensor);
    enu_pose_msg.header.stamp = header.stamp;
    enu_pose_msg.header.frame_id = "world";     // "enu" will more meaningful, but for viz
    enu_pose_msg.pose.position.x = estimator.enu_pos.x();
    enu_pose_msg.pose.position.y = estimator.enu_pos.y();
    enu_pose_msg.pose.position.z = estimator.enu_pos.z();
    enu_pose_msg.pose.orientation.x = enu_ori.x();
    enu_pose_msg.pose.orientation.y = enu_ori.y();
    enu_pose_msg.pose.orientation.z = enu_ori.z();
    enu_pose_msg.pose.orientation.w = enu_ori.w();
    pub_enu_pose->publish(enu_pose_msg);

    enu_path.header = enu_pose_msg.header;
    enu_path.poses.push_back(enu_pose_msg);
    pub_enu_path->publish(enu_path);

    // publish ENU-local tf
    Eigen::Quaterniond q_enu_world(estimator.R_enu_local);
    rclcpp::Node gnssResult("pubGnssResult");
    auto br = tf2_ros::TransformBroadcaster(gnssResult);

    geometry_msgs::msg::TransformStamped transform_enu_world_msg;
    // tf::Transform transform_enu_world;
    transform_enu_world_msg.header.stamp = header.stamp;
    transform_enu_world_msg.header.frame_id = "enu";
    transform_enu_world_msg.child_frame_id = "world";
    transform_enu_world_msg.transform.translation.x = 0.0;
    transform_enu_world_msg.transform.translation.y = 0.0;
    transform_enu_world_msg.transform.translation.z = 0.0;
    transform_enu_world_msg.transform.rotation.w = q_enu_world.w();
    transform_enu_world_msg.transform.rotation.x = q_enu_world.x();
    transform_enu_world_msg.transform.rotation.y = q_enu_world.y();
    transform_enu_world_msg.transform.rotation.z = q_enu_world.z();

    br.sendTransform(transform_enu_world_msg);

    // write GNSS result to file
    std::ofstream gnss_output(GNSS_RESULT_PATH, std::ios::app);
    gnss_output.setf(std::ios::fixed, std::ios::floatfield);
    gnss_output.precision(0);
    gnss_output << stamp2Sec(header.stamp) * 1e9 << ',';
    gnss_output << gnss_ts * 1e9 << ',';
    gnss_output.precision(5);
    gnss_output << estimator.ecef_pos(0) << ','
                << estimator.ecef_pos(1) << ','
                << estimator.ecef_pos(2) << ','
                << estimator.yaw_enu_local << ','
                << estimator.para_rcv_dt[(WINDOW_SIZE)*4+0] << ','
                << estimator.para_rcv_dt[(WINDOW_SIZE)*4+1] << ','
                << estimator.para_rcv_dt[(WINDOW_SIZE)*4+2] << ','
                << estimator.para_rcv_dt[(WINDOW_SIZE)*4+3] << ','
                << estimator.para_rcv_ddt[WINDOW_SIZE] << ','
                << estimator.anc_ecef(0) << ','
                << estimator.anc_ecef(1) << ','
                << estimator.anc_ecef(2) << '\n';
    gnss_output.close();
}

void pubKeyPoses(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    if (estimator.key_poses.size() == 0)
        return;
    visualization_msgs::msg::Marker key_poses;
    key_poses.header = header;
    key_poses.header.frame_id = "world";
    key_poses.ns = "key_poses";
    key_poses.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    key_poses.action = visualization_msgs::msg::Marker::ADD;
    key_poses.pose.orientation.w = 1.0;
    // key_poses.lifetime = rclcpp::Duration

    //static int key_poses_id = 0;
    key_poses.id = 0; //key_poses_id++;
    key_poses.scale.x = 0.05;
    key_poses.scale.y = 0.05;
    key_poses.scale.z = 0.05;
    key_poses.color.r = 1.0;
    key_poses.color.a = 1.0;

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        geometry_msgs::msg::Point pose_marker;
        Eigen::Vector3d correct_pose;
        correct_pose = estimator.key_poses[i];
        pose_marker.x = correct_pose.x();
        pose_marker.y = correct_pose.y();
        pose_marker.z = correct_pose.z();
        key_poses.points.push_back(pose_marker);
    }
    pub_key_poses->publish(key_poses);
}

void pubCameraPose(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    int idx2 = WINDOW_SIZE - 1;

    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        int i = idx2;
        Eigen::Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Eigen::Quaterniond R = Eigen::Quaterniond(estimator.Rs[i] * estimator.ric[0]);

        nav_msgs::msg::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();

        pub_camera_pose->publish(odometry);

        cameraposevisual.reset();
        cameraposevisual.add_pose(P, R);
        cameraposevisual.publish_by(pub_camera_pose_visual, odometry.header);
    }
}


void pubPointCloud(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    sensor_msgs::msg::PointCloud point_cloud, loop_point_cloud;
    point_cloud.header = header;
    loop_point_cloud.header = header;


    for (auto &it_per_id : estimator.f_manager.feature)
    {
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1)
            continue;
        int imu_i = it_per_id.start_frame;
        Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        Eigen::Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

        geometry_msgs::msg::Point32 p;
        p.x = w_pts_i(0);
        p.y = w_pts_i(1);
        p.z = w_pts_i(2);
        point_cloud.points.push_back(p);
    }
    pub_point_cloud->publish(point_cloud);


    // pub margined potin
    sensor_msgs::msg::PointCloud margin_cloud;
    margin_cloud.header = header;

    for (auto &it_per_id : estimator.f_manager.feature)
    { 
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        //if (it_per_id->start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id->solve_flag != 1)
        //        continue;

        if (it_per_id.start_frame == 0 && it_per_id.feature_per_frame.size() <= 2 
            && it_per_id.solve_flag == 1 )
        {
            int imu_i = it_per_id.start_frame;
            Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
            Eigen::Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

            geometry_msgs::msg::Point32 p;
            p.x = w_pts_i(0);
            p.y = w_pts_i(1);
            p.z = w_pts_i(2);
            margin_cloud.points.push_back(p);
        }
    }
    pub_margin_cloud->publish(margin_cloud);
}

void pubTF(const Estimator &estimator, const std_msgs::msg::Header &header)
{
    if( estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    rclcpp::Node n("pubTF");
    auto br = tf2_ros::StaticTransformBroadcaster(n);
    geometry_msgs::msg::TransformStamped transform_body_msg;
    // body frame
    Eigen::Vector3d correct_t;
    Eigen::Quaterniond correct_q;
    correct_t = estimator.Ps[WINDOW_SIZE];
    correct_q = estimator.Rs[WINDOW_SIZE];

    transform_body_msg.header.stamp = header.stamp;
    transform_body_msg.header.frame_id = "world";
    transform_body_msg.child_frame_id = "body";
    transform_body_msg.transform.translation.x = correct_t(0);
    transform_body_msg.transform.translation.y = correct_t(1);
    transform_body_msg.transform.translation.z = correct_t(2);

    transform_body_msg.transform.rotation.w = correct_q.w();
    transform_body_msg.transform.rotation.x = correct_q.x();
    transform_body_msg.transform.rotation.y = correct_q.y();
    transform_body_msg.transform.rotation.z = correct_q.z();

    br.sendTransform(transform_body_msg);


    // camera frame

    geometry_msgs::msg::TransformStamped transform_camera_msg;
    transform_camera_msg.header.stamp = header.stamp;
    transform_camera_msg.header.frame_id = "body";
    transform_camera_msg.child_frame_id = "camera";
    transform_camera_msg.transform.translation.x = estimator.tic[0].x();
    transform_camera_msg.transform.translation.y = estimator.tic[0].y();
    transform_camera_msg.transform.translation.z = estimator.tic[0].z();
    transform_camera_msg.transform.rotation.w = Eigen::Quaterniond(estimator.ric[0]).w();
    transform_camera_msg.transform.rotation.x = Eigen::Quaterniond(estimator.ric[0]).x();
    transform_camera_msg.transform.rotation.y = Eigen::Quaterniond(estimator.ric[0]).y();
    transform_camera_msg.transform.rotation.z = Eigen::Quaterniond(estimator.ric[0]).z();

    nav_msgs::msg::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = estimator.tic[0].x();
    odometry.pose.pose.position.y = estimator.tic[0].y();
    odometry.pose.pose.position.z = estimator.tic[0].z();
    Eigen::Quaterniond tmp_q{estimator.ric[0]};
    odometry.pose.pose.orientation.x = tmp_q.x();
    odometry.pose.pose.orientation.y = tmp_q.y();
    odometry.pose.pose.orientation.z = tmp_q.z();
    odometry.pose.pose.orientation.w = tmp_q.w();
    pub_extrinsic->publish(odometry);

}

void pubKeyframe(const Estimator &estimator)
{
    // pub camera pose, 2D-3D points of keyframe
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR && estimator.marginalization_flag == 0)
    {
        int i = WINDOW_SIZE - 2;
        //Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Eigen::Vector3d P = estimator.Ps[i];
        Eigen::Quaterniond R = Eigen::Quaterniond(estimator.Rs[i]);

        nav_msgs::msg::Odometry odometry;
        odometry.header = estimator.Headers[WINDOW_SIZE - 2];
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();
        //printf("time: %f t: %f %f %f r: %f %f %f %f\n", odometry.header.stamp.toSec(), P.x(), P.y(), P.z(), R.w(), R.x(), R.y(), R.z());

        pub_keyframe_pose->publish(odometry);


        sensor_msgs::msg::PointCloud point_cloud;
        point_cloud.header = estimator.Headers[WINDOW_SIZE - 2];
        for (auto &it_per_id : estimator.f_manager.feature)
        {
            int frame_size = it_per_id.feature_per_frame.size();
            if(it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - 2 && it_per_id.solve_flag == 1)
            {

                int imu_i = it_per_id.start_frame;
                Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                Eigen::Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0])
                                      + estimator.Ps[imu_i];
                geometry_msgs::msg::Point32 p;
                p.x = w_pts_i(0);
                p.y = w_pts_i(1);
                p.z = w_pts_i(2);
                point_cloud.points.push_back(p);

                int imu_j = WINDOW_SIZE - 2 - it_per_id.start_frame;
                sensor_msgs::msg::ChannelFloat32 p_2d;
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.y());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.y());
                p_2d.values.push_back(it_per_id.feature_id);
                point_cloud.channels.push_back(p_2d);
            }

        }
        pub_keyframe_point->publish(point_cloud);
    }
}