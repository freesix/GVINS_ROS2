#include "initial_ex_rotation.hpp"

InitialEXRotation::InitialEXRotation(){
    frame_count = 0;
    Rc.push_back(Eigen::Matrix3d::Identity());
    Rc_g.push_back(Eigen::Matrix3d::Identity());
    Rimu.push_back(Eigen::Matrix3d::Identity());
    ric = Eigen::Matrix3d::Identity();
}
/**
 * @brief 当外参完全不知道的时候，在线对其初步估计，会在后续优化过程中再次优化
*/
bool InitialEXRotation::CalibrationExRotation(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres,
            Eigen::Quaterniond delta_q_imu, Eigen::Matrix3d &calib_ric_result){
    frame_count++;
    Rc.push_back(solveRelativeR(corres));
    Rimu.push_back(delta_q_imu.toRotationMatrix());
    Rc_g.push_back(ric.inverse() * delta_q_imu * ric);

    Eigen::MatrixXd A(frame_count*4, 4);
    A.setZero();
    int sum_ok = 0;
    for(int i=0; i<=frame_count; i++){
        Eigen::Quaterniond r1(Rc[i]);
        Eigen::Quaterniond r2(Rc_g[i]);

        double angular_distance = 180 / M_PI * r1.angularDistance(r2);
        RCLCPP_DEBUG(rclcpp::get_logger("initial_ex_rotation"), "%d angular_distance: %f", i, angular_distance);
        
        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
        ++sum_ok;
        Eigen::Matrix4d L, R;

        double w = Eigen::Quaterniond(Rc[i]).w();
        Eigen::Vector3d v = Eigen::Quaterniond(Rc[i]).vec();

        L.block<3, 3>(0,0) = w * Eigen::Matrix3d::Identity() - skewSymmetric(v);
    }           
}


/**
* @brief 根据两帧3D特征点求解旋转矩阵
*/
Eigen::Matrix3d InitialEXRotation::solveRelativeR(const std::vector<std::pair<Eigen::Vector3d,
            Eigen::Vector3d>> &corres){
    if(corres.size() >= 9){
        std::vector<cv::Point2f> ll, rr;
        for(int i; i<corres.size(); i++){
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }

        // 求解两帧的本质矩阵 
        cv::Mat E = cv::findFundamentalMat(ll, rr);
        cv::Mat_<double> R1, R2, t1, t2;

        // 本质矩阵svd分解得到4组Rt解
        decomposeE(E, R1, R2, t1, t2);
        if(cv::determinant(R1) + 1.0 < 1e-09){
            E = -E;
            decomposeE(E, R1, R2, t1, t2);
        }
        double ratio1 = std::max(testTriangulation(ll, rr, R1, t1) ,testTriangulation(ll, rr, R1, t2));
        double ratio2 = std::max(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
        cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2;

        Eigen::Matrix3d ans_R_eigen;
        for(int i=0; i<3; i++){
            for(int j=0; j<3; j++){
                ans_R_eigen(j, i) = ans_R_cv(i, j);
            }
        return ans_R_eigen;
        }    
        return Eigen::Matrix3d::Identity();
    }            
            
}


double InitialEXRotation::testTriangulation(const std::vector<cv::Point2f> &l, 
        const std::vector<cv::Point2f> &r, cv::Mat_<double> R, cv::Mat_<double> t){

    cv::Mat pointcloud;
    cv::Matx34f P = cv::Matx34f(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0);
    cv::Matx34f P1 = cv::Matx34f(R(0,0), R(0,1), R(0,2), t(0),
                                 R(1,0), R(1,1), R(1,2), t(1),
                                 R(2,0), R(2,1), R(2,2), t(2));
    cv::triangulatePoints(P, P1, l, r, pointcloud);
    int front_count = 0;
    // 判断三角化得到的3D点是否还在两个相机前方
    for(int i=0; i<pointcloud.cols; i++){
        double normal_factor = pointcloud.col(i).at<float>(3); // z 
        cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor);
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor);
        if(p_3d_l(2) > 0 && p_3d_r(2) > 0){
            front_count++;
        } 
    }
    RCLCPP_DEBUG(rclcpp::get_logger("initial_ex_rotation"), "MotionEstimator: %f", 1.0*front_count/pointcloud.cols);
    return 1.0*front_count/pointcloud.cols;     
}


/**
 * @brief 分解本质矩阵得到可能得R和t
*/
void InitialEXRotation::decomposeE(cv::Mat E, cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                                   cv::Mat_<double> &t1, cv::Mat_<double> &t2){
    cv::SVD svd(E, cv::SVD::MODIFY_A);
    cv::Matx33d W(0, -1, 0,
                  1, 0,  0,
                  0, 0,  1);
    cv::Matx33d Wt(0, 1, 0,
                  -1, 0, 0,
                   0, 0, 1);
    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}