#include "initial/solve_5pts.hpp"


bool MotionEstimator::solveRelativeRT(const std::vector<std::pair<Eigen::Vector3d, 
        Eigen::Vector3d>> &corres, Eigen::Matrix3d &Rotation, Eigen::Vector3d &Translation){

    if(corres.size() >= 15){
        std::vector<cv::Point2f> ll, rr;
        for(int i=0; i<int(corres.size()); i++){
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }
        cv::Mat mask;
        /**
         * 通过RANSAC算法求解两幅图像之间的本质矩阵
         * 第一幅图像点数组
         * 第二幅图像点数组
         * RANSAC方法求解
         * 点到对极线的最大距离，超过此值会被舍弃
         * 矩阵正确的可信度
         * 在计算过程中没有被舍弃的点
        */
        cv::Mat E = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 0.3/460, 0.99, mask);
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        cv::Mat rot, trans;
        /**
         * 通过本质矩阵的到Rt
        */
        int inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask);

        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        for(int i=0; i<3; i++){
            T(i) = trans.at<double>(i, 0);
            for(int j=0; j<3; j++){
                R(i, j) = rot.at<double>(i, j);
            }
        }

        Rotation = R.transpose();
        Translation = -R.transpose() * T;
        if(inlier_cnt > 12)return true;
        else return false;
    }       
    return false;
}