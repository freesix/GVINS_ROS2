#pragma once 
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

class MotionEstimator{
public:
    /**
     * @brief 通过5点法求解两帧图像之间的相关位姿
     * @param corres 两帧图像之间相匹配的特征点
     * @param Rotation 求解得到的旋转矩阵
     * @param Translation 求解得到的平移向量
    */
    bool solveRelativeRT(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>
                        &corres, Eigen::Matrix3d &R, Eigen::Vector3d &T);

private:
    double testTriangulation(const std::vector<cv::Point2f> &l, 
                             const std::vector<cv::Point2f> &i,
                             cv::Mat_<double> R, cv::Mat_<double> t);

    void decomposeE(cv::Mat E,
                    cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                    cv::Mat_<double> &t1, cv::Mat_<double> &t2);

};