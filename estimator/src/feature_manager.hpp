#pragma once 
#include <list>
#include <algorithm>
#include <numeric>
#include <eigen3/Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <rcpputils/asserts.hpp>
#include "parameters.hpp"
// 特征点在每一帧上的属性
// 空间上特征点P映射到frame1或frame2上对应的图像坐标，特征点的跟踪速度，空间坐标等属性的封装
class FeaturePerFrame{
public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td){
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5);
        velocity.y() = _point(6);
        cur_td = td;
    }

    double cur_td;
    Eigen::Vector3d point;
    Eigen::Vector2d uv;
    Eigen::Vector2d velocity;
    double z;
    bool is_used;
    double parallax;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    double dep_gradient;
};
// 管理一个特征点
// 就特征点p1来说，它被两个帧观测到，第一个观测到的帧是frame1，即start_frame=1
// 最后一次观测到p1的帧为frame2，即end_frame=2，并把start_frame和end_frame之间的观测信息存储
class FeaturePerId{
public:
    const int feature_id; //特征点id
    int start_frame; // 特征点第一次被观测到的帧
    std::vector<FeaturePerFrame> feature_per_frame;

    int used_num; // 出现次数 
    bool is_outlier;
    bool is_margin;
    double estimated_depth; // 逆深度
    int solve_flag; // 是否被三角化 0: not solve now; 1: solve succ; 2: solve fail

    FeaturePerId(int _feature_id, int _start_frame) : 
        feature_id(_feature_id), start_frame(_start_frame), used_num(0),
        estimated_depth(-1.0), solve_flag(0)
        { 
        }

    int endFrame();
};

class FeatureManager{
public:
    FeatureManager(Eigen::Matrix3d _Rs[]);

    void setRic(Eigen::Matrix3d _ric[]);

    void clearState();

    int getFeatureCount();

    bool addFeatureCheckParallax(int frame_count, const std::map<int, 
        std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    void debugShow();
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> getCorresponding(
        int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    void setDepth(const Eigen::VectorXd &x);
    void removeFailures();
    void clearDepth(const Eigen::VectorXd &x);
    Eigen::VectorXd getDepthVector();
    void triangulate(Eigen::Vector3d Ps[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, 
        Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier();
    std::list<FeaturePerId> feature;
    int last_track_num;

private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Eigen::Matrix3d *Rs;
    Eigen::Matrix3d ric[NUM_OF_CAM];    
};
