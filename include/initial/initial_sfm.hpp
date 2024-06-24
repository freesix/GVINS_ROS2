#pragma once 

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <map>
#include <cstdlib>
#include <iostream>


struct SFMFeature
{
    bool state; // 特征点的状态，是否被三角化
    int id; 
    std::vector<std::pair<int, Eigen::Vector2d>> observation; // 所有观测到该特征点的图像帧id和图像坐标
    double position[3]; // 3d坐标
    double depth; // 深度
};

struct ReprojectionError3D{
    ReprojectionError3D(double observed_u, double observed_v): observed_u(observed_u), observed_v(observed_v){}

    // 计算三维点point的投影误差(残差)
    template<typename T>
    bool operator()(const T* camera_R, const T* camera_T, const T* point, T* residuals) const{
        T p[3];
        ceres::QuaternionRotatePoint(camera_R, point, p); // 对point点进行旋转
        p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2]; // 平移
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];
        residuals[0] = xp - T(observed_u);
        residuals[1] = yp - T(observed_v);
        return true;
    }
    // 定义残差优化的目标函数
    static ceres::CostFunction* Create(const double observed_x, 
                                       const double observed_y){
        // <残差函数、残差维度、残差函数的参数维度(相机位姿旋转四元数、相机位姿平移向量、三位点坐标)>
        return (new ceres::AutoDiffCostFunction<ReprojectionError3D, 2, 4, 3, 3>( 
                new ReprojectionError3D(observed_x, observed_y)));                                      
    } 

    double observed_u;
    double observed_v; 
};


class GlobalSFM
{
public:
	GlobalSFM();
	
	bool construct(int frame_num, Eigen::Quaterniond* q, Eigen::Vector3d* T, int l,
			  const Eigen::Matrix3d relative_R, const Eigen::Vector3d relative_T,
			  std::vector<SFMFeature> &sfm_f, std::map<int, Eigen::Vector3d> &sfm_tracked_points);

private:
	bool solveFrameByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, int i, std::vector<SFMFeature> &sfm_f);

	void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
							Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d);
	void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
							  int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
							  std::vector<SFMFeature> &sfm_f);

	int feature_num;
};