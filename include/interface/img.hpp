#pragma once 

#include <opencv2/opencv.hpp>
#include <filesystem>
#include "utility/tic_toc.h"
#include "feature_tracker/feature_tracker.hpp"
#include <glog/logging.h>
#include <regex>

extern FeatureTracker trackerData[NUM_OF_CAM];
extern bool RESTART_FLAG;

namespace Data{

struct ImgMsg{
    double timestamp{};
    cv::Mat img;
};
struct FeatureTimeMsg{
    double timestamp; // 时间戳
    std::vector<Eigen::Vector3d> points; // 特征点的空间位置
    std::vector<float> id_of_point; // 特征点的唯一标识id
    std::vector<float> u_of_point;  // 特征点在图像中的x位置
    std::vector<float> v_of_point;  // 特征点在图像中的y位置
    std::vector<float> velocity_x_of_point; // 特征点在归一化平面上相较于上一帧计算出的x方向上的运动速度
    std::vector<float> velocity_y_of_point; // 特征点在归一化平面上相较于上一帧计算出的y方向上的运动速度
};

/**
 * @brief 从文件中读取图像
 * @param path 包含图像的文件夹路径
 * @return 读取到的图像名容器
*/
std::vector<std::string> getImgsPath(const std::string& path);
/**
 * @brief 读取图像到buf中
 * @param imgs_path 图像路径
 * @param ImgMsg 结构存储的图像
*/
ImgMsg readImg(const std::string& img_path);
/**
 * @brief img回调函数
*/
// void imgCallback(const std::vector<std::string>& imgs_path, std::queue<Data::FeatureTimeMsg>& feature_buf);
void imgCallback(const std::vector<std::string>& imgs_path, 
    std::queue<FeatureTimeMsg>& feature_buf,  std::mutex& f_buf);


} // namespace Data