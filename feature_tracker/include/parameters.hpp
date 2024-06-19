#pragma once 
#include <rclcpp/rclcpp.hpp>
#include <opencv2/highgui/highgui.hpp>

extern int ROW;
extern int COL;
extern int FOCAL_LENGTH; // 焦点长度
const int NUM_OF_CAM = 1;

extern std::string IMAGE_TOPIC;
extern std::string IMU_TOPIC;
extern std::string FISHEYE_MASK; // 鱼眼mask
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST; // 特征点之间的最小距离
extern int WINDOW_SIZE;
extern int FREQ; // 发布跟踪结果频率，最少10hz，为0则和raw image一样频率
extern double F_THRESHOLD; // ransac阈值
extern int SHOW_TRACK; // 是否可视化跟踪结果
extern int STEREO_TRACK; 
extern int EQUALIZE; // 图像是否均衡化
extern int FISHEYE;
extern bool PUB_THIS_FRAME;

void readParameters(rclcpp::Node::SharedPtr n);
