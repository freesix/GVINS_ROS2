#pragma once

#include <cstdlib>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "utility/utility.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <glog/logging.h>

extern std::string IMAGE_FILE;  // 图片的文件夹路径
extern std::string FISHEYE_MASK; // 鱼眼mask
extern std::vector<std::string> CAM_NAMES; // 相机名字
extern int MAX_CNT;
extern int MIN_DIST; // 特征点之间的最小距离
extern int FREQ; // 发布跟踪结果频率，最少10hz，为0则和raw image一样频率
extern double F_THRESHOLD; // ransac阈值
extern int SHOW_TRACK; // 是否将跟踪图像作为话题发布
extern int STEREO_TRACK; 
extern int EQUALIZE; // 图像是否均衡化
extern int FISHEYE;
extern bool PUB_THIS_FRAME; // 是否将这一帧发布送入后端处理(频率控制)


const double FOCAL_LENGTH = 460.0; // 焦距
const int WINDOW_SIZE = 10;
const int NUM_OF_CAM = 1;
const int NUM_OF_F = 1000;
//#define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;
extern double MIN_PARALLAX;  // 最小视差
extern int ESTIMATE_EXTRINSIC; 

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string FACTOR_GRAPH_RESULT_PATH;
extern std::string IMU_FILE;
extern double TD;
extern int ESTIMATE_TD;
extern double ROW, COL;

extern bool GNSS_ENABLE;
extern std::string GNSS_EPHEM_TOPIC;
extern std::string GNSS_GLO_EPHEM_TOPIC;
extern std::string GNSS_MEAS_TOPIC;
extern std::string GNSS_IONO_PARAMS_TOPIC;
extern std::string GNSS_TP_INFO_TOPIC;
extern std::vector<double> GNSS_IONO_DEFAULT_PARAMS;
extern bool GNSS_LOCAL_ONLINE_SYNC;
extern std::string LOCAL_TRIGGER_INFO_TOPIC;
extern double GNSS_LOCAL_TIME_DIFF;
extern double GNSS_ELEVATION_THRES;
extern double GNSS_PSR_STD_THRES;
extern double GNSS_DOPP_STD_THRES;
extern uint32_t GNSS_TRACK_NUM_THRES;
extern double GNSS_DDT_WEIGHT;
extern std::string GNSS_RESULT_PATH;

void readParameters(const std::string& config_path);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7, // 位姿，旋转用四元数，因此为7
    SIZE_SPEEDBIAS = 9, // 速度和偏置
    SIZE_FEATURE = 1 // 特征点逆深度
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
