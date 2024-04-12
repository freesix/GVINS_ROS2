#include "parameters.hpp"

int ROW;
int COL;
int FOCAL_LENGTH; 

std::string IMAGE_TOPIC;
std::string IMU_TOPIC;
std::string FISHEYE_MASK; 
std::vector<std::string> CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ; 
double F_THRESHOLD; 
int SHOW_TRACK; 
int STEREO_TRACK; 
int EQUALIZE; 
int FISHEYE;
bool PUB_THIS_FRAME;

template<typename T>
T readParam(rclcpp::Node::SharedPtr &n, std::string name){
    std::cout<<n.get()<<std::endl;
    T ans;
    if(n->get_parameter(name, ans)){
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Loaded %s", name.c_str());  
    }
    else{
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Failed to load %s", name.c_str());
    }
    return ans;
}



void readParameters(rclcpp::Node::SharedPtr &n)
{
    std::string config_file;
    config_file = "/home/freesix/GVINS_ROS2_WS/src/GVINS/config/visensor_f9p/visensor_left_f9p_config.yaml";
    // config_file = readParam<std::string>(n, "config_file");
    std::cout<<"config_file: "<<config_file<<std::endl;
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    std::string GVINS_FOLDER_PATH = readParam<std::string>(n, "gvins_folder");

    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["imu_topic"] >> IMU_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    FREQ = fsSettings["freq"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    EQUALIZE = fsSettings["equalize"];
    FISHEYE = fsSettings["fisheye"];
    if (FISHEYE == 1)
        FISHEYE_MASK = GVINS_FOLDER_PATH + "config/fisheye_mask.jpg";
    CAM_NAMES.push_back(config_file);

    WINDOW_SIZE = 20;
    STEREO_TRACK = false;
    FOCAL_LENGTH = 460;
    PUB_THIS_FRAME = false;

    if (FREQ == 0)
        FREQ = 100;

    fsSettings.release();

}