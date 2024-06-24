#include <thread>
#include <queue>
#include "feature_tracker/feature_tracker.hpp"
#include "interface/img.hpp"
#include "interface/imu.hpp"

#define SHOW_UNDISTORTION 0

// std::queue<std::shared_ptr<Data::ImgMsg>> img_buf; // 图像缓存
std::queue<Data::FeatureTimeMsg> feature_buf; // 图像特征点缓存
std::queue<Data::ImuTimeMsg> imu_buf; // imu数据缓存

std::mutex m_buf; 
std::mutex f_buf; // 锁住特征点缓存
std::mutex i_buf; // 锁住imu数据缓存
std::mutex m_state;

void readImu(const std::string& imu_path){
    std::ifstream imu_file(imu_path);
    if(!imu_file.is_open()){
        LOG(ERROR) << "Open imu file failed: " << imu_path;
        return;
    }
    std::string line;
    while(std::getline(imu_file, line)){
        std::istringstream iss(line);
        Data::ImuTimeMsg imu_data;
        iss >> imu_data.timestamp;
        for(int i=0; i<3; i++){
            iss >> imu_data.linear_acceleration(i);
        }
        for(int i=0; i<3; i++){
            iss >> imu_data.angular_velocity(i);
        }
        std::cout<<"imu_data: "<<imu_data.timestamp<<": "
            <<imu_data.linear_acceleration<<","<<imu_data.angular_velocity<<std::endl;
        // 检查imu数据时间是否合法
        if(imu_data.timestamp <= Data::last_imu_t){
            LOG(ERROR) << "imu data disordered!";
            continue;
        }
        Data::last_imu_t = imu_data.timestamp;
        i_buf.lock();
        imu_buf.push(imu_data);
        i_buf.unlock();

        {
            std::lock_guard<std::mutex> lg(m_state);
            predict(imu_data);
            // 此处预测完后，发布tmp_P, tmp_Q, tmp_V, timestamp来做可视化
            std::cout<<"tmp_P: "<<Data::tmp_P.transpose()<<std::endl;
            std::cout<<"tmp_Q: "<<Data::tmp_Q.coeffs().transpose()<<std::endl;
            std::cout<<"tmp_V: "<<Data::tmp_V.transpose()<<std::endl; 
        }
        
    }
    imu_file.close();

}


int main(int argc, char** argv){
    if(argc !=2){
        LOG(ERROR) << "usage: ./${exe} path_to_config_path";
        return -1;
    }
    google::InitGoogleLogging(argv[0]); // 初始化glog日志系统
    FLAGS_minloglevel = google::INFO; // 设置日志级别

    readParameters(argv[1]); // 读取参数
    
    for(int i=0; i<NUM_OF_CAM; i++){
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]); // 读取相机参数
    }

    std::vector<std::string> imgs_path = Data::getImgsPath(IMAGE_FILE);
    // Data::imgCallback(imgs_path, feature_buf, m_buf);
    std::thread img_callback(Data::imgCallback, imgs_path, std::ref(feature_buf), std::ref(f_buf));
    std::thread imu_callback(Data::readImu, IMU_FILE);
    std::cout<<"feature_buf size: "<<feature_buf.size()<<std::endl;



    if(imu_callback.joinable()){
        imu_callback.join();
    }
    if(img_callback.joinable()){
        img_callback.join();
    }
    return 0;
}

