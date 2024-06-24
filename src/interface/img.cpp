#include "interface/img.hpp"

FeatureTracker trackerData[NUM_OF_CAM];
bool RESTART_FLAG = false;

bool first_image_flag = true;
double first_image_time;
double last_image_time = 0;
int pub_count = 1;
bool init_pub = false;

namespace Data{

std::vector<std::string> getImgsPath(const std::string& path){
    std::vector<std::string> imgs_path;
    if(!std::filesystem::exists(path)){
        LOG(ERROR) << "Path not exist: " << path;
        return imgs_path;
    }
    for(const auto& entry : std::filesystem::directory_iterator(path)){
        imgs_path.push_back(entry.path().string());
    }
    return imgs_path;
}

ImgMsg readImg(const std::string& img_path){
    ImgMsg img_msg;
    img_msg.img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    if(img_msg.img.empty()){
        LOG(ERROR) << "Read image failed: " << img_path;
    }
    // 将路径中的时间戳提取出来
    std::string img_name = img_path.substr(img_path.find_last_of('/') + 1);
    img_msg.timestamp = std::stod(img_name.substr(0, img_name.find('.')))
        + std::stod(img_name.substr(img_name.find('.') + 1)) * 1e-9;

    return img_msg; 
}


void imgCallback(const std::vector<std::string>& imgs_path, 
    std::queue<FeatureTimeMsg>& feature_buf, 
    std::mutex& f_buf){
    for(int i=0; i<imgs_path.size(); i++){
        ImgMsg img_msg = readImg(imgs_path[i]);
        if(first_image_flag) // 如果为第一张图像
        {
            first_image_flag = false;
            first_image_time = img_msg.timestamp;
            last_image_time = img_msg.timestamp;
            continue;
        }
        // 判断图片是否连续
        if(img_msg.timestamp < last_image_time){ // 此条件是读取文件中图像时，有时会将小时间读在后面
            continue;
        }
        if(img_msg.timestamp - last_image_time > 1.0){
            LOG(ERROR) << "image discontinue! reset the feature tracker";
            std::cout<<std::fixed<<std::setprecision(9) << "img_msg.timestamp: " 
                << img_msg.timestamp << " last_image_time: " << last_image_time;
            first_image_flag = true;
            last_image_time = 0;
            pub_count = 1; 
            // RESTART_FLAG = true; // 重启标志
            continue; 
        }
        last_image_time = img_msg.timestamp;

        // 频率控制
        if(round(1.0 * pub_count / (img_msg.timestamp - first_image_time)) <= FREQ){
            PUB_THIS_FRAME = true;
            // 重置频率控制
            if(abs(1.0 * pub_count / (img_msg.timestamp - first_image_flag) - FREQ) < 0.01 * FREQ){
                first_image_time = img_msg.timestamp;
                pub_count = 0;    
            }
        }
        else{
            PUB_THIS_FRAME = false;
        }

        cv::Mat show_img = img_msg.img;
        TicToc t_r;
        for(int i=0; i<NUM_OF_CAM; i++){
            LOG(INFO) << "feature tracker, processing camera: " << i;
            if(i !=1 || !STEREO_TRACK){
                trackerData[i].readImage(img_msg.img.rowRange(ROW * i, ROW * (i+1)), img_msg.timestamp);
            }
            else{
                if(EQUALIZE){
                    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                    clahe->apply(img_msg.img.rowRange(ROW * i, ROW * (i+1)), trackerData[i].cur_img);
                }
                else{
                    trackerData[i].cur_img = img_msg.img.rowRange(ROW * i, ROW * (i+1));    
                }
            }
#if SHOW_UNDISTORTION
            trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
        }

        for(unsigned int i=0; ; i++){
            bool completed = false;
            for(int j=0; j<NUM_OF_CAM; j++){
                if(i != 1 || !STEREO_TRACK){
                    completed = trackerData[i].updateID(i);
                }
            }
            if(!completed) break;
        }

        if(PUB_THIS_FRAME){
            pub_count++;
            FeatureTimeMsg feature_points;
            
            feature_points.timestamp = img_msg.timestamp;
            for(int i=0; i<NUM_OF_CAM; i++){
                auto& un_pts = trackerData[i].cur_un_pts;
                auto& cur_pts = trackerData[i].cur_pts;
                auto& ids = trackerData[i].ids;
                auto& pts_velocity = trackerData[i].pts_velocity;
                for(unsigned int j=0; j<ids.size(); j++){
                    int p_id = ids[j];
                    Eigen::Vector3d p;
                    p << un_pts[j].x, un_pts[j].y, 1;

                    feature_points.points.push_back(p);
                    feature_points.id_of_point.push_back(p_id * NUM_OF_CAM + i);
                    feature_points.u_of_point.push_back(cur_pts[j].x);
                    feature_points.v_of_point.push_back(cur_pts[j].y);
                    feature_points.velocity_x_of_point.push_back(pts_velocity[j].x);
                    feature_points.velocity_y_of_point.push_back(pts_velocity[j].y);
                }
            }
            LOG(INFO)<< "publish feature points: " << feature_points.timestamp;
            if(!init_pub){
                init_pub = true;
            }
            else{
                f_buf.lock();
                feature_buf.push(feature_points);                
                f_buf.unlock();
            }
            if(SHOW_TRACK){
                cv::Mat show_img;
                cv::cvtColor(img_msg.img, show_img, cv::COLOR_GRAY2BGR);
                for(int i=0; i<NUM_OF_CAM; i++){
                    for(unsigned int j=0; j<trackerData[i].ids.size(); j++){
                        cv::circle(show_img, trackerData[i].cur_pts[j], 2, cv::Scalar(0, 0, 255), 2);
                    }
                }
                cv::imshow("show_img", show_img);
                cv::waitKey(1);                
            }
        }
    }
}
} // namespace Data

