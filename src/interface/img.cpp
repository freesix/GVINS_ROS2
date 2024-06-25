#include "interface/img.hpp"


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

} // namespace Data

