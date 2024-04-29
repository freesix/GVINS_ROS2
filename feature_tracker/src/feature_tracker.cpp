#include "feature_tracker.hpp"

int FeatureTracker::n_id = 0;

/**
 * @brief 判断此点是否在图像边界内
*/
bool inBorder(const cv::Point2f &pt){
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y 
        && img_y < ROW - BORDER_SIZE;
}

/**
 * @brief 将status为0的点从v中删除
*/
void reduceVector(std::vector<cv::Point2f> &v, std::vector<uchar> status){
    int j = 0;
    for (int i = 0; i < int(v.size()); i++){
        if (status[i]){
            v[j++] = v[i];
        }
    }
    v.resize(j);
}
void reduceVector(std::vector<int> &v, std::vector<uchar> status){
    int j = 0;
    for(int i=0; i<int(v.size()); i++){
        if(status[i]){
            v[j++] = v[i];
        }
    }
    v.resize(j); 
}

/**
 * @brief 构造函数
*/
FeatureTracker::FeatureTracker(){}

/**
 * @brief 对跟踪点排序去除密集点，使用mask进行NMS，半径为30
*/
void FeatureTracker::setMask(){
    if(FISHEYE){
        mask = fisheye_mask.clone();
    }
    else{
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    }

    // 构造(cnt, pts, id)序列
    std::vector<std::pair<int, std::pair<cv::Point2f, int>>> cnt_pts_id;

    for(unsigned int i=0; i<forw_pts.size(); i++){
        cnt_pts_id.push_back(std::make_pair(track_cnt[i], std::make_pair(forw_pts[i], ids[i])));
    }
    // 按照cnt排序
    std::sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const std::pair<int, 
        std::pair<cv::Point2f, int>> &a, const std::pair<int, std::pair<cv::Point2f, int>> &b)
    {
        return a.first > b.first;
    });
    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for(auto &it:cnt_pts_id){
        if(mask.at<uchar>(it.second.first) == 255){
            // 当前特征点位置对应的mask值为255，则保留当前特征点
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            // 在mask中当前特征点周围半径为MIN_DIST的区域内mask值设为0
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

/**
 * @brief 添加新检测到的特征点
*/
void FeatureTracker::addPoints(){
    for(auto &p:n_pts){
        forw_pts.push_back(p);
        ids.push_back(-1); // 新提取特征点id设为-1
        track_cnt.push_back(1); // 新提取的特征点被跟踪次数为1
    }
}

/**
 * @brief 对图像使用光流法进行特征点跟踪
 *        calcOpticalFlowPyrLK() LK金字塔光流法
 *        setMask() 对跟踪点进行排序，设置mask
 *        rejectWithF() 通过基本矩阵剔除outliers
 *        goodFeaturesToTrack() 添加特征点(shi-tomasi角点)，确保每帧都有足够的特征点
 *        addPoints()添加新的追踪点
 *        undistortedPoints() 对角点图像坐标去畸变矫正，并计算每个角点的速度
 * @param _img 输入图像
 * @param _cur_time 当前时间
*/
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time){
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    if(EQUALIZE) // 是否均衡化
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "CLAHE costs: %f", t_c.toc());
    }
    else{
        img = _img;
    }

    if(forw_img.empty()){
        // forw_img为空，说明是第一帧图像，同时将读入图像赋给prev_img, cur_img
        prev_img = cur_img = forw_img = img;
    }
    else{
        forw_img = img;
    }

    forw_pts.clear();

    if(cur_pts.size() > 0){
        TicToc t_o;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21,21), 3);
        // 将位于图像边界外的点标记为0
        for(int i=0; i<int(forw_pts.size()); i++){
            if(status[i] && !inBorder(forw_pts[i])){
                status[i] = 0;
            }
        }
        // 根据status把跟踪失败的点去除
        // 不仅要从当前帧的forw_pts中去除，还要从cur_un_pts、prev_pts和cur_pts中去除
        // prev_pts和cur_pts中的特征点是一一对应的
        // 记录特征点id的ids和记录被跟踪次数的track_cnt也要相应去除
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "track costs: %f", t_o.toc());
    }

    // 光流追踪成功，特征点被跟踪的次数加1
    for(auto &n:track_cnt){
        n++;
    }

    // PUB_THIS_FRAME为true，说明需要发布跟踪结果
    if(PUB_THIS_FRAME){
        // 通过基本矩阵剔除外点
        rejectWithF();
 
        RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "set mask begins");
        TicToc t_m;
        setMask(); 
        RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "set mask costs: %f", t_m.toc());

        RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if(n_max_cnt > 0){
            if(mask.empty()){
                std::cout<<"mask is empty"<<std::endl;
            }
            if(mask.type() != CV_8UC1){
                std::cout<<"mask type wrong"<<std::endl;
            }
            if(mask.size() != forw_img.size()){
                std::cout<<"mask size wrong"<<std::endl;
                // for debug
                std::cout<<"mask size:"<<mask.size()<<std::endl;
                std::cout<<"forw_img size:"<<forw_img.size()<<std::endl;
            }
            /** 
             *void cv::goodFeaturesToTrack(    在mask中不为0的区域检测新的特征点
             *   InputArray  image,              输入图像
             *   OutputArray     corners,        存放检测到的角点的vector
             *   int     maxCorners,             返回的角点的数量的最大值
             *   double  qualityLevel,           角点质量水平的最低阈值（范围为0到1，质量最高角点的水平为1），小于该阈值的角点被拒绝
             *   double  minDistance,            返回角点之间欧式距离的最小值
             *   InputArray  mask = noArray(),   和输入图像具有相同大小，类型必须为CV_8UC1,用来描述图像中感兴趣的区域，只在感兴趣区域中检测角点
             *   int     blockSize = 3,          计算协方差矩阵时的窗口大小
             *   bool    useHarrisDetector = false,  指示是否使用Harris角点检测，如不指定则使用shi-tomasi算法
             *   double  k = 0.04                Harris角点检测需要的k值
             *)   
             */
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else{
            n_pts.clear();
        }
        RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "detect feature costs: %f", t_t.toc());

        RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "add feature begins");
        TicToc t_a;
        addPoints();
        RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "add feature costs: %f", t_a.toc());
    }

    // 当下一帧到来，当前帧数据就变为下一帧数据
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;    
}

/**
 * @brief 通过基本矩阵剔除外点，将图像坐标转化为归一化坐标
*/
void FeatureTracker::rejectWithF()
{
    if(forw_pts.size() >= 8){
        RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "FM ransac begins");
        TicToc t_f;
        std::vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for(unsigned int i=0; i<cur_pts.size(); i++){
            Eigen::Vector3d tmp_p;
            // 根据不同的相机模型将二维坐标转换到三维坐标
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            // 转换为归一化像素坐标
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        std::vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "FM ransac: %d -> %lu: %f", size_a, forw_pts.size(),
                    1.0 * forw_pts.size() / size_a);
        RCLCPP_DEBUG(rclcpp::get_logger("rclcpp"), "FM ransac costs: %f", t_f.toc());
    }
}

/**
 * @brief 更新特征点id
 * @param i 特征点id
*/
bool FeatureTracker::updateID(unsigned int i){
    if(i < ids.size()){
        if(ids[i] == -1){
            ids[i] = n_id++;
            return true;
        }
    }
    else{ 
        return false;
    }
}

/**
 * @brief 读取相机内参
*/
void FeatureTracker::readIntrinsicParameter(const std::string &calib_file){
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "reading paramerter of camera %s", calib_file.c_str());
    m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

/**
 * @berif 显示去畸变矫正后的特征点
 * @param name 图像帧名称
*/
void FeatureTracker::showUndistortion(const std::string &name){
    cv::Mat undistortedImg(ROW+600, COL+600, CV_8UC3, cv::Scalar(0));
    std::vector<Eigen::Vector2d> distortedp, undistortedp;
    for(int i=0; i<COL; i++){
        for(int j=0; j<ROW; j++){
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x()/b.z(), b.y()/b.z()));
        }
    }

    for(int i=0; i<int(undistortedp.size()); i++){
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2.0;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2.0;
        pp.at<float>(2, 0) = 1;

        if(pp.at<float>(1,0) + 300 >= 0 && pp.at<float>(1,0) + 300 < ROW + 600
            && pp.at<float>(0,0) + 300 >= 0 && pp.at<float>(0,0) + 300 < COL + 600){
            undistortedImg.at<uchar>(pp.at<float>(1,0)+300, pp.at<float>(0,0)+300) = 
                cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else{
            std::cout<<"undistorted point out of image"<<std::endl;
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);

}

/**
 * @brief 对角点图像坐标去畸变矫正，转化到归一化坐标系，并计算每个角点速度
*/
void FeatureTracker::undistortedPoints(){
    cur_un_pts.clear();
    cur_un_pts_map.clear();

    for(unsigned int i=0; i<cur_pts.size(); i++){
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;

        m_camera->liftProjective(a, b);

        // 再延伸到归一化平面上
        cur_un_pts.push_back(cv::Point2f(b.x()/b.z(), b.y()/b.z()));
        cur_un_pts_map.insert(std::make_pair(ids[i], cv::Point2f(b.x()/b.z(), b.y()/b.z())));
    }

    // 计算每个特征点的速度存到pts_velocity
    if(!prev_un_pts_map.empty()){
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for(unsigned int i=0; i<cur_un_pts.size(); i++){
            if(ids[i] != -1){
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if(it != prev_un_pts_map.end()){
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else{
                    pts_velocity.push_back(cv::Point2f(0, 0));
                }
            }
            else{
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else{
        for(unsigned int i=0; i<cur_un_pts.size(); i++){
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}