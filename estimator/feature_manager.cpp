#include "feature_manager.hpp"

/**
 * @brief 得到跟踪特征点最后一帧的id
*/
int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Eigen::Matrix3d _Rs[])
    : Rs(_Rs)
{
    for(int i=0; i<NUM_OF_CAM; i++){
        ric[i].setIdentity();
    }
}

void FeatureManager::setRic(Eigen::Matrix3d _ric[]){
    for(int i=0; i<NUM_OF_CAM; i++){
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState(){
    feature.clear();
}
/**
 * @brief 得到这一帧上特征点数量
*/
int FeatureManager::getFeatureCount(){
    int cnt = 0;
    for(auto &it : feature){
        it.used_num = it.feature_per_frame.size();
        if(it.used_num >=2 && it.start_frame < WINDOW_SIZE - 2){
            cnt++;
        }
    }  
    return cnt; 
}

/**
 * @brief 当前帧与之前帧进行视差比较，如果当前帧变化小，就会删去倒数第二帧，如果变化很大，就
 * 删去最旧帧，并把这一帧作为新的关键帧，保证滑窗内优化的除最后一帧可能不是关键帧外，其它均为
 * 关键帧。
*/
bool FeatureManager::addFeatureCheckParallax(int frame_count, const std::map<int, 
    std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td){

    RCUTILS_LOG_DEBUG("input feature: %d", (int)image.size());
    RCUTILS_LOG_DEBUG("num of feature: %d", getFeatureCount());

    double parallax_sum = 0; // 所有特征点视差总和
    int parallax_num = 0; // 满足某些条件的特征点个数
    last_track_num = 0; // 被跟踪点的个数
    // 把当前帧图像特征点加入feature容器中，feature按照特征点id组织数据，对于每个id的特征点
    // 记录它被滑动窗口中哪些帧观测到
    for(auto &id_pts : image){ // 便利每个特征点
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td); // 每一帧的属性

        int feature_id = id_pts.first;
        /**
         * @brief STL find_if()函数，查找让这个函数返回true的第一个元素
        */
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it){
            return it.feature_id == feature_id;
        });
        // 返回的是最后一个元素的迭代器， 说明该特征点是第一次出现，在feature容器中创建一个
        // FeaturePerId对象管理这个特征点
        if(it == feature.end()){
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        else if(it->feature_id == feature_id){
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;  // 当前帧跟踪到的特征点数
        }
    }
    // 当前帧帧号小于2，或者跟踪到的特征点数小于20，则把前一帧作为关键帧
    if(frame_count < 2 || last_track_num < 20){
        return false;
    }
    // 计算能被当前帧和其前两帧共同看到的特征点的视差
    for(auto &it_per_id : feature){
        if(it_per_id.start_frame < frame_count - 2 && it_per_id.start_frame 
            + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1){
            parallax_num += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;  
        }
    }

    if(parallax_num > 0){
        return true;
    }
    else{
        RCUTILS_LOG_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        RCUTILS_LOG_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}    

void FeatureManager::debugShow()
{
    RCUTILS_LOG_DEBUG("debug show");
    for (auto &it : feature)
    {
        assert(it.feature_per_frame.size() != 0);
        assert(it.start_frame >= 0);
        assert(it.used_num >= 0);

        RCUTILS_LOG_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            RCUTILS_LOG_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        assert(it.used_num == sum);
    }
}

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> FeatureManager::getCorresponding(
    int frame_count_l, int frame_count_r){
    
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
    for(auto &it : feature){
        if(it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r){
            Eigen::Vector3d a = Eigen::Vector3d::Zero(), b = Eigen::Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;
            b = it.feature_per_frame[idx_r].point;

            corres.push_back(std::make_pair(a, b));
        }
    }
    return corres;
}

/**
 * @brief 设置深度，在void Estimator::double2vector()中用了，如果失败，把solve_flag设置为2
*/
void FeatureManager::setDepth(const Eigen::VectorXd &x){
    int feature_index = -1;
    for(auto &it_per_id : feature){
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // 特征点没有在两帧及以上出现，开始出现帧不是小于滑动窗口大小减2
        if(!(it_per_id.used_num >=2 && it_per_id.start_frame < WINDOW_SIZE - 2)){
            continue;
        }

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        if(it_per_id.estimated_depth < 0){
            it_per_id.solve_flag = 2;
        }
        else{
            it_per_id.solve_flag = 1;
        } 
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth(const Eigen::VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

Eigen::VectorXd FeatureManager::getDepthVector()
{
    Eigen::VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}
/**
 * @brief 利用svd方法对双目进行三角化
*/
void FeatureManager::triangulate(Eigen::Vector3d Ps[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[])
{
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        if (it_per_id.estimated_depth > 0)
            continue;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        assert(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;

            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        assert(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }

    }
}

void FeatureManager::removeOutlier()
{
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Eigen::Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Eigen::Vector3d p_i = frame_i.point;
    Eigen::Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = std::max(ans, sqrt(std::min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}