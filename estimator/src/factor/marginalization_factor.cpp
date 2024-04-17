#include "marginalization_factor.hpp"

void ResidualBlockInfo::Evaluate(){
    residuals.resize(cost_function->num_residuals()); // 根据损失函数中优化变量块调整参数向量

    std::vector<int> block_sizes = cost_function->parameter_block_sizes(); // 获取每个参数块大小
    raw_jacobians = new double *[block_sizes.size()]; // 创建雅可比数据指针
    jacobians.resize(block_sizes.size());

    for(int i=0; i<static_cast<int>(block_sizes.size()); i++){ // 根据参数块数量一个个遍历
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]); // [残差项数量，块中参数数量]
        raw_jacobians[i] = jacobians[i].data();
    }
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

    if(loss_function){ // 核函数
        double residual_scaling_, alpha_sq_norm_;
        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm(); // 二范数
        loss_function->Evaluate(sq_norm, rho); // 核函数的参残差，存在rho

        double sqrt_rho1_ = sqrt(rho[1]);
        // 根据变量残差和核函数残差的一些条件判断加权权重
        if((sq_norm == 0.0) || (rho[2] <= 0.0)){
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        }
        else{
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }
        // 对雅可比矩阵加权
        for(int i=0; i<static_cast<int>(parameter_blocks.size()); i++){
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * 
                (residuals.transpose() * jacobians[i]));
        }
        // 对残差加权
        residuals *= residual_scaling_;
    }
}

MarginalizationInfo::~MarginalizationInfo(){
    for(auto it=parameter_block_data.begin(); it!=parameter_block_data.end(); ++it){
        delete[] it->second;
    }
    for(int i=0; i<(int)factors.size(); i++){
        delete[] factors[i]->raw_jacobians;
        delete factors[i]->cost_function;
        delete factors[i];
    }
}

/**
 * @brief 将residual_block_info 添加到 marginalization_info
 * 就是将不同损失函数对应的优化变量、边缘化位置存入parameter_block_sizes和parameter_block_idx
 * 但parameter_block_idx中存的仅是待边缘化的变量的内存地址，而其对应值全为0
*/
void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo *residual_block_info){
    factors.emplace_back(residual_block_info); // 将一个残差块对象添加到所有边缘化残差容器中

    std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks; // 获取这个此残差块数据指针
    std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes(); // 此残差块大小

    for(int i=0; i<static_cast<int> (residual_block_info->parameter_blocks.size()); i++){ 
        double *addr = parameter_blocks[i]; // 指向数据的指针
        int size = parameter_block_sizes[i]; // 仅有地址不行，需要地址指向数据的长度
        parameter_block_size[reinterpret_cast<long>(addr)] = size; // 将指针强转为数据地址(前面为double)
    }
    // 这里被边缘化变量的id
    for(int i=0; i<static_cast<int>(residual_block_info->drop_set.size()); i++){
        double *addr = parameter_blocks[residual_block_info->drop_set[i]];
        parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
    }
}

/**
 * @brief 前面addResidualBlockInfo已经确定边缘化变量的数量、存储位置、长度，以及被边缘化
 * 变量的数量以及存储位置，由此调用此函数进行边缘化前预处理
*/
void MarginalizationInfo::preMarginalize(){
    for(auto it : factors){
        it->Evaluate(); // 利用c++的多态分别计算各个状态变量构成的残差和雅可比，其实就是调用各个损失函数中的Evaluate()

        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();
        for(int i=0; i<static_cast<int>(block_sizes.size()); i++){
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]); // 优化变量的地址
            int size = block_sizes[i];
            if(parameter_block_data.find(addr) == parameter_block_data.end()){ // parameter_block_data全部变量数据
                double *data = new double[size];
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size); // 重新开辟一块内存
                parameter_block_data[addr] = data; // 将之前获得的地址和新开辟内存的数据关联
            }
        }
    }
}

int MarginalizationInfo::localSize(int size) const{
    return size == 7 ? 6 : size;
}
int MarginalizationInfo::globalSize(int size) const{
    return size == 6 ? 7 : size;
}

void* ThreadsConstructA(void* threadsstruct){
    ThreadsStruct* p = ((ThreadsStruct*)threadsstruct);
    for(auto it : p->sub_factors){
        for(int i=0; i<static_cast<int>(it->parameter_blocks.size()); i++){
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];
            if(size_i == 7){
                size_i == 6;
            }
            Eigen::MatrixXd jacobain_i = it->jacobians[i].leftCols(size_i);
            for(int j=i; i<static_cast<int>(it->parameter_blocks.size()); j++){
                int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
                if(size_j == 7){
                    size_j=6;
                }
                Eigen::MatrixXd jacobain_j = it->jacobians[j].leftCols(size_j);
                if(i == j){
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobain_i.transpose() * jacobain_j;
                }
                else{
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobain_i.transpose() * jacobain_j;
                    p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            p->b.segment(idx_i, size_i) += jacobain_i.transpose() * it->residuals;
        }
    }
    return threadsstruct;
}

void MarginalizationInfo::marginalize(){
    int pos = 0;
    for(auto &it : parameter_block_idx){ // 遍历被边缘的变量块id
        it.second = pos;
        pos += localSize(parameter_block_size[it.first]);
    }

    m = pos; // 需要被边缘掉的变量个数

    for(const auto &it : parameter_block_size){
        if(parameter_block_idx.find(it.first) == parameter_block_idx.end()){ // 如果这个变量不是被边缘化的
            parameter_block_idx[it.first] = pos; // 那这个变量id设为pos
            pos += localSize(it.second);  // pos 加上这个变量长度
        }
    }

    n = pos - m; // 被保留的变量个数
    // 上面两个for循环会进行一个伪排序，将被边缘化变量排在前面，保留变量排在后面

    TicToc t_summing;
    Eigen::MatrixXd A(pos, pos);
    Eigen::VectorXd b(pos);
    A.setZero();
    b.setZero();

    /*
    for (auto it : factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])]);
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])]);
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    A.block(idx_j, idx_i, size_j, size_i) = A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    ROS_INFO("summing up costs %f ms", t_summing.toc());
    */


    //multi thread
    TicToc t_thread_summing;
    pthread_t tids[NUM_THREADS];
    ThreadsStruct threadsstruct[NUM_THREADS];
    int i=0;
    for(auto it : factors){
        threadsstruct[i].sub_factors.push_back(it);
        i++;
        i = i % NUM_THREADS;
    }
    for(int i=0; i<NUM_THREADS; i++){
        TicToc zero_matrix;
        threadsstruct[i].A = Eigen::MatrixXd::Zero(pos, pos);
        threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
        threadsstruct[i].parameter_block_size = parameter_block_size;
        threadsstruct[i].parameter_block_idx = parameter_block_idx;
        int ret = pthread_create(&tids[i], NULL, ThreadsConstructA, (void*)&(threadsstruct[i]));
        if(ret != 0){
            RCUTILS_LOG_WARN("pthread_create error");
            // TODO: ros退出
        }
    }
    for(int i=NUM_THREADS-1; i>=0; i--){
        pthread_join(tids[i], NULL);
        A += threadsstruct[i].A;
        b += threadsstruct[i].b;
    }

    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd(
        (saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() 
        * saes.eigenvectors().transpose();

    Eigen::VectorXd bmm = b.segment(0, m);
    Eigen::MatrixXd Amr = A.block(0, m, m, m);
    Eigen::MatrixXd Arm = A.block(m, 0, m, m);
    Eigen::MatrixXd Arr = A.block(m, m, n, n);
    Eigen::VectorXd brr = b.segment(m, n);
    A = Arr - Arm * Amm_inv * Amr; // 这里的A和b大小应该是发生变化了的，经过上面的边缘化
    b = brr - Arm * Amm_inv * bmm;
    // 更新雅可比和残差
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(
        saes2.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(
        saes2.eigenvalues().array().inverse(), 0));

    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
}

std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double*> &addr_shift){
    std::vector<double *> keep_block_addr;
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();

    for(const auto &it : parameter_block_idx){
        if(it.second >= m){
            keep_block_size.push_back(parameter_block_size[it.first]);
            keep_block_idx.push_back(parameter_block_idx[it.first]);
            keep_block_data.push_back(parameter_block_data[it.first]);
            keep_block_addr.push_back(addr_shift[it.first]);
        } 
    }
    sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

    return keep_block_addr;
}

// 先验残差的损失函数
MarginalizationFactor::MarginalizationFactor(MarginalizationInfo* _marginalization_info):marginalization_info(_marginalization_info)
{
    int cnt = 0;
    for (auto it : marginalization_info->keep_block_size)
    {
        mutable_parameter_block_sizes()->push_back(it);
        cnt += it;
    }
    //printf("residual size: %d, %d\n", cnt, n);
    set_num_residuals(marginalization_info->n);
};

/**
 * @brief 计算所有变量构成的残差和雅可比矩阵
*/
bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const{
    int n = marginalization_info->n;
    int m = marginalization_info->m;
    Eigen::VectorXd dx(n);
    for(int i=0; i<static_cast<int>(marginalization_info->keep_block_size.size()); i++){
        int size = marginalization_info->keep_block_size[i];
        int idx = marginalization_info->keep_block_idx[i] - m;
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size);
        if(size != 7){
            dx.segment(idx, size) = x - x0;
        }
        else{
            dx.segment<3>(idx+0) = x.head<3>() - x0.head<3>();
            dx.segment<3>(idx+3) = 2.0 * Utility::positify(Eigen::Quaterniond(
                x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(
                x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * 
                Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(
                x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        } 
    }
    Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->linearized_jacobians + 
        marginalization_info->linearized_jacobians * dx;
    if(jacobians){
        for(int i=0; i<static_cast<int>(marginalization_info->keep_block_size.size()); i++){
            if (jacobians[i]){
                int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->localSize(size);
                int idx = marginalization_info->keep_block_idx[i] - m;
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
                jacobian.setZero();
                jacobian.leftCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size);
            }
        }
    }
    return true;
}
