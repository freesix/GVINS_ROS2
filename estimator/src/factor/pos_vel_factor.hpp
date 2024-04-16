#pragma once 

#include <map>
#include <Eigen/Dense>
#include <ceres/ceres.h>

/**
 * @param parameters[0] position(k) in ECEF(地心地固坐标)
 * @param parameters[1] position(k+1) in ECEF 
 * @param parameters[2] velocity(k) in ECEF 
 * @param parameters[3] velocity(k+1) in ECEF
*/
class PosVelFactor : public ceres::SizedCostFunction<3, 7, 7, 9, 9>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PosVelFactor() = delete;
    PosVelFactor(const double delta_t_);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
private:
    double delta_t;
    double info_coeff;
};
