#pragma once 

#include <Eigen/Dense>
#include <ceres/ceres.h>

/* 
**  parameters[0]: rev_ddt (t)     in light travelling distance (m)
**  parameters[1]: rev_ddt (t+1)   in light travelling distance (m)
 */
class DdtSmoothFactor : public ceres::SizedCostFunction<1, 1, 1>
{
    public: 
        DdtSmoothFactor(const double weight=1) : weight_(weight) {}
        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    private:
        double weight_;
};