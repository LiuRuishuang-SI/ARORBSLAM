#ifndef _PLANE_DETECTOR_HPP_
#define _PLANE_DETECTOR_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry> 

#include <tuple>
#include <vector>

namespace arslam
{
    namespace PlaneDetector
    {

        // Returns <plane_coefficients [a b c d], inlier_indices>
        std::tuple<Eigen::Vector4d, std::vector<int>> fitPlaneRANSAC(
            const std::vector<Eigen::Vector3d> &points,
            double distance_threshold = 0.05,
            int max_iters = 500);

    } // namespace PlaneDetector
} // namespace arslam

#endif // _PLANE_DETECTOR_HPP_
