#include "plane_detector.hpp"

#include <random>

namespace arslam
{
    namespace PlaneDetector
    {

        std::tuple<Eigen::Vector4d, std::vector<int>> fitPlaneRANSAC(
            const std::vector<Eigen::Vector3d> &pts,
            double distance_threshold,
            int max_iters)
        {
            std::default_random_engine rng;
            std::uniform_int_distribution<int> dist(0, (int)pts.size() - 1);

            Eigen::Vector4d best_plane;
            int best_inliers = 0;
            std::vector<int> best_inlier_idx;

            for (int iter = 0; iter < max_iters; ++iter)
            {
                int i1 = dist(rng);
                int i2 = dist(rng);
                int i3 = dist(rng);
                Eigen::Vector3d p1 = pts[i1];
                Eigen::Vector3d p2 = pts[i2];
                Eigen::Vector3d p3 = pts[i3];

                // manual cross to avoid linker issue
                Eigen::Vector3d v1 = p2 - p1;
                Eigen::Vector3d v2 = p3 - p1;
                Eigen::Vector3d n;
                n[0] = v1[1] * v2[2] - v1[2] * v2[1];
                n[1] = v1[2] * v2[0] - v1[0] * v2[2];
                n[2] = v1[0] * v2[1] - v1[1] * v2[0];

                if (n.norm() < 1e-8)
                    continue;
                n.normalize();
                double d = -n.dot(p1);

                int count = 0;
                std::vector<int> inlier_idx;
                for (int i = 0; i < (int)pts.size(); ++i)
                {
                    double dist_to_plane = std::abs(n.dot(pts[i]) + d);
                    if (dist_to_plane < distance_threshold)
                    {
                        count++;
                        inlier_idx.push_back(i);
                    }
                }

                if (count > best_inliers)
                {
                    best_inliers = count;
                    best_plane.head<3>() = n;
                    best_plane[3] = d;
                    best_inlier_idx = inlier_idx;
                }
            }
            return {best_plane, best_inlier_idx};
        }

    } // namespace PlaneDetector
} // namespace arslam
