#include "plane_detector.hpp"

// #include <pcl/ModelCoefficients.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <pcl/segmentation/sac_segmentation.h>

#include <random>

namespace arslam {
    namespace PlaneDetector {

        std::tuple<Eigen::Vector4d, std::vector<int>> fitPlaneRANSAC(
            const std::vector<Eigen::Vector3d>& pts, double distance_threshold,
            int max_iters) {
            std::default_random_engine rng;
            std::uniform_int_distribution<int> dist(0, (int)pts.size() - 1);

            Eigen::Vector4d best_plane;
            int best_inliers = 0;
            std::vector<int> best_inlier_idx;

            for (int iter = 0; iter < max_iters; ++iter) {
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

                if (n.norm() < 1e-8) continue;
                n.normalize();
                double d = -n.dot(p1);

                // count inliers and record indices
                int count = 0;
                std::vector<int> inlier_idx;
                for (int i = 0; i < (int)pts.size(); ++i) {
                    double dist_to_plane = std::abs(n.dot(pts[i]) + d);
                    if (dist_to_plane < distance_threshold) {
                        count++;
                        inlier_idx.push_back(i);
                    }
                }

                if (count > best_inliers) {
                    best_inliers = count;
                    best_plane.head<3>() = n;
                    best_plane[3] = d;
                    best_inlier_idx = inlier_idx;
                }
            }
            return {best_plane, best_inlier_idx};
        }

        // std::tuple<Eigen::Vector4d, std::vector<int>> fitPlanePCL(
        //     const std::vector<Eigen::Vector3d>& pts, double
        //     distance_threshold, int max_iters) { Eigen::Vector4d plane(0, 0,
        //     0, 0); int inliers = 0; std::vector<int> inlier_idx;

        //     if (pts.size() < 3) {
        //         return {plane, inliers, inlier_idx};
        //     }

        //     // Convert to PCL cloud
        //     using PointT = pcl::PointXYZ;
        //     pcl::PointCloud<PointT>::Ptr cloud(new
        //     pcl::PointCloud<PointT>()); cloud->reserve(pts.size()); for
        //     (const auto& p : pts) {
        //         cloud->push_back(
        //             PointT(float(p.x()), float(p.y()), float(p.z())));
        //     }

        //     // Configure SACSegmentation
        //     pcl::SACSegmentation<PointT> seg;
        //     seg.setOptimizeCoefficients(true);
        //     seg.setModelType(pcl::SACMODEL_PLANE);
        //     seg.setMethodType(pcl::SAC_RANSAC);
        //     seg.setDistanceThreshold(distance_threshold);
        //     seg.setMaxIterations(max_iters);
        //     seg.setInputCloud(cloud);

        //     pcl::ModelCoefficients coeffs;
        //     pcl::PointIndices inliers_idx;
        //     seg.segment(inliers_idx, coeffs);

        //     if (inliers_idx.indices.empty() || coeffs.values.size() < 4) {
        //         // Failed
        //         return {plane, 0, {}};
        //     }

        //     // Copy out plane coeffs (ax+by+cz+d=0)
        //     plane[0] = coeffs.values[0];
        //     plane[1] = coeffs.values[1];
        //     plane[2] = coeffs.values[2];
        //     plane[3] = coeffs.values[3];

        //     // Normalize (optional)
        //     const double normn = plane.head<3>().norm();
        //     if (normn > 1e-12) plane /= normn;

        //     // Copy inliers
        //     inliers = (int)inliers_idx.indices.size();
        //     inlier_idx.reserve(inliers);
        //     for (int idx : inliers_idx.indices) inlier_idx.push_back(idx);

        //     return {plane, inlier_idx};
        // }

    }  // namespace PlaneDetector
}  // namespace arslam
