#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include "System.h"
#include "MapPoint.h"

#include "plane_detector.hpp"

namespace fs = std::filesystem;

// ---------------- Helpers for ORB-SLAM3 return types ----------------
inline Eigen::Matrix4f ToMat4f(const Sophus::SE3f& se3) {
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3,3>(0,0) = se3.so3().matrix();
    T.block<3,1>(0,3) = se3.translation();
    return T;
}
inline Eigen::Matrix4f ToMat4f(const cv::Mat& m) {
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    if (m.empty()) return T;
    if (m.type() == CV_32F) {
        for (int r=0;r<4;++r) for (int c=0;c<4;++c) T(r,c) = m.at<float>(r,c);
    } else {
        for (int r=0;r<4;++r) for (int c=0;c<4;++c) T(r,c) = static_cast<float>(m.at<double>(r,c));
    }
    return T;
}
inline Eigen::Vector3f ToVec3f(const Eigen::Vector3f& v) { return v; }
inline Eigen::Vector3f ToVec3f(const cv::Mat& X) {
    if (X.type() == CV_32F) {
        return { X.at<float>(0), X.at<float>(1), X.at<float>(2) };
    } else {
        return { static_cast<float>(X.at<double>(0)),
                 static_cast<float>(X.at<double>(1)),
                 static_cast<float>(X.at<double>(2)) };
    }
}

// ---------------- EuRoC loader ----------------
static std::pair<std::vector<std::vector<std::string>>, std::vector<double>>
loadImages(const std::string& dataset_dir) {
    std::vector<std::vector<std::string>> image_lists(2);
    std::vector<double> timestamps;

    for (int cam = 0; cam < 2; ++cam) {
        std::string csv = dataset_dir + "/mav0/cam" + std::to_string(cam) + "/data.csv";
        std::ifstream ifs(csv);
        if (!ifs.is_open()) {
            std::cerr << "Failed to open " << csv << std::endl;
            return {{}, {}};
        }
        std::string line;
        std::string dirname = dataset_dir + "/mav0/cam" + std::to_string(cam) + "/data";

        while (std::getline(ifs, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ss(line);
            std::string ts_str, name;
            std::getline(ss, ts_str, ',');
            std::getline(ss, name, ',');
            while (!name.empty() && std::isspace(static_cast<unsigned char>(name.back())))
                name.pop_back();

            image_lists[cam].push_back(dirname + "/" + name);
            if (cam == 0) {
                double t = std::stod(ts_str) * 1e-9; // ns -> s
                timestamps.push_back(t);
            }
        }
    }
    return {image_lists, timestamps};
}

// ---------------- Dump utilities ----------------
static bool SavePlaneTXT(const std::string& path, const Eigen::Vector4d& plane_abcd) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::cerr << "[dump] Failed to open " << path << std::endl;
        return false;
    }
    Eigen::Vector3d n = plane_abcd.head<3>();
    double d = plane_abcd[3];
    const double nn = n.norm();
    if (nn > 1e-12) { n /= nn; d /= nn; }
    ofs << n[0] << " " << n[1] << " " << n[2] << " " << d << "\n";
    return true;
}

static bool SaveLabeledPointsCSV(const std::string& path,
                                 const std::vector<Eigen::Vector3d>& pts_world,
                                 const std::vector<int>& inlier_idx,
                                 const Eigen::Vector4d& plane_abcd)
{
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::cerr << "[dump] Failed to open " << path << std::endl;
        return false;
    }
    std::unordered_set<int> inliers(inlier_idx.begin(), inlier_idx.end());

    Eigen::Vector3d n = plane_abcd.head<3>();
    double d = plane_abcd[3];
    const double nn = n.norm();
    if (nn > 1e-12) { n /= nn; d /= nn; }

    ofs << "x,y,z,label,residual\n";
    for (int i = 0; i < (int)pts_world.size(); ++i) {
        const auto& p = pts_world[i];
        const int label = inliers.count(i) ? 1 : 0;
        const double r = std::abs(n.dot(p) + d);
        ofs << p.x() << "," << p.y() << "," << p.z() << "," << label << "," << r << "\n";
    }
    return true;
}

// ------------------------------- main ---------------------------------
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0]
                  << " /path/to/ORBvoc.txt /path/to/EuRoC_stereo.yaml /path/to/MH_XX\n";
        return -1;
    }
    const std::string voc_path      = argv[1];
    const std::string settings_path = argv[2];
    const std::string dataset_dir   = argv[3];

    ORB_SLAM3::System SLAM(
        voc_path, settings_path, ORB_SLAM3::System::STEREO, /*useViewer=*/true);

    auto lists_and_ts = loadImages(dataset_dir);
    auto& imgs = lists_and_ts.first;
    auto& ts   = lists_and_ts.second;

    if (imgs.empty() || imgs[0].empty() || imgs[1].size() != imgs[0].size() || ts.size() != imgs[0].size()) {
        std::cerr << "Failed to load EuRoC stereo image lists or timestamps.\n";
        return -1;
    }

    fs::create_directories("../export/points");
    fs::create_directories("../export/plane");

    const double target_fps = 30.0;
    const auto frame_dt     = std::chrono::duration<double>(1.0 / target_fps);

    for (size_t i = 0; i < imgs[0].size(); ++i) {
        const auto t_start = std::chrono::steady_clock::now();

        cv::Mat imL = cv::imread(imgs[0][i], cv::IMREAD_UNCHANGED);
        cv::Mat imR = cv::imread(imgs[1][i], cv::IMREAD_UNCHANGED);
        if (imL.empty() || imR.empty()) {
            std::cerr << "Failed to read images at idx=" << i << std::endl;
            continue;
        }

        const double t = ts[i];

        auto Tcw_raw = SLAM.TrackStereo(imL, imR, t);
        Eigen::Matrix4f Tcw = ToMat4f(Tcw_raw);
        (void)Tcw;

        std::vector<ORB_SLAM3::MapPoint*> tracked = SLAM.GetTrackedMapPoints();

        std::vector<Eigen::Vector3d> pts_world;
        pts_world.reserve(tracked.size());
        for (auto* mp : tracked) {
            if (!mp || mp->isBad()) continue;
            const auto Xraw = mp->GetWorldPos();
            const Eigen::Vector3f Xf = ToVec3f(Xraw);
            pts_world.emplace_back(Xf.cast<double>());
        }

        if (pts_world.size() >= 50) {
            const double dist_th = 0.02;
            const int    iters   = 1000;
            auto [plane_abcd, inliers] =
                arslam::PlaneDetector::fitPlaneRANSAC(pts_world, dist_th, iters);

            std::cout << "[Plane] a,b,c,d = " << plane_abcd.transpose()
                      << " | inliers " << inliers.size()
                      << "/" << pts_world.size() << std::endl;

            char fp_csv[256], fp_txt[256];
            std::snprintf(fp_csv, sizeof(fp_csv), "../export/points/%06zu.csv", i);
            std::snprintf(fp_txt, sizeof(fp_txt), "../export/plane/%06zu.txt",  i);
            SaveLabeledPointsCSV(fp_csv, pts_world, inliers, plane_abcd);
            SavePlaneTXT(fp_txt, plane_abcd);
        } else {
            std::cout << "[Info] too few map points this frame: "
                      << pts_world.size() << "\n";
        }

        const auto t_end = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration<double>(t_end - t_start);
        if (elapsed < frame_dt) {
            auto sleep_ms = std::chrono::duration_cast<std::chrono::milliseconds>(frame_dt - elapsed);
            std::this_thread::sleep_for(sleep_ms);
        }
    }

    SLAM.Shutdown();
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    return 0;
}
