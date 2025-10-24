#!/usr/bin/env python3
import os, glob
import numpy as np
import pandas as pd
import cv2

# ======= EDIT THESE =======
DATASET_DIR = "/path/to/MH_01_easy"
SETTINGS_YAML = "/path/to/EuRoC.yaml"
TRAJ_PATH = "/path/to/output/CameraTrajectory.txt"
POINTS_DIR = "/path/to/output/points"
TIME_TOL = 1.0 / 30.0  # seconds, accept pose within this gap
START_INDEX = 0
END_INDEX = None
DRAW_RADIUS = 3
INLIER_COLOR = (0, 255, 0)  # BGR
OUTLIER_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
# ==========================

def load_intrinsics_and_dist(yaml_path):
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(yaml_path)
    candidates = [
        ("Camera.fx", "Camera.fy", "Camera.cx", "Camera.cy",
         ["Camera.k1", "Camera.k2", "Camera.p1", "Camera.p2", "Camera.k3"]),
        ("Cam0.fx", "Cam0.fy", "Cam0.cx", "Cam0.cy",
         ["Cam0.k1", "Cam0.k2", "Cam0.p1", "Cam0.p2", "Cam0.k3"]),
        ("Camera1.fx", "Camera1.fy", "Camera1.cx", "Camera1.cy",
         ["Camera1.k1", "Camera1.k2", "Camera1.p1", "Camera1.p2", "Camera1.k3"]),
    ]
    K = None
    D = np.zeros(5, dtype=np.float64)
    for kfx, kfy, kcx, kcy, dk in candidates:
        nfx = fs.getNode(kfx)
        nfy = fs.getNode(kfy)
        ncx = fs.getNode(kcx)
        ncy = fs.getNode(kcy)
        if nfx.empty() or nfy.empty() or ncx.empty() or ncy.empty(): continue
        fx = float(nfx.real())
        fy = float(nfy.real())
        cx = float(ncx.real())
        cy = float(ncy.real())
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        for i, key in enumerate(dk):
            node = fs.getNode(key)
            if not node.empty(): D[i] = float(node.real())
        break
    fs.release()
    if K is None:
        raise RuntimeError("Failed to read K from YAML (fx,fy,cx,cy).")
    return K, D


def load_cam0_list(dataset_dir):
    csv = os.path.join(dataset_dir, "mav0/cam0/data.csv")
    imgdir = os.path.join(dataset_dir, "mav0/cam0/data")
    ts, paths = [], []
    with open(csv, "r") as f:
        for line in f:
            if not line or line[0] == '#': continue
            parts = line.strip().split(",")
            if len(parts) < 2: continue
            tns, name = parts[:2]
            ts.append(float(tns) * 1e-9)  # ns -> s
            paths.append(os.path.join(imgdir, name.strip()))
    return np.array(ts, dtype=np.float64), paths


def quat_to_rot(q):  # q = [w,x,y,z]
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    if Nq < 1e-12: return np.eye(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    return np.array([
        [1.0 - (yY + zZ), xY - wZ, xZ + wY],
        [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
        [xZ - wY, yZ + wX, 1.0 - (xX + yY)]
    ], dtype=np.float64)


def load_cameratraj(path):
    # Format: t tx ty tz qx qy qz qw  (Twc)
    times = []
    Twcs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == '#': continue
            parts = line.split()
            if len(parts) < 8: continue
            t = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])
            R = quat_to_rot(np.array([qw, qx, qy, qz], dtype=np.float64))
            Twc = np.eye(4, dtype=np.float64)
            Twc[:3, :3] = R
            Twc[:3, 3] = [tx, ty, tz]
            times.append(t)
            Twcs.append(Twc)
    if not times:
        raise RuntimeError(f"No poses read from {path}")
    return np.array(times, dtype=np.float64), np.stack(Twcs, axis=0)


def invert44(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def nearest_pose_by_time(t_img, traj_times):
    idx = int(np.argmin(np.abs(traj_times - t_img)))
    dt = float(abs(traj_times[idx] - t_img))
    return idx, dt


def project_points(Pw, K, Tcw):
    R = Tcw[:3, :3]
    t = Tcw[:3, 3]
    Pc = (R @ Pw.T + t.reshape(3, 1)).T
    z = Pc[:, 2]
    ok = z > 1e-6
    u = np.empty_like(z)
    v = np.empty_like(z)
    u.fill(np.nan)
    v.fill(np.nan)
    x = Pc[ok, 0] / z[ok]
    y = Pc[ok, 1] / z[ok]
    u[ok] = K[0, 0] * x + K[0, 2]
    v[ok] = K[1, 1] * y + K[1, 2]
    return u, v, ok


def main():
    K, D = load_intrinsics_and_dist(SETTINGS_YAML)
    ts_img, img_paths = load_cam0_list(DATASET_DIR)
    traj_times, Twcs = load_cameratraj(TRAJ_PATH)

    csvs = sorted(glob.glob(os.path.join(POINTS_DIR, "*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No CSVs under {POINTS_DIR}")
    idxs = sorted(int(os.path.splitext(os.path.basename(p))[0]) for p in csvs)
    if END_INDEX is not None:
        idxs = [i for i in idxs if START_INDEX <= i <= END_INDEX]
    else:
        idxs = [i for i in idxs if i >= START_INDEX]

    # precompute undistort maps once (assumes all frames same size)
    test_im = cv2.imread(img_paths[0], cv2.IMREAD_COLOR)
    h, w = test_im.shape[:2]
    newK = K.copy()
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, newK, (w, h), cv2.CV_32FC1)

    for i in idxs:
        if i >= len(img_paths): break
        img_raw = cv2.imread(img_paths[i], cv2.IMREAD_COLOR)
        if img_raw is None:
            print(f"[WARN] cannot read image {img_paths[i]}")
            continue

        img = cv2.remap(img_raw, map1, map2, interpolation=cv2.INTER_LINEAR)

        # load points for frame i
        df = pd.read_csv(os.path.join(POINTS_DIR, f"{i:06d}.csv"),
                         usecols=["x", "y", "z", "label"])
        Pw = df[["x", "y", "z"]].to_numpy(np.float64)
        labels = df["label"].to_numpy(np.int32)

        # find nearest pose by time
        t_img = ts_img[i]
        j, dt = nearest_pose_by_time(t_img, traj_times)
        if dt > TIME_TOL:
            # too far → skip (or relax TIME_TOL)
            info = f"frame {i}  NO POSE (Δt={dt * 1000:.1f}ms)"
            vis = img.copy()
            cv2.putText(vis, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)
            cv2.imshow("reprojection (cam0)", vis)
            if (cv2.waitKey(30) & 0xFF) in (ord('q'), 27): break
            continue

        Twc = Twcs[j]  # from CameraTraj (assumed Twc)
        Tcw = invert44(Twc)  # world to camera

        # project
        u, v, ok = project_points(Pw, newK, Tcw)
        H, W = img.shape[:2]
        inside = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        vis = img.copy()
        pts_in = np.where(inside & (labels == 1))[0]
        pts_out = np.where(inside & (labels == 0))[0]

        for idx in pts_out:
            cv2.circle(vis, (int(round(u[idx])), int(round(v[idx]))), DRAW_RADIUS, OUTLIER_COLOR, -1, cv2.LINE_AA)
        for idx in pts_in:
            cv2.circle(vis, (int(round(u[idx])), int(round(v[idx]))), DRAW_RADIUS, INLIER_COLOR, -1, cv2.LINE_AA)

        info = f"frame {i}  drawn:{len(pts_in) + len(pts_out)}  in:{len(pts_in)}  out:{len(pts_out)}  Δt={dt * 1000:.1f}ms"
        cv2.putText(vis, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.imshow("reprojection (cam0, undistorted)", vis)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord('q'), 27): break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
