DockerFile of ORB-SLAM3
=======================

### Environment

* Ubuntu 22.04


### Build the image

```bash
sudo docker build -t orb-slam3-22.04 .
```

### Usage of ORB_SLAM3

1. Set Permission for X Server

   ```bash
   xhost +
   ```

2. Run Image

   ```bash
   sudo docker run --name orb-slam3-22.04 -it \
      -e DISPLAY=$DISPLAY \
      -v /tmp/.X11-unix/:/tmp/.X11-unix 
      --privileged \
      --volume=/dev:/dev \
      -v <PATH_TO_DATASET>:<PATH_TO_DATASET> orb-slam3-22.04 /bin/bash
   ```

   If container is not started

   ```bash
   sudo docker start orb-slam3-22.04
   sudo docker exec -it orb-slam3-22.04 /bin/bash
   ```

   If multiple terminals are necessary (in another terminal)

   ```bash
   sudo docker exec -it orb-slam3-22.04 /bin/bash
   ```

3. Run ORB_SLAM3 Demo

   ```bash
   cd ORB_SLAM3
   ```

   Launch MH01 with Stereo sensor

   ```bash
   ./Examples/Stereo/stereo_euroc \
      ./Vocabulary/ORBvoc.txt \
      ./Examples/Stereo/EuRoC.yaml \
      <PATH_TO_DATASET>/MH_01_easy \
      ./Examples/Stereo/EuRoC_TimeStamps/MH01.txt \
      dataset-MH01_stereo
   ```

   Launch MH01 with Stereo-Inertial sensor

   ```bash
   ./Examples/Stereo-Inertial/stereo_inertial_euroc \
      ./Vocabulary/ORBvoc.txt \
      ./Examples/Stereo-Inertial/EuRoC.yaml \
      <PATH_TO_DATASET>/MH_01_easy \
      ./Examples/Stereo-Inertial/EuRoC_TimeStamps/MH01.txt \
      dataset-MH01_stereoi
   ```

### Usage of Our Script

1. Build

   ```bash
   cd ARORBSLAM
   mkdir build && cd build
   cmake ..
   make -j
   ```

2. Run

   ```bash
   # from ARORBSLAM/build
   ./stereo_euroc_plane \
      ../../ORB_SLAM3/Vocabulary/ORBvoc.txt \
      ../../ORB_SLAM3/Examples/Stereo/EuRoC.yaml \
      <PATH_TO_DATASET>/MH_01_easy
   ```

   Args:

   - Path to `ORBvoc.txt`
   - Path to EuRoC stereo YAML (camera intrinsics/extrinsics)
   - EuRoC dataset directory

   This will write point cloud of each frame and the estimated plane into `export` folder:
   - Per-frame point cloud (`export/points/000000.csv`)
      - Header: `x`,`y`,`z`,`label`,`residual`
      - `x`,`y`,`z`: 3D map point in world coordinates
      - label: `1` = inlier , `0` = outlier
      - residual: distance to the normalized plane
   - Per-frame plane (`export/plane/000000.txt`)
      - One line: `nx` `ny` `nz` `d`
      - Plane is normalized so `‖n‖=1`, equation is `n·X + d = 0`