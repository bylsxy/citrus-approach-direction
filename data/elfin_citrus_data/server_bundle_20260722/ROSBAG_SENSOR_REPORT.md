# ROSBAG Sensor Report

Dataset: `/home/aiseon/zzhslam/data/baiwan_dataset_2_5_1_6.bag`

## Summary

- Duration: 173 s
- Size: 8.9 GB
- Message count: 46752

## Available Topics

| Sensor / stream | Expected topic | Present | Actual topic | Message type | Count |
|---|---:|---:|---|---|---:|
| RGB image | `/camera/color/image_raw` | yes | `/camera/color/image_raw` | `sensor_msgs/Image` | 5198 |
| Aligned depth | `/camera/aligned_depth_to_color/image_raw` | yes | `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | 5198 |
| Rectified depth | `/camera/depth/image_rect_raw` | no | - | - | 0 |
| Camera info | `/camera/color/camera_info` | no | - | - | 0 |
| IMU | `/imu` | no | `/camera/imu` | `sensor_msgs/Imu` | 34622 |
| VINS odometry | `/vins_estimator/odometry` | no | not stored in raw bag | - | 0 |
| LiDAR point cloud | `/velodyne_points` | yes | `/velodyne_points` | `sensor_msgs/PointCloud2` | 1734 |

## Intrinsics Used For Stage 7

The raw rosbag does not contain `/camera/color/camera_info`. Stage 7 therefore uses the validated D435i configuration from:

`/home/aiseon/zzhslam/VINS-RGBD_1019/config/realsense/realsense_color_config_d435i.yaml`

- `fx = 617.1465454101562`
- `fy = 617.3543701171875`
- `cx = 318.7991638183594`
- `cy = 242.19847106933594`
- image size: `640 x 480`
- depth topic: `/camera/aligned_depth_to_color/image_raw`

## Notes

- RGB, aligned depth, and IMU are available in the raw rosbag.
- Camera intrinsics are not stored in the rosbag and must come from calibration/config files.
- VINS odometry is not stored in this raw rosbag. World-frame fruit fusion can use separately saved VINS trajectories from prior baseline runs, but the initial Stage 7 smoke test reconstructs fruit points in the camera frame.
