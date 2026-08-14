# Stage 7 Yield Estimation Pipeline Report

Root: `/home/aiseon/zzhslam/outputs/stage7_yield_estimation`

## Implemented Modules

| Module | Script | Status |
|---|---|---|
| RGB-D extraction from rosbag | `scripts/extract_rgbd_from_rosbag.py` | implemented and tested |
| RGB-D citrus point reconstruction | `scripts/rgbd_fruit_reconstruction.py` | implemented and tested with YOLOv8n |
| Fruit instance clustering | `scripts/fruit_instance_clustering.py` | implemented and tested |
| Multi-frame instance fusion | `scripts/fruit_temporal_fusion.py` | implemented and tested in camera-frame smoke mode |
| Fruit size estimation | `scripts/fruit_size_estimation.py` | implemented and tested |
| Yield estimation | `scripts/yield_estimation.py` | implemented and tested |
| Fruit map visualization | `scripts/visualize_fruit_map.py` | implemented; uses Open3D if installed, otherwise PCD can be opened with PCL/CloudCompare |
| One-command smoke pipeline | `run_yield_pipeline.sh` | implemented and tested |

## Rosbag Sensor Status

Source bag: `/home/aiseon/zzhslam/data/baiwan_dataset_2_5_1_6.bag`

- RGB: yes, `/camera/color/image_raw`
- aligned depth: yes, `/camera/aligned_depth_to_color/image_raw`
- IMU: yes, `/camera/imu`
- camera info: not stored in bag
- VINS odometry: not stored in raw bag
- LiDAR: yes, `/velodyne_points`

Detailed report: `reports/ROSBAG_SENSOR_REPORT.md`

## Smoke Test Result

Command:

```bash
/home/aiseon/zzhslam/outputs/stage7_yield_estimation/run_yield_pipeline.sh
```

The script extracts 30 synchronized RGB-D frames and automatically selects the frame with the largest YOLOv8n citrus mask area.

Selected frame:

- `frame_000010`

Single-frame reconstruction:

- fruit points: 17147
- point cloud: `results/frame_fruit_cloud.pcd`
- point csv: `results/fruit_points.csv`
- mask overlay: `visualization/frame_fruit_mask_overlay.png`

Single-frame clustering:

- fruit instances: 12
- noise points: 118
- instance table: `results/frame_instances.csv`
- colored instance PCD: `visualization/frame_instance_cloud.pcd`

Multi-frame smoke fusion:

- frames used: `frame_000009`, `frame_000010`, `frame_000011`
- frame instance counts: 10, 12, 10
- camera-frame fused candidates: 31
- output: `results/global_fruit_instances_multiframe_camera_frame.csv`

Important: the raw rosbag does not contain `/vins_estimator/odometry`, so this multi-frame fusion is a camera-frame smoke test. Metric global fusion requires timestamped VINS poses or another calibrated camera trajectory source.

Size estimation:

- method: PCA ellipsoid
- output: `results/fruit_size.csv`

Yield estimation:

- fruit count: 12
- estimated total mass: 11.058861 kg
- output: `results/yield_prediction.csv`
- summary: `reports/yield_summary.md`

The mass value uses placeholder density and calibration factor from `configs/yield.yaml`. It is a pipeline validation output, not a calibrated agronomic yield result.

## Generated Files

Key results:

- `results/frame_fruit_cloud.pcd`
- `results/fruit_points.csv`
- `results/frame_instances.csv`
- `results/frame_clustered_points.csv`
- `results/global_fruit_instances.csv`
- `results/global_fruit_instances_multiframe_camera_frame.csv`
- `results/fruit_size.csv`
- `results/yield_prediction.csv`

Visualization:

- `visualization/frame_fruit_mask_overlay.png`
- `visualization/frame_instance_cloud.pcd`
- `visualization/frame_000009_instance_cloud.pcd`
- `visualization/frame_000010_instance_cloud.pcd`
- `visualization/frame_000011_instance_cloud.pcd`

## Validation Status

- Python syntax check: passed with `python3 -m compileall scripts`
- RGB-D extraction: passed
- YOLOv8n citrus mask usage: passed
- 3D fruit point cloud generation: passed
- Fruit clustering: passed
- Camera-frame multi-frame fusion: passed
- Fruit size estimation: passed
- Yield table generation: passed

## Missing Calibration / Ground Truth For Final Use

The following data are still required before reporting final fruit count, size, and yield as validated experimental results:

1. Camera intrinsic source inside each rosbag, or a locked calibration file for every recording session.
2. Verified depth scale and depth alignment quality for the RealSense camera.
3. Timestamped VINS camera poses or another reliable trajectory synchronized to RGB-D frames.
4. Camera-to-body/world extrinsic convention for transforming points into a global frame.
5. Manual fruit count ground truth for the evaluated scene.
6. Manual fruit diameter or axis measurements for size validation.
7. Harvested fruit weight for density/calibration-factor fitting.
8. Occlusion protocol for partially observed fruits.
9. Multi-frame association ground truth or repeated-view validation for duplicate-count evaluation.

## Next Recommended Step

Use the existing YOLOv8n or YOLO11n segmentation masks with synchronized VINS odometry output saved during baseline runs, then rerun multi-frame fusion in a true world frame. After that, collect manual fruit count, diameter, and harvested mass to calibrate `density_kg_per_m3` and `calibration_factor`.
