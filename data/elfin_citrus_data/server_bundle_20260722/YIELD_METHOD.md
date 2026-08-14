# RGB-D Fruit Reconstruction and Yield Estimation Method

## RGB-D Fruit Reconstruction

The pipeline starts from synchronized RGB and aligned depth images. A semantic segmentation frontend provides a citrus mask. For each valid citrus pixel `(u, v)` with depth `Z`, the point is back-projected using the calibrated pinhole model:

`X = (u - cx) * Z / fx`

`Y = (v - cy) * Z / fy`

`Z = depth`

The initial Stage 7 implementation reconstructs fruit points in the camera frame. If timestamped VINS camera poses are provided, these camera-frame points can be transformed into a common world frame for multi-frame fusion.

## Spatial-Temporal Fruit Association

Single-frame fruit instances are obtained by DBSCAN-style Euclidean clustering on citrus 3D points. Multi-frame fusion associates frame-level instances by center distance and approximate diameter consistency. This removes duplicated observations of the same fruit when camera pose alignment is available.

## Fruit Size Estimation

For each fruit cluster, the default estimator fits a PCA ellipsoid. The three principal extents define ellipsoid axes `(a, b, c)`, and the equivalent diameter is computed as:

`d = 2 * (a * b * c)^(1/3)`

The scripts also preserve bounding-box diameter and can be extended with sphere fitting when fruit point coverage is close to complete.

## Yield Prediction

Fruit volume is estimated as:

`V = 4/3 * pi * a * b * c`

Mass is estimated using:

`mass = density * V * calibration_factor`

The density and calibration factor in `configs/yield.yaml` are placeholders until cultivar-specific measurements are available.

## Limitations

- Accurate global fusion requires reliable camera poses and camera/depth calibration.
- RGB-D observations often capture only the visible fruit surface; size estimates should be calibrated against manual diameter measurements.
- Yield estimates require real harvested weight calibration before agronomic use.
