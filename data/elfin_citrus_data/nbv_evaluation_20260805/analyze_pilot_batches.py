#!/usr/bin/env python3
"""Audit real citrus pilot batches and replay the Semantic NBV planners.

This script intentionally separates engineering proxy metrics from formal PCO.
The multi-view reference is the union of observations in the same captured
scene, so it is circular and must never be described as independent truth.
"""

from __future__ import division

import argparse
import csv
import hashlib
import json
import math
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from elfin_vision.nbv_evaluation import (
        STRATEGIES,
        _planner_order,
        _prepare_geometries,
        load_batch,
    )
except ImportError:
    from nbv_evaluation import (  # type: ignore
        STRATEGIES,
        _planner_order,
        _prepare_geometries,
        load_batch,
    )


VOXEL_SIZE_M = 0.003
COVERAGE_VOXEL_M = 0.010
MAX_ACTIONS = 10
SAMPLE_STRIDE = 4
MAX_POINTS = 20000
MAX_RANGE_M = 2.5
CLUSTER_EPS_M = 0.075

# Freeze clustering so the replay does not change with the installed sklearn
# version. The core evaluator records this backend in every attention trace.
os.environ["ELFIN_NBV_OPTICS_BACKEND"] = "deterministic"


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _mean(values):
    values = [float(value) for value in values if value is not None and
              math.isfinite(float(value))]
    return float(np.mean(values)) if values else None


def _median(values):
    values = [float(value) for value in values if value is not None and
              math.isfinite(float(value))]
    return float(np.median(values)) if values else None


def _percentile(values, percentile):
    values = [float(value) for value in values if value is not None and
              math.isfinite(float(value))]
    return float(np.percentile(values, percentile)) if values else None


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_image(archive, name, flags=cv2.IMREAD_UNCHANGED):
    if not name or name not in archive.namelist():
        return None
    encoded = np.frombuffer(archive.read(name), dtype=np.uint8)
    return cv2.imdecode(encoded, flags)


def _load_auxiliary(path, batch):
    auxiliary = {}
    with zipfile.ZipFile(path, "r") as archive:
        for view in batch["views"]:
            index = int(view["index"])
            record = view.get("record") or {}
            files = record.get("files") or {}
            color = _decode_image(archive, files.get("color"), cv2.IMREAD_COLOR)
            imu = {}
            imu_name = files.get("imu")
            if imu_name and imu_name in archive.namelist():
                imu = json.loads(archive.read(imu_name).decode("utf-8"))
            auxiliary[index] = {"color": color, "imu": imu}
    return auxiliary


def _stored_label(class_names, wanted):
    """Return the image label ID from load_batch's normalized class map."""
    wanted = str(wanted).strip().lower()
    for raw, name in class_names.items():
        if str(name).strip().lower() == wanted:
            return int(raw)
    return None


def _binary_entropy(probabilities):
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    if not values.size:
        return np.empty((0,), dtype=np.float64)
    return -(values * np.log2(values) + (1.0 - values) * np.log2(1.0 - values))


def _image_metrics(color):
    if color is None or color.ndim != 3:
        return {
            "brightness_mean": None,
            "contrast_std": None,
            "sharpness_laplacian_var": None,
            "dark_clip_ratio": None,
            "bright_clip_ratio": None,
            "saturation_mean": None,
            "red_clip_ratio": None,
        }
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    return {
        "brightness_mean": float(np.mean(gray)),
        "contrast_std": float(np.std(gray)),
        "sharpness_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "dark_clip_ratio": float(np.mean(gray <= 5)),
        "bright_clip_ratio": float(np.mean(gray >= 250)),
        "saturation_mean": float(np.mean(hsv[:, :, 1])),
        "red_clip_ratio": float(np.mean(color[:, :, 2] >= 250)),
    }


def _sample_rate(samples):
    stamps = [float(item.get("stamp_sec", 0.0)) for item in samples]
    stamps = np.asarray([value for value in stamps if math.isfinite(value) and value > 0.0])
    if len(stamps) < 2:
        return None
    deltas = np.diff(np.sort(stamps))
    deltas = deltas[deltas > 1e-6]
    return float(1.0 / np.median(deltas)) if len(deltas) else None


def _vector_array(samples, key):
    rows = []
    for sample in samples:
        value = sample.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            vector = np.asarray(value, dtype=np.float64)
            if np.all(np.isfinite(vector)):
                rows.append(vector)
    return np.asarray(rows, dtype=np.float64).reshape((-1, 3))


def _imu_metrics(imu):
    gyro_samples = list((imu or {}).get("gyro_samples") or [])
    accel_samples = list((imu or {}).get("accel_samples") or [])
    gyro = _vector_array(gyro_samples, "angular_velocity_rad_s")
    accel = _vector_array(accel_samples, "linear_acceleration_m_s2")
    gyro_norm = np.linalg.norm(gyro, axis=1) if len(gyro) else np.asarray([])
    accel_norm = np.linalg.norm(accel, axis=1) if len(accel) else np.asarray([])
    return {
        "imu_gyro_samples": int(len(gyro_samples)),
        "imu_accel_samples": int(len(accel_samples)),
        "imu_gyro_rate_hz": _sample_rate(gyro_samples),
        "imu_accel_rate_hz": _sample_rate(accel_samples),
        "gyro_norm_mean_rad_s": float(np.mean(gyro_norm)) if len(gyro_norm) else None,
        "gyro_norm_rms_rad_s": float(np.sqrt(np.mean(gyro_norm ** 2))) if len(gyro_norm) else None,
        "gyro_norm_p95_rad_s": float(np.percentile(gyro_norm, 95)) if len(gyro_norm) else None,
        "gyro_norm_max_rad_s": float(np.max(gyro_norm)) if len(gyro_norm) else None,
        "accel_norm_mean_m_s2": float(np.mean(accel_norm)) if len(accel_norm) else None,
        "accel_norm_std_m_s2": float(np.std(accel_norm)) if len(accel_norm) else None,
        "accel_norm_p95_m_s2": float(np.percentile(accel_norm, 95)) if len(accel_norm) else None,
        "accel_gravity_abs_error_m_s2": (
            float(abs(np.mean(accel_norm) - 9.80665)) if len(accel_norm) else None),
    }


def _detection_metrics(detections, image_shape):
    detections = list(detections or [])
    confidence = [float(item.get("confidence", 0.0)) for item in detections]
    valid = [item for item in detections if bool(item.get("target_point_valid"))]
    height, width = image_shape
    areas = []
    for item in detections:
        bbox = item.get("bbox") or []
        if len(bbox) == 4:
            x1, y1, x2, y2 = [float(value) for value in bbox]
            areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1) /
                         max(1.0, float(width * height)))
    return {
        "detection_count": int(len(detections)),
        "detection_depth_valid_count": int(len(valid)),
        "detection_depth_valid_ratio": (float(len(valid)) / len(detections)
                                          if detections else None),
        "detection_conf_mean": _mean(confidence),
        "detection_conf_median": _median(confidence),
        "detection_conf_min": min(confidence) if confidence else None,
        "detection_conf_max": max(confidence) if confidence else None,
        "detection_bbox_area_sum_ratio": float(sum(areas)),
        "detection_bbox_area_max_ratio": max(areas) if areas else None,
    }


def _frame_metrics(batch, view, aux):
    depth = np.asarray(view["depth_mm"])
    labels = np.asarray(view["semantic_labels"])
    confidence = np.asarray(view["semantic_confidence"], dtype=np.float64)
    instances = np.asarray(view["instance_ids"])
    color = aux.get("color")
    record = view.get("record") or {}
    class_names = batch.get("class_names") or {}
    citrus_label = _stored_label(class_names, "citrus")
    tree_label = _stored_label(class_names, "tree")
    finite_depth = np.isfinite(depth)
    nonzero_depth = finite_depth & (depth > 0)
    valid_depth = finite_depth & (depth >= 100) & (depth <= MAX_RANGE_M * 1000.0)
    out_of_range_depth = nonzero_depth & ~valid_depth
    citrus = labels == citrus_label if citrus_label is not None else np.zeros_like(labels, bool)
    tree = labels == tree_label if tree_label is not None else np.zeros_like(labels, bool)

    def mask_depth_ratio(mask):
        return float(np.mean(valid_depth[mask])) if np.any(mask) else None

    def mask_depth_values(mask):
        values = depth[mask & valid_depth].astype(np.float64)
        return values

    citrus_depth = mask_depth_values(citrus)
    tree_depth = mask_depth_values(tree)
    citrus_confidence = confidence[citrus]
    semantic_entropy = _binary_entropy(citrus_confidence)
    nonzero_instances = np.unique(instances[instances > 0])
    citrus_instance_depth_ratios = []
    citrus_instances_with_20_depth_pixels = 0
    citrus_instance_count = 0
    tree_instance_count = 0
    for instance_id in nonzero_instances:
        instance_mask = instances == instance_id
        citrus_pixels = int(np.sum(instance_mask & citrus))
        tree_pixels = int(np.sum(instance_mask & tree))
        if citrus_pixels > 0 and citrus_pixels >= tree_pixels:
            citrus_instance_count += 1
            valid_pixels = int(np.sum(instance_mask & citrus & valid_depth))
            citrus_instance_depth_ratios.append(
                float(valid_pixels) / float(citrus_pixels))
            if valid_pixels >= 20:
                citrus_instances_with_20_depth_pixels += 1
        elif tree_pixels > 0:
            tree_instance_count += 1
    sync = record.get("sync") or {}
    model = record.get("model") or {}
    row = {
        "group": int(batch["group_number"]),
        "scene_id": str(batch["scene_id"]),
        "view": int(view["index"]),
        "stamp_sec": float(view["stamp_sec"]),
        "depth_nonzero_ratio": float(np.mean(nonzero_depth)),
        "depth_valid_ratio": float(np.mean(valid_depth)),
        "depth_out_of_range_nonzero_ratio": float(np.mean(out_of_range_depth)),
        "citrus_pixel_ratio": float(np.mean(citrus)),
        "tree_pixel_ratio": float(np.mean(tree)),
        "citrus_depth_valid_ratio": mask_depth_ratio(citrus),
        "tree_depth_valid_ratio": mask_depth_ratio(tree),
        "citrus_depth_out_of_range_ratio": (
            float(np.mean(out_of_range_depth[citrus])) if np.any(citrus) else None),
        "tree_depth_out_of_range_ratio": (
            float(np.mean(out_of_range_depth[tree])) if np.any(tree) else None),
        "citrus_valid_depth_pixels": int(len(citrus_depth)),
        "tree_valid_depth_pixels": int(len(tree_depth)),
        "citrus_depth_median_mm": _median(citrus_depth.tolist()),
        "citrus_depth_p10_mm": _percentile(citrus_depth.tolist(), 10),
        "citrus_depth_p90_mm": _percentile(citrus_depth.tolist(), 90),
        "tree_depth_median_mm": _median(tree_depth.tolist()),
        "semantic_citrus_conf_mean": _mean(citrus_confidence.tolist()),
        "semantic_citrus_entropy_mean_bits": _mean(semantic_entropy.tolist()),
        "semantic_instance_count": int(len(nonzero_instances)),
        "semantic_citrus_instance_count": int(citrus_instance_count),
        "semantic_tree_instance_count": int(tree_instance_count),
        "semantic_citrus_instances_with_20_depth_pixels": int(
            citrus_instances_with_20_depth_pixels),
        "semantic_citrus_instance_20px_depth_support_ratio": (
            float(citrus_instances_with_20_depth_pixels) / citrus_instance_count
            if citrus_instance_count else None),
        "semantic_citrus_instance_depth_ratio_median": _median(
            citrus_instance_depth_ratios),
        "inference_ms": float(model.get("inference_ms", 0.0) or 0.0),
        "rgb_depth_delta_s": float(sync.get("rgb_depth_delta_s", 0.0) or 0.0),
        "target_color_delta_s": float(sync.get("target_color_delta_s", 0.0) or 0.0),
        "joint_stamp_delta_s": float(sync.get("joint_stamp_delta_s", 0.0) or 0.0),
        "timestamp_history_fallback": bool(sync.get("timestamp_history_fallback", False)),
        "model_status": str(model.get("status", "")),
    }
    row.update(_image_metrics(color))
    row.update(_imu_metrics(aux.get("imu") or {}))
    row.update(_detection_metrics(view.get("detections"), depth.shape))
    return row


def _rotation_angle_deg(first, second):
    relative = np.asarray(first, dtype=np.float64).T.dot(
        np.asarray(second, dtype=np.float64))
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _circular_coverage_deg(angles):
    values = sorted(float(value) % 360.0 for value in angles)
    if len(values) < 2:
        return 0.0
    gaps = [values[index + 1] - values[index] for index in range(len(values) - 1)]
    gaps.append(values[0] + 360.0 - values[-1])
    return float(360.0 - max(gaps))


def _pose_summary(batch, target_center):
    poses = [np.asarray(view["pose_matrix"], dtype=np.float64)
             for view in batch["views"]]
    origins = np.asarray([pose[:3, 3] for pose in poses], dtype=np.float64)
    translations = np.linalg.norm(np.diff(origins, axis=0), axis=1)
    rotations = [_rotation_angle_deg(poses[index - 1][:3, :3],
                                     poses[index][:3, :3])
                 for index in range(1, len(poses))]
    pairwise = []
    for first in range(len(origins)):
        for second in range(first + 1, len(origins)):
            pairwise.append(float(np.linalg.norm(origins[first] - origins[second])))
    delta = origins - np.asarray(target_center, dtype=np.float64).reshape(1, 3)
    azimuth = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    elevation = np.degrees(np.arctan2(delta[:, 2], np.maximum(horizontal, 1e-9)))
    near_duplicates = 0
    for first in range(len(poses)):
        for second in range(first + 1, len(poses)):
            distance = float(np.linalg.norm(origins[first] - origins[second]))
            angle = _rotation_angle_deg(poses[first][:3, :3], poses[second][:3, :3])
            if distance < 0.02 and angle < 5.0:
                near_duplicates += 1
    return {
        "origin_min_m": np.min(origins, axis=0).tolist(),
        "origin_max_m": np.max(origins, axis=0).tolist(),
        "origin_span_m": np.ptp(origins, axis=0).tolist(),
        "path_length_m": float(np.sum(translations)),
        "adjacent_translation_mean_m": _mean(translations.tolist()),
        "adjacent_translation_max_m": max(translations.tolist()) if len(translations) else 0.0,
        "adjacent_rotation_mean_deg": _mean(rotations),
        "adjacent_rotation_max_deg": max(rotations) if rotations else 0.0,
        "max_pose_baseline_m": max(pairwise) if pairwise else 0.0,
        "azimuth_deg": azimuth.tolist(),
        "azimuth_coverage_deg": _circular_coverage_deg(azimuth),
        "elevation_min_deg": float(np.min(elevation)),
        "elevation_max_deg": float(np.max(elevation)),
        "elevation_span_deg": float(np.ptp(elevation)),
        "near_duplicate_pose_pairs_2cm_5deg": int(near_duplicates),
    }


def _cross_group_pose_overlap(first, second):
    first_poses = [np.asarray(view["pose_matrix"], dtype=np.float64)
                   for view in first["views"]]
    second_poses = [np.asarray(view["pose_matrix"], dtype=np.float64)
                    for view in second["views"]]

    def nearest(source, target):
        rows = []
        for source_index, pose in enumerate(source):
            choices = []
            for target_index, other in enumerate(target):
                choices.append((
                    float(np.linalg.norm(pose[:3, 3] - other[:3, 3])),
                    _rotation_angle_deg(pose[:3, :3], other[:3, :3]),
                    target_index,
                ))
            distance, angle, target_index = min(choices, key=lambda item: item[0])
            rows.append({
                "source_view": source_index + 1,
                "nearest_view": target_index + 1,
                "translation_m": distance,
                "rotation_deg": angle,
            })
        return rows

    forward = nearest(first_poses, second_poses)
    reverse = nearest(second_poses, first_poses)
    combined = forward + reverse
    return {
        "group1_to_group2": forward,
        "group2_to_group1": reverse,
        "symmetric_translation_mean_m": _mean(
            [item["translation_m"] for item in combined]),
        "symmetric_translation_median_m": _median(
            [item["translation_m"] for item in combined]),
        "symmetric_rotation_mean_deg": _mean(
            [item["rotation_deg"] for item in combined]),
        "fraction_within_3cm": float(np.mean([
            item["translation_m"] <= 0.03 for item in combined])),
        "fraction_within_5cm": float(np.mean([
            item["translation_m"] <= 0.05 for item in combined])),
    }


def _target_observations(batches):
    rows = []
    for batch in batches:
        group = int(batch["group_number"])
        for view in batch["views"]:
            token = "g%04d_v%03d" % (group, int(view["index"]))
            for detection_index, detection in enumerate(view.get("detections") or []):
                if not detection.get("target_point_valid"):
                    continue
                point = np.asarray(detection.get("target_point"), dtype=np.float64)
                if point.shape != (3,) or not np.all(np.isfinite(point)):
                    continue
                rows.append({
                    "group": group,
                    "view": int(view["index"]),
                    "view_token": token,
                    "detection_index": int(detection_index),
                    "confidence": float(detection.get("confidence", 0.0)),
                    "point": point,
                })
    return rows


def _connected_clusters(observations, epsilon, min_samples=2):
    points = np.asarray([item["point"] for item in observations], dtype=np.float64)
    if not len(points):
        return np.empty((0,), dtype=np.int64), []
    neighbours = [set(np.flatnonzero(
        np.linalg.norm(points - points[index], axis=1) <= float(epsilon)).tolist())
        for index in range(len(points))]
    labels = np.full(len(points), -1, dtype=np.int64)
    cluster_id = 0
    for seed in range(len(points)):
        if labels[seed] >= 0 or len(neighbours[seed]) < int(min_samples):
            continue
        queue = [seed]
        labels[seed] = cluster_id
        while queue:
            current = queue.pop(0)
            for neighbour in sorted(neighbours[current]):
                if labels[neighbour] < 0:
                    labels[neighbour] = cluster_id
                    if len(neighbours[neighbour]) >= int(min_samples):
                        queue.append(neighbour)
        cluster_id += 1
    clusters = []
    for current_id in range(cluster_id):
        indices = np.flatnonzero(labels == current_id)
        cluster_points = points[indices]
        center = np.mean(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - center, axis=1)
        views = sorted(set(observations[index]["view_token"] for index in indices))
        groups = sorted(set(int(observations[index]["group"]) for index in indices))
        clusters.append({
            "id": int(current_id),
            "observation_count": int(len(indices)),
            "unique_view_count": int(len(views)),
            "groups": groups,
            "views": views,
            "center_m": center.tolist(),
            "rms_radius_m": float(np.sqrt(np.mean(distances ** 2))),
            "max_radius_m": float(np.max(distances)),
            "confidence_mean": _mean([
                observations[index]["confidence"] for index in indices]),
        })
    return labels, clusters


def _cluster_sensitivity(observations):
    result = {}
    for epsilon in (0.05, 0.075, 0.10):
        labels, clusters = _connected_clusters(observations, epsilon, min_samples=2)
        result["%.3f" % epsilon] = {
            "cluster_count": int(len(clusters)),
            "noise_observation_count": int(np.sum(labels < 0)),
            "clusters_repeated_in_two_or_more_views": int(sum(
                cluster["unique_view_count"] >= 2 for cluster in clusters)),
            "clusters_seen_in_both_groups": int(sum(
                len(cluster["groups"]) >= 2 for cluster in clusters)),
            "clusters": clusters,
        }
    return result


def _geometry_target_keys(geometry, label_id, resolution=COVERAGE_VOXEL_M):
    points = np.asarray(geometry.get("points"), dtype=np.float64)
    labels = np.asarray(geometry.get("labels"), dtype=np.int64)
    if points.ndim != 2 or labels.ndim != 1 or len(points) != len(labels):
        return set()
    points = points[labels == int(label_id)]
    if not len(points):
        return set()
    keys = np.floor(points / float(resolution)).astype(np.int64)
    return set(tuple(int(value) for value in row) for row in keys)


def _make_geometries(batches):
    geometries = []
    failures = []
    for batch in batches:
        current, current_failures = _prepare_geometries(
            batch["views"], VOXEL_SIZE_M, SAMPLE_STRIDE, MAX_POINTS,
            MAX_RANGE_M)
        group = int(batch["group_number"])
        for geometry in current:
            original_view = int(geometry["view_index"])
            geometry["group"] = group
            geometry["original_view_index"] = original_view
            geometry["view_token"] = "g%04d_v%03d" % (group, original_view)
            geometry["view_index"] = len(geometries) + 1
            geometries.append(geometry)
        for failure in current_failures:
            failures.append(dict(failure, group=group))
    return geometries, failures


def _stable_clusters_by_view(observations):
    labels, clusters = _connected_clusters(observations, CLUSTER_EPS_M, min_samples=2)
    stable_ids = set(cluster["id"] for cluster in clusters
                     if cluster["unique_view_count"] >= 2)
    by_view = {}
    for index, observation in enumerate(observations):
        cluster_id = int(labels[index]) if len(labels) else -1
        if cluster_id in stable_ids:
            by_view.setdefault(observation["view_token"], set()).add(cluster_id)
    return by_view, stable_ids, clusters


def _replay_one(geometries, class_names, citrus_label, reference_keys,
                clusters_by_view, stable_cluster_ids, strategy, initial_index,
                seed):
    dummy_truth = {
        "instances": [{"id": "proxy-citrus", "class": "citrus"}],
        "views": {},
    }
    order, trace, metadata = _planner_order(
        geometries, strategy, class_names, dummy_truth,
        seed=seed,
        voxel_size=VOXEL_SIZE_M,
        cluster_min_voxels=20,
        cluster_max_distance_m=0.06,
        object_box_size_m=0.06,
        main_stem_height_m=1.2,
        main_stem_width_m=0.05,
        candidate_count=len(geometries),
        max_actions=MAX_ACTIONS,
        ray_count=64,
        max_ray_voxels=128,
        sample_stride=SAMPLE_STRIDE,
        max_points=MAX_POINTS,
        initial_index=initial_index,
        attention_enabled=True,
    )
    observed_keys = set()
    discovered = set()
    coverage = [0.0]
    discovery = [0.0]
    path = [0.0]
    total_path = 0.0
    previous_origin = None
    selected = []
    for geometry_index in order:
        geometry = geometries[geometry_index]
        observed_keys.update(_geometry_target_keys(geometry, citrus_label))
        token = geometry["view_token"]
        discovered.update(clusters_by_view.get(token, set()))
        origin = np.asarray(geometry["origin"], dtype=np.float64)
        if previous_origin is not None:
            total_path += float(np.linalg.norm(origin - previous_origin))
        previous_origin = origin
        coverage.append(float(len(observed_keys)) / len(reference_keys)
                        if reference_keys else 0.0)
        discovery.append(float(len(discovered)) / len(stable_cluster_ids)
                         if stable_cluster_ids else 0.0)
        path.append(total_path)
        selected.append(token)
    return {
        "strategy": strategy,
        "initial_index": int(initial_index),
        "initial_view": geometries[initial_index]["view_token"],
        "selected_view_tokens": selected,
        "coverage_curve": coverage,
        "stable_hypothesis_discovery_curve": discovery,
        "path_length_curve_m": path,
        "coverage_auc": float(np.mean(coverage[1:])) if len(coverage) > 1 else 0.0,
        "discovery_auc": float(np.mean(discovery[1:])) if len(discovery) > 1 else 0.0,
        "final_coverage": coverage[-1],
        "final_stable_hypothesis_discovery": discovery[-1],
        "final_path_length_m": path[-1],
        "trace": trace,
        "planner_metadata": metadata,
    }


def _curve_at(curve, action):
    return float(curve[min(int(action), len(curve) - 1)]) if curve else 0.0


def _aggregate_replays(runs):
    summary = {}
    for strategy in STRATEGIES:
        selected = [run for run in runs if run["strategy"] == strategy]
        max_length = max((len(run["coverage_curve"]) for run in selected), default=1)
        coverage_curves = []
        discovery_curves = []
        path_curves = []
        for run in selected:
            def padded(name):
                values = list(run[name])
                return values + [values[-1]] * (max_length - len(values))
            coverage_curves.append(padded("coverage_curve"))
            discovery_curves.append(padded("stable_hypothesis_discovery_curve"))
            path_curves.append(padded("path_length_curve_m"))
        coverage_array = np.asarray(coverage_curves, dtype=np.float64)
        discovery_array = np.asarray(discovery_curves, dtype=np.float64)
        path_array = np.asarray(path_curves, dtype=np.float64)
        summary[strategy] = {
            "start_count": int(len(selected)),
            "coverage_curve_mean": np.mean(coverage_array, axis=0).tolist(),
            "coverage_curve_min": np.min(coverage_array, axis=0).tolist(),
            "coverage_curve_max": np.max(coverage_array, axis=0).tolist(),
            "discovery_curve_mean": np.mean(discovery_array, axis=0).tolist(),
            "path_curve_mean_m": np.mean(path_array, axis=0).tolist(),
            "coverage_auc_mean": _mean([run["coverage_auc"] for run in selected]),
            "coverage_auc_min": min([run["coverage_auc"] for run in selected], default=0.0),
            "coverage_auc_max": max([run["coverage_auc"] for run in selected], default=0.0),
            "coverage_action_3_mean": _mean([
                _curve_at(run["coverage_curve"], 3) for run in selected]),
            "coverage_action_5_mean": _mean([
                _curve_at(run["coverage_curve"], 5) for run in selected]),
            "coverage_action_7_mean": _mean([
                _curve_at(run["coverage_curve"], 7) for run in selected]),
            "final_coverage_mean": _mean([run["final_coverage"] for run in selected]),
            "final_discovery_mean": _mean([
                run["final_stable_hypothesis_discovery"] for run in selected]),
            "final_path_length_mean_m": _mean([
                run["final_path_length_m"] for run in selected]),
        }
    return summary


def _run_replay(name, batches, observations, starts):
    geometries, failures = _make_geometries(batches)
    class_names = batches[0].get("class_names") or {}
    citrus_label = _stored_label(class_names, "citrus")
    if citrus_label is None:
        raise ValueError("citrus class is absent")
    reference_keys = set()
    for geometry in geometries:
        reference_keys.update(_geometry_target_keys(geometry, citrus_label))
    clusters_by_view, stable_ids, clusters = _stable_clusters_by_view(observations)
    starts = [int(index) for index in starts if 0 <= int(index) < len(geometries)]
    starts = sorted(set(starts)) or [0]
    runs = []
    for initial_index in starts:
        for strategy in STRATEGIES:
            runs.append(_replay_one(
                geometries, class_names, citrus_label, reference_keys,
                clusters_by_view, stable_ids, strategy, initial_index,
                seed=20260805))
    return {
        "name": name,
        "claim": "fixed_view_pool_replay_with_circular_multiview_proxy_reference",
        "formal_pco": False,
        "candidate_count": int(len(geometries)),
        "max_actions": int(MAX_ACTIONS),
        "initial_indices": starts,
        "reference_citrus_voxel_count_1cm": int(len(reference_keys)),
        "stable_3d_hypothesis_count_proxy": int(len(stable_ids)),
        "geometry_failures": failures,
        "summary": _aggregate_replays(runs),
        "runs": runs,
        "clusters_75mm": clusters,
    }


def _aggregate_frames(frame_rows):
    numeric_fields = [
        "depth_nonzero_ratio", "depth_valid_ratio",
        "depth_out_of_range_nonzero_ratio",
        "citrus_pixel_ratio", "tree_pixel_ratio",
        "citrus_depth_valid_ratio", "tree_depth_valid_ratio",
        "citrus_depth_out_of_range_ratio", "tree_depth_out_of_range_ratio",
        "detection_count", "detection_depth_valid_ratio",
        "detection_conf_mean", "semantic_instance_count",
        "semantic_citrus_instance_count", "semantic_tree_instance_count",
        "semantic_citrus_instances_with_20_depth_pixels",
        "semantic_citrus_instance_20px_depth_support_ratio",
        "semantic_citrus_instance_depth_ratio_median",
        "semantic_citrus_conf_mean", "semantic_citrus_entropy_mean_bits",
        "inference_ms", "rgb_depth_delta_s", "target_color_delta_s",
        "joint_stamp_delta_s", "brightness_mean", "contrast_std",
        "sharpness_laplacian_var", "dark_clip_ratio", "bright_clip_ratio",
        "red_clip_ratio", "gyro_norm_rms_rad_s", "gyro_norm_max_rad_s",
        "accel_norm_mean_m_s2", "accel_norm_std_m_s2",
        "imu_gyro_samples", "imu_accel_samples", "imu_gyro_rate_hz",
        "imu_accel_rate_hz",
    ]
    result = {"view_count": int(len(frame_rows))}
    for field in numeric_fields:
        values = [row.get(field) for row in frame_rows]
        result[field + "_mean"] = _mean(values)
        result[field + "_median"] = _median(values)
        result[field + "_min"] = min(
            [float(value) for value in values if value is not None], default=None)
        result[field + "_max"] = max(
            [float(value) for value in values if value is not None], default=None)
    result["timestamp_history_fallback_count"] = int(sum(
        bool(row.get("timestamp_history_fallback")) for row in frame_rows))
    return result


def _write_csv(path, rows):
    fields = sorted(set(key for row in rows for key in row.keys()))
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _plot_quality(frame_rows, output):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {1: "#0072B2", 2: "#D55E00"}
    for group in sorted(set(int(row["group"]) for row in frame_rows)):
        rows = [row for row in frame_rows if int(row["group"]) == group]
        x = [row["view"] for row in rows]
        axes[0, 0].plot(x, [100.0 * row["depth_valid_ratio"] for row in rows],
                        "o-", color=colors.get(group), label="group %d global" % group)
        axes[0, 0].plot(x, [100.0 * (row["citrus_depth_valid_ratio"] or 0.0)
                              for row in rows], "s--", color=colors.get(group),
                        alpha=0.75, label="group %d citrus" % group)
        axes[0, 1].plot(x, [row["detection_count"] for row in rows], "o-",
                        color=colors.get(group), label="group %d" % group)
        axes[1, 0].plot(x, [row["sharpness_laplacian_var"] for row in rows], "o-",
                        color=colors.get(group), label="group %d" % group)
        axes[1, 1].plot(x, [row["gyro_norm_rms_rad_s"] for row in rows], "o-",
                        color=colors.get(group), label="group %d" % group)
    axes[0, 0].set(title="In-range depth validity (0.1-2.5 m)",
                   xlabel="View", ylabel="Valid pixels (%)")
    axes[0, 1].set(title="Citrus detections", xlabel="View", ylabel="Count")
    axes[1, 0].set(title="Image sharpness", xlabel="View", ylabel="Laplacian variance")
    axes[1, 1].set(title="IMU angular stability", xlabel="View",
                   ylabel="Gyro norm RMS (rad/s)")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_pose(batches, target_center, output):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    colors = {1: "#0072B2", 2: "#D55E00"}
    for batch in batches:
        group = int(batch["group_number"])
        origins = np.asarray([view["pose_matrix"][:3, 3]
                              for view in batch["views"]], dtype=np.float64)
        axes[0].plot(origins[:, 0], origins[:, 1], "o-",
                     color=colors.get(group), label="group %d" % group)
        axes[1].plot(range(1, len(origins) + 1), origins[:, 2], "o-",
                     color=colors.get(group), label="group %d" % group)
        for index, origin in enumerate(origins, start=1):
            axes[0].annotate(str(index), (origin[0], origin[1]), fontsize=7)
    axes[0].scatter([target_center[0]], [target_center[1]], marker="x", s=80,
                    color="black", label="robust target centre proxy")
    axes[0].set(title="Camera trajectory (top view)", xlabel="Base x (m)",
                ylabel="Base y (m)")
    axes[0].axis("equal")
    axes[1].set(title="Camera height", xlabel="View", ylabel="Base z (m)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_replay(replay, output):
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    palette = {
        "semantic_nbv": "#009E73",
        "volumetric_nbv": "#0072B2",
        "predefined_narrow": "#E69F00",
        "predefined_wide": "#CC79A7",
        "random": "#666666",
    }
    for strategy in STRATEGIES:
        item = replay["summary"][strategy]
        x = np.arange(len(item["coverage_curve_mean"]))
        mean = np.asarray(item["coverage_curve_mean"])
        low = np.asarray(item["coverage_curve_min"])
        high = np.asarray(item["coverage_curve_max"])
        color = palette[strategy]
        axes[0].plot(x, mean, marker="o", color=color, label=strategy)
        axes[0].fill_between(x, low, high, color=color, alpha=0.10)
        axes[1].plot(x, item["discovery_curve_mean"], marker="o",
                     color=color, label=strategy)
        axes[2].plot(x, item["path_curve_mean_m"], marker="o",
                     color=color, label=strategy)
    axes[0].set(title="Citrus voxel coverage proxy", xlabel="Selected views",
                ylabel="Fraction of 20-view union")
    axes[1].set(title="Unreliable 3-D point-cluster proxy (diagnostic only)",
                xlabel="Selected views", ylabel="Fraction discovered")
    axes[2].set(title="Recorded-pose path length", xlabel="Selected views",
                ylabel="Path length (m)")
    for axis in axes:
        axis.set_xlim(0, MAX_ACTIONS)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_detection(frame_rows, output):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = {1: "#0072B2", 2: "#D55E00"}
    for group in (1, 2):
        rows = [row for row in frame_rows if int(row["group"]) == group]
        axes[0].plot([row["view"] for row in rows],
                     [row["detection_conf_mean"] for row in rows], "o-",
                     color=colors[group], label="group %d" % group)
        axes[1].plot([row["view"] for row in rows],
                     [100.0 * (row["detection_depth_valid_ratio"] or 0.0)
                      for row in rows], "o-", color=colors[group],
                     label="group %d" % group)
    axes[0].set(title="Detector confidence", xlabel="View", ylabel="Mean confidence")
    axes[1].set(title="Detections with valid 3-D point", xlabel="View",
                ylabel="Valid detections (%)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="append", required=True,
                        help="completed semantic NBV ZIP; repeat twice")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    batch_paths = [str(Path(path).expanduser().resolve()) for path in args.batch]
    batch_pairs = [(path, load_batch(path, verify_checksums=True,
                                     strict_integrity=True))
                   for path in batch_paths]
    batch_pairs.sort(key=lambda item: int(item[1]["group_number"]))
    batch_paths = [item[0] for item in batch_pairs]
    batches = [item[1] for item in batch_pairs]
    if len(batches) != 2:
        raise ValueError("exactly two pilot batches are required")

    frame_rows = []
    archive_summary = []
    for path, batch in zip(batch_paths, batches):
        auxiliary = _load_auxiliary(path, batch)
        rows = [_frame_metrics(batch, view, auxiliary[int(view["index"])])
                for view in batch["views"]]
        frame_rows.extend(rows)
        archive_summary.append({
            "path": path,
            "sha256": _sha256(path),
            "size_bytes": os.path.getsize(path),
            "scene_id": batch["scene_id"],
            "group_number": batch["group_number"],
            "view_count": len(batch["views"]),
            "archive_integrity": batch["archive_integrity"],
            "class_names": batch["class_names"],
            "paper_usable_manifest_default": batch["paper_usable"],
        })

    observations = _target_observations(batches)
    target_center = (np.median(np.asarray([item["point"] for item in observations]), axis=0)
                     if observations else np.median(np.asarray([
                         view["pose_matrix"][:3, 3] for batch in batches
                         for view in batch["views"]]), axis=0))
    pose_summary = {
        "target_center_proxy_m": target_center.tolist(),
        "groups": {
            str(batch["group_number"]): _pose_summary(batch, target_center)
            for batch in batches
        },
        "cross_group_overlap": _cross_group_pose_overlap(batches[0], batches[1]),
    }

    observations_by_group = {
        int(batch["group_number"]): [item for item in observations
                                     if item["group"] == int(batch["group_number"])]
        for batch in batches
    }
    replay_group1 = _run_replay(
        "group1_10_candidates", [batches[0]], observations_by_group[1], [0, 5])
    replay_group2 = _run_replay(
        "group2_10_candidates", [batches[1]], observations_by_group[2], [0, 5])
    replay_combined = _run_replay(
        "combined_same_scene_20_candidates_select_10", batches, observations,
        [0, 5, 10, 15])

    result = {
        "schema_version": 1,
        "created_by": "analyze_pilot_batches.py",
        "claim_boundary": {
            "formal_pco": False,
            "reason": ("no independent per-instance 3-D truth or manual per-view truth; "
                       "all replay coverage uses the union of the same captured views"),
            "evaluation_mode": "fixed_view_pool_replay",
            "online_nbv_claim": False,
            "independent_scene_count": 1,
            "batch_count": 2,
        },
        "parameters": {
            "paper_voxel_size_m": VOXEL_SIZE_M,
            "coverage_proxy_voxel_size_m": COVERAGE_VOXEL_M,
            "max_actions": MAX_ACTIONS,
            "sample_stride": SAMPLE_STRIDE,
            "max_points": MAX_POINTS,
            "max_range_m": MAX_RANGE_M,
            "stable_detection_cluster_epsilon_m": CLUSTER_EPS_M,
            "optics_backend": "deterministic",
        },
        "archives": archive_summary,
        "frame_aggregate_all": _aggregate_frames(frame_rows),
        "frame_aggregate_by_group": {
            str(group): _aggregate_frames([
                row for row in frame_rows if int(row["group"]) == group])
            for group in (1, 2)
        },
        "pose": pose_summary,
        "valid_3d_detection_observation_count": len(observations),
        "detection_cluster_sensitivity": _cluster_sensitivity(observations),
        "replays": {
            "group1": replay_group1,
            "group2": replay_group2,
            "combined": replay_combined,
        },
    }

    _write_csv(output / "frame_metrics.csv", frame_rows)
    with open(output / "metrics.json", "w", encoding="utf-8") as stream:
        json.dump(_json_value(result), stream, indent=2, ensure_ascii=False,
                  allow_nan=False)
        stream.write("\n")
    _plot_quality(frame_rows, output / "data_quality_by_view.png")
    _plot_pose(batches, target_center, output / "camera_pose_coverage.png")
    _plot_replay(replay_combined, output / "nbv_proxy_replay.png")
    _plot_detection(frame_rows, output / "recognition_depth_support.png")
    print(json.dumps({
        "metrics": str(output / "metrics.json"),
        "frame_metrics": str(output / "frame_metrics.csv"),
        "combined_replay": replay_combined["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
