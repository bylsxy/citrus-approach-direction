#!/usr/bin/env python3
import json
import math
import os
import sys

import numpy as np
import rospy
import yaml
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

PROJECT = '/home/catas/ros_ws/src/elfin_vision'
sys.path.insert(0, os.path.join(PROJECT, 'src'))

from elfin_vision.approach_direction import select_target_fruit
from elfin_vision.harvest_logic import conservative_collision_boxes
from elfin_vision.nbv_evaluation import SemanticVoxelMap, load_batch, project_view_to_world

BATCH_PATH = '/home/catas/elfin_citrus_data/nbv_batches/semantic_nbv_batch_20260812-122205_g0040.zip'
RESULTS = {
    'm2': '/home/catas/elfin_citrus_data/approach_results/approach_20260812-122219-185031_approach-1786537338820-0010_planned.json',
    'm3': '/home/catas/elfin_citrus_data/approach_results/approach_20260812-122226-728207_approach-1786537346179-0011_planned.json',
}
APPROACH_CONFIG = os.path.join(PROJECT, 'config', 'approach_direction.yaml')
TOOL_CONFIG = os.path.join(PROJECT, 'config', 'harvester_tool.yaml')
FRAME = 'elfin_base_link'


def point(values):
    result = Point()
    result.x, result.y, result.z = (float(v) for v in values)
    return result


def color(r, g, b, a):
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


def cube_list(namespace, marker_id, values, resolution, rgba):
    marker = Marker()
    marker.header.frame_id = FRAME
    marker.header.stamp = rospy.Time(0)
    marker.ns = namespace
    marker.id = int(marker_id)
    marker.type = Marker.CUBE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = marker.scale.y = marker.scale.z = float(resolution)
    marker.color = rgba
    marker.points = [point(value) for value in values]
    marker.frame_locked = True
    marker.lifetime = rospy.Duration(0)
    return marker


def bounded(values, limit):
    if len(values) <= limit:
        return values
    indices = np.linspace(0, len(values) - 1, int(limit), dtype=np.int64)
    return [values[int(index)] for index in np.unique(indices)]


def semantic_points(voxel_map, target_keys, fruit_ids, cap=12000):
    target, citrus, scene, unknown = [], [], [], []
    for key, state in voxel_map.voxels.items():
        if not voxel_map.is_occupied(key):
            continue
        value = voxel_map.point_from_key(key)
        if not state.get('semantic_set'):
            unknown.append(value)
            continue
        label = int(state.get('semantic_label'))
        if tuple(key) in target_keys:
            target.append(value)
        elif label in fruit_ids:
            citrus.append(value)
        else:
            scene.append(value)
    target = bounded(target, cap)
    remaining = max(0, cap - len(target))
    citrus = bounded(citrus, remaining)
    remaining -= len(citrus)
    unknown_reserve = min(len(unknown), max(1, cap // 10), remaining) if unknown else 0
    scene = bounded(scene, max(0, remaining - unknown_reserve))
    remaining -= len(scene)
    unknown = bounded(unknown, remaining)
    return target, citrus, scene, unknown


def direction_markers(voxel_map, result, target_keys, fruit_ids):
    target, citrus, scene, unknown = semantic_points(voxel_map, target_keys, fruit_ids)
    resolution = float(voxel_map.resolution)
    markers = []
    delete = Marker()
    delete.action = Marker.DELETEALL
    markers.append(delete)
    delete.pose.orientation.w = 1.0
    if target:
        markers.append(cube_list('semantic_octree_target', 10, target, resolution, color(1.0, 0.18, 0.02, 0.95)))
    if citrus:
        markers.append(cube_list('semantic_octree_citrus', 11, citrus, resolution, color(1.0, 0.52, 0.02, 0.82)))
    if scene:
        markers.append(cube_list('semantic_octree_scene', 12, scene, resolution, color(0.12, 0.62, 0.28, 0.50)))
    if unknown:
        markers.append(cube_list('semantic_octree_unknown', 13, unknown, resolution, color(0.48, 0.48, 0.48, 0.16)))

    best = result['best']
    arrow = Marker()
    arrow.header.frame_id = FRAME
    arrow.header.stamp = rospy.Time(0)
    arrow.ns = 'approach_direction'
    arrow.id = 2
    arrow.type = Marker.ARROW
    arrow.action = Marker.ADD
    arrow.points = [point(best['preentry_point']), point(result['target_center'])]
    arrow.pose.orientation.w = 1.0
    arrow.scale.x = 0.014
    arrow.scale.y = 0.030
    arrow.scale.z = 0.045
    arrow.color = color(0.15, 0.90, 0.25, 0.95)
    arrow.frame_locked = True
    arrow.lifetime = rospy.Duration(0)
    delete.header.frame_id = FRAME
    delete.header.stamp = rospy.Time(0)
    markers.append(arrow)

    target_sphere = Marker()
    target_sphere.header.frame_id = FRAME
    target_sphere.header.stamp = rospy.Time(0)
    target_sphere.ns = 'target_sphere'
    target_sphere.id = 20
    target_sphere.type = Marker.SPHERE
    target_sphere.action = Marker.ADD
    target_sphere.pose.position = point(result['target_center'])
    target_sphere.pose.orientation.w = 1.0
    diameter = 2.0 * float(result.get('target_radius_m', 0.03))
    target_sphere.scale.x = target_sphere.scale.y = target_sphere.scale.z = diameter
    target_sphere.color = color(1.0, 0.18, 0.02, 0.18)
    target_sphere.frame_locked = True
    target_sphere.lifetime = rospy.Duration(0)
    markers.append(target_sphere)
    return MarkerArray(markers=markers)


def tool_markers(tool):
    collision = tool['collision_model']
    attach_link = str(collision['attach_link'])
    mesh = Marker()
    mesh.header.frame_id = attach_link
    mesh.header.stamp = rospy.Time(0)
    mesh.ns = 'harvester_exact_visual'
    mesh.id = 0
    mesh.type = Marker.MESH_RESOURCE
    mesh.action = Marker.ADD
    transform = collision['cad_to_attach']
    mesh.pose.position = point(transform['translation_m'])
    q = transform['quaternion_xyzw']
    mesh.pose.orientation.x, mesh.pose.orientation.y, mesh.pose.orientation.z, mesh.pose.orientation.w = [float(v) for v in q]
    mesh.scale.x, mesh.scale.y, mesh.scale.z = [float(v) for v in collision['mesh_scale']]
    mesh.color = color(0.82, 0.36, 0.12, 1.0)
    mesh.mesh_resource = str(collision['mesh_resource'])
    mesh.mesh_use_embedded_materials = False
    mesh.frame_locked = True
    mesh.lifetime = rospy.Duration(0)

    proxies = []
    for index, box in enumerate(conservative_collision_boxes(collision)):
        marker = Marker()
        marker.header.frame_id = attach_link
        marker.header.stamp = rospy.Time(0)
        marker.ns = 'harvester_collision_proxy'
        marker.id = index
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position = point(box['translation_m'])
        q = box['quaternion_xyzw']
        marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w = [float(v) for v in q]
        marker.scale.x, marker.scale.y, marker.scale.z = [float(v) for v in box['size_m']]
        marker.color = color(0.10, 0.55, 0.95, 0.18)
        marker.frame_locked = True
        marker.lifetime = rospy.Duration(0)
        proxies.append(marker)
    return mesh, MarkerArray(markers=proxies)


def main():
    rospy.init_node('fig8_archived_rviz_replay', anonymous=False, disable_signals=False)
    with open(APPROACH_CONFIG, encoding='utf-8') as stream:
        approach = yaml.safe_load(stream) or {}
    with open(TOOL_CONFIG, encoding='utf-8') as stream:
        tool = yaml.safe_load(stream) or {}
    reconstruction = approach.get('reconstruction') or {}
    planning = approach.get('planning') or {}
    batch = load_batch(BATCH_PATH, verify_checksums=True, strict_integrity=True)
    voxel_map = SemanticVoxelMap(
        resolution=float(reconstruction.get('resolution_m', 0.006)),
        max_voxels=int(reconstruction.get('maximum_voxels', 600000)))
    for view in batch['views']:
        geometry = project_view_to_world(
            view, voxel_size=voxel_map.resolution,
            stride=int(reconstruction.get('sample_stride', 4)),
            max_points=int(reconstruction.get('maximum_points_per_view', 24000)),
            max_range_m=float(reconstruction.get('maximum_range_m', 1.5)))
        voxel_map.insert_geometry(geometry, include_semantics=True, raycast=True,
                                  max_rays=600, max_ray_voxels=320)

    fruit_ids = {int(key) for key, name in batch['class_names'].items()
                 if str(name).strip().lower() in ('citrus', 'orange', 'tangerine', 'mandarin')}
    loaded = {}
    for method, path in RESULTS.items():
        with open(path, encoding='utf-8') as stream:
            loaded[method] = json.load(stream)
    anchor = loaded['m2']['target_center']
    target = select_target_fruit(
        voxel_map, anchor, fruit_ids=fruit_ids,
        search_radius=float(planning.get('target_search_radius_m', 0.14)),
        link_radius=float(planning.get('target_cluster_link_m', 0.0105)),
        minimum_radius=float(planning.get('fruit_radius_min_m', 0.025)),
        maximum_radius=float(planning.get('fruit_radius_max_m', 0.09)),
        minimum_voxels=int(planning.get('minimum_target_voxels', 12)))

    publishers = {
        method: rospy.Publisher('/fig8/%s/markers' % method, MarkerArray, queue_size=1, latch=True)
        for method in ('m2', 'm3')
    }
    mesh_pub = rospy.Publisher('/fig8/tool/visual_marker', Marker, queue_size=1, latch=True)
    proxy_pub = rospy.Publisher('/fig8/tool/collision_markers', MarkerArray, queue_size=1, latch=True)
    rospy.sleep(0.5)
    for method in ('m2', 'm3'):
        publishers[method].publish(direction_markers(
            voxel_map, loaded[method], target['keys'], fruit_ids))
    mesh, proxies = tool_markers(tool)
    mesh_pub.publish(mesh)
    proxy_pub.publish(proxies)
    rospy.loginfo('Fig.8 replay ready: batch=g0040 scene=%s voxels=%s M2=%s M3=%s',
                  batch['scene_id'], voxel_map.summary(),
                  loaded['m2']['moveit_plan_id'], loaded['m3']['moveit_plan_id'])
    print('Fig.8 replay ready: batch=g0040 M2=%s M3=%s' %
          (loaded['m2']['moveit_plan_id'], loaded['m3']['moveit_plan_id']), flush=True)
    rospy.spin()


if __name__ == '__main__':
    main()
