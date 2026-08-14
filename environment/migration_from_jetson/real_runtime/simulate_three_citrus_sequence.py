#!/usr/bin/env python3
"""Plan a sequential three-fruit cycle from carried simulated states only."""

import copy
import math
import sys
import time

import moveit_commander
import rospy
import tf2_geometry_msgs  # noqa: F401; registers PointStamped conversion
import tf2_ros
from geometry_msgs.msg import PointStamped, PoseStamped

from elfin_vision.cockpit_logic import (
    quaternion_axis_angle,
    quaternion_multiply,
    quaternion_normalize,
)
from elfin_vision.harvest_logic import (
    align_flange_z_to_horizontal_target,
    flange_points_for_target,
)
from elfin_vision.msg import CitrusTargetArray


def planned_result(result):
    trajectory = result
    success = None
    if isinstance(result, tuple):
        success = bool(result[0])
        trajectory = result[1]
    points = trajectory.joint_trajectory.points
    return (bool(points) if success is None else success), trajectory


def state_after(start_state, trajectory):
    result = copy.deepcopy(start_state)
    point = trajectory.joint_trajectory.points[-1]
    names = trajectory.joint_trajectory.joint_names
    positions = dict(zip(result.joint_state.name,
                         result.joint_state.position))
    positions.update(dict(zip(names, point.positions)))
    result.joint_state.position = [positions[name]
                                   for name in result.joint_state.name]
    result.joint_state.header.stamp = rospy.Time.now()
    return result


def duration(trajectory):
    points = trajectory.joint_trajectory.points
    return points[-1].time_from_start.to_sec() if points else 0.0


def pose_stamped(frame, position, quaternion):
    pose = PoseStamped()
    pose.header.frame_id = frame
    pose.header.stamp = rospy.Time.now()
    pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = position
    (pose.pose.orientation.x, pose.pose.orientation.y,
     pose.pose.orientation.z, pose.pose.orientation.w) = quaternion
    return pose


def transformed_point(buffer, point, source_frame, target_frame):
    message = PointStamped()
    message.header.frame_id = source_frame
    message.header.stamp = rospy.Time(0)
    message.point.x, message.point.y, message.point.z = point
    converted = buffer.transform(
        message, target_frame, timeout=rospy.Duration(0.5))
    return (converted.point.x, converted.point.y, converted.point.z)


def candidates(current_quaternion, target):
    radial = align_flange_z_to_horizontal_target(
        current_quaternion, target)
    result = [('radial', radial)]
    for offset in (-7.5, 7.5, -15.0, 15.0):
        delta = quaternion_axis_angle(
            (0.0, 0.0, 1.0), math.radians(offset))
        result.append(('radial%+.1f' % offset,
                       quaternion_normalize(
                           quaternion_multiply(delta, radial))))
    result.append(('current', quaternion_normalize(current_quaternion)))
    return result


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('simulate_three_citrus_sequence', anonymous=True,
                    disable_signals=True)
    group = moveit_commander.MoveGroupCommander('elfin_arm')
    group.set_end_effector_link('elfin_end_link')
    group.set_max_velocity_scaling_factor(1.0)
    group.set_max_acceleration_scaling_factor(1.0)
    group.set_planning_time(8.0)
    group.set_num_planning_attempts(5)
    frame = group.get_planning_frame()
    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
    listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(0.5)
    targets_message = rospy.wait_for_message(
        '/elfin_vision/citrus_rgbd_node/targets',
        CitrusTargetArray, timeout=8.0)
    if len(targets_message.targets) < 3:
        raise RuntimeError('need 3 targets')

    simulated_state = group.get_current_state()
    current_pose = group.get_current_pose('elfin_end_link').pose
    current_quaternion = (
        current_pose.orientation.x, current_pose.orientation.y,
        current_pose.orientation.z, current_pose.orientation.w)
    total_motion = 0.0
    total_planning = 0.0
    for index, target in enumerate(targets_message.targets[:3]):
        point = transformed_point(
            tf_buffer,
            (target.target_point.x, target.target_point.y,
             target.target_point.z),
            target.target_frame, frame)
        selected = None
        failures = []
        for strategy, quaternion in candidates(current_quaternion, point):
            pregrasp_point, final_point = flange_points_for_target(
                point, quaternion, (0.0, 0.0, 0.2), 0.15, 0.0)
            started = time.monotonic()
            group.clear_pose_targets()
            group.set_start_state(simulated_state)
            group.set_pose_target(pose_stamped(
                frame, pregrasp_point, quaternion))
            success, pregrasp = planned_result(group.plan())
            if not success:
                failures.append('%s:pregrasp' % strategy)
                continue
            pregrasp_state = state_after(simulated_state, pregrasp)
            group.clear_pose_targets()
            group.set_start_state(pregrasp_state)
            cartesian, fraction = group.compute_cartesian_path(
                [pose_stamped(frame, final_point, quaternion).pose],
                0.005, avoid_collisions=True)
            if fraction < 0.995:
                failures.append('%s:%.1f%%' % (strategy, fraction * 100.0))
                continue
            try:
                cartesian = group.retime_trajectory(
                    pregrasp_state, cartesian, 1.0, 1.0)
            except Exception:
                pass
            planning_s = time.monotonic() - started
            pregrasp_s = duration(pregrasp)
            insert_s = duration(cartesian)
            selected = (strategy, quaternion, pregrasp_state,
                        planning_s, pregrasp_s, insert_s)
            break
        if selected is None:
            raise RuntimeError('target %d failed: %s' %
                               (index, ', '.join(failures)))
        (strategy, current_quaternion, simulated_state,
         planning_s, pregrasp_s, insert_s) = selected
        motion_s = pregrasp_s + insert_s * 2.0
        total_planning += planning_s
        total_motion += motion_s
        print('TARGET index=%d strategy=%s planning_s=%.3f '
              'pregrasp_s=%.3f insert_s=%.3f retreat_s=%.3f '
              'motion_s=%.3f' % (
                  index, strategy, planning_s, pregrasp_s,
                  insert_s, insert_s, motion_s))

    fixed_tool = 3 * 10.0
    settle = 3 * 0.6
    rescan = 2 * 0.8
    deterministic = total_planning + total_motion + fixed_tool + settle + rescan
    print('SUMMARY planning_s=%.3f motion_s=%.3f tool_s=%.3f '
          'settle_s=%.3f rescan_s=%.3f deterministic_s=%.3f '
          'expected_with_0.5s_cloud_per_fruit_s=%.3f executed=False' % (
              total_planning, total_motion, fixed_tool, settle, rescan,
              deterministic, deterministic + 1.5))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    finally:
        moveit_commander.roscpp_shutdown()
