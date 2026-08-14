#!/usr/bin/env python3
"""Plan-only probe along the current Elfin tool +Z axis; never executes motion."""

import copy
import sys

import moveit_commander
import rospy
from moveit_msgs.msg import DisplayTrajectory, MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from tf.transformations import quaternion_matrix


def trajectory_points(plan_result):
    plan = plan_result
    success = None
    error_code = None
    if isinstance(plan_result, tuple):
        if len(plan_result) >= 2:
            success, plan = bool(plan_result[0]), plan_result[1]
        if len(plan_result) >= 4:
            error_code = getattr(plan_result[3], "val", None)
    points = len(getattr(getattr(plan, "joint_trajectory", None), "points", []))
    if success is None:
        success = points > 0
    return success, points, error_code, plan


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("elfin_canopy_plan_probe", anonymous=True, disable_signals=True)
    robot = moveit_commander.RobotCommander()
    group = moveit_commander.MoveGroupCommander("elfin_arm")
    group.set_end_effector_link("elfin_end_link")
    group.set_pose_reference_frame("world")
    group.set_planning_time(4.0)
    group.set_num_planning_attempts(2)
    group.set_start_state_to_current_state()

    current = group.get_current_pose("elfin_end_link").pose
    q = [current.orientation.x, current.orientation.y,
         current.orientation.z, current.orientation.w]
    axis = quaternion_matrix(q)[:3, 2]
    print("CURRENT p=(%.6f, %.6f, %.6f) q=(%.6f, %.6f, %.6f, %.6f)" % (
        current.position.x, current.position.y, current.position.z,
        q[0], q[1], q[2], q[3]))
    print("TOOL_PLUS_Z axis=(%.6f, %.6f, %.6f)" % tuple(axis))

    rospy.wait_for_service("/compute_ik", timeout=5.0)
    compute_ik = rospy.ServiceProxy("/compute_ik", GetPositionIK)
    seed = robot.get_current_state()
    best = None
    distances = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                 0.07, 0.08, 0.09, 0.10, 0.105, 0.11, 0.115, 0.12]
    for distance in distances:
        pose = copy.deepcopy(current)
        pose.position.x += float(axis[0]) * distance
        pose.position.y += float(axis[1]) * distance
        pose.position.z += float(axis[2]) * distance
        req = GetPositionIKRequest()
        req.ik_request.group_name = "elfin_arm"
        req.ik_request.robot_state = seed
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout = rospy.Duration(1.0)
        req.ik_request.ik_link_name = "elfin_end_link"
        req.ik_request.pose_stamped.header.frame_id = "world"
        req.ik_request.pose_stamped.header.stamp = rospy.Time.now()
        req.ik_request.pose_stamped.pose = pose
        response = compute_ik(req)
        ik_ok = response.error_code.val == MoveItErrorCodes.SUCCESS
        if not ik_ok:
            print("PROBE d=%.3f ik=False code=%d p=(%.4f, %.4f, %.4f)" % (
                distance, response.error_code.val,
                pose.position.x, pose.position.y, pose.position.z))
            continue

        names = response.solution.joint_state.name
        values = response.solution.joint_state.position
        joint_target = dict(zip(names, values))
        group.clear_pose_targets()
        group.set_start_state_to_current_state()
        group.set_joint_value_target(joint_target)
        result = group.plan()
        planned, points, plan_code, plan = trajectory_points(result)
        print("PROBE d=%.3f ik=True plan=%s points=%d code=%s p=(%.4f, %.4f, %.4f)" % (
            distance, planned, points, plan_code,
            pose.position.x, pose.position.y, pose.position.z))
        if planned:
            best = (distance, pose, plan, points)

    if best is None:
        print("BEST none")
        return 2

    distance, pose, plan, points = best
    tool_tip_length = 0.20
    tip = [
        pose.position.x + float(axis[0]) * tool_tip_length,
        pose.position.y + float(axis[1]) * tool_tip_length,
        pose.position.z + float(axis[2]) * tool_tip_length,
    ]
    print("BEST d=%.3f end=(%.6f, %.6f, %.6f) estimated_tip=(%.6f, %.6f, %.6f) points=%d" % (
        distance, pose.position.x, pose.position.y, pose.position.z,
        tip[0], tip[1], tip[2], points))

    display = DisplayTrajectory()
    display.trajectory_start = robot.get_current_state()
    display.trajectory.append(plan)
    pub = rospy.Publisher("/move_group/display_planned_path", DisplayTrajectory,
                          queue_size=1, latch=True)
    deadline = rospy.Time.now() + rospy.Duration(1.0)
    while pub.get_num_connections() == 0 and rospy.Time.now() < deadline:
        rospy.sleep(0.05)
    pub.publish(display)
    rospy.sleep(0.5)
    print("DISPLAY_PUBLISHED=True EXECUTED=False")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        moveit_commander.roscpp_shutdown()
