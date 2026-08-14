#!/usr/bin/env python3

"""Read-only preflight for the live E05 before any Servo On request."""

import json
import math
import sys

import rospy
from controller_manager_msgs.srv import ListControllers
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import SetBool


def fail(message, report):
    report['ok'] = False
    report.setdefault('problems', []).append(message)


def main():
    rospy.init_node('verify_elfin_servo_off', anonymous=True,
                    disable_signals=True)
    report = {'ok': True, 'problems': []}

    enable = rospy.wait_for_message(
        '/elfin_ros_control/elfin/enable_state', Bool, timeout=3.0)
    fault = rospy.wait_for_message(
        '/elfin_ros_control/elfin/fault_state', Bool, timeout=3.0)
    samples = [rospy.wait_for_message('/joint_states', JointState,
                                      timeout=3.0)
               for _unused in range(25)]

    expected = ['elfin_joint%d' % index for index in range(1, 7)]
    ordered = []
    maximum_velocity = 0.0
    for sample in samples:
        positions = dict(zip(sample.name, sample.position))
        velocities = dict(zip(sample.name, sample.velocity))
        if any(name not in positions for name in expected):
            fail('joint_states is missing an E05 joint', report)
            continue
        ordered.append([float(positions[name]) for name in expected])
        maximum_velocity = max(
            maximum_velocity,
            max(abs(float(velocities.get(name, 0.0))) for name in expected))

    maximum_span = float('inf')
    if ordered:
        maximum_span = max(
            max(values) - min(values) for values in zip(*ordered))
    final_positions = ordered[-1] if ordered else []

    rospy.wait_for_service('/elfin_ros_control/elfin/get_motion_state', 3.0)
    rospy.wait_for_service('/elfin_ros_control/elfin/get_pos_align_state', 3.0)
    moving = rospy.ServiceProxy(
        '/elfin_ros_control/elfin/get_motion_state', SetBool)(True)
    aligned = rospy.ServiceProxy(
        '/elfin_ros_control/elfin/get_pos_align_state', SetBool)(True)

    rospy.wait_for_service('/controller_manager/list_controllers', 3.0)
    controllers_response = rospy.ServiceProxy(
        '/controller_manager/list_controllers', ListControllers)()
    controllers = {item.name: item.state
                   for item in controllers_response.controller}

    validity = None
    validity_contacts = []
    try:
        rospy.wait_for_service('/check_state_validity', 5.0)
        request = GetStateValidityRequest()
        request.group_name = 'elfin_arm'
        request.robot_state = RobotState()
        request.robot_state.joint_state = samples[-1]
        validity_response = rospy.ServiceProxy(
            '/check_state_validity', GetStateValidity)(request)
        validity = bool(validity_response.valid)
        validity_contacts = [
            '%s<->%s' % (contact.contact_body_1, contact.contact_body_2)
            for contact in validity_response.contacts]
    except Exception as error:
        fail('MoveIt state validity unavailable: %s' % error, report)

    report.update({
        'servo_on': bool(enable.data),
        'fault': bool(fault.data),
        'driver_reports_moving': bool(moving.success),
        'position_aligned': bool(aligned.success),
        'sample_count': len(ordered),
        'maximum_position_span_rad': maximum_span,
        'maximum_reported_velocity_rad_s': maximum_velocity,
        'joint_positions_rad': final_positions,
        'controllers': controllers,
        'moveit_state_valid': validity,
        'moveit_contacts': validity_contacts,
    })

    if enable.data:
        fail('Servo On is already active', report)
    if fault.data:
        fail('the driver reports a fault', report)
    if moving.success:
        fail('the driver reports physical motion', report)
    if not aligned.success:
        fail('command and encoder positions are not aligned', report)
    if maximum_span > 0.001:
        fail('joint position span exceeds 0.001 rad', report)
    if maximum_velocity > 0.01:
        fail('reported joint velocity exceeds 0.01 rad/s', report)
    if controllers.get('joint_state_controller') != 'running':
        fail('joint_state_controller is not running', report)
    if controllers.get('elfin_arm_controller') != 'initialized':
        fail('position controller is unexpectedly active or unavailable', report)
    if controllers.get('elfin_freedrive_controller') == 'running':
        fail('freedrive controller is unexpectedly running', report)
    if validity is not True:
        fail('the current MoveIt state is invalid', report)
    if not all(math.isfinite(value) for value in final_positions):
        fail('a joint position is non-finite', report)

    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(json.dumps({'ok': False, 'fatal': str(error)},
                         ensure_ascii=False, sort_keys=True, indent=2))
        sys.exit(2)
