#!/usr/bin/env python3
"""Time three fixed PlanCitrus previews; this script never requests execution."""

import time
import threading

import rospy

from elfin_vision.msg import CitrusTargetArray
from elfin_vision.srv import PlanCitrus
from moveit_msgs.msg import DisplayTrajectory


class DisplayCollector(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.message = None

    def callback(self, message):
        with self.lock:
            self.message = message
        self.event.set()

    def reset(self):
        with self.lock:
            self.message = None
        self.event.clear()

    def wait(self, timeout_s=2.0):
        if not self.event.wait(timeout_s):
            return None
        with self.lock:
            return self.message


def main():
    rospy.init_node('benchmark_three_citrus_plan_only', anonymous=True)
    topic = '/elfin_vision/citrus_rgbd_node/targets'
    service_name = '/elfin_vision/citrus_moveit_planner/plan'
    rospy.wait_for_service(service_name, timeout=8.0)
    plan = rospy.ServiceProxy(service_name, PlanCitrus)
    display = DisplayCollector()
    subscriber = rospy.Subscriber(
        '/move_group/display_planned_path', DisplayTrajectory,
        display.callback, queue_size=3)
    rospy.sleep(0.25)
    targets = rospy.wait_for_message(topic, CitrusTargetArray, timeout=8.0)
    if len(targets.targets) < 3:
        raise RuntimeError('need at least 3 live targets, received %d' %
                           len(targets.targets))
    print('INPUT stamp=%.6f targets=%d status=%s' % (
        targets.header.stamp.to_sec(), len(targets.targets), targets.status))
    started_total = time.monotonic()
    successes = 0
    nominal_motion_total = 0.0
    for index in range(3):
        display.reset()
        started = time.monotonic()
        response = plan(target_index=index, execute=False)
        elapsed = time.monotonic() - started
        points = len(response.trajectory.joint_trajectory.points)
        pose = response.target_pose.pose.position
        display_message = display.wait()
        durations = []
        display_points = []
        if display_message is not None:
            for trajectory in display_message.trajectory:
                trajectory_points = trajectory.joint_trajectory.points
                display_points.append(len(trajectory_points))
                durations.append(
                    trajectory_points[-1].time_from_start.to_sec()
                    if trajectory_points else 0.0)
        nominal_motion_total += sum(durations)
        print('TARGET index=%d success=%s elapsed_s=%.3f points=%d '
              'flange=(%.4f,%.4f,%.4f) legs_points=%s legs_s=%s message=%s' % (
                  index, response.success, elapsed, points,
                  pose.x, pose.y, pose.z, display_points,
                  [round(value, 3) for value in durations],
                  response.message))
        successes += int(response.success)
    total = time.monotonic() - started_total
    print('SUMMARY successes=%d/3 total_plan_s=%.3f average_plan_s=%.3f '
          'nominal_motion_s=%.3f executed=False' % (
              successes, total, total / 3.0, nominal_motion_total))
    subscriber.unregister()
    return 0 if successes == 3 else 2


if __name__ == '__main__':
    raise SystemExit(main())
