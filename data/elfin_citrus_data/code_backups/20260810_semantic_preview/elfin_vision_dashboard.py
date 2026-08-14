#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight ROS dashboard for the Elfin citrus RGB-D pipeline."""

from __future__ import division

import json
import math
import os
import signal
import subprocess
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import dynamic_reconfigure.client
import numpy as np
import rospy
import yaml
try:
    import tf2_ros
except ImportError:  # allows offline archive/state tests without ROS TF bindings
    tf2_ros = None
import wx
from cv_bridge import CvBridge, CvBridgeError
from elfin_robot_msgs.srv import CockpitJog, SetInt16, SetString
from geometry_msgs.msg import Point, PoseStamped, Vector3
from moveit_msgs.msg import DisplayTrajectory
from sensor_msgs.msg import CameraInfo, Image, Imu, JointState
from std_msgs.msg import Bool, ColorRGBA, Header, String, UInt16
from std_srvs.srv import SetBool, Trigger
from visualization_msgs.msg import Marker, MarkerArray

from elfin_vision.approach_direction import (
    DEFAULT_CONFIG as APPROACH_DEFAULT_CONFIG,
    SemanticApproachSession,
    public_result as public_approach_result,
    render_spherical_heatmap,
    # The probe experiment's straight-line arm must emit a candidate with the
    # exact same field set as a scored one.  Reusing the serializer is what
    # keeps the CSV columns identical across all three methods.
    _serial_candidate as serial_approach_candidate,
)
from elfin_vision import probe_trials
from elfin_vision.camera_runtime import CameraRuntime
from elfin_vision.cockpit_logic import (
    FocusedCockpitInput,
    SinglePressGate,
    camera_goal_to_flange_pose,
    compose_pose,
    level_camera_quaternion,
    quaternion_rotate,
)
from elfin_vision.harvest_logic import (
    align_flange_z_to_direction,
    workspace_rejection,
)
from elfin_vision.calibration_install import (
    find_latest_quality_candidate,
    install_candidate,
    read_install_status,
)
from elfin_vision.depth_geometry import colorize_depth, depth_to_meters
from elfin_vision.hand_eye import (
    make_transform,
    rotation_matrix_to_quaternion_xyzw,
    quaternion_xyzw_to_rotation_matrix,
)
from elfin_vision.nbv_capture import (
    BatchRecorder,
    validate_capture_timing,
    validate_inference_timing,
)
from elfin_vision.panel_jog import LatestJogDispatcher
from elfin_vision.temporal_sync import (
    TimestampHistory,
    select_best_pair,
)
from elfin_vision.view_strategy import (
    WIDE_SWEEP_HALF_ANGLE_DEG,
    azimuth_span_deg,
    cockpit_cobra_steps,
    predefined_wide_sweep_poses,
    target_visual_priority_key,
)
from elfin_vision.msg import (
    CitrusTargetArray,
    CockpitStatus,
    HarvestStatus,
)
from elfin_vision.srv import (
    CockpitRecover,
    HarvestCommand,
    PlanApproachDirection,
    PlanCitrus,
    PlanObservationPose,
    RuntimeConfig,
)


class SharedState(object):
    """Thread-safe handoff from ROS callbacks to the wx main thread."""

    def __init__(self):
        self.lock = threading.Lock()
        # ROS callbacks arrive independently.  Keep a short raw history so a
        # throttled detector result can be matched to the RGB-D frame it
        # actually describes instead of being paired with the newest frame.
        self.history_size = 30
        self.raw_color_history = TimestampHistory(self.history_size)
        self.raw_depth_history = TimestampHistory(self.history_size)
        self.camera_info_history = TimestampHistory(self.history_size)
        self.semantic_histories = {
            'labels': TimestampHistory(self.history_size),
            'confidence': TimestampHistory(self.history_size),
            'instances': TimestampHistory(self.history_size),
        }
        self.target_history = TimestampHistory(self.history_size)
        self.joint_history = TimestampHistory(self.history_size)
        # IMU streams are much faster than RGB-D. Keep a wider bounded history
        # so one manual snapshot can archive the recent raw motion window.
        self.imu_histories = {
            'gyro': TimestampHistory(self.history_size * 8),
            'accel': TimestampHistory(self.history_size * 8),
        }
        self.imu_topics = {'gyro': '', 'accel': ''}
        self.images = {'color': None, 'depth': None, 'annotated': None}
        # Raw synchronized inputs are kept independently of the throttled
        # display images.  A paper capture must never archive a resized or
        # stale dashboard bitmap.
        self.raw_color = None
        self.raw_depth = None
        self.color_header = None
        self.depth_header = None
        self.annotated_header = None
        self.camera_info = None
        self.semantic_labels = None
        self.semantic_confidence = None
        self.semantic_instances = None
        self.semantic_available = False
        self.semantic_header = None
        self.raw_arrival = {}
        self.camera_info_arrival = 0.0
        self.semantic_arrival = 0.0
        self.joint_arrival = 0.0
        self.versions = {'color': 0, 'depth': 0, 'annotated': 0}
        self.last_arrival = {}
        self.fps = {'color': 0.0, 'depth': 0.0, 'annotated': 0.0}
        self.targets = None
        self.target_arrival = 0.0
        self.joints = None
        self.imu_latest = {'gyro': None, 'accel': None}
        self.vision_status = '等待视觉节点'
        self.planner_status = '等待 MoveIt 规划节点'
        self.action_status = '面板已启动；不会发送真机执行命令'
        self.nbv_status = {
            'enabled': False, 'scene_id': '', 'group_number': 1,
            'view_count': 0, 'max_views': 10, 'view_label': '1/10',
            'last_archive': '', 'last_error': '',
            'last_result': '论文采集模式未开启',
        }
        self.approach_status = {
            'state': 'empty', 'message': '等待十步语义重建',
            'view_count': 0, 'safe': False, 'computing': False,
            'result': None,
        }
        self.approach_heatmap = None
        self.approach_version = 0
        self.harvest_status = None
        self.harvest_status_version = 0
        self.cockpit_status = None
        self.events = deque(maxlen=400)
        self.event_version = 0
        self.trajectory_count = 0
        self.trajectory_points = 0

    def observe_image(self, key, stamp):
        value = stamp.to_sec() if stamp and stamp.to_sec() > 0.0 else time.monotonic()
        with self.lock:
            previous = self.last_arrival.get(key)
            if previous is not None and value > previous:
                measured = 1.0 / (value - previous)
                old = self.fps.get(key, 0.0)
                self.fps[key] = measured if old <= 0.0 else old * 0.8 + measured * 0.2
            self.last_arrival[key] = value

    def set_image(self, key, image, raw_depth=None):
        with self.lock:
            self.images[key] = image
            if raw_depth is not None:
                self.raw_depth = raw_depth
            self.versions[key] += 1

    def set_raw_color(self, image, header, arrival=None):
        with self.lock:
            value = np.ascontiguousarray(image).copy()
            arrival = float(arrival or time.monotonic())
            self.raw_color = value
            self.color_header = header
            self.raw_arrival['color'] = arrival
            self.raw_color_history.add(
                value, _stamp_sec(getattr(header, 'stamp', None)), arrival,
                header=header)

    def set_raw_depth(self, image, header, arrival=None):
        with self.lock:
            value = np.ascontiguousarray(image).copy()
            arrival = float(arrival or time.monotonic())
            self.raw_depth = value
            self.depth_header = header
            self.raw_arrival['depth'] = arrival
            self.raw_depth_history.add(
                value, _stamp_sec(getattr(header, 'stamp', None)), arrival,
                header=header)

    def set_camera_info(self, message, arrival=None):
        with self.lock:
            arrival = float(arrival or time.monotonic())
            self.camera_info = message
            self.camera_info_arrival = arrival
            self.camera_info_history.add(
                message,
                _stamp_sec(getattr(getattr(message, 'header', None),
                                   'stamp', None)),
                arrival)

    def set_semantic_images(self, labels, confidence, instances, available,
                            header, arrival=None):
        with self.lock:
            arrival = float(arrival or time.monotonic())
            self.semantic_labels = (None if labels is None else
                                    np.ascontiguousarray(labels).copy())
            self.semantic_confidence = (None if confidence is None else
                                        np.ascontiguousarray(confidence).copy())
            self.semantic_instances = (None if instances is None else
                                       np.ascontiguousarray(instances).copy())
            self.semantic_available = bool(available)
            self.semantic_header = header
            self.semantic_arrival = arrival
            stamp = _stamp_sec(getattr(header, 'stamp', None))
            for component, value in (
                    ('labels', self.semantic_labels),
                    ('confidence', self.semantic_confidence),
                    ('instances', self.semantic_instances)):
                if value is not None:
                    self.semantic_histories[component].add(
                        value, stamp, arrival, header=header)

    def set_semantic_component(self, component, value, header=None,
                               available=None, arrival=None):
        with self.lock:
            arrival = float(arrival or time.monotonic())
            if component == 'labels':
                self.semantic_labels = (None if value is None else
                                        np.ascontiguousarray(value).copy())
            elif component == 'confidence':
                self.semantic_confidence = (None if value is None else
                                             np.ascontiguousarray(value).copy())
            elif component == 'instances':
                self.semantic_instances = (None if value is None else
                                           np.ascontiguousarray(value).copy())
            elif component == 'available':
                self.semantic_available = bool(value)
            else:
                raise ValueError('unknown semantic component: %s' % component)
            if header is not None:
                self.semantic_header = header
            # The Bool topic only reports whether the frame had a positive
            # mask; it is not an image timestamp.  Never refresh semantic
            # freshness from that callback or a latched value could make an
            # old label image look current.
            if component != 'available':
                self.semantic_arrival = arrival
            if component in self.semantic_histories and value is not None:
                stored = getattr(self, 'semantic_' + component)
                self.semantic_histories[component].add(
                    stored, _stamp_sec(getattr(header, 'stamp', None)),
                    arrival, header=header)

    def record_targets(self, message, arrival=None):
        """Store a target message in the bounded capture history."""
        with self.lock:
            arrival = float(arrival or time.monotonic())
            self.targets = message
            self.target_arrival = arrival
            self.target_history.add(
                message,
                _stamp_sec(getattr(getattr(message, 'header', None),
                                   'stamp', None)),
                arrival)

    def record_joints(self, message, arrival=None):
        """Store a JointState message in the bounded capture history."""
        with self.lock:
            arrival = float(arrival or time.monotonic())
            self.joints = message
            self.joint_arrival = arrival
            self.joint_history.add(
                message,
                _stamp_sec(getattr(getattr(message, 'header', None),
                                   'stamp', None)),
                arrival)

    def record_imu(self, kind, message, topic, arrival=None):
        """Store a raw RealSense gyro/accelerometer sample by ROS timestamp."""
        if kind not in self.imu_histories:
            raise ValueError('unknown IMU stream: %s' % kind)
        arrival = float(arrival or time.monotonic())
        value = _imu_dict(message, topic)
        with self.lock:
            self.imu_latest[kind] = value
            self.imu_topics[kind] = str(topic or '')
            self.imu_histories[kind].add(
                value, value.get('stamp_sec', 0.0), arrival)

    def select_capture_inputs(self, now=None, rgbd_slop_s=0.08,
                              max_frame_age_s=0.50,
                              result_slop_s=0.25,
                              joint_slop_s=0.50,
                              imu_window_before_s=0.25,
                              imu_window_after_s=0.05,
                              imu_max_age_s=0.75):
        """Return timestamp-matched callback values for one capture.

        The returned values are references to callback-owned immutable copies
        (the arrays are copied when callbacks enter the state).  Callers must
        still copy them before doing slow archive work.
        """
        now = time.monotonic() if now is None else float(now)
        with self.lock:
            pair = select_best_pair(
                self.raw_color_history, self.raw_depth_history,
                max_delta_s=rgbd_slop_s, now=now,
                max_age_s=max_frame_age_s)
            if pair is None:
                return {'error': '历史缓存中没有新鲜且同步的 RGB-D 对'}
            color = pair['first']
            depth = pair['second']
            color_stamp = color['stamp_sec']
            selected = {
                'raw_color': color['value'],
                'raw_depth': depth['value'],
                'color_header': color.get('header'),
                'depth_header': depth.get('header'),
                'raw_arrival': {
                    'color': float(color.get('arrival', 0.0)),
                    'depth': float(depth.get('arrival', 0.0)),
                },
                'camera_info': self.camera_info,
                'camera_info_arrival': self.camera_info_arrival,
                'camera_info_static_stamp': False,
                'semantic_labels': self.semantic_labels,
                'semantic_confidence': self.semantic_confidence,
                'semantic_instances': self.semantic_instances,
                'semantic_header': self.semantic_header,
                'semantic_arrival': self.semantic_arrival,
                'semantic_available': bool(self.semantic_available),
                'semantic_component_stamps': {},
                'semantic_components_synchronized': True,
                'targets': self.targets,
                'target_arrival': self.target_arrival,
                'joints': self.joints,
                'joint_arrival': self.joint_arrival,
                'imu_samples': {'gyro': [], 'accel': []},
                'imu_topics': dict(self.imu_topics),
                'rgbd_pair_delta_s': float(pair['delta_s']),
            }

            imu_start = color_stamp - max(0.0, float(imu_window_before_s))
            imu_end = color_stamp + max(0.0, float(imu_window_after_s))
            for kind, history in self.imu_histories.items():
                samples = []
                for item in history.records():
                    stamp = float(item.get('stamp_sec', 0.0) or 0.0)
                    arrival = float(item.get('arrival', 0.0) or 0.0)
                    if not stamp or stamp < imu_start or stamp > imu_end:
                        continue
                    if arrival <= 0.0 or now - arrival > float(imu_max_age_s):
                        continue
                    sample = dict(item.get('value') or {})
                    sample['arrival_age_s'] = max(0.0, now - arrival)
                    samples.append(sample)
                selected['imu_samples'][kind] = samples
            info = self.camera_info_history.nearest(
                color_stamp, max_delta_s=rgbd_slop_s, now=now,
                max_age_s=max_frame_age_s, allow_zero_stamp=True)
            if info is None:
                # Some camera drivers repeat one startup stamp while still
                # publishing fresh CameraInfo messages.  Permit that only if
                # the bounded history proves all observed non-zero stamps are
                # effectively identical; a changing but mismatched stream is
                # still rejected by the normal synchronization gate.
                records = self.camera_info_history.records()
                stamps = [float(item.get('stamp_sec', 0.0) or 0.0)
                          for item in records
                          if float(item.get('stamp_sec', 0.0) or 0.0) > 0.0]
                static_stamp = bool(len(stamps) >= 2 and
                                    max(stamps) - min(stamps) <= 1e-9)
                if static_stamp:
                    info = self.camera_info_history.nearest(
                        color_stamp, max_delta_s=None, now=now,
                        max_age_s=max_frame_age_s, allow_zero_stamp=True)
                    selected['camera_info_static_stamp'] = info is not None
            if info is not None:
                selected['camera_info'] = info['value']
                selected['camera_info_arrival'] = info['arrival']

            for component in ('labels', 'confidence', 'instances'):
                record = self.semantic_histories[component].nearest(
                    color_stamp, max_delta_s=result_slop_s, now=now,
                    max_age_s=max_frame_age_s)
                selected['semantic_' + component] = (
                    None if record is None else record['value'])
                selected['semantic_component_stamps'][component] = (
                    None if record is None else record.get('stamp_sec', 0.0))
                if component == 'labels' and record is not None:
                    selected['semantic_header'] = record.get('header')
                    selected['semantic_arrival'] = record.get('arrival', 0.0)

            label_stamp = selected['semantic_component_stamps'].get('labels')
            if label_stamp:
                for component in ('confidence', 'instances'):
                    component_stamp = selected['semantic_component_stamps'].get(
                        component)
                    if (component_stamp is None or
                            not abs(float(component_stamp) - float(label_stamp))
                            <= float(result_slop_s)):
                        selected['semantic_components_synchronized'] = False

            target = self.target_history.nearest(
                color_stamp, max_delta_s=result_slop_s, now=now,
                max_age_s=max_frame_age_s)
            if target is not None:
                selected['targets'] = target['value']
                selected['target_arrival'] = target['arrival']
            else:
                selected['targets'] = None
                selected['target_arrival'] = 0.0

            joint = self.joint_history.nearest(
                color_stamp, max_delta_s=joint_slop_s, now=now,
                max_age_s=max_frame_age_s, allow_zero_stamp=True)
            if joint is not None:
                selected['joints'] = joint['value']
                selected['joint_arrival'] = joint['arrival']
            return selected

    def set_action_status(self, value):
        with self.lock:
            self.action_status = str(value)

    def set_nbv_status(self, value):
        with self.lock:
            self.nbv_status = dict(value or {})

    def set_approach_status(self, value, heatmap=None):
        with self.lock:
            self.approach_status = dict(value or {})
            if heatmap is not None:
                self.approach_heatmap = np.ascontiguousarray(heatmap).copy()
            self.approach_version += 1

    def reset_vision(self, status):
        with self.lock:
            self.images = {'color': None, 'depth': None, 'annotated': None}
            self.raw_color = None
            self.raw_depth = None
            self.color_header = None
            self.depth_header = None
            self.annotated_header = None
            self.camera_info = None
            self.semantic_labels = None
            self.semantic_confidence = None
            self.semantic_instances = None
            self.semantic_available = False
            self.semantic_header = None
            self.raw_arrival = {}
            self.camera_info_arrival = 0.0
            self.semantic_arrival = 0.0
            self.joint_arrival = 0.0
            self.raw_color_history.clear()
            self.raw_depth_history.clear()
            self.camera_info_history.clear()
            for history in self.semantic_histories.values():
                history.clear()
            self.target_history.clear()
            self.joint_history.clear()
            for history in self.imu_histories.values():
                history.clear()
            for key in self.versions:
                self.versions[key] += 1
            self.last_arrival = {}
            self.fps = {'color': 0.0, 'depth': 0.0, 'annotated': 0.0}
            self.targets = None
            self.target_arrival = 0.0
            self.joints = None
            self.imu_latest = {'gyro': None, 'accel': None}
            self.imu_topics = {'gyro': '', 'accel': ''}
            self.vision_status = str(status)

    def append_event(self, value):
        with self.lock:
            self.events.append(str(value))
            self.event_version += 1

    def snapshot(self):
        with self.lock:
            return {
                'images': dict(self.images),
                'raw_color': self.raw_color,
                'raw_depth': self.raw_depth,
                'color_header': self.color_header,
                'depth_header': self.depth_header,
                'annotated_header': self.annotated_header,
                'camera_info': self.camera_info,
                'semantic_labels': self.semantic_labels,
                'semantic_confidence': self.semantic_confidence,
                'semantic_instances': self.semantic_instances,
                'semantic_available': bool(self.semantic_available),
                'semantic_header': self.semantic_header,
                'raw_arrival': dict(self.raw_arrival),
                'camera_info_arrival': self.camera_info_arrival,
                'semantic_arrival': self.semantic_arrival,
                'joint_arrival': self.joint_arrival,
                'versions': dict(self.versions),
                'last_arrival': dict(self.last_arrival),
                'fps': dict(self.fps),
                'targets': self.targets,
                'target_arrival': self.target_arrival,
                'joints': self.joints,
                'imu_latest': dict(self.imu_latest),
                'imu_topics': dict(self.imu_topics),
                'vision_status': self.vision_status,
                'planner_status': self.planner_status,
                'harvest_status': self.harvest_status,
                'harvest_status_version': self.harvest_status_version,
                'cockpit_status': self.cockpit_status,
                'events': list(self.events),
                'event_version': self.event_version,
                'action_status': self.action_status,
                'nbv_status': dict(self.nbv_status),
                'approach_status': dict(self.approach_status),
                'approach_heatmap': self.approach_heatmap,
                'approach_version': self.approach_version,
                'trajectory_count': self.trajectory_count,
                'trajectory_points': self.trajectory_points,
            }


_READONLY_WRAP_STYLE = (
    wx.TE_MULTILINE | wx.TE_READONLY |
    getattr(wx, 'TE_CHARWRAP', wx.TE_WORDWRAP))


class StableReadOnlyTextCtrl(wx.TextCtrl):
    """Read-only wrapped text that does not overwrite an active selection."""

    def __init__(self, parent, value='', style=0, **kwargs):
        wx.TextCtrl.__init__(
            self, parent, value=value,
            style=_READONLY_WRAP_STYLE | style, **kwargs)
        self._pending_value = None
        self.SetToolTip(
            '文字会按控件宽度自动换行；点击或选择文字时暂停刷新，失去焦点后恢复')
        self.Bind(wx.EVT_KILL_FOCUS, self._on_kill_focus)

    def SetValue(self, value):
        value = str(value)
        selection = self.GetSelection()
        if self.HasFocus() or selection[0] != selection[1]:
            self._pending_value = value
            return
        self._pending_value = None
        if self.GetValue() != value:
            self.ChangeValue(value)

    def _on_kill_focus(self, event):
        if self._pending_value is not None:
            value = self._pending_value
            self._pending_value = None
            wx.CallAfter(self.ChangeValue, value)
        event.Skip()


class ProtectedLogView(wx.Panel):
    """Wrapped log with explicit follow control and copy-safe selection."""

    ISSUE_MARKERS = (
        ' ERROR ', '[ERROR]', 'ERROR ·', '错误 ·', '错误：',
        ' WARN ', '[WARN]', 'WARN ·', '警告 ·', '失败：', '拒绝：',
    )

    def __init__(self, parent, minimum_size=(300, 155)):
        wx.Panel.__init__(self, parent)
        self._pending_content = ''
        self._latest_issue = ''

        root = wx.BoxSizer(wx.VERTICAL)
        follow_row = wx.BoxSizer(wx.HORIZONTAL)
        self.follow_check = wx.CheckBox(self, label='跟随最新')
        self.follow_check.SetValue(True)
        self.follow_check.SetToolTip(
            '关闭后新日志仍会加入，但不会夺走选区或滚动位置；点击日志会自动关闭')
        self.follow_state = wx.StaticText(self, label='实时跟随')
        self.copy_button = wx.Button(self, label='复制选中/全部', size=(112, -1))
        self.locate_button = wx.Button(self, label='定位最近异常', size=(108, -1))
        self.locate_button.Enable(False)
        follow_row.Add(
            self.follow_check, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        follow_row.Add(self.follow_state, 0, wx.ALIGN_CENTER_VERTICAL)
        root.Add(follow_row, 0, wx.BOTTOM | wx.EXPAND, 3)
        action_row = wx.BoxSizer(wx.HORIZONTAL)
        action_row.AddStretchSpacer(1)
        action_row.Add(self.locate_button, 0, wx.RIGHT, 5)
        action_row.Add(self.copy_button, 0)
        root.Add(action_row, 0, wx.BOTTOM | wx.EXPAND, 4)

        self.issue_text = StableReadOnlyTextCtrl(
            self, value='最近警告/错误：暂无', style=wx.BORDER_SIMPLE)
        self.issue_text.SetMinSize((-1, 46))
        self.issue_text.SetMaxSize((-1, 58))
        self.issue_text.SetBackgroundColour(wx.Colour(255, 244, 228))
        root.Add(self.issue_text, 0, wx.BOTTOM | wx.EXPAND, 4)

        self.text = wx.TextCtrl(
            self, style=_READONLY_WRAP_STYLE | wx.TE_RICH2)
        self.text.SetMinSize((-1, 92))
        self.text.SetToolTip(
            '按控件宽度逐字符换行；点击、滚轮或键盘选择会暂停自动滚动，复制不受新日志打断')
        root.Add(self.text, 1, wx.EXPAND)
        self.SetSizer(root)
        self.SetMinSize(minimum_size)

        self.follow_check.Bind(wx.EVT_CHECKBOX, self._on_follow_changed)
        self.copy_button.Bind(wx.EVT_BUTTON, self._on_copy)
        self.locate_button.Bind(wx.EVT_BUTTON, self._on_locate_issue)
        self.text.Bind(wx.EVT_LEFT_DOWN, self._on_manual_interaction)
        self.text.Bind(wx.EVT_MOUSEWHEEL, self._on_manual_interaction)
        self.text.Bind(wx.EVT_KEY_DOWN, self._on_manual_interaction)

    @property
    def following(self):
        return bool(self.follow_check.GetValue())

    def _pause_follow(self, reason='已暂停，选区受保护'):
        if self.following:
            self.follow_check.SetValue(False)
        self.follow_state.SetLabel(reason)

    def _on_manual_interaction(self, event):
        self._pause_follow()
        event.Skip()

    def _on_follow_changed(self, _event):
        if self.following:
            self.follow_state.SetLabel('实时跟随')
            self._render(self._pending_content, force=True)
            self.text.SetInsertionPointEnd()
            self.text.ShowPosition(self.text.GetLastPosition())
        else:
            self.follow_state.SetLabel('已暂停，选区受保护')

    @classmethod
    def _find_latest_issue(cls, lines):
        for line in reversed(lines):
            if any(marker in str(line) for marker in cls.ISSUE_MARKERS):
                return str(line)
        return ''

    def update_lines(self, lines):
        lines = [str(line) for line in lines]
        content = '\n'.join(lines)
        self._pending_content = content
        issue = self._find_latest_issue(lines)
        if issue and issue != self._latest_issue:
            self._latest_issue = issue
            self.issue_text.SetValue('最近警告/错误：' + issue)
            self.locate_button.Enable(True)
        self._render(content)

    def _render(self, content, force=False):
        current = self.text.GetValue()
        if content == current:
            return
        selection = self.text.GetSelection()
        preserve = not self.following or selection[0] != selection[1]
        if preserve and current and not content.startswith(current) and not force:
            # The shared deque rolled over. Defer the destructive rebuild until
            # the operator explicitly resumes following.
            self.follow_state.SetLabel('已暂停；旧日志选区受保护')
            return
        scroll = self.text.GetScrollPos(wx.VERTICAL)
        if content.startswith(current):
            self.text.AppendText(content[len(current):])
        else:
            self.text.ChangeValue(content)
        if preserve and not force:
            end = self.text.GetLastPosition()
            self.text.SetSelection(min(selection[0], end),
                                   min(selection[1], end))
            self.text.SetScrollPos(wx.VERTICAL, scroll)
        else:
            self.text.SetInsertionPointEnd()
            self.text.ShowPosition(self.text.GetLastPosition())

    @staticmethod
    def _copy_to_clipboard(value):
        if not value or not wx.TheClipboard.Open():
            return False
        try:
            wx.TheClipboard.SetData(wx.TextDataObject(value))
            return True
        finally:
            wx.TheClipboard.Close()

    def _on_copy(self, _event):
        start, end = self.text.GetSelection()
        value = self.text.GetRange(start, end) if start != end else ''
        if not value:
            issue_start, issue_end = self.issue_text.GetSelection()
            if issue_start != issue_end:
                value = self.issue_text.GetRange(issue_start, issue_end)
        if not value:
            value = self.text.GetValue()
        self._pause_follow(
            '已复制；保持暂停' if self._copy_to_clipboard(value)
            else '剪贴板不可用；保持暂停')

    def _on_locate_issue(self, _event):
        if not self._latest_issue:
            return
        self._render(self._pending_content, force=True)
        start = self.text.GetValue().rfind(self._latest_issue)
        if start >= 0:
            self._pause_follow('已定位最近异常')
            self.text.SetFocus()
            self.text.SetSelection(start, start + len(self._latest_issue))
            self.text.ShowPosition(start)


def _stamp_sec(stamp):
    if stamp is None:
        return 0.0
    try:
        value = float(stamp.to_sec()) if hasattr(stamp, 'to_sec') else float(stamp)
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def _header_dict(header):
    if header is None:
        return {}
    return {
        'seq': int(getattr(header, 'seq', 0)),
        'stamp_sec': _stamp_sec(getattr(header, 'stamp', None)),
        'frame_id': str(getattr(header, 'frame_id', '') or ''),
    }


def _imu_vector(value):
    value = value or object()
    return [float(getattr(value, name, 0.0))
            for name in ('x', 'y', 'z')]


def _imu_covariance(value):
    values = list(value or [])
    return [float(item) for item in values]


def _imu_dict(message, topic=''):
    """Serialize one raw sensor_msgs/Imu message without fusing its pose."""
    header = getattr(message, 'header', None)
    orientation = getattr(message, 'orientation', None)
    orientation_covariance = _imu_covariance(
        getattr(message, 'orientation_covariance', []))
    orientation_valid = bool(
        len(orientation_covariance) == 9 and
        orientation_covariance[0] >= 0.0 and orientation is not None)
    return {
        'topic': str(topic or ''),
        'header': _header_dict(header),
        'stamp_sec': _stamp_sec(getattr(header, 'stamp', None)),
        'frame_id': str(getattr(header, 'frame_id', '') or ''),
        'orientation_xyzw': [
            float(getattr(orientation, 'x', 0.0)),
            float(getattr(orientation, 'y', 0.0)),
            float(getattr(orientation, 'z', 0.0)),
            float(getattr(orientation, 'w', 1.0)),
        ],
        'orientation_covariance': orientation_covariance,
        'orientation_valid': orientation_valid,
        'angular_velocity_rad_s': _imu_vector(
            getattr(message, 'angular_velocity', None)),
        'angular_velocity_covariance': _imu_covariance(
            getattr(message, 'angular_velocity_covariance', [])),
        'linear_acceleration_m_s2': _imu_vector(
            getattr(message, 'linear_acceleration', None)),
        'linear_acceleration_covariance': _imu_covariance(
            getattr(message, 'linear_acceleration_covariance', [])),
    }


def _camera_info_dict(message):
    if message is None:
        return {}
    return {
        'width': int(getattr(message, 'width', 0)),
        'height': int(getattr(message, 'height', 0)),
        'distortion_model': str(getattr(message, 'distortion_model', '') or ''),
        'K': [float(value) for value in list(getattr(message, 'K', []))],
        'D': [float(value) for value in list(getattr(message, 'D', []))],
        'R': [float(value) for value in list(getattr(message, 'R', []))],
        'P': [float(value) for value in list(getattr(message, 'P', []))],
        'header': _header_dict(getattr(message, 'header', None)),
    }


def _target_dict(target):
    return {
        'label': str(getattr(target, 'label', '') or ''),
        'confidence': float(getattr(target, 'confidence', 0.0)),
        'bbox': [int(getattr(target, name, 0)) for name in
                 ('xmin', 'ymin', 'xmax', 'ymax')],
        'pixel': [int(getattr(target, 'pixel_x', 0)),
                  int(getattr(target, 'pixel_y', 0))],
        'depth_m': float(getattr(target, 'depth_m', 0.0)),
        'camera_point': [float(getattr(target.camera_point, name, 0.0))
                         for name in ('x', 'y', 'z')],
        'target_point': [float(getattr(target.target_point, name, 0.0))
                         for name in ('x', 'y', 'z')],
        'target_frame': str(getattr(target, 'target_frame', '') or ''),
        'target_point_valid': bool(getattr(target, 'target_point_valid', False)),
    }


def _normalise_class_names(value):
    """Normalize the vision node's zero-based class-name parameter."""
    if isinstance(value, dict):
        pairs = []
        for key, label in value.items():
            try:
                identifier = int(key)
            except (TypeError, ValueError):
                continue
            pairs.append((identifier, str(label or '').strip()))
        return dict(sorted((key, label) for key, label in pairs if label))
    if isinstance(value, (list, tuple)):
        return dict((index, str(label or '').strip())
                    for index, label in enumerate(value) if str(label or '').strip())
    return {}


def _semantic_gap_for_classes(class_names):
    names = set(str(value).strip().lower().replace('-', '_').replace(' ', '_')
                for value in (class_names or {}).values())
    required_aliases = {
        'fruit_stem': ('fruit_stem', 'peduncle', 'fruitstem', '果梗', '果柄'),
        'petiole': ('petiole', 'leaf_stem', 'leafstalk', '叶柄'),
    }
    missing = [key for key, aliases in required_aliases.items()
               if not any(alias.lower() in names for alias in aliases)]
    if not missing:
        return ''
    return ('verified model classes do not include %s; no fruit-stem/petiole '
            'label is inferred from another class' % ', '.join(missing))


def _transform_dict(transform):
    value = transform.transform
    quaternion = [float(value.rotation.x), float(value.rotation.y),
                  float(value.rotation.z), float(value.rotation.w)]
    matrix = make_transform(
        quaternion_xyzw_to_rotation_matrix(quaternion),
        [value.translation.x, value.translation.y, value.translation.z])
    return matrix, quaternion


def _imu_bundle(samples, topics, reference_stamp, before_s, after_s):
    """Build a JSON-safe raw IMU window and stability summary.

    The summary is diagnostic only.  It deliberately does not turn the D455
    six-axis IMU into an absolute camera orientation or a compass heading.
    """
    samples = dict(samples or {})
    result = {
        'reference_stamp_sec': float(reference_stamp),
        'window_before_s': float(before_s),
        'window_after_s': float(after_s),
        'topics': dict(topics or {}),
        'pose_role': 'raw motion/stability audit; not an absolute heading',
        'gyro_samples': list(samples.get('gyro') or []),
        'accel_samples': list(samples.get('accel') or []),
    }
    result['sample_counts'] = {
        'gyro': len(result['gyro_samples']),
        'accel': len(result['accel_samples']),
    }
    gyro = np.asarray([
        item.get('angular_velocity_rad_s', [0.0, 0.0, 0.0])
        for item in result['gyro_samples']], dtype=np.float64)
    accel = np.asarray([
        item.get('linear_acceleration_m_s2', [0.0, 0.0, 0.0])
        for item in result['accel_samples']], dtype=np.float64)

    def summary(array, expected_norm=None):
        if array.size == 0:
            return {
                'sample_count': 0,
                'mean': None,
                'std': None,
                'norm_mean': None,
                'norm_max': None,
                'expected_norm_residual_mean': None,
            }
        norms = np.linalg.norm(array, axis=1)
        residual = (np.abs(norms - float(expected_norm))
                    if expected_norm is not None else None)
        return {
            'sample_count': int(array.shape[0]),
            'mean': np.mean(array, axis=0).tolist(),
            'std': np.std(array, axis=0).tolist(),
            'norm_mean': float(np.mean(norms)),
            'norm_max': float(np.max(norms)),
            'expected_norm_residual_mean': (
                None if residual is None else float(np.mean(residual))),
        }

    result['summary'] = {
        'gyro': summary(gyro),
        # 9.80665 m/s^2 is used only as a gravity/stationarity diagnostic.
        'accel': summary(accel, expected_norm=9.80665),
    }
    result['available'] = bool(result['gyro_samples'] and
                               result['accel_samples'])
    return result


class RosBridge(object):

    def __init__(self, state):
        self.state = state
        self.bridge = CvBridge()
        self.depth_min_m = float(rospy.get_param('~depth_min_m', 0.20))
        self.depth_max_m = float(rospy.get_param('~depth_max_m', 4.00))
        self.output_dir = os.path.abspath(os.path.expanduser(
            rospy.get_param('~capture_output_dir',
                            '/home/catas/elfin_citrus_data/captures')))
        self.demo_camera_tf = bool(rospy.get_param('~demo_camera_tf', False))
        self.camera_name = str(rospy.get_param('~camera_name', 'RealSense'))
        self.camera_serial = str(rospy.get_param('~camera_serial', ''))
        self.camera_device_type = str(rospy.get_param(
            '~camera_device_type', ''))
        self.camera_usb_port_id = str(rospy.get_param('~camera_usb_port_id', ''))
        self.camera_usb_type = str(rospy.get_param('~camera_usb_type', ''))
        self.nbv_expected_camera_type = str(rospy.get_param(
            '~nbv_expected_camera_type', 'd455') or '').strip().lower()
        self.nbv_allow_non_expected_camera = bool(rospy.get_param(
            '~nbv_allow_non_expected_camera', False))
        self.nbv_require_usb3 = bool(rospy.get_param(
            '~nbv_require_usb3', True))
        self.python_executable = str(rospy.get_param(
            '~python_executable', os.environ.get('ELFIN_VISION_PYTHON', '')) or '')
        self.camera_fps = max(1, int(rospy.get_param('~camera_fps', 15)))
        self.camera_connected = bool(rospy.get_param(
            '~camera_connected', False))
        self.camera_initial_on = bool(rospy.get_param(
            '~camera_initial_on', False))
        self.camera_auto_start = bool(rospy.get_param(
            '~camera_auto_start', False))
        self.camera_auto_start_usb3_only = bool(rospy.get_param(
            '~camera_auto_start_usb3_only', False))
        self.camera_usb2_auto_start_delay_s = max(0.0, float(
            rospy.get_param('~camera_usb2_auto_start_delay_s', 10.0)))
        self.target_frame = str(rospy.get_param(
            '~target_frame', 'elfin_base_link'))
        self.flange_frame = str(rospy.get_param(
            '~flange_frame', 'elfin_end_link'))
        # 宽幅扫描的球面半径。0.55 m 与 0.45 m 都在前置勘定里 10/10 通过，取较大
        # 值让外侧站位的法兰离目标更远、IK 余量更多。
        self.sweep_observation_distance_m = max(0.05, float(rospy.get_param(
            '~sweep_observation_distance_m', 0.55)))
        # 没有有效检测时的兜底目标点：沿当前视线前方这个距离取一点。
        self.sweep_fallback_distance_m = max(0.05, float(rospy.get_param(
            '~sweep_fallback_distance_m', 0.55)))
        self.camera_processing = threading.Event()
        self.camera_display_mode = 'full' if self.camera_initial_on else 'off'
        if self.camera_initial_on:
            self.camera_processing.set()
        else:
            self.state.reset_vision('摄像头已关闭；视觉识别未启动')
        self.display_image_hz = max(1.0, float(
            rospy.get_param('~display_image_hz', 10.0)))
        self.annotated_display_hz = max(1.0, float(
            rospy.get_param('~annotated_display_hz', 3.0)))
        self.depth_display_hz = max(0.5, float(
            rospy.get_param('~depth_display_hz', 2.0)))
        self.nbv_output_dir = os.path.abspath(os.path.expanduser(
            rospy.get_param('~nbv_output_dir',
                            '/home/catas/elfin_citrus_data/nbv_batches')))
        self.nbv_max_views = max(1, int(rospy.get_param(
            '~nbv_max_views', 10)))
        self.nbv_require_semantics = bool(rospy.get_param(
            '~nbv_require_semantics', True))
        # Keep pilot collection usable when the detector has class masks but
        # no instance-ID topic.  Enable this only when the project protocol
        # explicitly requires the optional per-pixel instance audit; missing
        # IDs are still recorded as an auditable gap.
        self.nbv_require_instance_ids = bool(rospy.get_param(
            '~nbv_require_instance_ids', False))
        # IMU is an auxiliary archive contract, not a replacement for the
        # calibrated base->camera pose.  Formal D455 batches require both raw
        # gyro and accelerometer samples so missing streams cannot go unnoticed.
        self.nbv_require_imu = bool(rospy.get_param(
            '~nbv_require_imu', False))
        self.nbv_imu_window_before_s = max(0.01, float(rospy.get_param(
            '~nbv_imu_window_before_s', 0.25)))
        self.nbv_imu_window_after_s = max(0.0, float(rospy.get_param(
            '~nbv_imu_window_after_s', 0.05)))
        self.nbv_imu_max_age_s = max(0.05, float(rospy.get_param(
            '~nbv_imu_max_age_s', 0.75)))
        self.nbv_require_calibration = bool(rospy.get_param(
            '~nbv_require_calibration', not self.demo_camera_tf))
        self.nbv_sync_slop_s = max(0.01, float(rospy.get_param(
            '~nbv_sync_slop_s', 0.08)))
        # Inference is intentionally throttled below the RGB rate (usually
        # 8 Hz versus 15 Hz). Keep its frame association auditable without
        # treating normal model latency as a bad RGB-D pair.
        self.nbv_target_sync_slop_s = max(self.nbv_sync_slop_s, float(
            rospy.get_param('~nbv_target_sync_slop_s', 0.25)))
        self.nbv_max_frame_age_s = max(0.05, float(rospy.get_param(
            '~nbv_max_frame_age_s', 0.50)))
        self.nbv_max_joint_age_s = max(0.05, float(rospy.get_param(
            '~nbv_max_joint_age_s', 0.75)))
        self.nbv_semantic_max_age_s = max(0.05, float(rospy.get_param(
            '~nbv_semantic_max_age_s', 0.50)))
        self.nbv_scene_default = str(rospy.get_param(
            '~nbv_scene_id', ''))
        self.nbv_auto_enable_on_cockpit = bool(rospy.get_param(
            '~nbv_auto_enable_on_cockpit', False))
        self.nbv_mode_initial = bool(rospy.get_param(
            '~nbv_capture_mode_initial', False))
        self.nbv_point_button_topic = str(rospy.get_param(
            '~nbv_point_button_topic',
            '/elfin_freedrive_manager/raw_di'))
        self.nbv_point_button_bit = max(0, min(15, int(rospy.get_param(
            '~nbv_point_button_bit', 4))))
        self.nbv_point_button_gate = SinglePressGate()
        self.nbv_point_button_initialized = False
        self.nbv_point_button_init_lock = threading.Lock()
        self.calibration_config_file = os.path.abspath(os.path.expanduser(
            rospy.get_param('~calibration_config_file',
                            '/home/catas/ros_ws/src/elfin_vision/config/camera_to_robot.yaml')))
        self.calibration_data_dir = os.path.abspath(os.path.expanduser(
            rospy.get_param('~calibration_data_dir',
                            '/home/catas/elfin_citrus_data/calibration')))
        self.calibration_launcher = os.path.abspath(os.path.expanduser(
            rospy.get_param('~calibration_launcher',
                            '/home/catas/ros_ws/src/elfin_vision/scripts/start_eye_in_hand_calibration.sh')))
        self.calibrated_tf_reload_service = rospy.get_param(
            '~calibrated_tf_reload_service',
            '/elfin_vision/publish_camera_tf/reload')
        self.calibration_process = None
        self.calibration_run_dir = ''
        self.calibration_lock = threading.Lock()
        self.nbv_lock = threading.Lock()
        self.nbv_capture_inflight = False
        self.nbv_recorder = BatchRecorder(
            self.nbv_output_dir, max_views=self.nbv_max_views)
        self.nbv_last_status = self.nbv_recorder.status()
        if self.nbv_mode_initial:
            self.nbv_last_status = self.nbv_recorder.enable(
                scene_id=self.nbv_scene_default, notes='', resume=True)
        self._publish_nbv_status(self.nbv_last_status)
        if tf2_ros is not None:
            self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(15.0))
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        else:
            self.tf_buffer = None
            self.tf_listener = None
        self.camera_info_topic = rospy.get_param(
            '~camera_info_topic', '/camera/color/camera_info')
        self.camera_optical_frame = str(rospy.get_param(
            '~camera_optical_frame', ''))
        self.model_sha256 = str(rospy.get_param(
            '/elfin_vision/citrus_rgbd_node/model_sha256', '') or '')
        self.semantic_class_names = _normalise_class_names(rospy.get_param(
            '/elfin_vision/citrus_rgbd_node/class_names',
            {'0': 'citrus', '1': 'tree'}))
        if not self.semantic_class_names:
            self.semantic_class_names = {0: 'citrus', 1: 'tree'}
        self.semantic_gap = _semantic_gap_for_classes(
            self.semantic_class_names)
        approach_parameters = dict(rospy.get_param(
            '~approach_direction', {}) or {})
        reconstruction = dict(approach_parameters.get('reconstruction') or {})
        planning = dict(approach_parameters.get('planning') or {})
        self.approach_config = dict(APPROACH_DEFAULT_CONFIG)
        self.approach_config.update(planning)
        approach_execution = dict(approach_parameters.get('execution') or {})
        self.approach_tool_timeout_s = max(
            1.0, float(approach_execution.get(
                'tool_status_timeout_s', 4.0)))
        self.approach_result_dir = os.path.abspath(os.path.expanduser(
            rospy.get_param(
                '~approach_result_dir',
                '/home/catas/elfin_citrus_data/approach_results')))
        self.approach_minimum_views = max(1, int(approach_parameters.get(
            'minimum_views_for_complete_map', self.nbv_max_views)))
        self.approach_session = SemanticApproachSession(
            class_names=self.semantic_class_names,
            resolution=float(reconstruction.get('resolution_m', 0.006)),
            max_voxels=int(reconstruction.get('maximum_voxels', 600000)),
            sample_stride=int(reconstruction.get('sample_stride', 4)),
            max_points=int(reconstruction.get('maximum_points_per_view',
                                               24000)),
            max_range_m=float(reconstruction.get('maximum_range_m', 4.0)))
        self.approach_group_number = None
        self.approach_discovered_targets = []
        self.approach_compute_lock = threading.Lock()
        self.approach_plan_cache = None
        # 柔性探针 48 次实测的落盘位置；方法排列种子与它同目录，断点续做靠这两个文件。
        self.probe_trials_csv = os.path.abspath(os.path.expanduser(
            rospy.get_param('~probe_trials_csv',
                            probe_trials.default_csv_path())))
        self.probe_lock = threading.Lock()
        # 当前待做试验的方法只存在内存里，界面从不读它；扫描完成后自动规划也用它。
        self.probe_active = None
        self.code_version = str(rospy.get_param(
            '~code_version', os.environ.get('ELFIN_VISION_CODE_VERSION', 'workspace')))
        self.semantic_labels_topic = rospy.get_param(
            '~semantic_labels_topic',
            '/elfin_vision/citrus_rgbd_node/semantic_labels')
        self.semantic_confidence_topic = rospy.get_param(
            '~semantic_confidence_topic',
            '/elfin_vision/citrus_rgbd_node/semantic_confidence')
        self.semantic_instances_topic = rospy.get_param(
            '~semantic_instances_topic',
            '/elfin_vision/citrus_rgbd_node/semantic_instances')
        self.semantic_available_topic = rospy.get_param(
            '~semantic_available_topic',
            '/elfin_vision/citrus_rgbd_node/semantic_available')
        self.gyro_topic = str(rospy.get_param(
            '~gyro_topic', '/camera_imu/gyro/sample'))
        self.accel_topic = str(rospy.get_param(
            '~accel_topic', '/camera_imu/accel/sample'))
        self.image_gate_lock = threading.Lock()
        self.last_image_process = {}
        self.plan_service = rospy.get_param(
            '~plan_service', '/elfin_vision/citrus_moveit_planner/plan')
        self.approach_plan_service = rospy.get_param(
            '~approach_plan_service',
            '/elfin_vision/citrus_moveit_planner/plan_approach_direction')
        # 预定义宽幅扫描的每个绝对站位都走这条服务；它内部与采摘共用同一条
        # 碰撞校验与执行门禁链，界面这边不允许再有第二条运动通路。
        self.observation_pose_service = rospy.get_param(
            '~observation_pose_service',
            '/elfin_vision/citrus_moveit_planner/plan_observation_pose')
        self.command_service = rospy.get_param(
            '~command_service', '/elfin_vision/harvest_coordinator/command')
        self.runtime_config_service = rospy.get_param(
            '~runtime_config_service',
            '/elfin_vision/citrus_moveit_planner/runtime_config')
        self.cockpit_active_service = rospy.get_param(
            '~cockpit_active_service',
            '/elfin_vision/cockpit_controller/set_active')
        self.planner_stop_service = rospy.get_param(
            '~planner_stop_service',
            '/elfin_vision/citrus_moveit_planner/stop')
        self.planner_cockpit_claim_service = rospy.get_param(
            '~planner_cockpit_claim_service',
            '/elfin_vision/citrus_moveit_planner/set_cockpit_active')
        self.active_recovery_service = rospy.get_param(
            '~active_recovery_service',
            '/elfin_vision/citrus_moveit_planner/cockpit_recover')
        self.panel_joint_service = rospy.get_param(
            '~panel_joint_service', '/elfin_basic_api/joint_teleop')
        self.panel_cart_service = rospy.get_param(
            '~panel_cart_service', '/elfin_basic_api/cart_teleop')
        self.panel_stop_service = rospy.get_param(
            '~panel_stop_service', '/elfin_basic_api/stop_teleop')
        self.panel_cockpit_service = rospy.get_param(
            '~panel_cockpit_service', '/elfin_basic_api/cockpit_teleop')
        # 上一次宽幅扫描实测达成的方位角跨度，进 manifest 的 session_metadata。
        self.sweep_last_azimuth_span_deg = None
        self.sweep_last_target_fallback = False
        self.panel_reference_service = rospy.get_param(
            '~panel_reference_service',
            '/elfin_basic_api/set_reference_link')
        self.panel_end_service = rospy.get_param(
            '~panel_end_service', '/elfin_basic_api/set_end_link')
        self.panel_dynamic_namespace = rospy.get_param(
            '~panel_dynamic_namespace', '/elfin_basic_api')
        self.panel_jog_message_lock = threading.Lock()
        self.panel_jog_message = 'Panel 实时点动待命'
        self.panel_end_prepared = False
        self.panel_speed_lock = threading.Lock()
        self.panel_speed_desired = None
        self.panel_speed_worker_running = False
        self.panel_requested_actions = ()
        self.panel_recovery_lock = threading.Lock()
        self.panel_recovery_inflight = False
        self.panel_failed_recovery_actions = None
        self.panel_start_recoverable = False
        self.panel_jog = LatestJogDispatcher(
            self._start_panel_jog, self._stop_panel_jog,
            self._panel_jog_result)
        # Published by the wx timer, not a rospy background timer. If the GUI
        # event loop freezes, this heartbeat stops as well.
        self.dashboard_heartbeat_pub = rospy.Publisher(
            '/elfin_vision/dashboard/heartbeat', Header, queue_size=1)
        self.approach_result_pub = rospy.Publisher(
            '/elfin_vision/approach_direction/result', String,
            queue_size=1, latch=True)
        self.approach_heatmap_pub = rospy.Publisher(
            '/elfin_vision/approach_direction/heatmap', Image,
            queue_size=1, latch=True)
        self.approach_marker_pub = rospy.Publisher(
            '/elfin_vision/approach_direction/markers', MarkerArray,
            queue_size=1, latch=True)

        rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback,
                         queue_size=1, buff_size=2 ** 22, tcp_nodelay=True)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image,
                         self.depth_callback, queue_size=1,
                         buff_size=2 ** 22, tcp_nodelay=True)
        rospy.Subscriber(self.camera_info_topic, CameraInfo,
                         self.camera_info_callback, queue_size=1)
        rospy.Subscriber('/elfin_vision/citrus_rgbd_node/annotated', Image,
                         self.annotated_callback, queue_size=1,
                         buff_size=2 ** 22, tcp_nodelay=True)
        rospy.Subscriber('/elfin_vision/citrus_rgbd_node/targets',
                         CitrusTargetArray, self.targets_callback, queue_size=1)
        rospy.Subscriber(self.semantic_labels_topic, Image,
                         self.semantic_labels_callback, queue_size=1,
                         buff_size=2 ** 22, tcp_nodelay=True)
        rospy.Subscriber(self.semantic_confidence_topic, Image,
                         self.semantic_confidence_callback, queue_size=1,
                         buff_size=2 ** 22, tcp_nodelay=True)
        rospy.Subscriber(self.semantic_instances_topic, Image,
                         self.semantic_instances_callback, queue_size=1,
                         buff_size=2 ** 22, tcp_nodelay=True)
        rospy.Subscriber(self.semantic_available_topic, Bool,
                         self.semantic_available_callback, queue_size=1)
        rospy.Subscriber('/elfin_vision/citrus_rgbd_node/status', String,
                         self.vision_status_callback, queue_size=1)
        rospy.Subscriber('/elfin_vision/citrus_moveit_planner/status', String,
                         self.planner_status_callback, queue_size=1)
        rospy.Subscriber('/elfin_vision/harvest_coordinator/status',
                         HarvestStatus, self.harvest_status_callback,
                         queue_size=1)
        rospy.Subscriber('/elfin_vision/harvest_coordinator/events', String,
                         self.harvest_event_callback, queue_size=20)
        rospy.Subscriber('/elfin_vision/cockpit_controller/status',
                         CockpitStatus, self.cockpit_status_callback,
                         queue_size=1)
        rospy.Subscriber('/joint_states', JointState, self.joints_callback,
                         queue_size=1)
        rospy.Subscriber(self.gyro_topic, Imu, self.gyro_callback,
                         queue_size=200)
        rospy.Subscriber(self.accel_topic, Imu, self.accel_callback,
                         queue_size=100)
        rospy.Subscriber(self.nbv_point_button_topic, UInt16,
                         self.point_button_callback, queue_size=1)
        rospy.Subscriber('/move_group/display_planned_path', DisplayTrajectory,
                         self.trajectory_callback, queue_size=1)
        if self.nbv_mode_initial:
            try:
                self._rebuild_approach_from_staging(
                    int(self.nbv_recorder.group_number))
            except Exception as error:
                self._approach_state(
                    'error', '启动时恢复语义方向图失败：%s' % error)

    def should_process_image(self, key, message):
        if not self.camera_processing.is_set():
            return False
        self.state.observe_image(key, message.header.stamp)
        now = time.monotonic()
        if key == 'depth':
            rate = self.depth_display_hz
        elif key == 'annotated':
            rate = self.annotated_display_hz
        else:
            # RGB-only is the low-load human-driving path and can afford a
            # smoother view. Full RGB-D mode spends its budget on depth,
            # pointcloud and inference, so dashboard painting stays slower.
            rate = 10.0 if self.camera_display_mode == 'rgb' \
                else self.display_image_hz
        with self.image_gate_lock:
            previous = self.last_image_process.get(key, 0.0)
            if now - previous < 1.0 / rate:
                return False
            self.last_image_process[key] = now
        return True

    def set_camera_processing(self, enabled):
        if enabled:
            self.state.reset_vision('相机正在启动；等待视觉识别节点')
            self.camera_processing.set()
        else:
            self.camera_processing.clear()
            self.state.reset_vision('摄像头已关闭；视觉识别已停止')

    def color_callback(self, message):
        if not self.camera_processing.is_set():
            return
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            image = np.ascontiguousarray(image)
            if self.camera_processing.is_set():
                self.state.set_raw_color(image, message.header)
            if not self.should_process_image('color', message):
                return
            self.state.set_image('color', image)
        except CvBridgeError as error:
            rospy.logwarn_throttle(5.0, 'dashboard color conversion failed: %s', error)

    def depth_callback(self, message):
        if not self.camera_processing.is_set():
            return
        try:
            raw = self.bridge.imgmsg_to_cv2(message, desired_encoding='passthrough')
            raw = np.ascontiguousarray(raw)
            if self.camera_processing.is_set():
                self.state.set_raw_depth(raw, message.header)
            if not self.should_process_image('depth', message):
                return
            depth_m = depth_to_meters(raw, message.encoding)
            visual = colorize_depth(depth_m, self.depth_min_m, self.depth_max_m)
            self.state.set_image('depth', visual, raw_depth=raw)
        except (CvBridgeError, ValueError) as error:
            rospy.logwarn_throttle(5.0, 'dashboard depth conversion failed: %s', error)

    def annotated_callback(self, message):
        if not self.camera_processing.is_set():
            return
        if not self.should_process_image('annotated', message):
            return
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            self.state.set_image('annotated', np.ascontiguousarray(image))
            with self.state.lock:
                self.state.annotated_header = message.header
        except CvBridgeError as error:
            rospy.logwarn_throttle(5.0, 'dashboard annotation conversion failed: %s', error)

    def camera_info_callback(self, message):
        self.state.set_camera_info(message)

    def _semantic_image_callback(self, component, message, encoding):
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding=encoding)
            self.state.set_semantic_component(
                component, np.ascontiguousarray(image), header=message.header)
        except (CvBridgeError, ValueError) as error:
            rospy.logwarn_throttle(
                5.0, 'dashboard semantic %s conversion failed: %s',
                component, error)

    def semantic_labels_callback(self, message):
        self._semantic_image_callback('labels', message, 'mono8')

    def semantic_confidence_callback(self, message):
        self._semantic_image_callback('confidence', message, '32FC1')

    def semantic_instances_callback(self, message):
        self._semantic_image_callback('instances', message, '16UC1')

    def semantic_available_callback(self, message):
        self.state.set_semantic_component('available', bool(message.data))

    def targets_callback(self, message):
        self.state.record_targets(message)

    def vision_status_callback(self, message):
        with self.state.lock:
            self.state.vision_status = message.data

    def planner_status_callback(self, message):
        with self.state.lock:
            self.state.planner_status = message.data

    def harvest_status_callback(self, message):
        with self.state.lock:
            self.state.harvest_status = message
            self.state.harvest_status_version += 1

    def harvest_event_callback(self, message):
        self.state.append_event(message.data)

    def cockpit_status_callback(self, message):
        with self.state.lock:
            self.state.cockpit_status = message

    def joints_callback(self, message):
        self.state.record_joints(message)

    def gyro_callback(self, message):
        self.state.record_imu('gyro', message, self.gyro_topic)

    def accel_callback(self, message):
        self.state.record_imu('accel', message, self.accel_topic)

    def point_button_callback(self, message):
        """Map the physical POINT rising edge to one validated NBV capture."""
        pressed = bool(
            int(message.data) & (1 << int(self.nbv_point_button_bit)))
        with self.nbv_point_button_init_lock:
            if not self.nbv_point_button_initialized:
                # raw_di is latched. Its first message only establishes the
                # current level so restarting the Dashboard while POINT is held
                # cannot create a snapshot.
                self.nbv_point_button_initialized = True
                if pressed:
                    self.nbv_point_button_gate.press()
                else:
                    self.nbv_point_button_gate.release()
                return
        if not pressed:
            self.nbv_point_button_gate.release()
            return
        if not self.nbv_point_button_gate.press():
            return
        if not self.nbv_recorder.enabled:
            self.state.set_action_status(
                '实体 POINT：论文采集未开启，本次未保存视角')
            return
        self.state.set_action_status(
            '实体 POINT：已请求一次论文视角快照')
        self.state.append_event('[NBV] 实体 POINT：已请求一次视角快照')
        self.capture_nbv_view_async(capture_trigger='tool_point_di4')

    def trajectory_callback(self, message):
        trajectories = list(message.trajectory)
        point_count = sum(len(item.joint_trajectory.points) for item in trajectories)
        with self.state.lock:
            self.state.trajectory_count = len(trajectories)
            self.state.trajectory_points = point_count

    @staticmethod
    def _target_request_fields(target_point, target_frame, target_label):
        point = Point()
        if target_point is not None:
            point.x, point.y, point.z = target_point
        return {
            'target_point': point,
            'target_frame': str(target_frame or ''),
            'target_label': str(target_label or ''),
            'target_point_valid': target_point is not None,
        }

    def request_plan(self, target_index, target_point=None,
                     target_frame='', target_label=''):
        self.state.set_action_status('正在请求状态机只规划目标 %d...' % target_index)
        try:
            rospy.wait_for_service(self.command_service, timeout=1.0)
            client = rospy.ServiceProxy(self.command_service, HarvestCommand)
            fields = self._target_request_fields(
                target_point, target_frame, target_label)
            response = client(command='plan', target_index=target_index,
                              continuous=False, patrol=False,
                              **fields)
            prefix = '请求已接收：' if response.accepted else '请求被拒绝：'
            self.state.set_action_status(prefix + response.message)
            return
        except Exception:
            # Keep the vision-only launcher backward compatible when the
            # harvesting coordinator is intentionally not started.
            pass
        try:
            rospy.wait_for_service(self.plan_service, timeout=3.0)
            client = rospy.ServiceProxy(self.plan_service, PlanCitrus)
            response = client(target_index=target_index, execute=False)
            prefix = '规划成功：' if response.success else '规划失败：'
            self.state.set_action_status(prefix + response.message)
        except Exception as error:
            self.state.set_action_status('规划服务不可用：' + str(error))

    def request_harvest_command(self, command, target_index=-1,
                                target_point=None, target_frame='',
                                target_label='', continuous=False,
                                patrol=False):
        self.state.set_action_status('正在发送采摘命令：%s' % command)
        try:
            rospy.wait_for_service(self.command_service, timeout=2.0)
            client = rospy.ServiceProxy(self.command_service, HarvestCommand)
            fields = self._target_request_fields(
                target_point, target_frame, target_label)
            response = client(command=command, target_index=int(target_index),
                              continuous=bool(continuous),
                              patrol=bool(patrol),
                              **fields)
            prefix = '命令已接收：' if response.accepted else '命令被拒绝：'
            self.state.set_action_status(prefix + response.message)
            return bool(response.accepted), str(response.message)
        except Exception as error:
            self.state.set_action_status('采摘状态机不可用：' + str(error))
            return False, str(error)

    def request_cockpit_active(self, enabled):
        action = '进入' if enabled else '退出'
        self.state.set_action_status('正在%s驾驶舱...' % action)
        try:
            rospy.wait_for_service(self.cockpit_active_service, timeout=2.0)
            response = rospy.ServiceProxy(
                self.cockpit_active_service, SetBool)(bool(enabled))
            prefix = '完成：' if response.success else '拒绝：'
            self.state.set_action_status(prefix + response.message)
            return bool(response.success), str(response.message)
        except Exception as error:
            message = '驾驶舱服务不可用：%s' % error
            self.state.set_action_status(message)
            return False, message

    def release_cockpit_for_moveit(self):
        """Stop Panel jog and release both cockpit ownership layers."""
        self.set_cockpit_actions(())
        snapshot = self.state.snapshot()
        cockpit = snapshot.get('cockpit_status')
        controller_result = (True, '驾驶舱节点未报告 ACTIVE')
        if cockpit is not None:
            controller_result = self.request_cockpit_active(False)
            if not controller_result[0]:
                return False, '无法停止驾驶舱节点：' + controller_result[1]
        try:
            rospy.wait_for_service(
                self.planner_cockpit_claim_service, timeout=1.0)
            response = rospy.ServiceProxy(
                self.planner_cockpit_claim_service, SetBool)(False)
            if not response.success:
                return False, 'MoveIt 控制权释放失败：' + str(response.message)
        except Exception as error:
            return False, 'MoveIt 驾驶舱控制权服务不可用：%s' % error
        message = ('已停止 Panel 点动并释放驾驶舱独占；'
                   '方向规划/执行结束前不会自动重新接管')
        self.state.set_action_status(message)
        self.state.append_event('[伸入方向] ' + message)
        return True, message

    @staticmethod
    def _service_response(response):
        return bool(response.success), str(response.message)

    def _panel_service(self, name, service_type, value):
        rospy.wait_for_service(name, timeout=1.0)
        return rospy.ServiceProxy(name, service_type)(value)

    def _prepare_panel_end(self):
        if self.panel_end_prepared:
            return True, '末端坐标系已准备'
        response = self._panel_service(
            self.panel_end_service, SetString, 'elfin_end_link')
        success, message = self._service_response(response)
        if success:
            self.panel_end_prepared = True
        return success, message

    @staticmethod
    def _signed_action(actions, positive, negative):
        held = set(actions)
        return float(positive in held) - float(negative in held)

    def _start_panel_jog(self, actions, _specs):
        self.panel_start_recoverable = False
        success, message = self._prepare_panel_end()
        if not success:
            return False, '无法设置 Panel 末端坐标系：' + message

        base = self._signed_action(actions, 'base_right', 'base_left')
        cart_actions = set(actions) - {'base_left', 'base_right'}
        if abs(base) > 1e-6:
            if cart_actions:
                return False, 'Z/X J1 点动不能与笛卡尔组合键同时使用'
            response = self._panel_service(
                self.panel_joint_service, SetInt16,
                1 if base > 0.0 else -1)
            return self._service_response(response)

        values = {
            'forward': self._signed_action(actions, 'forward', 'back'),
            'strafe': self._signed_action(actions, 'right', 'left'),
            'vertical': self._signed_action(actions, 'up', 'down'),
            # Keep the established Panel signs: LEFT=world Rz+,
            # UP=camera Rx+, E=flange Rz+.
            'yaw': self._signed_action(
                actions, 'yaw_left', 'yaw_right'),
            'pitch': self._signed_action(
                actions, 'pitch_up', 'pitch_down'),
            'roll': self._signed_action(
                actions, 'roll_right', 'roll_left'),
        }
        if max(abs(value) for value in values.values()) <= 1e-6:
            return False, '相反方向互相抵消；已保持停止'
        rospy.wait_for_service(self.panel_cockpit_service, timeout=1.0)
        response = rospy.ServiceProxy(
            self.panel_cockpit_service, CockpitJog)(
                camera_frame='camera_cockpit_optical_frame', **values)
        result = self._service_response(response)
        self.panel_start_recoverable = not result[0]
        return result

    def _stop_panel_jog(self):
        response = self._panel_service(
            self.panel_stop_service, SetBool, True)
        return self._service_response(response)

    def _panel_jog_result(self, operation, action, success, message):
        if operation == 'stop':
            text = ('STOP 已发送：' if success else 'STOP 失败：') + message
        else:
            text = (
                ('点动已启动：' if success else '点动被拒绝：') +
                str(action) + '；' + message)
        with self.panel_jog_message_lock:
            self.panel_jog_message = text
        self.state.set_action_status(text)
        if operation == 'start' and not success and \
                getattr(self, 'panel_start_recoverable', False):
            self._schedule_panel_recovery(tuple(action), message)

    def _panel_recovery_values(self, actions):
        return {
            'forward': self._signed_action(actions, 'forward', 'back'),
            'strafe': self._signed_action(actions, 'right', 'left'),
            'vertical': self._signed_action(actions, 'up', 'down'),
            'view_yaw': self._signed_action(
                actions, 'yaw_left', 'yaw_right'),
            'view_pitch': self._signed_action(
                actions, 'pitch_up', 'pitch_down'),
            'view_roll': self._signed_action(
                actions, 'roll_right', 'roll_left'),
            'base_yaw': self._signed_action(
                actions, 'base_right', 'base_left'),
            'level_only': False,
            'automatic': False,
        }

    def _perform_panel_recovery(self, actions):
        values = self._panel_recovery_values(actions)
        if abs(values['base_yaw']) > 1e-6 and max(
                abs(values[key]) for key in (
                    'forward', 'strafe', 'vertical', 'view_yaw',
                    'view_pitch', 'view_roll')) <= 1e-6:
            return False, 'J1 直接点动已到关节边界，请换向'
        released = False
        recovery = None
        error = None
        reclaim_error = None
        try:
            stopped, stop_message = self._stop_panel_jog()
            if not stopped:
                raise RuntimeError('主动脱困前 Panel STOP 失败：%s' % stop_message)
            rospy.wait_for_service(
                self.planner_cockpit_claim_service, timeout=1.0)
            claim_client = rospy.ServiceProxy(
                self.planner_cockpit_claim_service, SetBool)
            release = claim_client(False)
            if not release.success:
                raise RuntimeError('无法暂停驾驶舱独占：%s' % release.message)
            released = True
            rospy.wait_for_service(self.active_recovery_service, timeout=2.0)
            recovery = rospy.ServiceProxy(
                self.active_recovery_service, CockpitRecover)(**values)
        except Exception as exception:
            error = str(exception)
        finally:
            if released:
                try:
                    reclaim = rospy.ServiceProxy(
                        self.planner_cockpit_claim_service, SetBool)(True)
                    if not reclaim.success:
                        reclaim_error = str(reclaim.message)
                except Exception as exception:
                    reclaim_error = str(exception)
        if reclaim_error:
            return False, '主动脱困后无法恢复驾驶舱接管：' + reclaim_error
        if error:
            return False, '主动脱困异常：' + error
        return bool(recovery.success), str(recovery.message)

    def _schedule_panel_recovery(self, actions, rejection):
        actions = tuple(sorted(actions))
        with self.panel_recovery_lock:
            if self.panel_recovery_inflight or \
                    actions == self.panel_failed_recovery_actions:
                return False
            self.panel_recovery_inflight = True
        with self.panel_jog_message_lock:
            self.panel_jog_message = (
                '点动卡滞：%s；正在主动整理关节构型' % rejection)

        def worker():
            success, message = self._perform_panel_recovery(actions)
            with self.panel_recovery_lock:
                self.panel_recovery_inflight = False
                current = tuple(self.panel_requested_actions)
                # One physical hold gets at most one reorganization. A key-up
                # or changed direction clears this signature.
                self.panel_failed_recovery_actions = actions
            with self.panel_jog_message_lock:
                self.panel_jog_message = message
            self.state.set_action_status(message)
            if not success or current != actions:
                return
            # The rejected dispatcher generation clears itself immediately
            # after reporting. Wait for that bounded cleanup before resuming
            # only the keys that are still physically held.
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                snapshot = self.panel_jog.snapshot()
                if not snapshot['desired_actions'] and \
                        not snapshot['motion_possible']:
                    break
                time.sleep(0.01)
            with self.panel_recovery_lock:
                still_held = tuple(self.panel_requested_actions) == actions
            if still_held:
                self.panel_jog.set_actions(actions)

        threading.Thread(
            target=worker, name='panel-active-recovery', daemon=True).start()
        return True

    def start_cockpit_jog(self, action):
        started = self.panel_jog.press(action)
        if started:
            with self.panel_jog_message_lock:
                self.panel_jog_message = (
                    '按下已接收：%s；正在提交 Panel 点动' % action)
        return started

    def set_cockpit_actions(self, actions):
        normalized = tuple(sorted(set(str(action) for action in actions)))
        with self.panel_recovery_lock:
            self.panel_requested_actions = normalized
            if not normalized or normalized != self.panel_failed_recovery_actions:
                self.panel_failed_recovery_actions = None
        changed = self.panel_jog.set_actions(normalized)
        if changed:
            with self.panel_jog_message_lock:
                self.panel_jog_message = (
                    '组合键已接收：%s；正在提交唯一 Panel 点动' %
                    ('+'.join(actions) if actions else 'STOP'))
        return changed

    def stop_cockpit_jog(self, action=None):
        return self.panel_jog.release(action)

    def cockpit_jog_message(self):
        with self.panel_jog_message_lock:
            return self.panel_jog_message

    def set_panel_jog_speed(self, percent):
        percent = max(1, min(100, int(percent)))
        with self.panel_speed_lock:
            self.panel_speed_desired = percent
            if self.panel_speed_worker_running:
                return
            self.panel_speed_worker_running = True

        def worker():
            while True:
                with self.panel_speed_lock:
                    requested = self.panel_speed_desired
                    self.panel_speed_desired = None
                try:
                    client = dynamic_reconfigure.client.Client(
                        self.panel_dynamic_namespace, timeout=1.0)
                    client.update_configuration(
                        {'velocity_scaling': requested / 100.0})
                    message = (
                        'Panel 点动最高速度已设为 %d%%' % requested)
                except Exception as error:
                    message = 'Panel 点动速度设置失败：%s' % error
                with self.panel_jog_message_lock:
                    self.panel_jog_message = message
                self.state.set_action_status(message)
                with self.panel_speed_lock:
                    if self.panel_speed_desired is None:
                        self.panel_speed_worker_running = False
                        return

        threading.Thread(target=worker, daemon=True).start()

    def publish_dashboard_heartbeat(self):
        message = Header()
        message.stamp = rospy.Time.now()
        self.dashboard_heartbeat_pub.publish(message)

    def request_planner_stop(self):
        try:
            rospy.wait_for_service(self.planner_stop_service, timeout=1.0)
            response = rospy.ServiceProxy(
                self.planner_stop_service, Trigger)()
            return bool(response.success), str(response.message)
        except Exception as error:
            return False, 'MoveIt Stop 不可用：%s' % error

    def run_flange_snake_maneuver(self, reverse=False,
                                  base_duration_s=0.15):
        """Run the shared bounded Panel observation steps.

        The cockpit stays active and every started step is followed by the
        same Panel STOP service used by key release.  This deliberately avoids
        the legacy MoveIt Cobra pose and any direct J1 command.
        """
        direction = '右侧' if reverse else '左侧'
        self.state.set_action_status(
            '正在执行%s法兰原点蛇形观察短步...' % direction)
        try:
            # Drain any held keyboard/mouse command before issuing a timed
            # manoeuvre through the same CockpitJog service.
            self.panel_jog.set_actions(())
            time.sleep(0.03)
            success, message = self._prepare_panel_end()
            if not success:
                return False, '无法设置法兰末端坐标系：' + message
            steps = cockpit_cobra_steps(reverse=bool(reverse))
            for index, step in enumerate(steps):
                response = None
                try:
                    rospy.wait_for_service(
                        self.panel_cockpit_service, timeout=1.0)
                    response = rospy.ServiceProxy(
                        self.panel_cockpit_service, CockpitJog)(
                            forward=float(step.forward),
                            strafe=float(step.strafe),
                            vertical=float(step.vertical),
                            yaw=float(step.yaw),
                            pitch=float(step.pitch),
                            roll=0.0,
                            camera_frame='camera_cockpit_optical_frame')
                    if not response.success:
                        return False, '第 %d 步被拒绝：%s' % (
                            index + 1, response.message)
                    deadline = time.monotonic() + max(
                        0.02, float(base_duration_s) *
                        float(step.duration_scale))
                    while time.monotonic() < deadline and \
                            not rospy.is_shutdown():
                        time.sleep(min(0.01, max(
                            0.0, deadline - time.monotonic())))
                finally:
                    stop_success, stop_message = self._stop_panel_jog()
                    if not stop_success:
                        return False, '第 %d 步 STOP 失败：%s' % (
                            index + 1, stop_message)
            message = (
                '%s法兰原点蛇形观察完成：后撤/抬升/俯视与短侧移；'
                '未调用 MoveIt 眼镜蛇或 J1' % direction)
            self.state.set_action_status(message)
            return True, message
        except Exception as error:
            try:
                self._stop_panel_jog()
            except Exception:
                pass
            message = '法兰原点蛇形观察失败：%s' % error
            self.state.set_action_status(message)
            return False, message

    def _auto_sweep_progress(self):
        """Return (group_number, view_count) as the only success evidence.

        `capture_nbv_view_async` 是异步的，真正的记录计数在 BatchRecorder 上。
        第 10 张封存后 records 会清零并把 group_number +1，所以「记满」必须靠
        组号推进来判断，不能只看计数递增。
        """
        status = self.nbv_recorder.status()
        return (int(status.get('group_number', 0)),
                int(status.get('view_count', 0)))

    def _auto_sweep_capture_one(self, view_number, scene_id, notes,
                                timeout_s, poll_s=0.05, retry_s=0.5):
        """Poll one snapshot through the unchanged capture gate until it lands.

        门禁标准一个都不放宽：这里既不改 `validate_capture_timing`，也不改任何
        `nbv_*` 阈值，只是把「不满足就 raise 让人重按」换成「等一小会儿再试」。
        判定成功只认 `_auto_sweep_progress()` 的推进，不靠 sleep 猜。
        """
        start_group, start_count = self._auto_sweep_progress()
        deadline = time.monotonic() + max(1.0, float(timeout_s))
        attempts = 0
        while not rospy.is_shutdown():
            attempts += 1
            self.capture_nbv_view_async(
                scene_id=scene_id, notes=notes,
                capture_trigger='auto_ten_view_sweep')
            attempt_deadline = min(deadline, time.monotonic() + float(retry_s))
            while time.monotonic() < attempt_deadline and \
                    not rospy.is_shutdown():
                group, count = self._auto_sweep_progress()
                if group > start_group or count > start_count:
                    return True, '第 %d 个视角已记录（尝试 %d 次）' % (
                        view_number, attempts)
                time.sleep(poll_s)
            if time.monotonic() >= deadline:
                break
        status = self.nbv_recorder.status()
        reason = str(status.get('last_error') or
                     status.get('last_result') or '门禁未通过且无错误记录')
        return False, '第 %d 个视角在 %.0f s 内未通过采集门禁（尝试 %d 次）：%s' % (
            view_number, float(timeout_s), attempts, reason)

    def run_auto_ten_view_sweep(self, scene_id='', notes='', reverse=False,
                                base_duration_s=0.15, settle_s=0.5,
                                capture_timeout_s=10.0):
        """Drive the predefined wide sweep and record every view unattended.

        十个站位由 `predefined_wide_sweep_poses()` 以目标点为球心给出绝对相机位
        姿，换算成法兰位姿后逐个交给 `plan_observation_pose` 服务规划并执行，每
        到位再调 `_auto_sweep_capture_one()` 拍一张。运动不再走 CockpitJog 速度点
        动：那条路径实测只走出 1.2° 方位角跨度，且没有碰撞校验。

        settle_s 默认 0.5 s：轨迹执行结束后要等 RGB-D 与 joint_states 都刷出停稳
        后的新帧，`nbv_max_frame_age_s` 量级的新鲜度门禁才可能过；0.5 s 覆盖
        30 fps 下十几帧，又不会让十个视角的总耗时显著变长。
        capture_timeout_s 默认 10 s：足够跨过一次语义推理或 IMU 窗口的短暂缺口
        （单次门禁重试间隔 0.5 s，即最多 20 次尝试），又不至于在相机真的掉线时
        让作者干等太久。

        任何一个站位规划失败、执行失败或拍照超时都立即整次失败，并报出第几个视
        角、已记录几个、原因。不允许以残缺批次收尾。
        """
        maximum = int(self.nbv_recorder.max_views)
        if not self.nbv_recorder.enabled:
            message = '论文采集模式未开启，无法自动扫描'
            self.state.set_action_status(message)
            return False, message
        try:
            target_point, target_detail, fallback = self._sweep_target_point()
            self.state.set_action_status(
                '正在自动执行预定义宽幅十视角扫描（共 %d 个视角，半角 %.0f°）；'
                '%s' % (maximum, float(WIDE_SWEEP_HALF_ANGLE_DEG),
                        target_detail))
            if fallback:
                self.state.append_event('[NBV] 宽幅扫描使用' + target_detail)
            flange_to_optical = self._flange_to_optical_pose()
            stations = predefined_wide_sweep_poses(
                target_point, self.sweep_observation_distance_m,
                view_count=maximum, reverse=bool(reverse))
            recorded = 0
            reached_azimuths = []
            for index, station in enumerate(stations):
                pose, _position, _quaternion = self._sweep_flange_pose(
                    station, flange_to_optical)
                ok, detail, _response = self._request_observation_pose(
                    pose, index + 1, execute=True)
                if not ok:
                    return False, (
                        '自动扫描第 %d 个视角规划或执行失败：%s'
                        '（已记录 %d/%d 个视角，本批次作废）' % (
                            index + 1, detail, recorded, maximum))
                reached_azimuths.append(float(station.azimuth_deg))
                time.sleep(max(0.0, float(settle_s)))
                ok, detail = self._auto_sweep_capture_one(
                    index + 1, scene_id, notes, capture_timeout_s)
                if not ok:
                    return False, '自动扫描中止：%s（已记录 %d/%d 个视角）' % (
                        detail, recorded, maximum)
                recorded += 1
            if recorded < maximum:
                return False, (
                    '自动扫描只记录了 %d/%d 个视角；本批次不完整，请重试' % (
                        recorded, maximum))
            self.sweep_last_azimuth_span_deg = azimuth_span_deg(
                reached_azimuths)
            self.sweep_last_target_fallback = bool(fallback)
            message = (
                '预定义宽幅扫描完成：已自动记录 %d/%d 个视角；'
                '实测方位角跨度 %.1f°（标称 %.0f°）' % (
                    recorded, maximum, self.sweep_last_azimuth_span_deg,
                    2.0 * float(WIDE_SWEEP_HALF_ANGLE_DEG)))
            self.state.set_action_status(message)
            self.state.append_event('[NBV] ' + message)
            return True, message
        except Exception as error:
            message = '自动十视角扫描失败：%s' % error
            self.state.set_action_status(message)
            return False, message

    def request_runtime_config(self, operation, values=None):
        labels = {
            'get': '读取运行参数',
            'apply': '临时应用运行参数',
            'save': '应用并保存运行参数',
            'reload': '重新载入运行参数',
            'reset': '恢复项目默认参数',
        }
        action = labels.get(operation, operation)
        self.state.set_action_status('正在%s...' % action)
        try:
            rospy.wait_for_service(self.runtime_config_service, timeout=3.0)
            client = rospy.ServiceProxy(
                self.runtime_config_service, RuntimeConfig)
            response = client(
                operation=str(operation),
                values_json=json.dumps(values or {}, ensure_ascii=False))
            result = {
                'success': bool(response.success),
                'message': str(response.message),
                'values': json.loads(response.values_json or '{}'),
                'schema': json.loads(response.schema_json or '{}'),
                'profile_path': str(response.profile_path),
            }
            if result['success'] and 'tool_timeout_s' in result['values']:
                self.approach_grasp_dwell_s = max(
                    0.01, float(result['values']['tool_timeout_s']))
            prefix = '完成：' if result['success'] else '未应用：'
            self.state.set_action_status(prefix + result['message'])
            return result
        except Exception as error:
            message = '运行参数服务不可用：%s' % error
            self.state.set_action_status(message)
            return {
                'success': False,
                'message': message,
                'values': {},
                'schema': {},
                'profile_path': '',
            }

    # ------------------------------------------------------------------
    # Paper capture mode
    # ------------------------------------------------------------------
    def _publish_nbv_status(self, status):
        status = dict(status or {})
        calibration = read_install_status(self.calibration_config_file)
        camera = {
            'name': self.camera_name,
            'serial': self.camera_serial,
            'device_type': self.camera_device_type,
            'usb_type': self.camera_usb_type,
            'usb_port_id': self.camera_usb_port_id,
        }
        actual_type = str(self.camera_device_type or '').strip().lower()
        type_matches = bool(
            not self.nbv_expected_camera_type or
            (actual_type and self.nbv_expected_camera_type in actual_type) or
            self.nbv_allow_non_expected_camera)
        serial_matches = bool(
            self.camera_serial and calibration.get('camera_serial') and
            self.camera_serial == calibration.get('camera_serial'))
        usb3 = str(self.camera_usb_type or '').strip().startswith('3.')
        status['camera'] = camera
        status['calibration'] = calibration
        state_snapshot = self.state.snapshot()
        if not isinstance(state_snapshot, dict):
            state_snapshot = {}
        imu_latest = state_snapshot.get('imu_latest') or {}
        status['imu'] = {
            'gyro_seen': bool(imu_latest.get('gyro')),
            'accel_seen': bool(imu_latest.get('accel')),
            'required': bool(getattr(self, 'nbv_require_imu', False)),
            'topics': dict(state_snapshot.get('imu_topics') or {}),
        }
        status['readiness'] = {
            'camera_type_matches': type_matches,
            'calibration_serial_matches': serial_matches,
            'usb3': usb3,
            'ready': bool(
                self.camera_serial and type_matches and
                calibration.get('configured', False) and
                calibration.get('quality_passed', False) and
                calibration.get('installed', False) and serial_matches and
                (usb3 or not self.nbv_require_usb3)),
            'semantics_required': bool(self.nbv_require_semantics),
            'instance_ids_required': bool(self.nbv_require_instance_ids),
            'imu_required': bool(getattr(self, 'nbv_require_imu', False)),
        }
        self.nbv_last_status = status
        self.state.set_nbv_status(self.nbv_last_status)

    def nbv_status(self):
        return dict(self.nbv_last_status)

    def refresh_nbv_status(self):
        self._publish_nbv_status(self.nbv_recorder.status())
        return self.nbv_status()

    def _approach_state(self, state, message, result=None):
        summary = self.approach_session.status()
        payload = {
            'state': str(state),
            'message': str(message),
            'computing': state == 'computing',
            'safe': bool((result or {}).get('safe', False)),
            'view_count': int(summary.get('view_count', 0)),
            'minimum_views': int(self.approach_minimum_views),
            'reconstruction_complete': bool(
                summary.get('view_count', 0) >= self.approach_minimum_views),
            'map_summary': summary,
            'reconstructed_targets': list(self.approach_discovered_targets),
            'result': result,
        }
        blank = (np.zeros((360, 720, 3), dtype=np.uint8)
                 if result is None else None)
        self.state.set_approach_status(payload, heatmap=blank)
        return payload

    def _clear_approach_markers(self):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.target_frame
        marker.action = Marker.DELETEALL
        self.approach_marker_pub.publish(MarkerArray(markers=[marker]))

    def reset_approach_map(self, scene_id='', group_number=None):
        summary = self.approach_session.reset(scene_id=scene_id)
        self.approach_discovered_targets = []
        self.approach_plan_cache = None
        self.approach_group_number = (
            None if group_number is None else int(group_number))
        self._clear_approach_markers()
        self._approach_state(
            'empty', '球面通道图已清空；等待有效语义视角', result=None)
        return summary

    def _insert_approach_snapshot(self, snapshot, group_number):
        group_number = int(group_number)
        if self.approach_group_number != group_number:
            self.reset_approach_map(
                scene_id=self.nbv_recorder.scene_id,
                group_number=group_number)
        status = self.approach_session.insert_snapshot(snapshot)
        self.approach_plan_cache = None
        self.approach_discovered_targets = \
            self.approach_session.discover_targets(self.approach_config)
        count = int(status.get('view_count', 0))
        complete = count >= self.approach_minimum_views
        message = ('语义体素图已完成 %d 个视角，可计算正式候选方向' % count
                   if complete else
                   '语义体素图已融合 %d/%d 个视角；当前结果仅作中途诊断' %
                   (count, self.approach_minimum_views))
        self._approach_state('ready' if complete else 'mapping', message)
        # Keep the semantic octree visible throughout the ten-view capture.
        # There is no selected fruit yet, so all reconstructed citrus voxels
        # use the target colour until the operator chooses one for planning.
        self._publish_approach_markers({'provisional_semantic_map': True})
        self.state.append_event('[伸入方向] ' + message)
        return status

    def _rebuild_approach_from_staging(
            self, group_number, staging=None, records=None, scene_id=None):
        records = (list(self.nbv_recorder.records or ())
                   if records is None else list(records or ()))
        staging = self.nbv_recorder.staging_dir if staging is None else str(staging)
        scene_id = (self.nbv_recorder.scene_id
                    if scene_id is None else str(scene_id or ''))
        rebuild_started = time.monotonic()
        view_timings = []
        self.reset_approach_map(
            scene_id=scene_id,
            group_number=group_number)
        if not records or not staging:
            return self.approach_session.status()
        for record in records:
            view_started = time.monotonic()
            files = dict(record.get('files') or {})
            depth_path = os.path.join(staging, files.get('depth_mm', ''))
            label_path = os.path.join(staging, files.get(
                'semantic_labels', ''))
            confidence_path = os.path.join(staging, files.get(
                'semantic_confidence', ''))
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            labels = cv2.imread(label_path, cv2.IMREAD_UNCHANGED) \
                if os.path.isfile(label_path) else None
            confidence = np.load(confidence_path, allow_pickle=False) \
                if os.path.isfile(confidence_path) else None
            if depth is None or labels is None:
                raise ValueError(
                    'staging 视角 %s 缺少深度或语义标签' %
                    record.get('index', '?'))
            snapshot = {
                'depth': depth,
                'semantic_labels': labels,
                'semantic_confidence': confidence,
                'pose_matrix': record.get('pose_matrix'),
                'camera_info': record.get('camera_info') or {},
                'stamp_sec': record.get('stamp_sec', 0.0),
            }
            self.approach_session.insert_snapshot(snapshot)
            view_timings.append({
                'view': int(record.get('index', len(view_timings) + 1)),
                'seconds': time.monotonic() - view_started,
            })
        self.approach_discovered_targets = \
            self.approach_session.discover_targets(self.approach_config)
        count = self.approach_session.status().get('view_count', 0)
        elapsed = time.monotonic() - rebuild_started
        message = ('已从保留的 staging 恢复 %d 个语义体素视角（%.2f s）' %
                   (count, elapsed))
        self._approach_state(
            'ready' if count >= self.approach_minimum_views else 'mapping',
            message)
        self._publish_approach_markers({'provisional_semantic_map': True})
        self.state.append_event('[伸入方向] ' + message)
        rospy.loginfo(
            '[APPROACH_TIMING] map_rebuild group=%s total=%.3fs views=%s source=%s',
            group_number, elapsed,
            json.dumps(view_timings, ensure_ascii=False, sort_keys=True),
            staging)
        return self.approach_session.status()

    def _latest_complete_staging(self):
        candidates = []
        if not os.path.isdir(self.nbv_output_dir):
            return None
        try:
            entries = list(os.scandir(self.nbv_output_dir))
        except OSError:
            return None
        for entry in entries:
            if not entry.is_dir() or not entry.name.startswith('.nbv_staging_g'):
                continue
            manifest_path = os.path.join(entry.path, 'manifest.json')
            try:
                with open(manifest_path, 'r', encoding='utf-8') as stream:
                    manifest = json.load(stream)
                records = list(manifest.get('views') or ())
                count = int(manifest.get('view_count', len(records)) or 0)
                group = int(manifest.get('group_number', 0) or 0)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if not manifest.get('complete') or \
                    count < self.approach_minimum_views or \
                    len(records) < self.approach_minimum_views:
                continue
            candidates.append({
                'group_number': group,
                'staging': entry.path,
                'manifest_path': manifest_path,
                'manifest': manifest,
                'records': records,
                'modified_at': float(os.path.getmtime(manifest_path)),
            })
        return max(candidates, key=lambda item: (
            item['group_number'], item['modified_at'])) if candidates else None

    def load_latest_complete_approach_map(self):
        if not self.approach_compute_lock.acquire(False):
            return False, '球面方向计算或执行正在进行，暂不能切换语义地图'
        try:
            latest = self._latest_complete_staging()
            if latest is None:
                raise ValueError('没有找到包含完整十视角 manifest.json 的 staging')
            manifest = latest['manifest']
            status = self._rebuild_approach_from_staging(
                latest['group_number'], staging=latest['staging'],
                records=latest['records'],
                scene_id=manifest.get('scene_id', ''))
            message = (
                '已载入最近完整十视角：g%04d，%d 个视角，%d 个占据体素；无需重新拍摄' %
                (latest['group_number'], int(status.get('view_count', 0)),
                 int(status.get('occupied_voxel_count', 0))))
            self.state.set_action_status(message)
            self.state.append_event('[伸入方向] ' + message)
            return True, message
        except Exception as error:
            message = '载入最近十视角失败：%s' % error
            self.state.set_action_status(message)
            self.state.append_event('[伸入方向] ' + message)
            return False, message
        finally:
            self.approach_compute_lock.release()

    @staticmethod
    def _marker_point(values):
        point = Point()
        point.x, point.y, point.z = (float(value) for value in values)
        return point

    def _semantic_voxel_marker(self, namespace, marker_id, points, resolution,
                               colour):
        """Build one bounded CUBE_LIST for the reconstructed semantic map."""
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.target_frame
        marker.ns = str(namespace)
        marker.id = int(marker_id)
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        side = max(0.001, float(resolution) * 0.90)
        marker.scale.x = marker.scale.y = marker.scale.z = side
        marker.color = colour
        marker.points = [self._marker_point(point) for point in points]
        marker.lifetime = rospy.Duration(0)
        return marker

    def _semantic_voxel_marker_points(self, result):
        """Return target/other-citrus/scene voxel centers for RViz.

        The reconstructed fruit is shown as its actual occupied semantic
        voxels.  A deterministic cap protects RViz from a large ten-view map;
        the selected fruit is kept intact before the scene is downsampled.
        """
        result = result or {}
        target_keys = set(tuple(int(value) for value in key) for key in
                          (result.get('_target_voxel_keys') or ()))
        provisional = bool(result.get('provisional_semantic_map', False))
        with self.approach_session.lock:
            voxel_map = self.approach_session.voxel_map
            fruit_ids = set(int(value) for value in
                            self.approach_session.fruit_ids)
            target_points = []
            other_citrus_points = []
            scene_points = []
            include_unknown = bool(self.approach_config.get(
                'visualization_include_unknown', True))
            unknown_points = []
            for key, state in voxel_map.voxels.items():
                if not voxel_map.is_occupied(key):
                    continue
                point = voxel_map.point_from_key(key)
                if not state.get('semantic_set'):
                    if include_unknown:
                        unknown_points.append(point)
                    continue
                label = int(state.get('semantic_label'))
                if label in fruit_ids:
                    (target_points if provisional or tuple(key) in target_keys else
                     other_citrus_points).append(point)
                else:
                    scene_points.append(point)
            resolution = float(voxel_map.resolution)

        cap = max(1000, int(self.approach_config.get(
            'visualization_max_voxels', 12000)))

        def bounded(points, limit):
            if len(points) <= limit:
                return points
            indices = np.linspace(0, len(points) - 1, int(limit),
                                  dtype=np.int64)
            return [points[int(index)] for index in np.unique(indices)]

        # Never discard target voxels unless a pathological false-positive map
        # itself exceeds the cap. Reserve a small visible slice for unknown
        # occupancy before the much larger scene class consumes the budget.
        target_points = bounded(target_points, cap)
        remaining = max(0, cap - len(target_points))
        other_citrus_points = bounded(other_citrus_points, remaining)
        remaining -= len(other_citrus_points)
        unknown_reserve = (min(len(unknown_points), max(1, cap // 10),
                               remaining)
                           if include_unknown and unknown_points else 0)
        scene_points = bounded(scene_points,
                               max(0, remaining - unknown_reserve))
        remaining -= len(scene_points)
        unknown_points = bounded(unknown_points, remaining)
        return (target_points, other_citrus_points, scene_points,
                unknown_points, resolution)

    def _publish_approach_markers(self, result):
        result = result or {}
        markers = []
        delete = Marker()
        delete.header.stamp = rospy.Time.now()
        delete.header.frame_id = self.target_frame
        delete.action = Marker.DELETEALL
        markers.append(delete)
        target = result.get('target_center') or (0.0, 0.0, 0.0)

        (target_voxels, other_citrus_voxels, scene_voxels,
         unknown_voxels, resolution) = self._semantic_voxel_marker_points(result)
        if target_voxels:
            markers.append(self._semantic_voxel_marker(
                'semantic_octree_target', 10, target_voxels, resolution,
                ColorRGBA(r=1.0, g=0.18, b=0.02, a=0.95)))
        if other_citrus_voxels:
            markers.append(self._semantic_voxel_marker(
                'semantic_octree_citrus', 11, other_citrus_voxels, resolution,
                ColorRGBA(r=1.0, g=0.52, b=0.02, a=0.82)))
        if scene_voxels:
            markers.append(self._semantic_voxel_marker(
                'semantic_octree_scene', 12, scene_voxels, resolution,
                ColorRGBA(r=0.12, g=0.62, b=0.28, a=0.50)))
        if unknown_voxels:
            markers.append(self._semantic_voxel_marker(
                'semantic_octree_unknown', 13, unknown_voxels, resolution,
                ColorRGBA(r=0.48, g=0.48, b=0.48, a=0.16)))

        best = result.get('best') or {}
        if best.get('preentry_point'):
            arrow = Marker()
            arrow.header.stamp = rospy.Time.now()
            arrow.header.frame_id = self.target_frame
            arrow.ns = 'approach_direction'
            arrow.id = 2
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            # Arrow direction is the actual outside-to-fruit insertion vector.
            arrow.points = [self._marker_point(best['preentry_point']),
                            self._marker_point(target)]
            arrow.scale.x = 0.014
            arrow.scale.y = 0.030
            arrow.scale.z = 0.045
            arrow.color = (ColorRGBA(r=0.15, g=0.90, b=0.25, a=0.95)
                           if result.get('safe') else
                           ColorRGBA(r=0.95, g=0.15, b=0.10, a=0.95))
            arrow.lifetime = rospy.Duration(0)
            markers.append(arrow)

        arrays = (result.get('_arrays') or {})
        directions = np.asarray(arrays.get('directions', []), dtype=np.float64)
        scores = np.asarray(arrays.get('local_score', []), dtype=np.float64)
        eligible = np.asarray(arrays.get('eligible', []), dtype=bool)
        if len(directions) and len(scores) == len(directions):
            screen = Marker()
            screen.header.stamp = rospy.Time.now()
            screen.header.frame_id = self.target_frame
            screen.ns = 'approach_direction'
            screen.id = 3
            screen.type = Marker.POINTS
            screen.action = Marker.ADD
            screen.scale.x = screen.scale.y = 0.025
            # The displayed screen uses the same radius as the calculation,
            # which is intentionally larger than the 1.5 m model tree.
            display_radius = float((result.get('config') or {}).get(
                'screen_radius_m', self.approach_config.get(
                    'screen_radius_m', 1.60)))
            maximum = max(1e-9, float(np.max(scores)))
            for index in range(0, len(directions), 2):
                location = np.asarray(target) + directions[index] * display_radius
                screen.points.append(self._marker_point(location))
                value = float(np.clip(scores[index] / maximum, 0.0, 1.0))
                screen.colors.append(ColorRGBA(
                    r=value, g=(0.85 if eligible[index] else 0.25) * value,
                    b=1.0 - value, a=0.80))
            screen.lifetime = rospy.Duration(0)
            markers.append(screen)
        self.approach_marker_pub.publish(MarkerArray(markers=markers))

    def _validate_approach_with_moveit(self, result, target_point,
                                       target_frame,
                                       use_fitted_surface=False):
        result['moveit_validation_attempted'] = False
        result['moveit_validation_available'] = False
        result['moveit_validated'] = False
        result['moveit_failures'] = []
        try:
            rospy.wait_for_service(self.approach_plan_service, timeout=0.75)
        except (rospy.ROSException, rospy.ROSInterruptException) as error:
            result['moveit_message'] = 'MoveIt 方向服务不可用：%s' % error
            return result
        result['moveit_validation_available'] = True
        result['moveit_validation_attempted'] = True
        client = rospy.ServiceProxy(
            self.approach_plan_service, PlanApproachDirection)
        candidates = list(result.get('candidates') or ())
        if not candidates and result.get('best'):
            candidates = [dict(result['best'])]
        # Preserve the pure geometry winner for the heatmap and paper audit;
        # ``best`` will become the fastest MoveIt-feasible candidate below.
        geometry_best = dict(result.get('best') or {})
        result['geometry_best'] = dict(geometry_best)
        validated = []
        for candidate_index, candidate in enumerate(candidates):
            candidate = dict(candidate)
            candidates[candidate_index] = candidate
            candidate_started = time.monotonic()
            outward_values = candidate.get('outward_direction') or ()
            if len(outward_values) != 3:
                candidate['display_label'] = 'REJECTED: BAD DIRECTION'
                continue
            outward = Vector3()
            outward.x, outward.y, outward.z = (
                float(value) for value in outward_values)
            plan_target = tuple(float(value) for value in target_point)
            if use_fitted_surface:
                center = result.get('target_center') or plan_target
                radius = float(result.get('target_radius_m', 0.0))
                plan_target = tuple(
                    float(center[axis]) + radius * float(outward_values[axis])
                    for axis in range(3))
            point = Point()
            point.x, point.y, point.z = plan_target
            try:
                response = client(
                    target_point=point,
                    target_frame=str(target_frame or self.target_frame),
                    outward_direction=outward,
                    execute=False,
                    command='plan',
                    plan_id='')
                try:
                    parsed_timings = json.loads(
                        str(getattr(response, 'timings_json', '') or '{}'))
                except (TypeError, ValueError):
                    parsed_timings = {
                        'unparsed': str(getattr(
                            response, 'timings_json', '') or '')}
                candidate['moveit_timings'] = parsed_timings
                if response.success:
                    candidate['moveit_message'] = str(response.message)
                    candidate['moveit_validated'] = True
                    candidate['moveit_feasible'] = True
                    candidate['moveit_candidate_rank'] = candidate_index + 1
                    candidate['endpoint_condition_number'] = float(
                        response.endpoint_condition_number)
                    candidate['moveit_target_point'] = list(plan_target)
                    candidate['moveit_plan_id'] = str(response.plan_id)
                    # The physical-probe trial records the pose the flange
                    # actually reaches before the straight insertion begins.
                    # An older planner without the pose fields must still
                    # yield a feasible candidate, so the lookup is tolerant.
                    candidate['moveit_entry_pose'] = self._pose_dict(
                        getattr(response, 'pregrasp_pose', None))
                    candidate['moveit_final_pose'] = self._pose_dict(
                        getattr(response, 'final_pose', None))
                    try:
                        candidate['moveit_planning_duration_s'] = float(
                            response.planning_duration_s)
                    except (AttributeError, TypeError, ValueError):
                        candidate['moveit_planning_duration_s'] = 0.0
                    # New planners expose these at the top level.  The roll
                    # record fallback keeps compatibility with older nodes.
                    timing_source = parsed_timings
                    rolls = parsed_timings.get('roll_candidates', []) \
                        if isinstance(parsed_timings, dict) else []
                    selected_strategy = parsed_timings.get(
                        'selected_strategy') if isinstance(
                            parsed_timings, dict) else None
                    for roll in rolls:
                        if not isinstance(roll, dict) or not roll.get('success'):
                            continue
                        if selected_strategy is None or str(roll.get(
                                'strategy')) == str(selected_strategy):
                            timing_source = dict(parsed_timings)
                            timing_source.update(roll)
                            break
                    distance = timing_source.get('entry_distance_m')
                    eta = timing_source.get('max_speed_entry_eta_s')
                    # Joint-space displacement of the entry leg travels with
                    # the candidate so the trial row does not have to re-parse
                    # the nested roll records later.
                    for field in ('joint_delta_l2_rad', 'joint_delta_inf_rad',
                                  'joint_delta_weighted'):
                        candidate[field] = timing_source.get(field)
                    try:
                        distance = float(distance)
                    except (TypeError, ValueError):
                        distance = float('inf')
                    try:
                        eta = float(eta)
                    except (TypeError, ValueError):
                        eta = float('inf')
                    if not math.isfinite(distance) or distance < 0.0:
                        distance = float('inf')
                    if not math.isfinite(eta) or eta < 0.0:
                        eta = float('inf')
                    candidate['distance_m'] = (
                        distance if math.isfinite(distance) else None)
                    candidate['distance_score'] = (
                        math.exp(-distance) if math.isfinite(distance) else 0.0)
                    candidate['distance_score_exp_neg_d'] = candidate[
                        'distance_score']
                    candidate['max_speed_eta_s'] = (
                        eta if math.isfinite(eta) else None)
                    candidate['eta_available'] = math.isfinite(eta)
                    candidate['display_label'] = 'FEASIBLE: PENDING'
                    candidate['moveit_validation_duration_s'] = (
                        time.monotonic() - candidate_started)
                    validated.append(candidate)
                    continue
                result['moveit_failures'].append({
                    'candidate_rank': candidate_index + 1,
                    'message': str(response.message),
                    'duration_s': time.monotonic() - candidate_started,
                })
                candidate['moveit_feasible'] = False
                candidate['moveit_validated'] = False
                candidate['moveit_message'] = str(response.message)
                candidate['display_label'] = 'INFEASIBLE: MOVEIT'
            except Exception as error:
                result['moveit_failures'].append({
                    'candidate_rank': candidate_index + 1,
                    'message': str(error),
                    'duration_s': time.monotonic() - candidate_started,
                })
                candidate['moveit_feasible'] = False
                candidate['moveit_validated'] = False
                candidate['moveit_message'] = str(error)
                candidate['display_label'] = 'INFEASIBLE: MOVEIT'

        result['candidates'] = candidates
        if validated:
            def finite_metric(item, name, fallback=float('inf')):
                try:
                    value = float(item.get(name))
                except (TypeError, ValueError):
                    return fallback
                return value if math.isfinite(value) else fallback

            # ETA is the final timestamp of the entry trajectory after a
            # 1.0/1.0 MoveIt retime, not the wall-clock planning-call time.
            selected = min(
                validated,
                key=lambda item: (
                    finite_metric(item, 'max_speed_eta_s'),
                    -finite_metric(item, 'distance_score', 0.0),
                    -finite_metric(item, 'utility', 0.0),
                    int(item.get('moveit_candidate_rank', 0))))
            selected_id = str(selected.get('moveit_plan_id', ''))
            has_eta = any(bool(item.get('eta_available')) for item in validated)
            for candidate in validated:
                if str(candidate.get('moveit_plan_id', '')) == selected_id:
                    candidate['display_label'] = 'SELECTED: FASTEST' \
                        if has_eta else 'SELECTED: FEASIBLE'
                elif finite_metric(candidate, 'max_speed_eta_s') < float('inf'):
                    selected_distance = finite_metric(selected, 'distance_m')
                    candidate_distance = finite_metric(candidate, 'distance_m')
                    candidate['display_label'] = (
                        'FEASIBLE: FARTHER' if candidate_distance >
                        selected_distance + 1e-3 else 'FEASIBLE: SLOWER')
                else:
                    candidate['display_label'] = 'FEASIBLE: ETA UNKNOWN'
            result['best'] = dict(selected)
            result['selected_candidate'] = dict(selected)
            result['moveit_validated'] = True
            result['moveit_candidate_rank'] = int(
                selected.get('moveit_candidate_rank', 0))
            result['moveit_endpoint_condition_number'] = float(
                selected.get('endpoint_condition_number', float('inf')))
            result['moveit_message'] = (
                'MoveIt 可行候选 %d/%d；按最高速入口 ETA 选择 #%d'
                % (len(validated), len(candidates),
                   result['moveit_candidate_rank']))
            result['moveit_plan_id'] = str(selected.get('moveit_plan_id', ''))
            result['moveit_planning_duration_s'] = float(
                selected.get('moveit_planning_duration_s', 0.0))
            # The response timing is retained on the selected candidate; this
            # summary field is deliberately the selected service-call time.
            result['moveit_timings'] = selected.get('moveit_timings') or {}
            result['moveit_candidate_duration_s'] = sum(
                float(item.get('moveit_validation_duration_s', 0.0))
                for item in candidates)
            result['moveit_target_source'] = (
                'fitted_entry_surface' if use_fitted_surface
                else 'live_rgbd_surface')
            result['moveit_selection_metric'] = 'max_speed_entry_eta_s'
            result['moveit_entry_pose'] = dict(
                selected.get('moveit_entry_pose') or {})
            result['moveit_final_pose'] = dict(
                selected.get('moveit_final_pose') or {})
            result['moveit_selected_eta_s'] = selected.get('max_speed_eta_s')
            result['moveit_selected_distance_m'] = selected.get('distance_m')
            result['moveit_selected_distance_score'] = selected.get(
                'distance_score')
            if has_eta:
                try:
                    display_response = self._select_cached_approach_plan(
                        result['moveit_plan_id'])
                    result['moveit_selected_display_refreshed'] = bool(
                        display_response.success)
                except Exception as error:
                    # A display refresh is useful but must not turn a valid
                    # plan-only result into a second planning failure.
                    result['moveit_selected_display_refreshed'] = False
                    result['moveit_message'] += '；RViz缓存显示刷新失败：%s' % error
            else:
                result['moveit_selected_display_refreshed'] = False
            return result
        details = '；'.join(
            '#%d %s' % (int(item.get('candidate_rank', 0)),
                         item.get('message', '未知原因'))
            for item in result['moveit_failures'])
        result['moveit_message'] = (
            'MoveIt 已拒绝全部 %d 个几何候选%s' %
            (len(candidates), ('：' + details) if details else ''))
        return result

    def _call_cached_approach_stage(self, plan_id, command):
        """Execute one cached MoveIt stage without sending geometry again."""
        try:
            rospy.wait_for_service(self.approach_plan_service, timeout=0.75)
            client = rospy.ServiceProxy(
                self.approach_plan_service, PlanApproachDirection)
            response = client(
                target_point=Point(), target_frame='',
                outward_direction=Vector3(), execute=True,
                command=str(command), plan_id=str(plan_id))
            return response
        except (rospy.ServiceException, rospy.ROSException) as error:
            raise RuntimeError('MoveIt 缓存方向执行服务失败：%s' % error)

    def _select_cached_approach_plan(self, plan_id):
        """Make a previously planned candidate the RViz/display plan."""
        try:
            rospy.wait_for_service(self.approach_plan_service, timeout=0.75)
            client = rospy.ServiceProxy(
                self.approach_plan_service, PlanApproachDirection)
            response = client(
                target_point=Point(), target_frame='',
                outward_direction=Vector3(), execute=False,
                command='select', plan_id=str(plan_id))
            if not response.success:
                raise RuntimeError(str(response.message))
            return response
        except (rospy.ServiceException, rospy.ROSException) as error:
            raise RuntimeError('MoveIt 缓存方向显示选择失败：%s' % error)

    def _execute_approach_with_moveit(self, result, _target_point=None,
                                      _target_frame='',
                                      use_fitted_surface=False):
        """Compatibility wrapper: execute the cached outbound stage only."""
        del use_fitted_surface
        result['execution_requested'] = True
        result['execution_attempted'] = False
        result['executed'] = False
        plan_id = str(result.get('moveit_plan_id', '') or '')
        if not plan_id:
            result['moveit_message'] = '当前结果没有缓存 plan_id；请先只规划'
            return result
        try:
            response = self._call_cached_approach_stage(
                plan_id, 'execute_outbound')
            result['execution_attempted'] = bool(response.execution_attempted)
            result['executed'] = bool(response.executed)
            result['moveit_message'] = str(response.message)
        except RuntimeError as error:
            result['moveit_message'] = str(error)
        return result

    def _persist_approach_result(self, result, phase):
        """Atomically persist a public planning/execution result for audit."""
        try:
            os.makedirs(self.approach_result_dir, exist_ok=True)
            plan_id = str((result or {}).get('moveit_plan_id', 'unplanned'))
            safe_plan_id = ''.join(
                character if character.isalnum() or character in '-_' else '_'
                for character in plan_id) or 'unplanned'
            stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            path = os.path.join(
                self.approach_result_dir,
                'approach_%s_%s_%s.json' % (stamp, safe_plan_id, phase))
            payload = dict(result or {})
            payload['result_phase'] = str(phase)
            payload['result_json_path'] = path
            payload['saved_at'] = datetime.now().isoformat(timespec='milliseconds')
            temporary = path + '.tmp.%d' % os.getpid()
            with open(temporary, 'w', encoding='utf-8') as stream:
                json.dump(payload, stream, ensure_ascii=False, sort_keys=True,
                          indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            result['result_json_path'] = path
            return path
        except Exception as error:
            rospy.logwarn('Cannot persist approach result JSON: %s', error)
            return ''

    def _straight_line_candidate(self, result, robot_position, blind=True):
        """Replace the scored candidate set with the flange-to-fruit ray.

        M3 in the probe experiment ignores the transmission field entirely, so
        the only direction offered to MoveIt is the one pointing from the fruit
        centre back at the current flange origin.  The nearest sampled
        direction is used so the candidate carries exactly the same measured
        fields as a scored one; at 1024 Fibonacci samples that snap is under
        four degrees, and the snapped value is what the CSV reports.
        """
        arrays = result.get('_arrays') or {}
        directions = np.asarray(arrays.get('directions', ()), dtype=np.float64)
        center = np.asarray(result.get('target_center') or (), dtype=np.float64)
        if robot_position is None or directions.size == 0 or center.size != 3:
            raise ValueError(
                '直线基线需要当前法兰原点和已建立的球面采样；请确认 TF 可用')
        offset = np.asarray(robot_position, dtype=np.float64).reshape(3) - center
        norm = float(np.linalg.norm(offset))
        if not math.isfinite(norm) or norm <= 1e-6:
            raise ValueError('法兰原点与目标中心重合，无法定义直线基线')
        outward = offset / norm
        index = int(np.argmax(directions.dot(outward)))
        candidate = serial_approach_candidate(
            index, directions[index], center, result['config'], arrays)
        # 盲评要求屏幕上不出现方法身份。候选表和球面热图都渲染 display_label，
        # 而操作者必须看着热图选目标，所以这里用与 M1/M2 几何优胜者完全相同的
        # 标签；基线身份只存在于 CSV 和结果 JSON 里。
        candidate['display_label'] = (
            'GEOMETRY BEST' if blind else 'BASELINE CANDIDATE')
        candidate['straight_line_baseline'] = True
        candidate['straight_line_snap_deg'] = math.degrees(math.acos(
            max(-1.0, min(1.0, float(directions[index].dot(outward))))))
        result['candidates'] = [candidate]
        result['best'] = dict(candidate)
        result['geometry_best'] = dict(candidate)
        result['geometry_best_sample_index'] = index
        # The sphere scorer's patch verdict does not gate this baseline: the
        # direction is fixed by the flange, and MoveIt remains the only judge.
        result['safe'] = True
        result['reason'] = 'straight line from the current flange origin'
        return result

    def calculate_approach_direction(self, target_index, target_point,
                                     target_frame='', target_label='',
                                     execute=False, probe_method=None):
        if execute:
            # Compatibility callers now consume the last plan instead of
            # silently repeating sphere scoring and MoveIt planning.
            return self.execute_cached_approach()
        if not self.approach_compute_lock.acquire(False):
            return False, '上一项球面通道计算仍在进行'
        total_started = time.monotonic()
        try:
            self.approach_plan_cache = None
            handoff_started = time.monotonic()
            released, release_message = self.release_cockpit_for_moveit()
            if not released:
                raise ValueError(release_message)
            handoff_duration = time.monotonic() - handoff_started
            map_status = self.approach_session.status()
            view_count = int(map_status.get('view_count', 0))
            if view_count <= 0:
                raise ValueError('尚未融合任何有效语义视角')
            if str(target_frame or self.target_frame).lstrip('/') != \
                    self.target_frame.lstrip('/'):
                raise ValueError('目标坐标系 %s 与重建坐标系 %s 不一致' %
                                 (target_frame, self.target_frame))
            self._approach_state(
                'computing', '正在计算目标 %d 的球面可见性与软树冠空隙...' %
                int(target_index))
            self._clear_approach_markers()
            target_point = tuple(float(value) for value in target_point)
            preferred = (-target_point[0], -target_point[1], 0.0)
            geometry_started = time.monotonic()
            robot_position = self._lookup_flange_origin()
            # The probe experiment's three methods differ only here, so every
            # later stage (MoveIt validation, caching, persistence) is shared.
            plan_config = dict(self.approach_config)
            method = probe_trials.method_spec(probe_method)
            if method is not None:
                plan_config['robot_proximity_weight'] = float(
                    method['proximity_weight'])
            result = self.approach_session.plan(
                target_point, preferred_outward=preferred,
                config=plan_config,
                robot_position=robot_position)
            if method is not None and method['straight_line']:
                result = self._straight_line_candidate(
                    result, robot_position, blind=probe_method is not None)
            result['probe_method'] = (
                None if method is None else str(method['name']))
            result['robot_proximity_weight_used'] = float(
                plan_config['robot_proximity_weight'])
            result['robot_position'] = (
                None if robot_position is None else list(robot_position))
            geometry_duration = time.monotonic() - geometry_started
            result['target_index'] = int(target_index)
            result['target_label'] = str(target_label or '')
            result['target_source'] = (
                'reconstructed_semantic_cluster'
                if str(target_label or '').startswith('reconstructed_citrus_')
                else 'live_rgbd_detection')
            result['target_frame'] = self.target_frame
            result['reconstruction_view_count'] = view_count
            result['reconstruction_complete'] = bool(
                view_count >= self.approach_minimum_views)
            result['geometry_candidate_ready'] = bool(
                result.get('safe') and result['reconstruction_complete'])
            if result['geometry_candidate_ready']:
                moveit_started = time.monotonic()
                result = self._validate_approach_with_moveit(
                    result, target_point, target_frame,
                    use_fitted_surface=(
                        result['target_source'] ==
                        'reconstructed_semantic_cluster'))
                result.setdefault('timings_s', {})[
                    'moveit_candidate_search_s'] = (
                        time.monotonic() - moveit_started)
            result['direction_candidate_ready'] = bool(
                result.get('geometry_candidate_ready') and
                result.get('moveit_validated'))
            result['execution_requested'] = False
            result['execution_attempted'] = False
            result['executed'] = False
            result['actionable'] = bool(result['direction_candidate_ready'])
            timings = result.setdefault('timings_s', {})
            timings['control_handoff_s'] = handoff_duration
            timings.setdefault('total_geometry_s', geometry_duration)
            timings['dashboard_geometry_call_s'] = geometry_duration
            timings['total_plan_pipeline_s'] = time.monotonic() - total_started
            serial = public_approach_result(result)
            if result.get('direction_candidate_ready'):
                self.approach_plan_cache = {
                    'plan_id': str(result.get('moveit_plan_id', '')),
                    'result': json.loads(json.dumps(
                        serial, ensure_ascii=False)),
                    'target_index': int(target_index),
                    'target_point': list(target_point),
                    'target_frame': str(target_frame or self.target_frame),
                    'target_label': str(target_label or ''),
                    'group_number': self.approach_group_number,
                    'view_count': view_count,
                    'created_at': time.monotonic(),
                }
            render_started = time.monotonic()
            heatmap = render_spherical_heatmap(result)
            timings['heatmap_render_s'] = time.monotonic() - render_started
            if result.get('direction_candidate_ready') and probe_method:
                # 候选个数与所选序号是盲评旁证：直线基线恒为 1/1 第 1 个。试验期
                # 间只说轨迹已就绪，计数照旧进结果 JSON。
                message = ('候选已通过 MoveIt 只规划，plan_id=%s；'
                           '执行按钮直接复用缓存，不再计算' %
                           result.get('moveit_plan_id', '<missing>'))
                state = 'plan_validated'
            elif result.get('direction_candidate_ready'):
                feasible_count = sum(
                    1 for item in (result.get('candidates') or ())
                    if item.get('moveit_feasible'))
                message = ('%d/%d 个几何洞口通过 MoveIt 只规划；'
                           '按最高速入口 ETA 选择第 %d 个，plan_id=%s；'
                           '执行按钮直接复用缓存，不再计算' %
                           (feasible_count,
                            len(result.get('candidates') or ()),
                            int(result.get('moveit_candidate_rank', 1)),
                            result.get('moveit_plan_id', '<missing>')))
                state = 'plan_validated'
            elif result.get('geometry_candidate_ready'):
                message = ('可见连续空隙满足几何条件，但 %s；'
                           '拒绝形成机械臂规划候选' %
                           result.get('moveit_message', 'MoveIt 尚未验证'))
                state = 'moveit_rejected'
            elif result.get('safe'):
                message = ('已找到几何亮斑，但重建只有 %d/%d 个视角；'
                           '当前方向仅作中途诊断' %
                           (view_count, self.approach_minimum_views))
                state = 'provisional'
            else:
                message = ('未找到达到宽松局部光照、连续面积和最低观测证据的方向；'
                           '暂不生成可执行候选')
                state = 'rejected'
            heatmap_message = self.bridge.cv2_to_imgmsg(
                heatmap, encoding='bgr8')
            heatmap_message.header.stamp = rospy.Time.now()
            heatmap_message.header.frame_id = self.target_frame
            self.approach_heatmap_pub.publish(heatmap_message)
            marker_started = time.monotonic()
            self._publish_approach_markers(result)
            timings['rviz_marker_publish_s'] = time.monotonic() - marker_started
            timings['total_plan_pipeline_s'] = time.monotonic() - total_started
            serial['timings_s'] = dict(timings)
            result_path = self._persist_approach_result(serial, 'planned')
            if self.approach_plan_cache is not None:
                self.approach_plan_cache['result'] = json.loads(json.dumps(
                    serial, ensure_ascii=False))
            status = {
                'state': state,
                'message': message,
                'computing': False,
                'safe': bool(result.get('safe')),
                'view_count': view_count,
                'minimum_views': int(self.approach_minimum_views),
                'reconstruction_complete': result['reconstruction_complete'],
                'map_summary': map_status,
                'reconstructed_targets': list(
                    self.approach_discovered_targets),
                'result': serial,
            }
            self.state.set_approach_status(status, heatmap=heatmap)
            self.approach_result_pub.publish(String(
                data=json.dumps(serial, ensure_ascii=False, sort_keys=True)))
            rospy.loginfo(
                '[APPROACH_TIMING] plan_id=%s stages=%s',
                result.get('moveit_plan_id', ''),
                json.dumps(timings, ensure_ascii=False, sort_keys=True))
            self.state.set_action_status('[伸入方向] ' + message)
            self.state.append_event('[伸入方向] ' + message)
            return True, message
        except Exception as error:
            message = '球面通道计算拒绝：%s' % error
            self._approach_state('error', message)
            self._clear_approach_markers()
            self.approach_result_pub.publish(String(data=json.dumps({
                'success': False,
                'safe': False,
                'actionable': False,
                'message': message,
            }, ensure_ascii=False, sort_keys=True)))
            self.state.set_action_status('[伸入方向] ' + message)
            self.state.append_event('[伸入方向] ' + message)
            return False, message
        finally:
            self.approach_compute_lock.release()

    def _wait_for_tool_completion(self, baseline_version, action_label):
        deadline = time.monotonic() + self.approach_tool_timeout_s
        last_message = ''
        while time.monotonic() < deadline and not rospy.is_shutdown():
            snapshot = self.state.snapshot()
            version = int(snapshot.get('harvest_status_version', 0))
            status = snapshot.get('harvest_status')
            if status is not None:
                last_message = str(getattr(status, 'message', '') or '')
                phase = str(getattr(status, 'phase', '') or '')
                if version > baseline_version and phase == 'COCKPIT_TOOL_FAILED':
                    raise RuntimeError('%s失败：%s' % (action_label, last_message))
                if version > baseline_version and \
                        not bool(getattr(status, 'tool_active', False)) and \
                        phase == 'COCKPIT_TOOL_DONE':
                    return last_message
            time.sleep(0.02)
        raise RuntimeError('%s在 %.1f s 内没有完成状态%s' % (
            action_label, self.approach_tool_timeout_s,
            ('；最后状态：' + last_message) if last_message else ''))

    def _run_cached_tool_action(self, command, action_label):
        baseline = int(self.state.snapshot().get(
            'harvest_status_version', 0))
        accepted, message = self.request_harvest_command(command, -1)
        if not accepted:
            raise RuntimeError('%s被拒绝：%s' % (action_label, message))
        return self._wait_for_tool_completion(baseline, action_label)

    @staticmethod
    def _response_timings(response):
        try:
            return json.loads(str(getattr(response, 'timings_json', '') or '{}'))
        except (TypeError, ValueError):
            return {'unparsed': str(getattr(response, 'timings_json', '') or '')}

    @staticmethod
    def _response_execution_duration_s(response):
        """Return the planner-measured motion wall clock for one stage.

        The service already times its own controller calls.  Recording it here
        keeps the paper's measured execution time out of the dashboard's
        transport and gate-check overhead.
        """
        try:
            value = float(getattr(response, 'execution_duration_s', 0.0))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) and value >= 0.0 else None

    @staticmethod
    def _pose_dict(pose_stamped):
        """Serialize a PoseStamped into the audit JSON without ROS types."""
        try:
            pose = pose_stamped.pose
            return {
                'frame_id': str(pose_stamped.header.frame_id or ''),
                'position': [float(pose.position.x), float(pose.position.y),
                             float(pose.position.z)],
                'orientation_xyzw': [
                    float(pose.orientation.x), float(pose.orientation.y),
                    float(pose.orientation.z), float(pose.orientation.w)],
            }
        except AttributeError:
            return {}

    def execute_cached_approach(self, skip_tool_pulses=False):
        """Run cached outbound, J, dwell, cached return and K exactly once.

        ``skip_tool_pulses`` omits both DO0 cockpit pulses.  It exists for the
        flexible-probe trials, where the shear tool the pulses were measured
        against is not fitted; motion, gates and recovery are unchanged.
        """
        if not self.approach_compute_lock.acquire(False):
            return False, '上一项球面通道计算或执行仍在进行'
        skip_tool_pulses = bool(skip_tool_pulses)
        cycle_started = time.monotonic()
        outbound_completed = False
        close_completed = False
        return_completed = False
        open_completed = False
        primary_error = None
        recovery_notes = []
        timings = {}
        result = {}
        plan_id = ''
        cycle_dwell_s = None
        try:
            cache = self.approach_plan_cache
            if not cache or not cache.get('plan_id'):
                raise RuntimeError('没有可执行的缓存轨迹；请先点击“只规划选中柑橘”')
            result = json.loads(json.dumps(
                cache.get('result') or {}, ensure_ascii=False))
            plan_id = str(cache['plan_id'])
            timings = dict(result.get('execution_timings_s') or {})
            runtime = self.request_runtime_config('get')
            if not runtime.get('success') or \
                    'tool_timeout_s' not in runtime.get('values', {}):
                raise RuntimeError(
                    '无法读取主面板“夹取后停留/s”当前值：%s' %
                    runtime.get('message', '运行参数服务没有返回该字段'))
            cycle_dwell_s = max(
                0.01, float(runtime['values']['tool_timeout_s']))
            map_status = self.approach_session.status()
            self.state.set_approach_status({
                'state': 'executing',
                'message': '正在直接执行缓存 %s；不会重新计算球面或MoveIt轨迹' % plan_id,
                'computing': True,
                'safe': bool(result.get('safe')),
                'view_count': int(map_status.get('view_count', 0)),
                'minimum_views': int(self.approach_minimum_views),
                'reconstruction_complete': True,
                'map_summary': map_status,
                'reconstructed_targets': list(self.approach_discovered_targets),
                'result': result,
            })

            stage_started = time.monotonic()
            released, release_message = self.release_cockpit_for_moveit()
            timings['control_handoff_s'] = time.monotonic() - stage_started
            if not released:
                raise RuntimeError(release_message)

            stage_started = time.monotonic()
            response = self._call_cached_approach_stage(
                plan_id, 'execute_outbound')
            timings['outbound_moveit_s'] = time.monotonic() - stage_started
            timings['outbound_detail'] = self._response_timings(response)
            timings['outbound_execution_duration_s'] = (
                self._response_execution_duration_s(response))
            if not response.success or not response.executed:
                raise RuntimeError(str(response.message))
            outbound_completed = True

            stage_started = time.monotonic()
            if skip_tool_pulses:
                # No shear tool is fitted, so the measured J pulse would drive
                # an end effector that is not on the flange.
                close_completed = True
            else:
                self._run_cached_tool_action('cockpit_close', 'J闭合')
                close_completed = True
            timings['tool_close_j_s'] = time.monotonic() - stage_started

            stage_started = time.monotonic()
            dwell_deadline = stage_started + cycle_dwell_s
            while time.monotonic() < dwell_deadline and not rospy.is_shutdown():
                time.sleep(max(
                    0.0, min(0.05, dwell_deadline - time.monotonic())))
            timings['grasp_dwell_s'] = time.monotonic() - stage_started
            if rospy.is_shutdown():
                raise RuntimeError('ROS 正在关闭，夹取后停留被中断')

            stage_started = time.monotonic()
            response = self._call_cached_approach_stage(
                plan_id, 'execute_return')
            timings['return_moveit_s'] = time.monotonic() - stage_started
            timings['return_detail'] = self._response_timings(response)
            timings['return_execution_duration_s'] = (
                self._response_execution_duration_s(response))
            if not response.success or not response.executed:
                raise RuntimeError(str(response.message))
            return_completed = True

            stage_started = time.monotonic()
            if skip_tool_pulses:
                open_completed = True
            else:
                self._run_cached_tool_action('cockpit_open', 'K张开')
                open_completed = True
            timings['tool_open_k_s'] = time.monotonic() - stage_started
        except Exception as error:
            primary_error = str(error)
            if outbound_completed and not return_completed and not rospy.is_shutdown():
                try:
                    recovery_started = time.monotonic()
                    response = self._call_cached_approach_stage(
                        str((self.approach_plan_cache or {}).get('plan_id', '')),
                        'execute_return')
                    timings['recovery_return_s'] = (
                        time.monotonic() - recovery_started)
                    if not response.success or not response.executed:
                        raise RuntimeError(str(response.message))
                    return_completed = True
                    recovery_notes.append('已用缓存轨迹完成返程')
                except Exception as recovery_error:
                    recovery_notes.append('缓存返程失败：%s' % recovery_error)
            if close_completed and return_completed and not open_completed and \
                    not skip_tool_pulses and not rospy.is_shutdown():
                try:
                    recovery_started = time.monotonic()
                    self._run_cached_tool_action('cockpit_open', '恢复K张开')
                    timings['recovery_tool_open_s'] = (
                        time.monotonic() - recovery_started)
                    open_completed = True
                    recovery_notes.append('返程后已执行K张开')
                except Exception as recovery_error:
                    recovery_notes.append('恢复K张开失败：%s' % recovery_error)
        finally:
            timings['total_execution_cycle_s'] = time.monotonic() - cycle_started
            cache = self.approach_plan_cache
            result = json.loads(json.dumps(
                (cache or {}).get('result') or {}, ensure_ascii=False))
            plan_id = str((cache or {}).get('plan_id', '') or '')
            success = bool(outbound_completed and close_completed and
                           return_completed and open_completed and
                           primary_error is None)
            result.update({
                'execution_requested': True,
                'execution_attempted': bool(outbound_completed or primary_error),
                'executed': success,
                'grasp_cycle_completed': success,
                'outbound_completed': outbound_completed,
                'tool_close_completed': close_completed,
                'return_completed': return_completed,
                'tool_open_completed': open_completed,
                'grasp_dwell_requested_s': cycle_dwell_s,
                'execution_timings_s': timings,
            })
            if primary_error:
                message = '缓存抓取周期失败：%s' % primary_error
                if recovery_notes:
                    message += '；' + '；'.join(recovery_notes)
                state_name = 'execution_rejected'
            else:
                message = (
                    '已复用 %s 完成入口、直线伸入15cm、J闭合、等待%.1fs、'
                    '原路退出、返回起点和K张开；全程未重新规划' %
                    (plan_id, cycle_dwell_s))
                state_name = 'executed'
            result['moveit_message'] = message
            self._persist_approach_result(
                result, 'executed' if success else 'execution_failed')
            if cache is not None:
                cache['result'] = json.loads(json.dumps(
                    result, ensure_ascii=False))
            map_status = self.approach_session.status()
            self.state.set_approach_status({
                'state': state_name,
                'message': message,
                'computing': False,
                'safe': bool(result.get('safe')),
                'view_count': int(map_status.get('view_count', 0)),
                'minimum_views': int(self.approach_minimum_views),
                'reconstruction_complete': bool(
                    int(map_status.get('view_count', 0)) >=
                    self.approach_minimum_views),
                'map_summary': map_status,
                'reconstructed_targets': list(self.approach_discovered_targets),
                'result': result,
            })
            self.approach_result_pub.publish(String(
                data=json.dumps(result, ensure_ascii=False, sort_keys=True)))
            self.state.set_action_status('[伸入方向] ' + message)
            self.state.append_event('[伸入方向] ' + message)
            rospy.loginfo(
                '[APPROACH_TIMING] grasp_cycle plan_id=%s success=%s stages=%s',
                plan_id, success,
                json.dumps(timings, ensure_ascii=False, sort_keys=True))
            self.approach_compute_lock.release()
        return success, message

    def _probe_batch_archive(self, group_number):
        """Locate the sealed ten-view ZIP for one group number.

        每条戳刺记录都要能回指产生该场景的归档，所以这里按组号在采集目录里找
        `*_g%04d.zip`；只有 staging、还没封箱时返回空字符串，由 probe_trials
        原样落盘，而不是猜一个不存在的路径。
        """
        try:
            group_number = int(group_number)
        except (TypeError, ValueError):
            return ''
        recorder_archive = str(
            getattr(self.nbv_recorder, 'last_archive', '') or '')
        suffix = '_g%04d.zip' % group_number
        if recorder_archive.endswith(suffix):
            return recorder_archive
        try:
            names = sorted(name for name in os.listdir(self.nbv_output_dir)
                           if name.endswith(suffix))
        except (IOError, OSError):
            return ''
        return (os.path.join(self.nbv_output_dir, names[-1])
                if names else '')

    def probe_status(self, layout_index, repeat_index):
        """Return only the blind facts the probe dialog is allowed to show."""
        rows = probe_trials.read_rows(self.probe_trials_csv)
        seed = probe_trials.load_or_create_seed(self.probe_trials_csv)
        assignment = probe_trials.next_assignment(
            rows, layout_index, repeat_index, seed)
        recommended = probe_trials.recommended_indices(rows)
        map_status = self.approach_session.status()
        view_count = int(map_status.get('view_count', 0))
        cache = self.approach_plan_cache or {}
        with self.probe_lock:
            armed = self.probe_active is not None
        return {
            'summary': probe_trials.display_summary(
                rows, layout_index, repeat_index, assignment),
            'block_complete': assignment is None,
            'armed': armed,
            'progress': probe_trials.progress(rows),
            'view_count': view_count,
            'minimum_views': int(self.approach_minimum_views),
            'reconstruction_complete': bool(
                view_count >= self.approach_minimum_views),
            'plan_cached': bool(cache.get('plan_id')),
            'recommended_layout_index': recommended[0],
            'recommended_repeat_index': recommended[1],
            'csv_path': self.probe_trials_csv,
        }

    def probe_start_scan(self, layout_index, repeat_index, scene_id='',
                         notes=''):
        """Draw the blind method, then auto-run the whole ten-view batch.

        方法在这一步就定下来，但只存在内存里；界面拿到的返回值不含 M1/M2/M3
        和 w_pi。开好批次后直接跑预定义宽幅自动扫描，十个视角一键记满，作者不用
        再按 Enter；记满后 `capture_nbv_view` 会自动接着规划。自动扫描走预定义宽幅
        绝对站位，逐站经 `plan_observation_pose` 规划与校验。自动扫描失败时本次
        试验保持可重试，Enter 手动路径仍然可用作兜底。
        """
        rows = probe_trials.read_rows(self.probe_trials_csv)
        seed = probe_trials.load_or_create_seed(self.probe_trials_csv)
        assignment = probe_trials.next_assignment(
            rows, layout_index, repeat_index, seed)
        if assignment is None:
            message = '本布局/重复的三次已记录完；请改布局或重复序号'
            self.state.set_action_status('[柔性探针] ' + message)
            return False, message
        with self.probe_lock:
            self.probe_active = {
                'assignment': dict(assignment),
                'layout_index': int(layout_index),
                'repeat_index': int(repeat_index),
                'trial_id': 'L%02dR%dO%d-%s' % (
                    int(layout_index), int(repeat_index),
                    int(assignment['order_in_block']),
                    datetime.now().strftime('%Y%m%d-%H%M%S')),
                'scene_id': str(scene_id or ''),
                'recorded': False,
            }
        status = self.set_nbv_mode(True, scene_id=scene_id, notes=notes)
        if not status.get('enabled'):
            with self.probe_lock:
                self.probe_active = None
            message = '无法开启十视角采集：%s' % status.get(
                'last_error', '论文采集模式被拒绝')
            self.state.set_action_status('[柔性探针] ' + message)
            return False, message
        swept, sweep_message = self.run_auto_ten_view_sweep(
            scene_id=scene_id, notes=notes)
        if not swept:
            # 失败不清 probe_active，也不置 recorded：本次试验仍可原样重按 F5
            # 重试，或退回 Enter 手动路径把这批补满。
            message = ('布局 %d 重复 %d 的自动扫描失败：%s；'
                       '可再按 F5 重试，或用 Enter/实体 POINT 手动补满' %
                       (int(layout_index), int(repeat_index), sweep_message))
            self.state.set_action_status('[柔性探针] ' + message)
            self.state.append_event('[柔性探针] ' + message)
            return False, message
        message = ('已为布局 %d 重复 %d 自动记满 %d 个视角；%s；'
                   '结果会自动出现' %
                   (int(layout_index), int(repeat_index),
                    int(self.approach_minimum_views), sweep_message))
        self.state.set_action_status('[柔性探针] ' + message)
        self.state.append_event('[柔性探针] ' + message)
        return True, message

    def probe_drive_in(self, target_index, target_point, target_frame='',
                       target_label=''):
        """Plan with the blind method and drive the probe in, in one key press.

        规划和执行合成一步：作者只按一个键，探针就伸进去。DO0 脉冲被跳过，因为
        本次实验没有装剪切工具。MoveIt 找不到可行候选是一种正常结果，返回
        `no_feasible_plan`，交给对话框记一行空分数，而不是重试或崩掉。
        """
        with self.probe_lock:
            active = None if self.probe_active is None else dict(
                self.probe_active)
        if active is None:
            message = '还没有开始一次试验；请先按“自动扫描”(F5)'
            self.state.set_action_status('[柔性探针] ' + message)
            return {'success': False, 'outcome': '', 'message': message}
        method_id = str(active['assignment']['method_id'])
        planned, plan_message = self.calculate_approach_direction(
            target_index, target_point, target_frame=target_frame,
            target_label=target_label, execute=False,
            probe_method=method_id)
        cache = self.approach_plan_cache or {}
        if not planned or not cache.get('plan_id'):
            # 探针根本没有伸出，操作者无从目视评分。
            message = '本次无可行轨迹：%s' % plan_message
            self.state.set_action_status('[柔性探针] ' + message)
            self.state.append_event('[柔性探针] ' + message)
            return {
                'success': False,
                'outcome': probe_trials.OUTCOME_NO_FEASIBLE_PLAN,
                'message': message,
                'needs_score': False,
            }
        executed, execute_message = self.execute_cached_approach(
            skip_tool_pulses=True)
        outcome = (probe_trials.OUTCOME_EXECUTED if executed
                   else probe_trials.OUTCOME_EXECUTION_FAILED)
        self.state.set_action_status('[柔性探针] ' + execute_message)
        return {
            'success': bool(executed),
            'outcome': outcome,
            'message': execute_message,
            'needs_score': bool(executed),
        }

    def probe_record(self, layout_index, repeat_index, outcome, score=None,
                     note=''):
        """Write one trial row and disarm; the next trial needs a new scan."""
        with self.probe_lock:
            active = None if self.probe_active is None else dict(
                self.probe_active)
        if active is None:
            message = '没有待记录的试验；请先按“自动扫描”(F5)'
            self.state.set_action_status('[柔性探针] ' + message)
            return False, message
        cache = self.approach_plan_cache or {}
        result = dict(cache.get('result') or {})
        group_number = cache.get('group_number', self.approach_group_number)
        recommended = probe_trials.recommended_indices(
            probe_trials.read_rows(self.probe_trials_csv))
        context = {
            'trial_id': str(active['trial_id']),
            'scene_id': str(active.get('scene_id', '')),
            'group_number': group_number,
            'batch_archive_path': self._probe_batch_archive(group_number),
            'batch_output_dir': self.nbv_output_dir,
            'tool_pulse_bypassed': True,
            'dry_run': False,
        }
        operator = {
            'layout_index': int(layout_index),
            'repeat_index': int(repeat_index),
            'outcome': str(outcome),
            'score': score,
            'note': str(note or ''),
            'indices_match_recommendation': bool(
                (int(layout_index), int(repeat_index)) == recommended),
        }
        try:
            row = probe_trials.build_row(
                result, active['assignment'], operator, context=context)
            path = probe_trials.append_row(row, self.probe_trials_csv)
        except Exception as error:
            message = '记录失败，未写入任何行：%s' % error
            self.state.set_action_status('[柔性探针] ' + message)
            self.state.append_event('[柔性探针] ' + message)
            return False, message
        with self.probe_lock:
            self.probe_active = None
        rows = probe_trials.read_rows(self.probe_trials_csv)
        overall = probe_trials.progress(rows)
        message = ('已记录第 %d / %d 次（%s）；已写入 %s' %
                   (overall['completed'], overall['total'],
                    str(outcome), path))
        self.state.set_action_status('[柔性探针] ' + message)
        self.state.append_event('[柔性探针] ' + message)
        return True, message

    def set_nbv_mode(self, enabled, scene_id='', notes=''):
        """Enable/disable the ten-view recorder without touching robot motion."""
        try:
            if enabled:
                scene_id = str(scene_id or self.nbv_scene_default or '').strip()
                status = self.nbv_recorder.enable(
                    scene_id=scene_id, notes=str(notes or ''), resume=True)
                group_number = int(status.get('group_number', 1))
                if self.approach_group_number != group_number or \
                        self.approach_session.status().get('view_count', 0) != \
                        int(status.get('view_count', 0)):
                    self._rebuild_approach_from_staging(group_number)
                self.state.set_action_status(
                    '论文采集模式已开启：按 Enter 或机械臂实体 POINT 记录当前视角；' +
                    '不要在未确认画面/位姿时连续按键')
            else:
                status = self.nbv_recorder.disable(keep_staging=True)
                self.state.set_action_status(
                    '论文采集模式已关闭；未完成批次保留在 staging，可恢复')
            self._publish_nbv_status(status)
            return status
        except Exception as error:
            status = self.nbv_recorder.status()
            status['last_error'] = '论文采集模式切换失败：%s' % error
            status['last_result'] = status['last_error']
            self._publish_nbv_status(status)
            self.state.set_action_status(status['last_error'])
            return status

    def update_nbv_context(self, scene_id='', notes=''):
        """Persist current GUI scene/notes before the next manual snapshot."""
        if not self.nbv_recorder.enabled:
            return self.nbv_recorder.status()
        status = self.nbv_recorder.set_context(
            scene_id=str(scene_id or '').strip() or None,
            notes=str(notes or ''))
        self._publish_nbv_status(status)
        return status

    def _calibration_metadata(self):
        path = self.calibration_config_file
        status = read_install_status(path)
        text_value = ''
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as stream:
                    text_value = stream.read()
            except (IOError, OSError) as error:
                status['read_error'] = str(error)
        return status, text_value

    def update_camera_identity(self, device):
        """Refresh serial/USB metadata discovered by CameraRuntime."""
        device = dict(device or {})
        changed = False
        for attribute, key in (
                ('camera_serial', 'serial'),
                ('camera_device_type', 'device_type'),
                ('camera_usb_port_id', 'usb_port_id'),
                ('camera_usb_type', 'usb_type')):
            value = str(device.get(key, '') or '').strip()
            if value and value != getattr(self, attribute):
                setattr(self, attribute, value)
                changed = True
        if changed:
            self._publish_nbv_status(self.nbv_recorder.status())

    def _latest_joint_record(self, state_snapshot):
        message = state_snapshot.get('joints')
        if message is None or not getattr(message, 'name', None):
            raise ValueError('没有最新 /joint_states')
        arrival = float(state_snapshot.get('joint_arrival', 0.0) or 0.0)
        if arrival <= 0.0 or time.monotonic() - arrival > self.nbv_max_joint_age_s:
            raise ValueError('关节状态过旧（%.2fs），请保持机器人状态流动' %
                             (time.monotonic() - arrival if arrival else 999.0))
        return {
            'names': [str(value) for value in message.name],
            'positions_rad': [float(value) for value in message.position],
            'velocities_rad_s': [float(value) for value in message.velocity],
            'efforts': [float(value) for value in message.effort],
            'stamp_sec': _stamp_sec(getattr(message.header, 'stamp', None)),
            'frame_id': str(getattr(message.header, 'frame_id', '') or ''),
        }

    def _lookup_camera_pose(self, color_header):
        stamp = _stamp_sec(getattr(color_header, 'stamp', None))
        if stamp <= 0.0:
            raise ValueError('RGB 时间戳为零，不能查询相机 6D 位姿')
        source_frame = str(getattr(color_header, 'frame_id', '') or
                           self.camera_optical_frame or
                           'camera_color_optical_frame').lstrip('/')
        target_frame = self.target_frame.lstrip('/')
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rospy.Time.from_sec(stamp),
                rospy.Duration(min(0.15, self.nbv_sync_slop_s)))
        except Exception as error:
            raise ValueError('没有新鲜 TF %s -> %s：%s' %
                             (target_frame, source_frame, error))
        matrix, quaternion = _transform_dict(transform)
        return matrix, quaternion, target_frame, source_frame, stamp

    def _lookup_flange_origin(self):
        """Return the current flange origin in the reconstruction frame.

        The task-conditioned direction term is optional, so an unavailable or
        stale TF returns None and the planner falls back to pure geometry.
        """
        if self.tf_buffer is None:
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame.lstrip('/'), self.flange_frame.lstrip('/'),
                rospy.Time(0), rospy.Duration(0.15))
        except Exception as error:
            rospy.logwarn('没有可用法兰 TF %s -> %s：%s', self.target_frame,
                          self.flange_frame, error)
            return None
        matrix, _quaternion = _transform_dict(transform)
        return [float(value) for value in matrix[:3, 3]]

    def _lookup_pose(self, target_frame, source_frame):
        """Return (position, quaternion_xyzw) of source expressed in target."""
        if self.tf_buffer is None:
            raise RuntimeError('TF 缓冲不可用')
        transform = self.tf_buffer.lookup_transform(
            str(target_frame).lstrip('/'), str(source_frame).lstrip('/'),
            rospy.Time(0), rospy.Duration(0.25))
        matrix, quaternion = _transform_dict(transform)
        position = tuple(float(value) for value in matrix[:3, 3])
        return position, tuple(quaternion)

    def _flange_to_optical_pose(self):
        """Return the fixed flange -> cockpit optical pose.

        换算链不在这里重新推导：先直接查 TF（`publish_camera_tf.py` 常驻发布
        `elfin_end_link -> camera_link -> camera_cockpit_optical_frame`）；TF 缺
        失时退回用同一份已通过质量校验的标定文件，与那条固定 REP-103 旋转按
        `compose_pose` 复合，结果与 TF 一致。
        """
        try:
            return self._lookup_pose(
                self.flange_frame, 'camera_cockpit_optical_frame')
        except Exception:
            pass
        with open(self.calibration_config_file, 'r') as stream:
            document = yaml.safe_load(stream) or {}
        translation = [float(value) for value in document['translation_m']]
        quaternion = [float(value) for value in document['quaternion_xyzw']]
        if str(document.get('parent_frame', '')) != self.flange_frame:
            raise RuntimeError('标定父坐标系 %s 不是法兰 %s' % (
                document.get('parent_frame'), self.flange_frame))
        # publish_camera_tf.py:114-120 的固定 camera_link -> 光学帧旋转。
        return compose_pose(tuple(translation), tuple(quaternion),
                            (0.0, 0.0, 0.0), (-0.5, 0.5, -0.5, 0.5))

    def _sweep_target_point(self):
        """Return (target_point, description, is_fallback) for the sweep centre.

        首选实时检测里视觉优先级最高的那颗（最大最近），并且只认已经变换到
        `target_frame` 且 `target_point_valid` 为真的目标。没有可用检测时退回当
        前视线前方一个默认距离，调用方必须把这一点写进状态栏。
        """
        message = self.state.snapshot().get('targets')
        candidates = []
        for target in list(getattr(message, 'targets', ()) or ()):
            if not bool(getattr(target, 'target_point_valid', False)):
                continue
            frame = str(getattr(target, 'target_frame', '') or '')
            if frame.lstrip('/') != self.target_frame.lstrip('/'):
                continue
            point = getattr(target, 'target_point', None)
            values = (float(getattr(point, 'x', 0.0)),
                      float(getattr(point, 'y', 0.0)),
                      float(getattr(point, 'z', 0.0)))
            if not all(math.isfinite(value) for value in values):
                continue
            candidates.append((target, values))
        if candidates:
            target, values = max(
                candidates, key=lambda item: target_visual_priority_key(
                    item[0]))
            return values, '实时检测目标（%s，置信度 %.2f，深度 %.3f m）' % (
                str(getattr(target, 'label', '') or '?'),
                float(getattr(target, 'confidence', 0.0)),
                float(getattr(target, 'depth_m', 0.0))), False
        position, quaternion = self._lookup_pose(
            self.target_frame, 'camera_cockpit_optical_frame')
        axis = quaternion_rotate(quaternion, (0.0, 0.0, 1.0))
        distance = float(self.sweep_fallback_distance_m)
        values = tuple(position[index] + axis[index] * distance
                       for index in range(3))
        return values, '兜底目标点：当前视线前方 %.2f m（无有效检测）' % distance, True

    def _sweep_flange_pose(self, station, flange_to_optical):
        """Convert one absolute camera station into a flange PoseStamped.

        姿态先用 `align_flange_z_to_direction` 把光轴转到该站位的 `view_axis`，
        再用 `level_camera_quaternion` 去掉滚转；位置与姿态一起经
        `camera_goal_to_flange_pose` 换算到法兰。
        """
        camera_quaternion = level_camera_quaternion(
            align_flange_z_to_direction(
                (0.0, 0.0, 0.0, 1.0), station.view_axis))
        position, orientation = camera_goal_to_flange_pose(
            tuple(float(value) for value in station.camera_position),
            camera_quaternion, flange_to_optical[0], flange_to_optical[1])
        pose = PoseStamped()
        pose.header.frame_id = self.target_frame
        pose.header.stamp = rospy.Time(0)
        pose.pose.position.x = float(position[0])
        pose.pose.position.y = float(position[1])
        pose.pose.position.z = float(position[2])
        pose.pose.orientation.x = float(orientation[0])
        pose.pose.orientation.y = float(orientation[1])
        pose.pose.orientation.z = float(orientation[2])
        pose.pose.orientation.w = float(orientation[3])
        return pose, position, camera_quaternion

    def _request_observation_pose(self, pose, view_index, execute=True,
                                  timeout_s=2.0):
        """Send one station to the planner's validated plan/execute service."""
        rospy.wait_for_service(self.observation_pose_service,
                               timeout=float(timeout_s))
        response = rospy.ServiceProxy(
            self.observation_pose_service, PlanObservationPose)(
                flange_pose=pose, execute=bool(execute),
                view_index=int(view_index))
        return bool(response.success), str(response.message or ''), response

    def _build_nbv_snapshot(self, capture_trigger='keyboard_enter'):
        state_snapshot = self.state.snapshot()
        selected = self.state.select_capture_inputs(
            now=time.monotonic(),
            rgbd_slop_s=self.nbv_sync_slop_s,
            max_frame_age_s=self.nbv_max_frame_age_s,
            result_slop_s=self.nbv_target_sync_slop_s,
            joint_slop_s=self.nbv_max_frame_age_s,
            imu_window_before_s=self.nbv_imu_window_before_s,
            imu_window_after_s=self.nbv_imu_window_after_s,
            imu_max_age_s=self.nbv_imu_max_age_s)
        if selected.get('error'):
            # Keep a narrow compatibility path for tests or legacy launchers
            # that populate SharedState directly rather than through ROS
            # callbacks.  Live callbacks always populate the history above.
            if (state_snapshot.get('raw_color') is None or
                    state_snapshot.get('raw_depth') is None or
                    state_snapshot.get('color_header') is None or
                    state_snapshot.get('depth_header') is None):
                raise ValueError(selected['error'])
            selected = {
                'raw_color': state_snapshot.get('raw_color'),
                'raw_depth': state_snapshot.get('raw_depth'),
                'color_header': state_snapshot.get('color_header'),
                'depth_header': state_snapshot.get('depth_header'),
                'raw_arrival': dict(state_snapshot.get('raw_arrival') or {}),
                'camera_info': state_snapshot.get('camera_info'),
                'camera_info_arrival': state_snapshot.get(
                    'camera_info_arrival', 0.0),
                'semantic_labels': state_snapshot.get('semantic_labels'),
                'semantic_confidence': state_snapshot.get(
                    'semantic_confidence'),
                'semantic_instances': state_snapshot.get('semantic_instances'),
                'semantic_header': state_snapshot.get('semantic_header'),
                'semantic_arrival': state_snapshot.get('semantic_arrival', 0.0),
                'semantic_available': state_snapshot.get(
                    'semantic_available', False),
                'targets': state_snapshot.get('targets'),
                'target_arrival': state_snapshot.get('target_arrival', 0.0),
                'joints': state_snapshot.get('joints'),
                'joint_arrival': state_snapshot.get('joint_arrival', 0.0),
                'imu_samples': {
                    'gyro': ([
                        (state_snapshot.get('imu_latest') or {}).get('gyro')]
                             if (state_snapshot.get('imu_latest') or {}).get(
                                 'gyro') else []),
                    'accel': ([
                        (state_snapshot.get('imu_latest') or {}).get('accel')]
                              if (state_snapshot.get('imu_latest') or {}).get(
                                  'accel') else []),
                },
                'imu_topics': dict(state_snapshot.get('imu_topics') or {}),
                'rgbd_pair_delta_s': None,
                'history_fallback': True,
            }
        state_snapshot.update(selected)
        color = state_snapshot.get('raw_color')
        depth = state_snapshot.get('raw_depth')
        color_header = state_snapshot.get('color_header')
        depth_header = state_snapshot.get('depth_header')
        if color is None or depth is None or color_header is None or depth_header is None:
            raise ValueError('尚未收到完整 RGB-D 原始帧')
        color_stamp = _stamp_sec(getattr(color_header, 'stamp', None))
        depth_stamp = _stamp_sec(getattr(depth_header, 'stamp', None))
        arrivals = state_snapshot.get('raw_arrival', {})
        now = time.monotonic()
        camera_info = state_snapshot.get('camera_info')
        info_dict = _camera_info_dict(camera_info)
        if len(info_dict.get('K', [])) != 9:
            raise ValueError('没有有效 CameraInfo 内参')
        joint_message = state_snapshot.get('joints')
        joint_stamp = _stamp_sec(
            getattr(getattr(joint_message, 'header', None), 'stamp', None))
        info_stamp = _stamp_sec(
            getattr(getattr(camera_info, 'header', None), 'stamp', None))
        info_stamp_for_timing = (0.0 if state_snapshot.get(
            'camera_info_static_stamp', False) else info_stamp)
        timing_ok, timing_reason, timing = validate_capture_timing(
            color_stamp, depth_stamp, now=now, raw_arrival=arrivals,
            sync_slop_s=self.nbv_sync_slop_s,
            max_frame_age_s=self.nbv_max_frame_age_s,
            camera_info_stamp=info_stamp_for_timing,
            camera_info_arrival=state_snapshot.get('camera_info_arrival', 0.0),
            joint_stamp=joint_stamp,
            joint_arrival=state_snapshot.get('joint_arrival', 0.0),
            max_joint_age_s=self.nbv_max_joint_age_s)
        if not timing_ok:
            raise ValueError(timing_reason + '；请等待同步数据稳定后再按 Enter')

        pose, quaternion, pose_parent, pose_child, stamp = \
            self._lookup_camera_pose(color_header)
        joint_state = self._latest_joint_record(state_snapshot)
        imu = _imu_bundle(
            state_snapshot.get('imu_samples'),
            state_snapshot.get('imu_topics'), color_stamp,
            self.nbv_imu_window_before_s, self.nbv_imu_window_after_s)
        if self.nbv_require_imu and not imu.get('available', False):
            counts = imu.get('sample_counts') or {}
            raise ValueError(
                'D455 IMU 原始窗口不完整（gyro=%d, accel=%d）；'
                '请确认 %s 与 %s 持续发布' %
                (int(counts.get('gyro', 0)), int(counts.get('accel', 0)),
                 self.gyro_topic, self.accel_topic))
        targets_message = state_snapshot.get('targets')
        target_stamp = _stamp_sec(
            getattr(getattr(targets_message, 'header', None), 'stamp', None))
        detections = []
        model = {}
        target_sync_ok = False
        target_timing = {}
        target_arrival = float(state_snapshot.get('target_arrival', 0.0) or 0.0)
        target_age = max(0.0, now - target_arrival) if target_arrival else float('inf')
        if targets_message is not None:
            detections = [_target_dict(item) for item in targets_message.targets]
            model = {
                'model_path': str(getattr(targets_message, 'model_path', '') or ''),
                'status': str(getattr(targets_message, 'status', '') or ''),
                'inference_ms': float(getattr(targets_message, 'inference_ms', 0.0)),
                'configured_sha256': self.model_sha256,
                'class_names': self.semantic_class_names,
                'code_version': self.code_version,
            }
            target_sync_ok, _target_reason, target_timing = \
                validate_inference_timing(
                    target_stamp, color_stamp, target_arrival, now=now,
                    max_delta_s=self.nbv_target_sync_slop_s,
                    max_age_s=self.nbv_max_frame_age_s)
        if not target_sync_ok:
            # Keep the image/pose sample auditable, but do not pretend its
            # detection result belongs to this frame.
            detections = []
            model['synchronization_gap'] = True

        semantic_labels = state_snapshot.get('semantic_labels')
        semantic_confidence = state_snapshot.get('semantic_confidence')
        semantic_instances = state_snapshot.get('semantic_instances')
        semantic_available = bool(state_snapshot.get('semantic_available', False))
        semantic_header = state_snapshot.get('semantic_header')
        semantic_stamp = _stamp_sec(getattr(semantic_header, 'stamp', None))
        semantic_arrival = float(state_snapshot.get('semantic_arrival', 0.0) or 0.0)
        semantic_age = max(0.0, now - semantic_arrival) if semantic_arrival else float('inf')
        # ``semantic_available`` means that the detector produced at least
        # one positive mask, not that the semantic topic exists.  An all-zero
        # mask is a valid negative observation and must remain in the batch.
        semantic_ok = (semantic_labels is not None and
                       semantic_labels.shape == depth.shape and
                       semantic_stamp > 0.0 and
                       abs(semantic_stamp - color_stamp) <= self.nbv_semantic_max_age_s and
                       semantic_age <= self.nbv_semantic_max_age_s and
                       bool(state_snapshot.get(
                           'semantic_components_synchronized', True)))
        if not semantic_ok:
            semantic_labels = None
            semantic_confidence = None
            semantic_instances = None

        calibration, calibration_text = self._calibration_metadata()
        if self.nbv_require_calibration and not calibration.get('configured', False):
            raise ValueError('当前 eye-in-hand 标定未配置，不能采集正式位姿')
        if self.nbv_require_calibration and not calibration.get('quality_passed', False):
            raise ValueError('当前 eye-in-hand 标定质量未通过，不能采集正式位姿')
        if self.nbv_require_calibration and not calibration.get('installed', False):
            raise ValueError(
                '当前 eye-in-hand 标定没有已安装/可回滚记录；请通过 D455 标定入口安装后再采集')
        if self.nbv_require_calibration and not calibration_text:
            raise ValueError('当前标定 YAML 无法读取，不能采集正式位姿')
        camera_serial = self.camera_serial
        if self.nbv_require_calibration and not camera_serial:
            raise ValueError('尚未确认当前 RealSense 序列号，不能采集正式位姿')
        if (self.nbv_require_usb3 and self.nbv_require_calibration and not str(
                self.camera_usb_type or '').strip().startswith('3.')):
            raise ValueError(
                '当前 RealSense 不是 USB3（检测值 %s）；请更换接口/线缆后再采集' %
                (self.camera_usb_type or '<unknown>'))
        actual_type = str(self.camera_device_type or '').strip().lower()
        if (self.nbv_require_calibration and
                self.nbv_expected_camera_type and
                not self.nbv_allow_non_expected_camera and
                actual_type and
                self.nbv_expected_camera_type not in actual_type):
            raise ValueError(
                '当前相机型号 %s 不是实验预期的 %s；D455 更换后必须重新标定' %
                (self.camera_device_type or '<unknown>',
                 self.nbv_expected_camera_type))
        if (self.nbv_require_calibration and
                self.nbv_expected_camera_type and not actual_type and
                not self.nbv_allow_non_expected_camera):
            raise ValueError('尚未确认当前 RealSense 型号，不能采集正式位姿')
        if (self.nbv_require_calibration and
                not calibration.get('camera_serial', '')):
            raise ValueError('当前标定没有相机序列号，不能采集正式位姿')
        camera = {
            'name': self.camera_name,
            'serial': camera_serial,
            'device_type': self.camera_device_type,
            'usb_port_id': self.camera_usb_port_id,
            'usb_type': self.camera_usb_type,
            'configured_calibration_serial': calibration.get('camera_serial', ''),
        }
        sync = {
            'rgb_stamp_sec': color_stamp,
            'depth_stamp_sec': depth_stamp,
            'rgb_depth_delta_s': abs(color_stamp - depth_stamp),
            'timestamp_history_pair_delta_s': state_snapshot.get(
                'rgbd_pair_delta_s'),
            'timestamp_history_fallback': bool(
                state_snapshot.get('history_fallback', False)),
            'camera_info_stamp_sec': info_stamp,
            'camera_info_static_stamp': bool(state_snapshot.get(
                'camera_info_static_stamp', False)),
            'targets_stamp_sec': target_stamp,
            'target_color_delta_s': (abs(target_stamp - color_stamp)
                                     if target_stamp > 0.0 else None),
            'target_sync_slop_s': self.nbv_target_sync_slop_s,
            'semantic_stamp_sec': semantic_stamp,
            'targets_synchronized': bool(target_sync_ok),
            'semantic_synchronized': bool(semantic_ok),
            'semantic_available_topic': bool(semantic_available),
            'semantic_empty': bool(semantic_ok and
                                   not np.any(np.asarray(semantic_labels))),
            'semantic_component_stamps': state_snapshot.get(
                'semantic_component_stamps', {}),
            'imu_sample_counts': dict(imu.get('sample_counts') or {}),
            'imu_available': bool(imu.get('available', False)),
            'imu_reference_stamp_sec': float(color_stamp),
            'semantic_components_synchronized': bool(state_snapshot.get(
                'semantic_components_synchronized', True)),
            'target_arrival_age_s': (None if not math.isfinite(target_age)
                                     else target_age),
            'semantic_arrival_age_s': (None if not math.isfinite(semantic_age)
                                       else semantic_age),
        }
        sync.update(timing)
        sync.update({'target_timing': target_timing})
        if (camera_serial and calibration.get('camera_serial') and
                camera_serial != calibration.get('camera_serial')):
            sync['calibration_serial_mismatch'] = True
            if self.nbv_require_calibration:
                raise ValueError(
                    '当前相机序列号 %s 与标定序列号 %s 不一致；D455 必须重新标定' %
                    (camera_serial, calibration.get('camera_serial')))
        semantic_gap = '' if semantic_ok else (
            'pixel-level semantic mask unavailable or unsynchronized; '
            'do not use this view for formal Semantic NBV PCO')
        if semantic_ok and semantic_instances is None:
            semantic_gap = (
                'pixel-level instance ID map unavailable; engineering replay is '
                'allowed; class-labelled 3-D PCO remains eligible when independent '
                'OOI truth is present')
        session_metadata = {
            'capture_mode': (
                'manual_tool_point'
                if capture_trigger == 'tool_point_di4' else
                'auto_predefined_wide_sweep'
                if capture_trigger == 'auto_ten_view_sweep' else
                'manual_cockpit_enter'),
            'capture_trigger': str(capture_trigger),
            # 标称跨度是 2 × 半角；实测跨度由已达成的十个站位方位角算出，作业空
            # 间或 IK 拒掉外侧站位时会小于标称值。论文报实测值。
            'sweep_nominal_azimuth_span_deg': (
                2.0 * float(WIDE_SWEEP_HALF_ANGLE_DEG)),
            'sweep_measured_azimuth_span_deg': (
                None if self.sweep_last_azimuth_span_deg is None
                else float(self.sweep_last_azimuth_span_deg)),
            'sweep_observation_distance_m': float(
                self.sweep_observation_distance_m),
            'sweep_target_point_fallback': bool(
                self.sweep_last_target_fallback),
            'input_contract': 'RGB-D images plus base-frame camera 6D pose',
            'auxiliary_channels': [
                'joint_states', 'realsense_gyro', 'realsense_accel'],
            'imu_required': bool(self.nbv_require_imu),
            'imu_role': (
                'raw motion/stability audit; no magnetometer or absolute '
                'camera heading is inferred'),
            'pose_provenance': (
                'TF lookup at RGB timestamp; TF is produced from the installed '
                'eye-in-hand calibration and robot state'),
            'pose_parent_frame': pose_parent,
            'pose_child_frame': pose_child,
            'class_names': self.semantic_class_names,
            'semantic_class_names': self.semantic_class_names,
            'semantic_classes_available': list(self.semantic_class_names.values()),
            'semantic_ooi_gap': self.semantic_gap,
            'semantic_instance_ids_available': bool(semantic_instances is not None),
            'sample_recommendation': {
                'pilot_batches': 2,
                'engineering_minimum_batches': 18,
                'trend_minimum_batches': 24,
                'formal_target_independent_scenes': '10-12',
            },
        }
        return {
            'color': np.asarray(color).copy(),
            'depth': np.asarray(depth).copy(),
            'annotated': (None if state_snapshot['images'].get('annotated') is None
                          else state_snapshot['images']['annotated'].copy()),
            'semantic_labels': semantic_labels,
            'semantic_confidence': semantic_confidence,
            'instance_ids': semantic_instances,
            'semantic_present': bool(semantic_ok and semantic_labels is not None),
            'semantic_gap': semantic_gap,
            'stamp_sec': color_stamp,
            'wall_time': datetime.now().isoformat(timespec='milliseconds'),
            'color_frame_id': str(getattr(color_header, 'frame_id', '') or ''),
            'depth_frame_id': str(getattr(depth_header, 'frame_id', '') or ''),
            'pose_frame_id': '%s -> %s' % (pose_parent, pose_child),
            'pose_parent_frame': pose_parent,
            'pose_child_frame': pose_child,
            'pose_matrix': pose,
            'pose_quaternion_xyzw': quaternion,
            'joint_state': joint_state,
            'imu': imu,
            'camera_info': info_dict,
            'camera': camera,
            'calibration': calibration,
            'calibration_yaml_text': calibration_text,
            'model': model,
            'detections': detections,
            'sync': sync,
            'session_metadata': session_metadata,
        }

    def capture_nbv_view(self, require_semantics=None,
                         require_instance_ids=None, scene_id=None,
                         notes=None, capture_trigger='keyboard_enter'):
        """Capture one validated view; this method itself never commands motion."""
        if require_semantics is None:
            require_semantics = self.nbv_require_semantics
        if require_instance_ids is None:
            require_instance_ids = self.nbv_require_instance_ids
        with self.nbv_lock:
            if self.nbv_capture_inflight:
                return {'accepted': False, 'reason': '上一张采集快照仍在写入'}
            self.nbv_capture_inflight = True
        try:
            if not self.nbv_recorder.enabled:
                reason = '请先打开论文采集模式，再按 Enter 或实体 POINT'
                self.state.set_action_status(reason)
                return {'accepted': False, 'reason': reason,
                        'status': self.nbv_recorder.status()}
            if scene_id is not None or notes is not None:
                self.update_nbv_context(scene_id=scene_id or '',
                                        notes=notes or '')
            try:
                snapshot = self._build_nbv_snapshot(
                    capture_trigger=capture_trigger)
            except Exception as error:
                reason = '本次未记录：%s' % error
                status = self.nbv_recorder.status()
                status['last_error'] = reason
                status['last_result'] = reason
                self._publish_nbv_status(status)
                self.state.set_action_status(reason)
                self.state.append_event('[NBV] ' + reason)
                return {'accepted': False, 'reason': reason, 'status': status}
            capture_group = int(self.nbv_recorder.group_number)
            result = self.nbv_recorder.capture(
                snapshot, require_semantics=bool(require_semantics),
                require_instance_ids=bool(require_instance_ids),
                require_imu=bool(getattr(self, 'nbv_require_imu', False)))
            status = result.get('status', self.nbv_recorder.status())
            self._publish_nbv_status(status)
            if result.get('accepted'):
                expected_view = int(result.get('view_index', 0) or 0)
                approach_count = int(
                    self.approach_session.status().get('view_count', 0))
                if (self.approach_group_number != capture_group or
                        approach_count < expected_view):
                    try:
                        self._insert_approach_snapshot(
                            snapshot, group_number=capture_group)
                    except Exception as error:
                        # The research archive remains authoritative. A map
                        # insertion failure is reported but never rolls back a
                        # successfully sealed RGB-D view.
                        map_error = '语义体素融合失败，原始快照仍已保留：%s' % error
                        self._approach_state('error', map_error)
                        self.state.append_event('[伸入方向] ' + map_error)
                if result.get('completed'):
                    message = ('[NBV] 批次已完成并保存：%s；已自动准备下一批，'
                               '可重整场景或保持当前场景继续' %
                               result.get('archive_path', ''))
                    with self.probe_lock:
                        armed = self.probe_active is not None
                    if armed:
                        # 十张记满即重建完成，候选目标已经在对话框列表里；
                        # 操作者只需确认目标再按“伸入”，中间不再有别的按键。
                        message += '；柔性探针试验已就绪，请确认目标后按“伸入”'
                else:
                    message = '[NBV] 已记录当前视角 %s' % status.get('view_count')
                self.state.set_action_status(message)
                self.state.append_event(message)
            else:
                reason = result.get('reason', '未知写入错误')
                self.state.set_action_status('论文采集失败：' + str(reason))
                self.state.append_event('[NBV] 失败：' + str(reason))
            return result
        finally:
            with self.nbv_lock:
                self.nbv_capture_inflight = False

    def capture_nbv_view_async(self, scene_id=None, notes=None,
                               capture_trigger='keyboard_enter'):
        thread = threading.Thread(
            target=self.capture_nbv_view,
            kwargs={'scene_id': scene_id, 'notes': notes,
                    'capture_trigger': capture_trigger},
                                   name='nbv-view-capture', daemon=True)
        thread.start()
        return thread

    def start_calibration(self, install_live=True):
        """Launch the read-only Charuco collector from the dashboard."""
        with self.calibration_lock:
            if self.calibration_process is not None and \
                    self.calibration_process.poll() is None:
                message = '手眼标定窗口已经运行'
                self.state.set_action_status(message)
                return False, message
            command = [self.calibration_launcher]
            if install_live:
                command.append('--install-live')
            try:
                run_root = self.calibration_data_dir
                os.makedirs(run_root, exist_ok=True)
                stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
                run_dir = os.path.join(
                    run_root, 'dashboard_eye_in_hand_' + stamp)
                suffix = 1
                while os.path.exists(run_dir):
                    run_dir = os.path.join(
                        run_root,
                        'dashboard_eye_in_hand_%s_%02d' % (stamp, suffix))
                    suffix += 1
                self.calibration_run_dir = run_dir
                command.extend([
                    '--output-dir', run_dir,
                    '--live-config', self.calibration_config_file,
                ])
                environment = os.environ.copy()
                environment.pop('ROS_NAMESPACE', None)
                if self.python_executable:
                    environment['ELFIN_VISION_PYTHON'] = self.python_executable
                self.calibration_process = subprocess.Popen(
                    command, start_new_session=True, env=environment)
                process = self.calibration_process
            except Exception as error:
                self.calibration_process = None
                message = '手眼标定入口启动失败：%s' % error
                self.state.set_action_status(message)
                return False, message
        threading.Thread(
            target=self._watch_calibration_process, args=(process,),
            name='watch-eye-hand-calibration', daemon=True).start()
        message = ('已启动 D455 手眼标定：只读采样，不发送机械臂运动；'
                   '质量通过且序列号匹配后自动备份并安装新外参')
        self.state.set_action_status(message)
        self.state.append_event('[CALIBRATION] ' + message)
        return True, message

    def _watch_calibration_process(self, process):
        return_code = process.wait()
        with self.calibration_lock:
            if self.calibration_process is process:
                self.calibration_process = None
            run_dir = self.calibration_run_dir
        result_document = {}
        result_path = os.path.join(run_dir, 'calibration_result.yaml') \
            if run_dir else ''
        if result_path and os.path.isfile(result_path):
            try:
                with open(result_path, 'r', encoding='utf-8') as stream:
                    result_document = yaml.safe_load(stream) or {}
            except Exception as error:
                result_document = {'status': 'result_read_failed',
                                   'message': str(error)}
        result_status = str(result_document.get('status', '') or '')
        status = self.refresh_nbv_status()
        calibration = status.get('calibration') or {}
        if return_code == 0 and result_status == 'installed':
            reload_ok, reload_message = self.reload_calibrated_tf()
            message = ('D455 标定已通过并安装；live 标定序列号 %s；结果：%s' % (
                calibration.get('camera_serial') or '<missing>',
                result_path or '<missing>'))
            if not reload_ok:
                message += '；' + reload_message
        elif return_code == 0 and result_status == 'candidate_passed':
            message = ('D455 候选质量通过但未安装 live；结果：%s；live 未刷新' %
                       (result_path or '<missing>'))
        else:
            message = ('D455 标定未完成（退出码 %s，状态=%s）；live 配置未被假定更新；'
                       '结果：%s' % (return_code, result_status or '<unknown>',
                                   result_path or '<missing>'))
        self.state.set_action_status(message)
        self.state.append_event('[CALIBRATION] ' + message)

    def stop_calibration(self):
        with self.calibration_lock:
            process = self.calibration_process
            self.calibration_process = None
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=4.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except OSError:
                pass

    def reload_calibrated_tf(self):
        """Ask the live TF publisher to atomically re-read the new YAML."""
        try:
            rospy.wait_for_service(self.calibrated_tf_reload_service, timeout=1.0)
            response = rospy.ServiceProxy(
                self.calibrated_tf_reload_service, Trigger)()
            message = str(response.message or '')
            self.state.set_action_status(
                ('手眼外参已刷新：' if response.success else '手眼外参刷新失败：') +
                message)
            return bool(response.success), message
        except Exception as error:
            message = '手眼外参刷新服务不可用：%s' % error
            self.state.set_action_status(message)
            return False, message

    def use_current_or_latest_calibration(self):
        """Reuse live calibration, or install the newest valid matching one."""
        try:
            current = read_install_status(self.calibration_config_file)
            expected_serial = str(
                self.camera_serial or current.get('camera_serial', '') or '')
            latest = find_latest_quality_candidate(
                self.calibration_data_dir, expected_serial=expected_serial)
            current_valid = bool(
                current.get('configured') and current.get('quality_passed') and
                current.get('installed') and
                (not expected_serial or
                 current.get('camera_serial') == expected_serial))
            current_source = os.path.abspath(os.path.expanduser(
                str(current.get('source_candidate', '') or ''))) \
                if current.get('source_candidate') else ''
            latest_path = str((latest or {}).get('path', '') or '')
            installed = False
            if latest_path and (not current_valid or
                                os.path.abspath(latest_path) != current_source):
                install_candidate(
                    latest_path, self.calibration_config_file,
                    expected_serial=expected_serial,
                    installer='dashboard_reuse_latest')
                installed = True
            elif not current_valid:
                raise ValueError('当前 live 外参无效，且没有找到序列号匹配的质量通过候选')
            reload_ok, reload_message = self.reload_calibrated_tf()
            if not reload_ok:
                raise RuntimeError(reload_message)
            status = self.refresh_nbv_status()
            calibration = status.get('calibration') or {}
            message = (
                ('已安装并使用最新质量通过标定' if installed else
                 '已直接使用当前已安装标定') +
                '：序列号 %s，日期 %s；不需要重新拍摄 Charuco' %
                (calibration.get('camera_serial') or expected_serial or '<missing>',
                 calibration.get('installed_at') or
                 (latest or {}).get('calibration_date') or '<missing>'))
            self.state.set_action_status(message)
            self.state.append_event('[CALIBRATION] ' + message)
            return True, message
        except Exception as error:
            message = '使用现有标定失败：%s' % error
            self.state.set_action_status(message)
            self.state.append_event('[CALIBRATION] ' + message)
            return False, message

    def calibration_running(self):
        with self.calibration_lock:
            return bool(self.calibration_process is not None and
                        self.calibration_process.poll() is None)

    def save_capture(self):
        snapshot = self.state.snapshot()
        color = snapshot['images']['color']
        annotated = snapshot['images']['annotated']
        depth = snapshot['raw_depth']
        if color is None or depth is None:
            self.state.set_action_status('保存失败：尚未收到完整 RGB-D 画面')
            return
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        target_dir = os.path.join(self.output_dir, stamp)
        try:
            os.makedirs(target_dir, exist_ok=False)
            cv2.imwrite(os.path.join(target_dir, 'color.png'), color)
            cv2.imwrite(os.path.join(target_dir, 'depth_raw.png'), depth)
            if annotated is not None:
                cv2.imwrite(os.path.join(target_dir, 'annotated.png'), annotated)
            metadata = {
                'captured_at': datetime.now().isoformat(timespec='seconds'),
                'vision_status': snapshot['vision_status'],
                'planner_status': snapshot['planner_status'],
                'depth_unit': 'millimetres for 16-bit RealSense input',
                'execution_requested': False,
            }
            with open(os.path.join(target_dir, 'metadata.json'), 'w',
                      encoding='utf-8') as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2)
                handle.write('\n')
            self.state.set_action_status('RGB-D 样本已保存：' + target_dir)
        except Exception as error:
            self.state.set_action_status('保存失败：' + str(error))


class ImageView(wx.Panel):

    def __init__(self, parent, title):
        wx.Panel.__init__(self, parent)
        self.bgr = None
        self.title = wx.StaticText(self, label=title)
        title_font = self.title.GetFont()
        title_font.SetWeight(wx.FONTWEIGHT_BOLD)
        self.title.SetFont(title_font)
        self.bitmap = wx.StaticBitmap(self, size=(360, 240))
        # Yield vertical space to diagnostics on 1366x768 displays. The
        # source image is still aspect-fitted into whatever area is available.
        self.bitmap.SetMinSize((240, 145))
        self.bitmap.SetBackgroundColour(wx.Colour(24, 28, 32))
        self.resize_later = None
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.title, 0, wx.BOTTOM, 6)
        sizer.Add(self.bitmap, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Bind(wx.EVT_SIZE, self.on_resize)

    def set_bgr(self, image):
        self.bgr = image
        self.refresh_bitmap()

    def clear(self):
        self.bgr = np.zeros((225, 300, 3), dtype=np.uint8)
        self.refresh_bitmap()

    def on_resize(self, event):
        event.Skip()
        if self.bgr is not None:
            if self.resize_later is not None:
                self.resize_later.Stop()
            self.resize_later = wx.CallLater(80, self._refresh_after_resize)

    def _refresh_after_resize(self):
        self.resize_later = None
        self.refresh_bitmap()

    def refresh_bitmap(self):
        if self.bgr is None or not self.bitmap:
            return
        width, height = self.bitmap.GetClientSize()
        if width < 2 or height < 2:
            return
        source_height, source_width = self.bgr.shape[:2]
        scale = min(float(width) / source_width, float(height) / source_height)
        draw_width = max(1, int(source_width * scale))
        draw_height = max(1, int(source_height * scale))
        resized = cv2.resize(self.bgr, (draw_width, draw_height),
                             interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        x = (width - draw_width) // 2
        y = (height - draw_height) // 2
        canvas[y:y + draw_height, x:x + draw_width] = rgb
        wx_image = wx.Image(width, height)
        wx_image.SetData(np.ascontiguousarray(canvas).tobytes())
        self.bitmap.SetBitmap(wx.Bitmap(wx_image))


class RuntimeFieldRow(wx.Panel):
    """One schema-driven runtime setting with synchronized slider and input."""

    def __init__(self, parent, key, spec):
        wx.Panel.__init__(self, parent)
        self.key = key
        self.spec = dict(spec)
        self.scale = float(self.spec.get('display_scale', 1.0))
        self.value_type = self.spec.get('type', 'float')
        self.nullable = bool(self.spec.get('nullable', False))
        self._updating = False
        self.limit_check = None
        self.slider = None
        self.spin = None

        tooltip = str(self.spec.get('help', '')).strip()
        source_hint = str(self.spec.get('source_hint', '')).strip()
        if source_hint:
            tooltip += ('\n' if tooltip else '') + '源码位置：' + source_hint

        layout = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(self, label=str(self.spec.get('label', key)))
        label.SetMinSize((150, -1))
        label.SetToolTip(tooltip)
        layout.Add(label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)

        if self.value_type == 'bool':
            self.check = wx.CheckBox(self, label='启用')
            self.check.SetToolTip(tooltip)
            layout.Add(self.check, 0, wx.ALIGN_CENTER_VERTICAL)
            layout.AddStretchSpacer(1)
            self.SetSizer(layout)
            return

        minimum = float(self.spec['minimum']) * self.scale
        maximum = float(self.spec['maximum']) * self.scale
        step = float(self.spec.get('step') or 1.0) * self.scale
        self.display_minimum = minimum
        self.display_maximum = maximum
        self.display_step = step
        self.position_count = max(1, int(round((maximum - minimum) / step)))

        if self.nullable:
            self.limit_check = wx.CheckBox(self, label='启用上限')
            self.limit_check.SetToolTip(tooltip)
            self.limit_check.Bind(wx.EVT_CHECKBOX, self._on_limit_toggle)
            layout.Add(self.limit_check, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        else:
            layout.AddSpacer(80)

        self.slider = wx.Slider(
            self, value=0, minValue=0, maxValue=self.position_count,
            style=wx.SL_HORIZONTAL)
        self.slider.SetMinSize((250, -1))
        self.slider.SetToolTip(tooltip)
        layout.Add(self.slider, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)

        self.spin = wx.TextCtrl(
            self, value=self._format_display(minimum), size=(108, -1),
            style=wx.TE_PROCESS_ENTER | wx.TE_RIGHT)
        self.spin.Bind(wx.EVT_TEXT, self._on_spin)
        self.spin.SetToolTip(tooltip)
        self.slider.Bind(wx.EVT_SLIDER, self._on_slider)
        layout.Add(self.spin, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)

        unit = wx.StaticText(self, label=str(self.spec.get('unit', '')))
        unit.SetMinSize((32, -1))
        unit.SetToolTip(tooltip)
        layout.Add(unit, 0, wx.ALIGN_CENTER_VERTICAL)
        self.SetSizer(layout)

    def _display_from_position(self, position):
        value = self.display_minimum + int(position) * self.display_step
        return min(self.display_maximum, max(self.display_minimum, value))

    def _position_from_display(self, value):
        position = int(round(
            (float(value) - self.display_minimum) / self.display_step))
        return min(self.position_count, max(0, position))

    def _set_numeric_enabled(self, enabled):
        self.slider.Enable(bool(enabled))
        self.spin.Enable(bool(enabled))

    def _format_display(self, value):
        if self.value_type == 'int':
            return str(int(round(value)))
        decimals = int(self.spec.get('decimals', 3))
        return ('%%.%df' % decimals) % float(value)

    def _parsed_display(self):
        text = self.spin.GetValue().strip()
        try:
            value = float(text)
        except ValueError:
            raise ValueError('%s 不是有效数字' % self.spec.get('label', self.key))
        if not np.isfinite(value):
            raise ValueError('%s 不是有限数字' % self.spec.get('label', self.key))
        if value < self.display_minimum or value > self.display_maximum:
            raise ValueError('%s 必须在 %s 到 %s 之间' % (
                self.spec.get('label', self.key),
                self._format_display(self.display_minimum),
                self._format_display(self.display_maximum)))
        if self.value_type == 'int' and value != int(value):
            raise ValueError('%s 必须为整数' % self.spec.get('label', self.key))
        return value

    def _on_limit_toggle(self, _event):
        self._set_numeric_enabled(self.limit_check.IsChecked())

    def _on_slider(self, _event):
        if self._updating:
            return
        self._updating = True
        try:
            value = self._display_from_position(self.slider.GetValue())
            self.spin.ChangeValue(self._format_display(value))
        finally:
            self._updating = False

    def _on_spin(self, _event):
        if self._updating:
            return
        try:
            value = self._parsed_display()
        except ValueError:
            return
        self._updating = True
        try:
            self.slider.SetValue(self._position_from_display(value))
        finally:
            self._updating = False

    def set_value(self, value):
        if self.value_type == 'bool':
            self.check.SetValue(bool(value))
            return
        enabled = value is not None
        if self.limit_check is not None:
            self.limit_check.SetValue(enabled)
        display = self.display_minimum if value is None else float(value) * self.scale
        display = min(self.display_maximum, max(self.display_minimum, display))
        self._updating = True
        try:
            self.spin.ChangeValue(self._format_display(display))
            self.slider.SetValue(self._position_from_display(display))
        finally:
            self._updating = False
        self._set_numeric_enabled(enabled or not self.nullable)

    def get_value(self):
        if self.value_type == 'bool':
            return bool(self.check.IsChecked())
        if self.limit_check is not None and not self.limit_check.IsChecked():
            return None
        value = self._parsed_display() / self.scale
        if self.value_type == 'int':
            return int(round(value))
        return round(value, 12)


class ExperimentConsoleDialog(wx.Dialog):
    """Modeless staged-operation and runtime-parameter console."""

    STAGES = (
        ('demo_lock', '1  锁存目标',
         '锁存当前稳定目标和随后到达的环境点云；不移动机械臂。'),
        ('demo_preview', '2  完整预览',
         '计算预抓取、直线伸入和撤回三段路径，只在 RViz 显示。'),
        ('demo_pregrasp', '3  前往预抓取',
         '执行第一段真机轨迹；碰撞、控制器与硬件状态门禁保持生效。'),
        ('demo_insert', '4  直线伸入',
         '从预抓取点沿法兰法线执行碰撞检查后的直线伸入。'),
        ('demo_tool', '5  末端动作',
         '在最终夹取位触发一次 DO0 上升沿，并等待末端固定流程。'),
        ('demo_retreat', '6  原路撤回',
         '重新检查反向轨迹的碰撞和当前起点后，撤回预抓取点。'),
        ('demo_unlock', '7  解除锁存',
         '清除演示目标；在最终夹取位时必须先撤回。'),
        ('stop', '停止当前流程',
         '请求采摘状态机和 MoveIt 停止当前阶段；硬件急停仍由主 Panel 负责。'),
    )

    STAGE_NAMES = {
        'UNLOCKED': '未锁存',
        'LOCKED': '目标已锁存',
        'PREGRASP': '位于预抓取点',
        'INSERTED': '位于最终夹取点',
        'TOOL_DONE': '末端动作已完成',
        'RETREATED': '已撤回预抓取点',
    }

    def __init__(self, owner):
        wx.Dialog.__init__(
            self, owner, title='视觉采摘实验控制台', size=(1040, 760),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.SetMinSize((850, 620))
        self.owner = owner
        self.ros_bridge = owner.ros_bridge
        self.closed = False
        self.request_in_flight = False
        self.harvest_busy = False
        self.schema = {}
        self.field_rows = {}

        root = wx.BoxSizer(wx.VERTICAL)
        self.notebook = wx.Notebook(self)
        self.stage_page = self._build_stage_page(self.notebook)
        self.parameter_page = self._build_parameter_page(self.notebook)
        self.notebook.AddPage(self.stage_page, '分阶段演示')
        self.notebook.AddPage(self.parameter_page, '运行参数')
        self.notebook.SetSelection(1)
        root.Add(self.notebook, 1, wx.ALL | wx.EXPAND, 10)

        close_button = wx.Button(self, wx.ID_CLOSE, label='关闭')
        close_button.Bind(wx.EVT_BUTTON, lambda _event: self.Close())
        footer = wx.BoxSizer(wx.HORIZONTAL)
        footer.AddStretchSpacer(1)
        footer.Add(close_button, 0)
        root.Add(footer, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)
        self.SetSizer(root)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self.Bind(wx.EVT_CHAR_HOOK, self.on_char_hook)
        self._request_runtime('get')

    def _build_stage_page(self, parent):
        page = wx.Panel(parent)
        layout = wx.BoxSizer(wx.VERTICAL)
        self.stage_status = StableReadOnlyTextCtrl(page)
        self.stage_status.SetMinSize((-1, 125))
        layout.Add(self.stage_status, 0, wx.ALL | wx.EXPAND, 12)

        grid = wx.GridSizer(rows=2, cols=4, vgap=10, hgap=10)
        self.stage_buttons = {}
        for command, label, tooltip in self.STAGES:
            button = wx.Button(page, label=label, size=(175, 66))
            button.SetToolTip(tooltip)
            button.Bind(
                wx.EVT_BUTTON,
                lambda _event, selected=command: self._on_stage(selected))
            self.stage_buttons[command] = button
            grid.Add(button, 1, wx.EXPAND)
        layout.Add(grid, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 12)
        log_box = wx.StaticBoxSizer(wx.VERTICAL, page, '采摘流程日志')
        self.stage_log = ProtectedLogView(page, minimum_size=(-1, 175))
        log_box.Add(self.stage_log, 1, wx.EXPAND)
        layout.Add(log_box, 1, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 12)
        page.SetSizer(layout)
        return page

    def _build_parameter_page(self, parent):
        page = wx.Panel(parent)
        layout = wx.BoxSizer(wx.VERTICAL)

        profile_row = wx.BoxSizer(wx.HORIZONTAL)
        profile_row.Add(wx.StaticText(page, label='保存位置'), 0,
                        wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        self.profile_path = wx.TextCtrl(page, style=wx.TE_READONLY)
        profile_row.Add(self.profile_path, 1, wx.EXPAND)
        layout.Add(profile_row, 0, wx.ALL | wx.EXPAND, 10)

        self.group_book = wx.Notebook(page)
        loading = wx.Panel(self.group_book)
        loading_layout = wx.BoxSizer(wx.VERTICAL)
        loading_layout.AddStretchSpacer(1)
        loading_layout.Add(wx.StaticText(loading, label='正在读取运行参数...'),
                           0, wx.ALIGN_CENTER_HORIZONTAL)
        loading_layout.AddStretchSpacer(1)
        loading.SetSizer(loading_layout)
        self.group_book.AddPage(loading, '加载中')
        layout.Add(self.group_book, 1, wx.LEFT | wx.RIGHT | wx.EXPAND, 10)

        self.parameter_status = wx.StaticText(page, label='')
        layout.Add(self.parameter_status, 0, wx.ALL | wx.EXPAND, 10)

        buttons = wx.BoxSizer(wx.HORIZONTAL)
        self.apply_button = wx.Button(page, label='临时应用', size=(135, 42))
        self.save_button = wx.Button(page, label='应用并保存', size=(135, 42))
        self.reload_button = wx.Button(page, label='重新载入', size=(135, 42))
        self.reset_button = wx.Button(page, label='恢复项目默认', size=(145, 42))
        self.apply_button.SetToolTip('应用到当前进程；重启后恢复上次保存值。')
        self.save_button.SetToolTip('应用到当前进程，并写入下次启动自动加载的用户配置。')
        self.reload_button.SetToolTip('放弃界面未应用修改，重新读取已保存的用户配置。')
        self.reset_button.SetToolTip('删除用户覆盖文件并恢复项目 YAML 默认值。')
        for button in (self.apply_button, self.save_button,
                       self.reload_button, self.reset_button):
            buttons.Add(button, 0, wx.RIGHT, 8)
        buttons.AddStretchSpacer(1)
        layout.Add(buttons, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)

        self.apply_button.Bind(wx.EVT_BUTTON,
                               lambda _event: self._apply_values('apply'))
        self.save_button.Bind(wx.EVT_BUTTON,
                              lambda _event: self._apply_values('save'))
        self.reload_button.Bind(wx.EVT_BUTTON,
                                lambda _event: self._request_runtime('reload'))
        self.reset_button.Bind(wx.EVT_BUTTON, self._on_reset)
        page.SetSizer(layout)
        self._refresh_parameter_buttons()
        return page

    def _build_parameter_groups(self, schema):
        self.group_book.DeleteAllPages()
        self.field_rows = {}
        grouped = []
        by_group = {}
        for key, spec in schema.items():
            group = str(spec.get('group', '其他'))
            if group not in by_group:
                grouped.append(group)
                by_group[group] = []
            by_group[group].append((key, spec))

        for group in grouped:
            scroll = wx.ScrolledWindow(self.group_book, style=wx.VSCROLL)
            scroll.SetScrollRate(0, 12)
            rows = wx.BoxSizer(wx.VERTICAL)
            for index, (key, spec) in enumerate(by_group[group]):
                row = RuntimeFieldRow(scroll, key, spec)
                if index % 2:
                    row.SetBackgroundColour(wx.Colour(246, 248, 250))
                rows.Add(row, 0, wx.ALL | wx.EXPAND, 7)
                self.field_rows[key] = row
            rows.AddStretchSpacer(1)
            scroll.SetSizer(rows)
            scroll.FitInside()
            self.group_book.AddPage(scroll, group)
        self.parameter_page.Layout()

    def _set_values(self, values):
        for key, row in self.field_rows.items():
            if key in values:
                row.set_value(values[key])

    def _collect_values(self):
        return {key: row.get_value() for key, row in self.field_rows.items()}

    def _refresh_parameter_buttons(self):
        enabled = bool(self.field_rows) and not self.request_in_flight \
            and not self.harvest_busy
        for button in (getattr(self, 'apply_button', None),
                       getattr(self, 'save_button', None),
                       getattr(self, 'reload_button', None),
                       getattr(self, 'reset_button', None)):
            if button is not None:
                button.Enable(enabled)

    def _request_runtime(self, operation, values=None):
        if self.request_in_flight:
            return
        self.request_in_flight = True
        self.parameter_status.SetLabel('正在请求：%s' % operation)
        self._refresh_parameter_buttons()

        def worker():
            result = self.ros_bridge.request_runtime_config(operation, values)
            wx.CallAfter(self._finish_runtime_request, result)

        threading.Thread(target=worker, daemon=True).start()

    def _finish_runtime_request(self, result):
        if self.closed:
            return
        self.request_in_flight = False
        schema = result.get('schema') or {}
        if schema and schema != self.schema:
            self.schema = schema
            self._build_parameter_groups(schema)
        values = result.get('values') or {}
        if values and self.field_rows:
            self._set_values(values)
        self.profile_path.SetValue(result.get('profile_path') or '')
        self.parameter_status.SetLabel(result.get('message') or '')
        colour = wx.Colour(25, 110, 50) if result.get('success') \
            else wx.Colour(175, 35, 35)
        self.parameter_status.SetForegroundColour(colour)
        self._refresh_parameter_buttons()
        self.parameter_page.Layout()

    def _apply_values(self, operation):
        if self.harvest_busy:
            self.parameter_status.SetLabel('采摘阶段正在运行，参数未修改')
            return
        try:
            values = self._collect_values()
        except ValueError as error:
            self.parameter_status.SetLabel(str(error))
            self.parameter_status.SetForegroundColour(wx.Colour(175, 35, 35))
            return
        self._request_runtime(operation, values)

    def _on_reset(self, _event):
        answer = wx.MessageBox(
            '将删除用户运行参数覆盖文件，并立即恢复项目 YAML 默认值。继续？',
            '恢复项目默认参数', wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION,
            parent=self)
        if answer == wx.YES:
            self._request_runtime('reset')

    def _on_stage(self, command):
        if command == 'demo_tool':
            answer = wx.MessageBox(
                '将立即触发一次 OUTPUT0 上升沿，夹爪、剪刀和释放流程会真实动作。继续？',
                '确认末端动作', wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING,
                parent=self)
            if answer != wx.YES:
                return
        self.owner.start_demo_command(command)

    def update_status(self, snapshot, selected_valid):
        self.stage_log.update_lines(snapshot.get('events') or [])
        harvest = snapshot.get('harvest_status')
        if harvest is None:
            self.stage_status.SetValue('采摘状态机：未启动\n分阶段命令：不可用')
            for button in self.stage_buttons.values():
                button.Enable(False)
            self.harvest_busy = False
            self._refresh_parameter_buttons()
            return

        stage = str(harvest.manual_stage or 'UNLOCKED')
        locked = bool(harvest.target_locked)
        busy = bool(harvest.busy)
        requested = bool(harvest.execution_requested)
        ready = bool(harvest.execution_ready)
        blockers = '；'.join(harvest.blockers) if harvest.blockers else '无'
        text = (
            '当前阶段：%s（%s）\n'
            '流程状态：%s / %s；%s\n'
            '真机执行：%s；门禁：%s'
        ) % (self.STAGE_NAMES.get(stage, stage),
             '目标已锁存' if locked else '未锁存',
             harvest.mode, harvest.phase, harvest.message,
             '已授权' if requested else '只规划', blockers)
        if self.stage_status.GetValue() != text:
            self.stage_status.SetValue(text)

        self.stage_buttons['demo_lock'].Enable(
            selected_valid and not locked and not busy)
        self.stage_buttons['demo_preview'].Enable(
            (locked or selected_valid) and not busy)
        self.stage_buttons['demo_pregrasp'].Enable(
            locked and requested and ready and not busy and
            stage in ('LOCKED', 'PREGRASP'))
        self.stage_buttons['demo_insert'].Enable(
            locked and requested and ready and not busy and stage == 'PREGRASP')
        self.stage_buttons['demo_tool'].Enable(
            locked and requested and ready and not busy and stage == 'INSERTED')
        self.stage_buttons['demo_retreat'].Enable(
            locked and requested and ready and not busy and
            stage in ('INSERTED', 'TOOL_DONE'))
        self.stage_buttons['demo_unlock'].Enable(
            locked and not busy and stage not in ('INSERTED', 'TOOL_DONE'))
        self.stage_buttons['stop'].Enable(busy)
        self.harvest_busy = busy
        self._refresh_parameter_buttons()

    def _on_close(self, event):
        self.closed = True
        self.owner.experiment_dialog = None
        event.Skip()

    def on_char_hook(self, event):
        if event.ControlDown() and int(event.GetKeyCode()) in (
                ord('C'), ord('c')):
            self.owner.Close()
            return
        event.Skip()


class CockpitDialog(wx.Dialog):
    """Modeless, focus-local keyboard cockpit with a strict input watchdog."""

    KEY_ACTIONS = {
        ord('W'): 'forward',
        ord('S'): 'back',
        ord('A'): 'left',
        ord('D'): 'right',
        ord('w'): 'forward',
        ord('s'): 'back',
        ord('a'): 'left',
        ord('d'): 'right',
        ord('Q'): 'roll_left',
        ord('E'): 'roll_right',
        ord('q'): 'roll_left',
        ord('e'): 'roll_right',
        # The installed base axis turns opposite to the original keyboard
        # legend. Keep the semantic jog directions unchanged and swap only
        # the operator-facing Z/X bindings.
        ord('Z'): 'base_right',
        ord('X'): 'base_left',
        ord('z'): 'base_right',
        ord('x'): 'base_left',
        wx.WXK_LEFT: 'yaw_left',
        wx.WXK_RIGHT: 'yaw_right',
        wx.WXK_UP: 'pitch_up',
        wx.WXK_DOWN: 'pitch_down',
        wx.WXK_SHIFT: 'up',
        wx.WXK_CONTROL: 'down',
    }
    ACTION_LABELS = {
        'forward': 'W', 'back': 'S', 'left': 'A', 'right': 'D',
        'yaw_left': 'LEFT', 'yaw_right': 'RIGHT',
        'pitch_up': 'UP', 'pitch_down': 'DOWN',
        'roll_left': 'Q', 'roll_right': 'E',
        'base_left': 'X/J1-', 'base_right': 'Z/J1+',
        'up': 'SHIFT/UP', 'down': 'CTRL/DOWN',
    }

    def __init__(self, owner):
        wx.Dialog.__init__(
            self, owner, title='Elfin E05 驾驶舱', size=(980, 780),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.SetMinSize((820, 620))
        self.owner = owner
        self.state = owner.state
        self.ros_bridge = owner.ros_bridge
        self.closed = False
        self.activation_inflight = False
        self.activation_succeeded = False
        self.activation_paused = False
        self.activation_retry_at = 0.0
        self.input_state = FocusedCockpitInput()
        self.input_speed_scale = max(
            0.01, min(1.0, float(owner.cockpit_speed_percent) / 100.0))
        self.seen_image_version = -1
        self.speed_apply_later = None
        self.recovery_speed_apply_later = None
        self.operation_message = '等待操作'
        self.tool_keys_held = set()
        self.nbv_enter_gate = SinglePressGate()

        outer = wx.BoxSizer(wx.VERTICAL)
        self.camera_view = ImageView(self, '驾驶舱视野')
        self.camera_view.SetMinSize((-1, 235))
        outer.Add(self.camera_view, 1, wx.ALL | wx.EXPAND, 10)

        controls = wx.BoxSizer(wx.HORIZONTAL)
        controls.Add(self._direction_group(), 0, wx.RIGHT, 12)
        controls.Add(self._view_group(), 0, wx.RIGHT, 12)
        controls.Add(self._mode_group(), 1, wx.EXPAND)
        outer.Add(controls, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)

        paper_row = wx.BoxSizer(wx.VERTICAL)
        paper_controls = wx.BoxSizer(wx.HORIZONTAL)
        self.nbv_button = wx.ToggleButton(
            self, label='论文采集 OFF', size=(145, 42))
        self.nbv_button.SetToolTip(
            '开启后按 Enter 或机械臂实体 POINT，记录一张经过 RGB-D、TF 和关节新鲜度校验的视角')
        paper_controls.Add(self.nbv_button, 0, wx.RIGHT, 8)
        paper_controls.AddStretchSpacer(1)
        paper_row.Add(paper_controls, 0, wx.EXPAND)
        self.nbv_status = wx.StaticText(
            self, label='论文采集未开启；Enter/实体POINT 不会保存单张样本', size=(-1, 128),
            style=wx.ST_NO_AUTORESIZE)
        self.nbv_status.Wrap(640)
        paper_row.Add(self.nbv_status, 0, wx.TOP | wx.EXPAND, 6)
        outer.Add(paper_row, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)

        self.status = StableReadOnlyTextCtrl(self)
        self.status.SetMinSize((-1, 125))
        outer.Add(self.status, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)
        self.SetSizer(outer)

        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Bind(wx.EVT_ACTIVATE, self.on_activate)
        self.Bind(wx.EVT_ICONIZE, self.on_iconize)
        # EVT_CHAR_HOOK sees key-down events even while a slider, button, or
        # read-only status field owns focus.  Key-up is bound to every child
        # because wx key events do not reliably propagate to a parent.
        self.Bind(wx.EVT_CHAR_HOOK, self.on_key_down)
        self._bind_key_up(self)
        self.nbv_button.Bind(wx.EVT_TOGGLEBUTTON, self.on_nbv_toggle)
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)
        # Image/status work stays at 10 Hz. Motion itself is event-driven:
        # one Panel teleop request on key-down and one Stop on key-up.
        self.timer.Start(100)

    @staticmethod
    def _key_badge(parent, label):
        badge = wx.StaticText(
            parent, label=label, size=(70, 62),
            style=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL |
            wx.ST_NO_AUTORESIZE | wx.BORDER_SIMPLE)
        badge.SetToolTip('键盘提示；不会捕获鼠标')
        return badge

    def _direction_group(self):
        box = wx.StaticBoxSizer(wx.VERTICAL, self, '平移')
        panel = box.GetStaticBox()
        grid = wx.GridSizer(rows=3, cols=3, vgap=6, hgap=6)
        grid.Add(self._key_badge(panel, 'SHIFT\n升'), 0, wx.EXPAND)
        grid.Add(self._key_badge(panel, 'W'), 0, wx.EXPAND)
        grid.Add(70, 62)
        grid.Add(self._key_badge(panel, 'A'), 0, wx.EXPAND)
        grid.Add(self._key_badge(panel, 'S'), 0, wx.EXPAND)
        grid.Add(self._key_badge(panel, 'D'), 0, wx.EXPAND)
        grid.Add(70, 62)
        grid.Add(self._key_badge(panel, 'CTRL\n降'), 0, wx.EXPAND)
        grid.Add(70, 62)
        box.Add(grid, 0, wx.ALL, 8)
        return box

    def _view_group(self):
        box = wx.StaticBoxSizer(wx.VERTICAL, self, '法兰原点转头（与 Z/X 分离）')
        panel = box.GetStaticBox()
        grid = wx.GridSizer(rows=2, cols=3, vgap=6, hgap=6)
        grid.Add(self._key_badge(panel, 'Q ROLL'), 0, wx.EXPAND)
        self.view_up_button = wx.Button(panel, label='↑ 抬头', size=(86, 62))
        grid.Add(self.view_up_button, 0, wx.EXPAND)
        grid.Add(self._key_badge(panel, 'E ROLL'), 0, wx.EXPAND)
        self.view_left_button = wx.Button(panel, label='← 左看', size=(86, 62))
        self.view_down_button = wx.Button(panel, label='↓ 低头', size=(86, 62))
        self.view_right_button = wx.Button(panel, label='→ 右看', size=(86, 62))
        grid.Add(self.view_left_button, 0, wx.EXPAND)
        grid.Add(self.view_down_button, 0, wx.EXPAND)
        grid.Add(self.view_right_button, 0, wx.EXPAND)
        for button in (self.view_left_button, self.view_right_button,
                       self.view_up_button, self.view_down_button):
            button.SetToolTip(
                '保持法兰位置不变，只绕法兰原点改变视角；不会调用 Z/X 的 J1 直接点动')
        self._bind_hold_button(self.view_left_button, 'yaw_left')
        self._bind_hold_button(self.view_right_button, 'yaw_right')
        self._bind_hold_button(self.view_up_button, 'pitch_up')
        self._bind_hold_button(self.view_down_button, 'pitch_down')
        box.Add(grid, 0, wx.ALL, 8)
        return box

    def _mode_group(self):
        box = wx.StaticBoxSizer(wx.VERTICAL, self, '模式')
        panel = box.GetStaticBox()
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.cobra_button = wx.Button(
            panel, label='蛇形观察·左', size=(135, 58))
        row.Add(self.cobra_button, 0)
        self.reverse_cobra_button = wx.Button(
            panel, label='蛇形观察·右', size=(135, 58))
        row.Add(self.reverse_cobra_button, 0, wx.LEFT, 6)
        for button in (self.cobra_button, self.reverse_cobra_button):
            button.SetToolTip(
                '复用 AUTO 的法兰原点短步：后撤、抬升、俯视并侧移；'
                '保持驾驶舱接管，不调用 MoveIt 眼镜蛇姿态或 J1')
        self.tool_button = wx.Button(
            panel, label='J · 闭合', size=(135, 58))
        self.tool_button.SetToolTip(
            '发送 DO0 高电平 50 ms，只闭合；两次信号之间至少保持 10 ms 低电平')
        row.Add(self.tool_button, 0, wx.LEFT, 6)
        self.open_tool_button = wx.Button(
            panel, label='K · 张开', size=(135, 58))
        self.open_tool_button.SetToolTip(
            '发送 DO0 高电平约 110 ms，只张开；两次信号之间至少保持 10 ms 低电平')
        row.Add(self.open_tool_button, 0, wx.LEFT, 6)
        box.Add(row, 0, wx.ALL, 8)

        scan_row = wx.BoxSizer(wx.HORIZONTAL)
        self.base_left_button = wx.Button(
            panel, label='按住 X · J1-', size=(145, 48))
        self.base_right_button = wx.Button(
            panel, label='按住 Z · J1+', size=(145, 48))
        self.base_left_button.SetToolTip(
            '可按住鼠标或键盘 X；按下一次 Start，松开立即发送 Panel STOP')
        self.base_right_button.SetToolTip(
            '可按住鼠标或键盘 Z；按下一次 Start，松开立即发送 Panel STOP')
        scan_row.Add(self.base_left_button, 0, wx.RIGHT, 8)
        scan_row.Add(self.base_right_button, 0)
        box.Add(scan_row, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        speed_row = wx.BoxSizer(wx.HORIZONTAL)
        speed_row.Add(wx.StaticText(panel, label='按键点动速度'), 0,
                      wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        initial = int(self.owner.cockpit_speed_percent)
        self.speed_slider = wx.Slider(
            panel, value=initial, minValue=1, maxValue=100,
            size=(190, -1))
        self.speed_label = wx.StaticText(panel, label='%d%%' % initial,
                                         size=(48, -1))
        speed_row.Add(self.speed_slider, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        speed_row.Add(self.speed_label, 0, wx.ALIGN_CENTER_VERTICAL)
        box.Add(speed_row, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 8)

        plan_speed_row = wx.BoxSizer(wx.HORIZONTAL)
        plan_speed_row.Add(
            wx.StaticText(panel, label='采摘规划速度'), 0,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        plan_initial = int(self.owner.global_speed_percent)
        self.plan_speed_slider = wx.Slider(
            panel, value=plan_initial, minValue=1, maxValue=100,
            size=(190, -1))
        self.plan_speed_label = wx.StaticText(
            panel, label='%d%%' % plan_initial, size=(48, -1))
        plan_speed_row.Add(
            self.plan_speed_slider, 1,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        plan_speed_row.Add(
            self.plan_speed_label, 0, wx.ALIGN_CENTER_VERTICAL)
        box.Add(
            plan_speed_row, 0,
            wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 8)

        recovery_speed_row = wx.BoxSizer(wx.HORIZONTAL)
        recovery_speed_row.Add(
            wx.StaticText(panel, label='主动脱困速度'), 0,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        recovery_initial = int(self.owner.recovery_speed_percent)
        self.recovery_speed_slider = wx.Slider(
            panel, value=recovery_initial, minValue=1, maxValue=100,
            size=(190, -1))
        self.recovery_speed_label = wx.StaticText(
            panel, label='%d%%' % recovery_initial, size=(48, -1))
        recovery_speed_row.Add(
            self.recovery_speed_slider, 1,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        recovery_speed_row.Add(
            self.recovery_speed_label, 0, wx.ALIGN_CENTER_VERTICAL)
        box.Add(
            recovery_speed_row, 0,
            wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 8)

        self.cobra_button.Bind(wx.EVT_BUTTON, self.on_cobra)
        self.reverse_cobra_button.Bind(
            wx.EVT_BUTTON, self.on_reverse_cobra)
        self.tool_button.Bind(
            wx.EVT_BUTTON, lambda event: self.on_tool(event, 'close'))
        self.open_tool_button.Bind(
            wx.EVT_BUTTON, lambda event: self.on_tool(event, 'open'))
        self._bind_hold_button(self.base_left_button, 'base_left')
        self._bind_hold_button(self.base_right_button, 'base_right')
        self.speed_slider.Bind(wx.EVT_SLIDER, self.on_speed)
        self.plan_speed_slider.Bind(
            wx.EVT_SLIDER, self.on_plan_speed)
        self.recovery_speed_slider.Bind(
            wx.EVT_SLIDER, self.on_recovery_speed)
        return box

    def _bind_hold_button(self, button, action):
        button.Bind(
            wx.EVT_LEFT_DOWN,
            lambda event: self._on_hold_down(event, button, action))
        button.Bind(
            wx.EVT_LEFT_UP,
            lambda event: self._on_hold_up(event, button, action))
        button.Bind(
            wx.EVT_MOUSE_CAPTURE_LOST,
            lambda event: self._on_hold_lost(event, action))

    def _on_hold_down(self, event, button, action):
        self.input_state.set_focused(True)
        if self.input_state.press(action):
            self.ros_bridge.set_cockpit_actions(
                self.input_state.pressed_actions())
            self.operation_message = '鼠标按下：%s' % action
        if not button.HasCapture():
            button.CaptureMouse()

    def _on_hold_up(self, event, button, action):
        if self.input_state.release(action):
            self.ros_bridge.set_cockpit_actions(
                self.input_state.pressed_actions())
        self.operation_message = '鼠标松开：已立即请求 STOP'
        if button.HasCapture():
            button.ReleaseMouse()

    def _on_hold_lost(self, _event, action):
        if self.input_state.release(action):
            self.ros_bridge.set_cockpit_actions(
                self.input_state.pressed_actions())
        self.operation_message = '鼠标捕获丢失：已立即请求 STOP'

    def _bind_key_up(self, window):
        window.Bind(wx.EVT_KEY_UP, self.on_key_up)
        for child in window.GetChildren():
            self._bind_key_up(child)

    def _publish_zero(self):
        self.input_state.clear()
        self.tool_keys_held.clear()
        self.nbv_enter_gate.reset()
        self.ros_bridge.set_cockpit_actions(())

    def on_nbv_toggle(self, _event):
        desired = bool(self.nbv_button.GetValue())
        scene_id = self.owner.nbv_scene_text.GetValue().strip()
        notes = self.owner.nbv_notes_text.GetValue().strip()
        self.nbv_button.Enable(False)

        def worker():
            status = self.ros_bridge.set_nbv_mode(desired, scene_id, notes)
            wx.CallAfter(self._nbv_toggle_finished, status)

        threading.Thread(target=worker, name='cockpit-nbv-toggle', daemon=True).start()

    def _nbv_toggle_finished(self, status):
        self.nbv_button.Enable(True)
        self._update_nbv_status(status)

    def _update_nbv_status(self, status):
        status = dict(status or {})
        enabled = bool(status.get('enabled', False))
        self.nbv_button.SetValue(enabled)
        self.nbv_button.SetLabel('论文采集 ON' if enabled else '论文采集 OFF')
        count = int(status.get('view_count', 0))
        maximum = int(status.get('max_views', 10))
        camera = status.get('camera') or {}
        readiness = status.get('readiness') or {}
        camera_text = '%s / %s / USB %s' % (
            camera.get('device_type') or camera.get('name') or '相机未识别',
            camera.get('serial') or '无序列号',
            camera.get('usb_type') or '未知')
        calibration_text = ('标定匹配' if readiness.get(
            'calibration_serial_matches') else '标定未匹配')
        ready_text = 'HW READY' if readiness.get('ready') else 'HW NOT READY'
        semantic_text = ('语义完整门禁' if readiness.get(
            'semantics_required') else '诊断采集')
        imu = status.get('imu') or {}
        imu_text = ('IMU gyro/accel 已收到' if
                    imu.get('gyro_seen') and imu.get('accel_seen') else
                    'IMU 未收到；试拍保留缺口，正式采集请用 --require-imu')
        self.nbv_status.SetLabel(
            '场景 %s / 批次 %04d / 当前视角 %d/%d；按 Enter 或实体 POINT 记录一次。'
            '\n相机 %s；%s；%s；%s。'
            '\n%s。'
            '\n改变距离、偏航或俯仰，覆盖遮挡区域。'
            % (status.get('scene_id') or '未命名',
               int(status.get('group_number', 1)),
               min(count + 1, maximum), maximum,
               camera_text, calibration_text, ready_text, semantic_text,
               imu_text))
        self.nbv_status.Wrap(max(360, self.nbv_status.GetSize().width))

    def on_activate(self, event):
        focused = bool(event.GetActive()) and not self.closed
        self.input_state.set_focused(focused)
        if not focused:
            self._publish_zero()
        event.Skip()

    def on_iconize(self, event):
        if event.IsIconized():
            self.input_state.set_focused(False)
            self._publish_zero()
        event.Skip()

    def on_key_down(self, event):
        if event.ControlDown() and int(event.GetKeyCode()) in (
                ord('C'), ord('c')):
            self.owner.Close()
            return
        key_code = int(event.GetKeyCode())
        enter_keys = {int(getattr(wx, 'WXK_RETURN', 13)),
                      int(getattr(wx, 'WXK_NUMPAD_ENTER', 13))}
        if key_code in enter_keys and self.input_state.is_focused():
            if self.nbv_enter_gate.press():
                self.ros_bridge.capture_nbv_view_async(
                    scene_id=self.owner.nbv_scene_text.GetValue().strip(),
                    notes=self.owner.nbv_notes_text.GetValue().strip())
                self.operation_message = 'Enter：已请求一次论文视角快照'
            return
        tool_key_actions = {
            ord('J'): 'close', ord('j'): 'close',
            ord('K'): 'open', ord('k'): 'open',
        }
        if key_code in tool_key_actions and self.input_state.is_focused():
            if key_code not in self.tool_keys_held:
                self.tool_keys_held.add(key_code)
                self.on_tool(None, tool_key_actions[key_code])
            return
        action = self.KEY_ACTIONS.get(key_code)
        if action is not None:
            if self.input_state.press(action):
                self.ros_bridge.set_cockpit_actions(
                    self.input_state.pressed_actions())
                self.operation_message = '键盘按下：%s' % action
            if self.input_state.is_focused():
                # Consume OS key-repeat locally without creating ROS work.
                return
        event.Skip()

    def on_key_up(self, event):
        key_code = int(event.GetKeyCode())
        enter_keys = {int(getattr(wx, 'WXK_RETURN', 13)),
                      int(getattr(wx, 'WXK_NUMPAD_ENTER', 13))}
        if key_code in enter_keys:
            self.nbv_enter_gate.release()
            if self.input_state.is_focused():
                return
        if key_code in (ord('J'), ord('j'), ord('K'), ord('k')):
            self.tool_keys_held.discard(key_code)
            if self.input_state.is_focused():
                return
        action = self.KEY_ACTIONS.get(key_code)
        if action is not None:
            changed = self.input_state.release(action)
            if self.input_state.is_focused() and changed:
                self.ros_bridge.set_cockpit_actions(
                    self.input_state.pressed_actions())
                self.operation_message = '键盘松开：已立即请求 STOP'
                return
        event.Skip()

    def on_tool(self, _event, action='close'):
        key = 'J' if action == 'close' else 'K'
        self.operation_message = (
            '%s：正在发送 DO0 %s信号（%s）' %
            (key, '闭合' if action == 'close' else '张开',
             '50 ms' if action == 'close' else '110 ms'))

        def worker():
            result = self.ros_bridge.request_harvest_command(
                'cockpit_close' if action == 'close' else 'cockpit_open', -1)
            wx.CallAfter(self._tool_finished, result, action)

        threading.Thread(target=worker, daemon=True).start()

    def _tool_finished(self, result, action='close'):
        key = 'J' if action == 'close' else 'K'
        self.operation_message = (
            (key + ' 已接收：' if result[0] else key + ' 被拒绝：') +
            str(result[1]))

    def _request_activation(self):
        if self.closed or self.activation_paused or self.activation_inflight or \
                self.activation_succeeded:
            return
        self.activation_inflight = True

        def worker():
            result = self.ros_bridge.request_cockpit_active(True)
            wx.CallAfter(self._activation_finished, result)

        threading.Thread(target=worker, daemon=True).start()

    def _activation_finished(self, result):
        self.activation_inflight = False
        success, _message = result
        if self.activation_paused:
            if success:
                threading.Thread(
                    target=self.ros_bridge.request_cockpit_active,
                    args=(False,), daemon=True).start()
            return
        if self.closed:
            if success:
                threading.Thread(
                    target=self.ros_bridge.request_cockpit_active,
                    args=(False,), daemon=True).start()
            return
        if success:
            self.activation_succeeded = True
            self.ros_bridge.set_panel_jog_speed(
                int(round(self.input_speed_scale * 100.0)))
        else:
            # Startup dependencies may arrive in any order. Keep the window
            # responsive and claim control as soon as they are all ready.
            self.activation_retry_at = time.monotonic() + 0.5

    def suspend_for_moveit(self):
        self.activation_paused = True
        self.activation_succeeded = False
        self.input_state.set_focused(False)
        self._publish_zero()
        self.operation_message = '方向规划/执行正在使用 MoveIt；驾驶舱已暂停接管'

    def resume_after_moveit(self):
        if self.closed:
            return
        self.activation_paused = False
        self.activation_succeeded = False
        self.activation_retry_at = time.monotonic()
        self.operation_message = '方向流程已结束；正在恢复驾驶舱接管'

    def on_speed(self, _event):
        value = int(self.speed_slider.GetValue())
        self.speed_label.SetLabel('%d%%' % value)
        self.input_speed_scale = value / 100.0
        self.owner.cockpit_speed_percent = value
        self.ros_bridge.set_panel_jog_speed(value)
        self.operation_message = (
            '按键与蛇形观察速度设为 %d%%；不改变 AUTO/MoveIt 规划速度' % value)

    def on_plan_speed(self, _event):
        value = int(self.plan_speed_slider.GetValue())
        self.plan_speed_label.SetLabel('%d%%' % value)
        self.owner.global_speed_percent = value
        if self.speed_apply_later is not None:
            self.speed_apply_later.Stop()
        self.speed_apply_later = wx.CallLater(
            350, self.owner.apply_global_speed, value)
        self.operation_message = (
            'AUTO/MoveIt 规划速度设为 %d%%；不改变按键与蛇形观察速度' % value)

    def on_recovery_speed(self, _event):
        value = int(self.recovery_speed_slider.GetValue())
        self.recovery_speed_label.SetLabel('%d%%' % value)
        self.owner.recovery_speed_percent = value
        if self.recovery_speed_apply_later is not None:
            self.recovery_speed_apply_later.Stop()
        self.recovery_speed_apply_later = wx.CallLater(
            350, self.owner.apply_recovery_speed, value)
        self.operation_message = (
            '主动脱困速度设为 %d%%；驾驶舱与 AUTO 共用，不改变正常轨迹速度' %
            value)

    def on_cobra(self, _event):
        self._start_cobra(reverse=False)

    def on_reverse_cobra(self, _event):
        self._start_cobra(reverse=True)

    def _start_cobra(self, reverse):
        label = '右侧蛇形观察' if reverse else '左侧蛇形观察'
        answer = wx.MessageBox(
            '将保持驾驶舱接管，通过法兰原点 Panel 短步执行%s：'
            '后撤、抬升、俯视并向一侧移动。不会调用 MoveIt 眼镜蛇姿态，'
            '也不会用 Z/X 的 J1 旋转。确认后上方与侧方净空？' % label,
            '确认%s' % label,
            wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING)
        if answer != wx.YES:
            return
        self._publish_zero()
        self.cobra_button.Enable(False)
        self.reverse_cobra_button.Enable(False)
        self.operation_message = (
            '%s：正在停止旧点动并执行共享法兰短步' % label)

        def worker():
            result = self.ros_bridge.run_flange_snake_maneuver(
                reverse=reverse,
                base_duration_s=0.15)
            wx.CallAfter(self._cobra_finished, result)

        threading.Thread(target=worker, daemon=True).start()

    def _cobra_finished(self, cobra_result):
        self.operation_message = (
            ('蛇形观察完成：' if cobra_result[0] else '蛇形观察失败：') +
            str(cobra_result[1]))
        if not self.closed:
            self.cobra_button.Enable(True)
            self.reverse_cobra_button.Enable(True)

    def on_timer(self, _event):
        snapshot = self.state.snapshot()
        self._update_nbv_status(snapshot.get('nbv_status', {}))
        annotated = snapshot['images']['annotated']
        color = snapshot['images']['color']
        arrivals = snapshot['last_arrival']
        annotated_is_live = annotated is not None and \
            arrivals.get('annotated', 0.0) >= arrivals.get('color', 0.0) - 0.25
        if annotated_is_live:
            image = annotated
            version = ('annotated', snapshot['versions']['annotated'])
        else:
            image = color
            version = ('color', snapshot['versions']['color'])
        if image is not None and version != self.seen_image_version:
            self.camera_view.set_bgr(image)
            self.seen_image_version = version

        cockpit = snapshot['cockpit_status']
        active = bool(cockpit.active) if cockpit is not None else False
        harvest = snapshot.get('harvest_status')
        tool_active = bool(
            harvest.tool_active) if harvest is not None else False
        tool_allowed = bool(
            harvest is not None and harvest.execution_requested and
            not harvest.busy)
        self.tool_button.Enable(tool_allowed)
        self.open_tool_button.Enable(tool_allowed)
        if tool_active:
            self.operation_message = 'J/K：DO0 信号正在运行；结束后仍需 10 ms 低电平'
        if not active and not self.activation_paused and \
                not self.activation_succeeded and \
                not self.activation_inflight and \
                time.monotonic() >= self.activation_retry_at:
            self._request_activation()
        if cockpit is None:
            text = '驾驶舱节点尚未启动；正在自动等待'
        else:
            blockers = ('\n等待：' + '；'.join(cockpit.blockers)) \
                if cockpit.blockers else ''
            prefix = 'ACTIVE' if active else (
                '方向流程暂时释放' if self.activation_paused else
                '正在自动进入' if not self.activation_succeeded else '已停止')
            text = '驾驶舱：%s\n%s\n手动控制：%s%s' % (
                prefix,
                cockpit.message, cockpit.servo_status_text, blockers)
        pressed = self.input_state.pressed_actions()
        text += '\n本窗口收到的按键：%s；控制链：Panel 单次 Start / 单次 Stop' % (
            '+'.join(self.ACTION_LABELS.get(item, item)
                     for item in pressed) if pressed else '无')
        text += '\nPanel 点动：' + self.ros_bridge.cockpit_jog_message()
        text += '\n操作：' + self.operation_message
        if self.status.GetValue() != text:
            self.status.SetValue(text)

    def on_close(self, event):
        if self.closed:
            event.Skip()
            return
        self.closed = True
        self.timer.Stop()
        self.input_state.set_focused(False)
        self._publish_zero()
        threading.Thread(target=self.ros_bridge.request_cockpit_active,
                         args=(False,), daemon=True).start()
        self.owner.cockpit_dialog = None
        event.Skip()


class ApproachDirectionDialog(wx.Dialog):
    """Inspect, plan and execute a reconstructed citrus approach direction."""

    def __init__(self, owner):
        wx.Dialog.__init__(
            self, owner, title='柑橘最优伸入方向', size=(1260, 820),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.SetMinSize((980, 660))
        self.owner = owner
        self.state = owner.state
        self.ros_bridge = owner.ros_bridge
        self.closed = False
        self.seen_version = -1
        self.reconstructed_targets = []

        root = wx.Panel(self)
        layout = wx.BoxSizer(wx.VERTICAL)
        header = wx.BoxSizer(wx.HORIZONTAL)
        self.target_label = wx.StaticText(root, label='目标：等待主面板选择')
        target_font = self.target_label.GetFont()
        target_font.SetWeight(wx.FONTWEIGHT_BOLD)
        self.target_label.SetFont(target_font)
        self.target_source_choice = wx.Choice(
            root, choices=['主面板当前目标'], size=(155, -1))
        self.target_source_choice.SetSelection(0)
        self.target_source_choice.SetToolTip(
            '可选择当前 RGB-D 目标，或十步语义体素图中重建出的柑橘簇')
        self.compute_button = wx.Button(
            root, label='只规划选中柑橘', size=(145, 42))
        self.compute_button.SetToolTip(
            '计算球面可见空隙并请求 MoveIt 只规划')
        self.execute_button = wx.Button(
            root, label='执行缓存并夹取', size=(145, 42))
        self.execute_button.SetToolTip(
            '直接复用上次只规划的四段轨迹：入口、直线伸入15cm、J闭合、'
            '按主面板“夹取后停留/s”等待、退出返回、K张开；不重新规划')
        self.reset_button = wx.Button(
            root, label='清空方向图', size=(120, 42))
        self.reset_button.SetToolTip(
            '只清空内存中的方向体素图；不会删除已保存的 RGB-D/ZIP/staging 数据')
        close_button = wx.Button(root, wx.ID_CLOSE, label='关闭', size=(90, 42))
        header.Add(self.target_label, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 12)
        header.Add(self.target_source_choice, 0,
                   wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        header.Add(self.compute_button, 0, wx.RIGHT, 8)
        header.Add(self.execute_button, 0, wx.RIGHT, 8)
        header.Add(self.reset_button, 0, wx.RIGHT, 8)
        header.Add(close_button, 0)
        layout.Add(header, 0, wx.ALL | wx.EXPAND, 12)

        body = wx.BoxSizer(wx.HORIZONTAL)
        self.heatmap_view = ImageView(root, '球面方向效用图')
        self.heatmap_view.bitmap.SetToolTip(
            'GEOMETRY BEST=几何最佳；LIGHT MEAN=全局平均光；'
            'SELECTED: FASTEST=最高速入口 ETA 最短；其他十字为可行备选')
        body.Add(self.heatmap_view, 3, wx.EXPAND | wx.RIGHT, 10)

        diagnostics = wx.BoxSizer(wx.VERTICAL)
        self.status_text = StableReadOnlyTextCtrl(root)
        self.status_text.SetMinSize((340, 270))
        diagnostics.Add(self.status_text, 2, wx.EXPAND | wx.BOTTOM, 8)
        # 候选列表展示的是“外向光斑方向”，不是机械臂反向伸入轴。
        self.candidate_list = wx.ListCtrl(
            root, style=wx.LC_REPORT | wx.BORDER_SUNKEN)
        for index, (label, width) in enumerate((
                ('候选', 52), ('标签', 150), ('效用', 62), ('d/m', 68),
                ('e^-d', 62), ('ETAmax/s', 82), ('软树冠空隙/mm', 105),
                ('外向 XYZ', 185))):
            self.candidate_list.InsertColumn(index, label, width=width)
        diagnostics.Add(self.candidate_list, 1, wx.EXPAND)
        body.Add(diagnostics, 2, wx.EXPAND)
        layout.Add(body, 1, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 12)
        root.SetSizer(layout)

        self.compute_button.Bind(wx.EVT_BUTTON, self.on_compute)
        self.execute_button.Bind(wx.EVT_BUTTON, self.on_execute)
        self.reset_button.Bind(wx.EVT_BUTTON, self.on_reset)
        close_button.Bind(wx.EVT_BUTTON, lambda _event: self.Close())
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)
        self.timer.Start(250)
        self.on_timer(None)

    @staticmethod
    def _vector_text(values):
        if values is None or len(values) != 3:
            return '--'
        return '%.3f, %.3f, %.3f' % tuple(float(value) for value in values)

    def _selected_target(self):
        selection = int(self.target_source_choice.GetSelection())
        if selection > 0 and selection - 1 < len(self.reconstructed_targets):
            target = self.reconstructed_targets[selection - 1]
            center = tuple(float(value) for value in target.get('center', ()))
            if len(center) == 3:
                return (
                    -selection, center, self.ros_bridge.target_frame,
                    'reconstructed_citrus_%d' % selection)
        return self.owner._selected_target_request()

    def _update_target_sources(self, status):
        previous = self.target_source_choice.GetStringSelection()
        self.reconstructed_targets = list(
            (status or {}).get('reconstructed_targets') or ())
        choices = ['主面板当前目标']
        for index, target in enumerate(self.reconstructed_targets, 1):
            choices.append('重建柑橘 %d (%d体素)' % (
                index, int(target.get('voxel_count', 0))))
        self.target_source_choice.Set(choices)
        if previous in choices:
            self.target_source_choice.SetSelection(choices.index(previous))
        else:
            self.target_source_choice.SetSelection(0)

    def on_compute(self, _event):
        self._start_calculation(execute=False)

    def on_execute(self, _event):
        self._start_execution()

    def _pause_cockpit(self):
        dialog = self.owner.cockpit_dialog
        if dialog is not None and not dialog.closed:
            dialog.suspend_for_moveit()

    def _resume_cockpit(self):
        dialog = self.owner.cockpit_dialog
        if dialog is not None and not dialog.closed:
            dialog.resume_after_moveit()

    def _start_calculation(self, execute):
        del execute
        selected = self._selected_target()
        if selected is None:
            self.state.set_action_status(
                '请先在主面板选择一个具有机器人坐标的柑橘目标')
            return
        self.compute_button.Enable(False)
        self.execute_button.Enable(False)
        self._pause_cockpit()

        def worker():
            result = self.ros_bridge.calculate_approach_direction(
                *selected, execute=False)
            wx.CallAfter(self._calculation_finished, result)

        threading.Thread(
            target=worker, name='approach-direction-compute',
            daemon=True).start()

    def _start_execution(self):
        snapshot = self.state.snapshot()
        result = (snapshot.get('approach_status') or {}).get('result') or {}
        if not result.get('moveit_plan_id'):
            self.state.set_action_status('当前没有缓存 plan_id；请先点击“只规划选中柑橘”')
            return
        self.compute_button.Enable(False)
        self.execute_button.Enable(False)
        self._pause_cockpit()

        def worker():
            response = self.ros_bridge.execute_cached_approach()
            wx.CallAfter(self._calculation_finished, response)

        threading.Thread(
            target=worker, name='approach-direction-execute-cached',
            daemon=True).start()

    def _calculation_finished(self, _result):
        self._resume_cockpit()
        if not self.closed:
            self.compute_button.Enable(True)

    def on_reset(self, _event):
        answer = wx.MessageBox(
            '只会清空内存中的语义方向图，不会删除论文采集 ZIP 或 staging。继续？',
            '清空方向图', wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION,
            parent=self)
        if answer != wx.YES:
            return
        status = self.ros_bridge.nbv_status()
        self.ros_bridge.reset_approach_map(
            scene_id=status.get('scene_id', ''),
            group_number=status.get('group_number'))

    def _status_message(self, status):
        status = dict(status or {})
        result = status.get('result') or {}
        map_summary = status.get('map_summary') or {}
        lines = [
            '状态：%s' % (status.get('message') or '等待重建'),
            '重建：%d/%d 视角，%d 个占据体素，%d 个语义体素%s' % (
                int(status.get('view_count', 0)),
                int(status.get('minimum_views', 10)),
                int(map_summary.get('occupied_voxel_count', 0)),
                int(map_summary.get('semantic_voxel_count', 0)),
                '（地图已截断）' if map_summary.get('truncated') else ''),
        ]
        if not result:
            lines.append('结论：尚无方向结果；所有已采集原始数据保持不变。')
            return '\n'.join(lines)
        best = result.get('best') or {}
        lines.extend([
            '目标中心/m：%s；拟合半径 %.1f mm；目标体素 %d' % (
                self._vector_text(result.get('target_center')),
                1000.0 * float(result.get('target_radius_m', 0.0)),
                int(result.get('target_voxel_count', 0))),
            # 这里把“果实 -> 外部亮斑”和“机械臂实际伸入”分开显示，避免正负号混淆。
            '果实 -> 外部亮斑：%s' % self._vector_text(
                best.get('outward_direction')),
            '机械臂实际伸入：%s' % self._vector_text(
                best.get('insertion_direction')),
            '硬净空：未启用（柑橘、叶片和普通枝条均按可挤压软占据处理）',
            '软树冠最近距离：%.1f mm（只用于最大空隙排序，允许穿过）' % (
                1000.0 * float(best.get('soft_clearance_m', 0.0))),
            '光透过率：%.1f%%；局部光：%.1f%%；已观测：%.1f%%；未知：%.1f%%' % (
                100.0 * float(best.get('light_transmission', 0.0)),
                100.0 * float(best.get('local_light', 0.0)),
                100.0 * float(best.get('observed_fraction', 0.0)),
                100.0 * float(best.get('unknown_fraction', 0.0))),
            '亮斑等效角半径：%.1f°（要求 >= %.1f°）' % (
                float(best.get('patch_angular_radius_deg', 0.0)),
                float(result.get('minimum_patch_angular_radius_deg', 0.0))),
            '距离评分：d=%.3f m；e^-d=%.4f；最高速入口 ETA：%s' % (
                float(best.get('distance_m', 0.0)
                      if best.get('distance_m') is not None else 0.0),
                float(best.get('distance_score', 0.0)
                      if best.get('distance_score') is not None else 0.0),
                ('%.3f s' % float(best.get('max_speed_eta_s'))
                 if best.get('max_speed_eta_s') is not None and
                 math.isfinite(float(best.get('max_speed_eta_s')))
                 else 'unknown')),
            # moveit_message 带候选个数与所选序号，这是盲评旁证；试验期间只报是
            # 否通过验证，原文照旧写进结果 JSON 与 CSV。
            'MoveIt：%s' % (
                ('已通过验证' if result.get('moveit_validated') else
                 '未通过验证') if result.get('probe_method') else
                (result.get('moveit_message') or
                 ('尚未验证' if result.get('geometry_candidate_ready')
                  else '几何候选未完成，不进入 IK/碰撞验证'))),
            '判定：%s。' % (
                '已执行伸入并返回' if result.get('executed') else
                '候选已通过，可执行' if result.get(
                    'direction_candidate_ready') else
                '仅诊断或已拒绝'),
        ])
        return '\n'.join(lines)

    def _update_candidates(self, result):
        self.candidate_list.DeleteAllItems()
        if (result or {}).get('probe_method'):
            # 候选行数是盲评的旁证：直线基线恒为 1 行，另两种方法通常多行。试验
            # 期间这张表对三种方法一律留空，全部数值仍进 CSV 与结果 JSON。
            return
        for row, candidate in enumerate((result or {}).get('candidates') or ()):
            index = self.candidate_list.InsertItem(row, str(row + 1))
            self.candidate_list.SetItem(
                index, 1, str(candidate.get(
                    'display_label', 'GEOMETRY CANDIDATE')))
            self.candidate_list.SetItem(
                index, 2, '%.3f' % float(candidate.get('utility', 0.0)))
            distance = candidate.get('distance_m')
            distance_score = candidate.get('distance_score')
            eta = candidate.get('max_speed_eta_s')
            self.candidate_list.SetItem(
                index, 3, ('%.3f' % float(distance)
                           if distance is not None and
                           math.isfinite(float(distance)) else '--'))
            self.candidate_list.SetItem(
                index, 4, ('%.4f' % float(distance_score)
                           if distance_score is not None else '--'))
            self.candidate_list.SetItem(
                index, 5, ('%.3f' % float(eta)
                           if eta is not None and
                           math.isfinite(float(eta)) else '--'))
            self.candidate_list.SetItem(
                index, 6, '%.1f' % (1000.0 * float(
                    candidate.get('soft_clearance_m', 0.0))))
            self.candidate_list.SetItem(
                index, 7, self._vector_text(
                    candidate.get('outward_direction')))

    def on_timer(self, _event):
        if self.closed:
            return
        snapshot = self.state.snapshot()
        version = int(snapshot.get('approach_version', 0))
        status = snapshot.get('approach_status') or {}
        if version != self.seen_version:
            self._update_target_sources(status)
        selected = self._selected_target()
        if selected is None:
            self.target_label.SetLabel('目标：等待主面板选择')
        else:
            self.target_label.SetLabel(
                '目标 %d / %s / %s' % (
                    selected[0], selected[3],
                    self._vector_text(selected[1])))
        self.compute_button.Enable(
            not status.get('computing', False) and selected is not None)
        cached_plan_id = str(
            ((status.get('result') or {}).get('moveit_plan_id', '') or ''))
        self.execute_button.Enable(
            not status.get('computing', False) and bool(cached_plan_id))
        if version == self.seen_version:
            return
        self.seen_version = version
        self.status_text.SetValue(self._status_message(status))
        self._update_candidates(status.get('result') or {})
        heatmap = snapshot.get('approach_heatmap')
        if heatmap is None:
            self.heatmap_view.clear()
        else:
            self.heatmap_view.set_bgr(heatmap)

    def on_close(self, event):
        if self.closed:
            event.Skip()
            return
        self.closed = True
        self.timer.Stop()
        self.owner.approach_dialog = None
        event.Skip()


class ProbeTrialDialog(wx.Dialog):
    """Run one blind flexible-probe trial with one key press per step.

    对话框只显示布局、重复和组内第几次；方法编号、w_pi 一律不出现，保证目视
    评分不被暗示。48 次全部靠三个键完成：扫描、伸入、记录。
    """

    def __init__(self, owner):
        wx.Dialog.__init__(
            self, owner, title='柔性探针实测（盲评）', size=(720, 560),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.SetMinSize((640, 480))
        self.owner = owner
        self.state = owner.state
        self.ros_bridge = owner.ros_bridge
        self.closed = False
        self.busy = False
        self.pending_outcome = ''
        self.pending_auto_record = False
        # 本次试验开扫时确认过的布局/重复；记录时复用，避免中途拨号写错区块。
        self.trial_indices = None

        root = wx.Panel(self)
        layout = wx.BoxSizer(wx.VERTICAL)
        header = wx.BoxSizer(wx.HORIZONTAL)
        self.layout_spin = wx.SpinCtrl(
            root, min=1, max=probe_trials.LAYOUT_COUNT, initial=1,
            size=(70, -1))
        self.layout_spin.SetToolTip('第几种摆放布局；换布局前请先记完本组三次')
        self.repeat_spin = wx.SpinCtrl(
            root, min=1, max=probe_trials.REPEAT_COUNT, initial=1,
            size=(70, -1))
        self.repeat_spin.SetToolTip('同一布局的第几次重复')
        header.Add(wx.StaticText(root, label='布局'), 0,
                   wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        header.Add(self.layout_spin, 0, wx.RIGHT, 12)
        header.Add(wx.StaticText(root, label='重复'), 0,
                   wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        header.Add(self.repeat_spin, 0, wx.RIGHT, 12)
        close_button = wx.Button(root, wx.ID_CLOSE, label='关闭', size=(90, 42))
        header.Add(close_button, 0)
        layout.Add(header, 0, wx.ALL | wx.EXPAND, 12)

        self.status_text = StableReadOnlyTextCtrl(root)
        self.status_text.SetMinSize((-1, 150))
        layout.Add(self.status_text, 1,
                   wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 12)

        steps = wx.BoxSizer(wx.HORIZONTAL)
        self.scan_button = wx.Button(
            root, label='1 自动扫描 (F5)', size=(190, 48))
        self.scan_button.SetToolTip(
            '一键自动完成十视角扫描：开新批次后按预定义宽幅路线（半角 54°）'
            '逐站规划、校验、执行并记录，不用再按 Enter；记满十张后重建结果会自动出现，'
            '再在“伸入方向”窗口确认目标。失败会报原因，可再按一次重试')
        self.drive_button = wx.Button(
            root, label='2 伸入 (F6)', size=(190, 48))
        self.drive_button.SetToolTip(
            '按当前选中目标规划并直接伸入；不触发 DO0 脉冲（未装剪切工具）')
        steps.Add(self.scan_button, 0, wx.RIGHT, 10)
        steps.Add(self.drive_button, 0, wx.RIGHT, 10)
        layout.Add(steps, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

        score_box = wx.StaticBoxSizer(
            wx.VERTICAL, root, '3 目视评分（看到什么就填什么）')
        self.score_choice = wx.Choice(root, choices=[
            '2 = 干净命中：探针尖碰到果实，没有推开叶片或枝条',
            '1 = 污染命中：碰到果实，但叶片或枝条被推开',
            '0 = 未命中：没有碰到果实'])
        # 不预选任何一项：预选“2 = 干净命中”会让一次没看清就按 F7 的试验被记成
        # 满分，而 F7 不看按钮的 enable 状态。
        self.score_choice.SetSelection(wx.NOT_FOUND)
        score_box.Add(self.score_choice, 0, wx.ALL | wx.EXPAND, 6)
        self.note_input = wx.TextCtrl(root, value='', size=(-1, 56),
                                      style=wx.TE_MULTILINE)
        self.note_input.SetToolTip('可选备注；不填也能保存')
        score_box.Add(self.note_input, 0, wx.ALL | wx.EXPAND, 6)
        self.record_button = wx.Button(
            root, label='保存本次 (F7)', size=(190, 48))
        self.record_button.SetToolTip('把本次评分与全部离线指标写入 CSV 一行')
        score_box.Add(self.record_button, 0, wx.ALL, 6)
        layout.Add(score_box, 0,
                   wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 12)
        root.SetSizer(layout)

        self.scan_button.Bind(wx.EVT_BUTTON, self.on_scan)
        self.drive_button.Bind(wx.EVT_BUTTON, self.on_drive)
        self.record_button.Bind(wx.EVT_BUTTON, self.on_record)
        close_button.Bind(wx.EVT_BUTTON, lambda _event: self.Close())
        self.Bind(wx.EVT_CHAR_HOOK, self.on_char_hook)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)
        self.timer.Start(500)
        self.on_timer(None)

    def _indices(self):
        return int(self.layout_spin.GetValue()), int(self.repeat_spin.GetValue())

    def _recommended(self):
        layout_index, repeat_index = self._indices()
        try:
            status = self.ros_bridge.probe_status(layout_index, repeat_index)
        except Exception as error:
            # 种子文件损坏或丢失时整个试验都已被拒绝，落盘走不到；这里只要不挡住
            # 界面，错误原因由状态框显示。
            self.state.set_action_status('[柔性探针] %s' % error)
            return layout_index, repeat_index
        return (int(status.get('recommended_layout_index', layout_index)),
                int(status.get('recommended_repeat_index', repeat_index)))

    def _confirmed_indices(self):
        """Return the indices to use, asking once when they differ from the plan.

        软件按已落盘的行推荐下一个布局/重复区块，并在记满三次后自动推进控件；手拨
        到别的值时必须先确认，否则换布局忘拨号会把两个布局并进同一个
        layout_index，而布局内配对检验正是按这个字段分组。
        """
        layout_index, repeat_index = self._indices()
        recommended = self._recommended()
        if (layout_index, repeat_index) == recommended:
            return layout_index, repeat_index
        answer = wx.MessageBox(
            '你正要写入布局 %d 重复 %d，而软件推荐的是布局 %d 重复 %d。\n'
            '继续会把本次试验记到手填的区块里；换布局忘拨号会让两个布局合并，'
            '事后无法修复。确认继续？' % (
                layout_index, repeat_index, recommended[0], recommended[1]),
            '确认布局与重复序号', wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING,
            self)
        if answer != wx.YES:
            self.state.set_action_status(
                '[柔性探针] 已取消；软件推荐布局 %d 重复 %d' % recommended)
            return None
        return layout_index, repeat_index

    def on_char_hook(self, event):
        code = int(event.GetKeyCode())
        if code == wx.WXK_F5:
            self.on_scan(None)
            return
        if code == wx.WXK_F6:
            self.on_drive(None)
            return
        if code == wx.WXK_F7:
            self.on_record(None)
            return
        event.Skip()

    def _run_async(self, worker, name):
        if self.busy:
            self.state.set_action_status('[柔性探针] 上一步还没结束；请稍等')
            return
        self.busy = True
        self._enable_steps(False)

        def runner():
            try:
                worker()
            finally:
                wx.CallAfter(self._step_finished)

        threading.Thread(target=runner, name=name, daemon=True).start()

    def _step_finished(self):
        self.busy = False
        if self.closed:
            return
        if self.pending_auto_record:
            # 这一步必须等 worker 的 finally 放掉 busy 之后才能跑，否则 _run_async
            # 的忙闲门禁会把这一行无声吞掉。此刻仍在 GUI 线程，且伸入线程已结束。
            self.pending_auto_record = False
            self._record_no_feasible_plan()
            return
        self.on_timer(None)

    def _record_no_feasible_plan(self):
        """Commit the scoreless row for a trial whose probe never went in."""
        layout_index, repeat_index = self.trial_indices or self._indices()
        note = str(self.note_input.GetValue())

        def worker():
            self.ros_bridge.probe_record(
                layout_index, repeat_index,
                probe_trials.OUTCOME_NO_FEASIBLE_PLAN,
                score=None, note=note)
            wx.CallAfter(self._record_finished)

        self._run_async(worker, 'probe-trial-record-no-plan')

    def _enable_steps(self, enabled):
        for control in (self.scan_button, self.drive_button,
                        self.record_button):
            control.Enable(bool(enabled))

    def on_scan(self, _event):
        indices = self._confirmed_indices()
        if indices is None:
            return
        layout_index, repeat_index = indices
        self.trial_indices = (layout_index, repeat_index)
        self.pending_outcome = ''

        def worker():
            self.ros_bridge.probe_start_scan(
                layout_index, repeat_index,
                scene_id='probe_L%02dR%d' % (layout_index, repeat_index))

        self._run_async(worker, 'probe-trial-scan')

    def on_drive(self, _event):
        selected = self.owner._selected_target_request()
        dialog = self.owner.approach_dialog
        if dialog is not None and not dialog.closed:
            selected = dialog._selected_target()
        if selected is None:
            self.state.set_action_status(
                '[柔性探针] 请先确认一个有机器人坐标的目标再按“伸入”')
            return

        def worker():
            outcome = self.ros_bridge.probe_drive_in(*selected)
            wx.CallAfter(self._drive_finished, outcome)

        self._run_async(worker, 'probe-trial-drive-in')

    def _drive_finished(self, outcome):
        outcome = dict(outcome or {})
        self.pending_outcome = str(outcome.get('outcome', '') or '')
        # 无可行轨迹是一类正式结果，不重试也不报错；以空分数落盘。这个回调与
        # worker 的 _step_finished 同在 GUI 队列里，此时 busy 还是 True，所以只
        # 置标志，由 _step_finished 接着触发落盘。
        self.pending_auto_record = (
            self.pending_outcome == probe_trials.OUTCOME_NO_FEASIBLE_PLAN)

    def _record_finished(self):
        self.pending_outcome = ''
        self.trial_indices = None
        if self.closed:
            return
        self.note_input.SetValue('')
        self.score_choice.SetSelection(wx.NOT_FOUND)
        # 本组三次记满后由软件自己推进到下一个区块，操作者不必拨号。
        layout_index, repeat_index = self._recommended()
        self.layout_spin.SetValue(layout_index)
        self.repeat_spin.SetValue(repeat_index)

    def on_record(self, _event):
        # 探针没有伸出的结果本身就没有目视评分，这一行以空分数落盘。
        scoreless = bool(self.pending_outcome) and \
            self.pending_outcome != probe_trials.OUTCOME_EXECUTED
        selection = int(self.score_choice.GetSelection())
        if not scoreless and selection == wx.NOT_FOUND:
            # F7 不看按钮的 enable 状态，所以未评分必须在这里挡住，且不抛异常。
            self.state.set_action_status(
                '[柔性探针] 请先选择 2/1/0 评分再保存')
            return
        if self.trial_indices is None:
            indices = self._confirmed_indices()
            if indices is None:
                return
        else:
            indices = self.trial_indices
        layout_index, repeat_index = indices
        outcome = self.pending_outcome or probe_trials.OUTCOME_EXECUTED
        score = None if scoreless else 2 - selection
        note = str(self.note_input.GetValue())

        def worker():
            self.ros_bridge.probe_record(
                layout_index, repeat_index, outcome, score=score, note=note)
            wx.CallAfter(self._record_finished)

        self._run_async(worker, 'probe-trial-record')

    def _status_message(self, status):
        status = dict(status or {})
        lines = [str(status.get('summary', '') or '')]
        lines.append('重建：%d/%d 视角%s' % (
            int(status.get('view_count', 0)),
            int(status.get('minimum_views', 10)),
            '（已完成，可伸入）' if status.get('reconstruction_complete')
            else '（继续按 Enter 记录视角）'))
        lines.append('当前试验：%s；缓存轨迹：%s' % (
            '已就绪' if status.get('armed') else '未开始',
            '有' if status.get('plan_cached') else '无'))
        if self.pending_outcome == probe_trials.OUTCOME_NO_FEASIBLE_PLAN:
            lines.append('上次结果：无可行轨迹，已按空分数记录')
        lines.append('落盘：%s' % str(status.get('csv_path', '') or ''))
        return '\n'.join(lines)

    def on_timer(self, _event):
        if self.closed:
            return
        layout_index, repeat_index = self._indices()
        try:
            status = self.ros_bridge.probe_status(layout_index, repeat_index)
        except Exception as error:
            # 种子文件不可用是拒绝开工的条件；把原因写在状态框里，并让三个步骤都
            # 按不下去，而不是每 500 ms 抛一次栈。
            self.status_text.SetValue('无法开始试验：%s' % error)
            self._enable_steps(False)
            return
        self.status_text.SetValue(self._status_message(status))
        if self.busy:
            return
        self.scan_button.Enable(not status.get('armed', False) and
                                not status.get('block_complete', False))
        self.drive_button.Enable(bool(status.get('armed')) and
                                 bool(status.get('reconstruction_complete')))
        self.record_button.Enable(bool(status.get('armed')) and
                                  bool(status.get('plan_cached')))

    def on_close(self, event):
        if self.closed:
            event.Skip()
            return
        self.closed = True
        self.timer.Stop()
        self.owner.probe_dialog = None
        event.Skip()


class DashboardFrame(wx.Frame):

    def __init__(self, state, ros_bridge):
        wx.Frame.__init__(self, None, title='Elfin E05 柑橘视觉与规划终端',
                         size=(1500, 900))
        self.SetMinSize((1120, 680))
        self.state = state
        self.ros_bridge = ros_bridge
        self.closing = False
        self.external_shutdown_requested = False
        self.experiment_dialog = None
        self.cockpit_dialog = None
        self.approach_dialog = None
        self.probe_dialog = None
        self.rviz_process = None
        initial_device = {
            'name': ros_bridge.camera_name,
            'serial': ros_bridge.camera_serial,
            'device_type': ros_bridge.camera_device_type,
            'usb_port_id': ros_bridge.camera_usb_port_id,
            'usb_type': ros_bridge.camera_usb_type,
        }
        self.camera_runtime = CameraRuntime(
            requested_serial=ros_bridge.camera_serial,
            initial_running=ros_bridge.camera_initial_on,
            initial_connected=ros_bridge.camera_connected,
            initial_device=initial_device,
            camera_fps=ros_bridge.camera_fps,
            target_frame=ros_bridge.target_frame,
            python_executable=ros_bridge.python_executable,
            processing_callback=ros_bridge.set_camera_processing,
            auto_start=ros_bridge.camera_auto_start,
            auto_start_usb3_only=ros_bridge.camera_auto_start_usb3_only,
            usb2_auto_start_delay_s=
            ros_bridge.camera_usb2_auto_start_delay_s)
        self._camera_ui_running = None
        self.rviz_config = str(rospy.get_param(
            '~rviz_config',
            '/home/catas/ros_ws/src/elfin_vision/config/citrus_moveit.rviz'))
        self.rviz_egl_default = bool(rospy.get_param(
            '~rviz_egl_default', False))
        self.rviz_auto_start = bool(rospy.get_param(
            '~rviz_auto_start', False))
        self.global_speed_percent = int(max(1, min(100, rospy.get_param(
            '~global_speed_percent', 5))))
        self.recovery_speed_percent = int(max(
            1, min(100, rospy.get_param('~recovery_speed_percent', 25))))
        self.cockpit_speed_percent = int(max(
            1, min(100, rospy.get_param('~cockpit_speed_percent', 50))))
        self.execute_request_inflight = False
        self.auto_request_inflight = False
        self.tool_wait_apply_inflight = False
        self.batch_limit_apply_inflight = False
        self.nbv_toggle_inflight = False
        self.calibration_request_inflight = False
        self.calibration_reuse_inflight = False
        self.latest_map_load_inflight = False
        self.speed_apply_later = None
        self.recovery_speed_apply_later = None
        self.speed_slider = None
        self.speed_label = None
        self.recovery_speed_slider = None
        self.recovery_speed_label = None
        self.seen_versions = {'color': -1, 'depth': -1, 'annotated': -1}

        root = wx.Panel(self)
        root.SetBackgroundColour(wx.Colour(238, 241, 244))
        outer = wx.BoxSizer(wx.VERTICAL)

        header = wx.BoxSizer(wx.HORIZONTAL)
        title = wx.StaticText(root, label='Elfin E05 柑橘采摘研究终端')
        title_font = title.GetFont()
        title_font.SetPointSize(16)
        title_font.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(title_font)
        safety_label = ('仿真相机 TF；真机执行关闭' if ros_bridge.demo_camera_tf
                        else '等待采摘状态机执行门禁')
        self.safety = wx.StaticText(root, label=safety_label)
        self.safety.SetForegroundColour(wx.Colour(154, 62, 0))
        self.mode_badge = wx.StaticText(
            root, label='系统待命', size=(250, 42),
            style=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL |
            wx.ST_NO_AUTORESIZE | wx.BORDER_SIMPLE)
        badge_font = self.mode_badge.GetFont()
        badge_font.SetPointSize(12)
        badge_font.SetWeight(wx.FONTWEIGHT_BOLD)
        self.mode_badge.SetFont(badge_font)
        self.mode_badge.SetBackgroundColour(wx.Colour(205, 209, 214))
        self.mode_badge.SetForegroundColour(wx.Colour(55, 60, 65))
        header_status = wx.BoxSizer(wx.VERTICAL)
        header_status.Add(self.mode_badge, 0, wx.ALIGN_RIGHT | wx.BOTTOM, 4)
        header_status.Add(self.safety, 0, wx.ALIGN_RIGHT)
        header.Add(title, 1, wx.ALIGN_CENTER_VERTICAL)
        header.Add(header_status, 0, wx.ALIGN_CENTER_VERTICAL)
        outer.Add(header, 0, wx.ALL | wx.EXPAND, 12)

        images = wx.FlexGridSizer(rows=1, cols=3, vgap=10, hgap=10)
        self.color_view = ImageView(root, 'RGB 原图')
        self.depth_view = ImageView(root, '对齐深度（近暖远冷）')
        self.annotated_view = ImageView(root, '模型识别结果')
        for view in (self.color_view, self.depth_view, self.annotated_view):
            images.Add(view, 1, wx.EXPAND)
        for column in range(3):
            images.AddGrowableCol(column, 1)
        images.AddGrowableRow(0, 1)
        outer.Add(images, 3, wx.LEFT | wx.RIGHT | wx.EXPAND, 12)

        lower = wx.BoxSizer(wx.HORIZONTAL)

        target_box = wx.StaticBoxSizer(wx.VERTICAL, root, '当前柑橘目标')
        self.target_list = wx.ListCtrl(root, style=wx.LC_REPORT | wx.BORDER_SUNKEN)
        columns = [('序号', 44), ('类别', 72), ('置信度', 60), ('深度/m', 60),
                   ('相机坐标 XYZ/m', 195), ('机器人坐标 XYZ/m', 205)]
        for index, (label, width) in enumerate(columns):
            self.target_list.InsertColumn(index, label, width=width)
        target_box.Add(self.target_list, 1, wx.EXPAND)
        lower.Add(target_box, 3, wx.EXPAND | wx.RIGHT, 10)

        # Status and log share the full lower height side-by-side. Stacking
        # them made the log body only one or two lines tall on 768 px screens.
        right_lower = wx.BoxSizer(wx.HORIZONTAL)

        status_box = wx.StaticBoxSizer(wx.VERTICAL, root, '系统状态')
        self.status_text = StableReadOnlyTextCtrl(root)
        self.status_text.SetMinSize((190, 120))
        self.joint_text = wx.StaticText(root, label='关节状态：等待 /joint_states')
        self.joint_text.Wrap(420)
        status_box.Add(self.status_text, 1, wx.EXPAND | wx.BOTTOM, 8)
        status_box.Add(self.joint_text, 0, wx.EXPAND)
        right_lower.Add(status_box, 2, wx.EXPAND | wx.RIGHT, 8)

        event_box = wx.StaticBoxSizer(wx.VERTICAL, root, '采摘流程日志')
        self.event_log = ProtectedLogView(root, minimum_size=(280, 185))
        self.event_text = self.event_log.text
        event_box.Add(self.event_log, 1, wx.EXPAND)
        right_lower.Add(event_box, 3, wx.EXPAND)
        lower.Add(right_lower, 2, wx.EXPAND)
        lower.SetMinSize((-1, 225))
        outer.Add(lower, 2, wx.ALL | wx.EXPAND, 12)

        command_box = wx.StaticBoxSizer(wx.HORIZONTAL, root, '采摘控制')
        command_panel = command_box.GetStaticBox()
        self.execute_button = wx.ToggleButton(
            command_panel, label='EXECUTE OFF', size=(118, 50))
        self.execute_button.SetToolTip('开启时在本窗口完成二次清场确认；关闭即撤销真机授权')
        command_box.Add(self.execute_button, 0, wx.ALL, 6)
        command_box.Add(wx.StaticText(command_panel, label='目标'), 0,
                        wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 6)
        self.target_choice = wx.Choice(command_panel, choices=['--'], size=(72, -1))
        self.target_choice.SetSelection(0)
        self.target_valid = {}
        self.current_targets = None
        self.selected_target_point = None
        button_size = (108, 50)
        self.plan_button = wx.Button(command_panel, label='只规划', size=button_size)
        self.plan_button.SetToolTip('请求 MoveIt 计算路径；此按钮固定 execute=false，不驱动真机')
        self.run_button = wx.Button(command_panel, label='采摘所选', size=button_size)
        self.run_button.SetToolTip(
            '执行锁存、预抓取、按参数中心 TCP XYZ 计算的直线伸入、DO0、等待和撤回全过程')
        self.auto_button = wx.ToggleButton(command_panel, label='AUTO OFF', size=button_size)
        self.auto_button.SetToolTip(
            '连续模式会一次锁定同一视野内多颗基座坐标、按绿框面积从大到小夹取；'
            '每颗结束后沿两段实际轨迹原路返回，并用法兰原点转头居中边缘目标')
        auto_options = wx.BoxSizer(wx.VERTICAL)
        self.continuous_check = wx.CheckBox(command_panel, label='连续夹取')
        self.continuous_check.SetValue(True)
        self.continuous_check.SetToolTip(
            '开启时一次锁定当前视野多颗目标，后续被树冠遮挡也继续逐颗完整预检；'
            '关闭时只完成一颗后退出')
        self.patrol_check = wx.CheckBox(command_panel, label='无目标自动找果')
        self.patrol_check.SetValue(True)
        self.patrol_check.SetToolTip(
            '先建立低位后撤的广角追踪原点，再用可逆法兰短步从上次视角继续绕树；'
            '到累计上限或真实门禁边界才返回，不调用 MoveIt 眼镜蛇或 J1')
        auto_options.Add(self.continuous_check, 0, wx.BOTTOM, 2)
        batch_limit_row = wx.BoxSizer(wx.HORIZONTAL)
        batch_limit_row.Add(
            wx.StaticText(command_panel, label='连续颗数'), 0,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 3)
        self.batch_limit_spin = wx.SpinCtrl(
            command_panel, value='3', min=1, max=50, initial=3,
            size=(72, -1))
        self.batch_limit_spin.SetToolTip(
            '单次 AUTO 最多成功夹取的颗数；同时作为当前视野最大锁点数')
        self.batch_limit_apply_button = wx.Button(
            command_panel, label='应用', size=(54, -1))
        batch_limit_row.Add(self.batch_limit_spin, 0, wx.RIGHT, 3)
        batch_limit_row.Add(self.batch_limit_apply_button, 0)
        auto_options.Add(batch_limit_row, 0, wx.BOTTOM, 2)
        auto_options.Add(self.patrol_check, 0)
        dwell_options = wx.BoxSizer(wx.VERTICAL)
        dwell_options.Add(
            wx.StaticText(command_panel, label='夹取后停留/s'), 0,
            wx.ALIGN_CENTER_HORIZONTAL | wx.BOTTOM, 2)
        dwell_row = wx.BoxSizer(wx.HORIZONTAL)
        self.tool_wait_spin = wx.SpinCtrlDouble(
            command_panel, value='0.10', min=0.01, max=120.0,
            initial=0.10, inc=0.01, size=(82, -1))
        self.tool_wait_spin.SetDigits(2)
        self.tool_wait_spin.SetToolTip(
            'DO0 脉冲结束后到开始撤回的停留时间；仅在状态机空闲时应用')
        self.tool_wait_apply_button = wx.Button(
            command_panel, label='应用', size=(54, -1))
        dwell_row.Add(self.tool_wait_spin, 0, wx.RIGHT, 3)
        dwell_row.Add(self.tool_wait_apply_button, 0)
        dwell_options.Add(dwell_row, 0)
        speed_options = wx.BoxSizer(wx.VERTICAL)
        speed_header = wx.BoxSizer(wx.HORIZONTAL)
        speed_header.Add(
            wx.StaticText(command_panel, label='采摘轨迹速度'), 0,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.speed_label = wx.StaticText(
            command_panel, label='%d%%' % self.global_speed_percent,
            size=(38, -1), style=wx.ALIGN_RIGHT)
        speed_header.Add(self.speed_label, 0, wx.ALIGN_CENTER_VERTICAL)
        speed_options.Add(speed_header, 0, wx.ALIGN_CENTER_HORIZONTAL)
        self.speed_slider = wx.Slider(
            command_panel, value=self.global_speed_percent,
            minValue=1, maxValue=100, size=(118, -1),
            style=wx.SL_HORIZONTAL)
        self.speed_slider.SetToolTip(
            '同时调整采摘 MoveIt 速度和加速度倍率；每段执行前都按最新值重计时')
        speed_options.Add(self.speed_slider, 0, wx.TOP, 2)
        recovery_speed_options = wx.BoxSizer(wx.VERTICAL)
        recovery_speed_header = wx.BoxSizer(wx.HORIZONTAL)
        recovery_speed_header.Add(
            wx.StaticText(command_panel, label='主动脱困速度'), 0,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.recovery_speed_label = wx.StaticText(
            command_panel, label='%d%%' % self.recovery_speed_percent,
            size=(38, -1), style=wx.ALIGN_RIGHT)
        recovery_speed_header.Add(
            self.recovery_speed_label, 0, wx.ALIGN_CENTER_VERTICAL)
        recovery_speed_options.Add(
            recovery_speed_header, 0, wx.ALIGN_CENTER_HORIZONTAL)
        self.recovery_speed_slider = wx.Slider(
            command_panel, value=self.recovery_speed_percent,
            minValue=1, maxValue=100, size=(118, -1),
            style=wx.SL_HORIZONTAL)
        self.recovery_speed_slider.SetToolTip(
            '驾驶舱或 AUTO 卡滞时，短程 MoveIt 关节构型整理的独立速度')
        recovery_speed_options.Add(
            self.recovery_speed_slider, 0, wx.TOP, 2)
        self.stop_button = wx.Button(command_panel, label='停止', size=(84, 50))
        self.stop_button.SetToolTip('停止自动循环并请求 MoveIt 停止当前轨迹；不会切断硬件电源')
        command_box.Add(self.target_choice, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 6)
        command_box.Add(self.plan_button, 0, wx.ALL, 6)
        command_box.Add(self.run_button, 0, wx.ALL, 6)
        command_box.Add(self.auto_button, 0, wx.ALL, 6)
        command_box.Add(auto_options, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        command_box.Add(dwell_options, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        command_box.Add(speed_options, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        command_box.Add(
            recovery_speed_options, 0,
            wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        command_box.Add(self.stop_button, 0, wx.ALL, 6)
        outer.Add(command_box, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 12)

        tools_box = wx.StaticBoxSizer(wx.VERTICAL, root, '视图与实验')
        tools_panel = tools_box.GetStaticBox()
        camera_row = wx.BoxSizer(wx.HORIZONTAL)
        self.camera_status_label = wx.StaticText(
            tools_panel, label='摄像头：后台检测中', size=(-1, 30),
            style=wx.ST_NO_AUTORESIZE)
        self.camera_status_label.Wrap(900)
        camera_row.Add(self.camera_status_label, 1,
                       wx.ALL | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 6)
        tools_box.Add(camera_row, 0, wx.EXPAND)
        control_row = wx.BoxSizer(wx.HORIZONTAL)
        self.rgb_camera_button = wx.ToggleButton(
            tools_panel, label='睁眼 RGB', size=(112, 48))
        self.rgb_camera_button.SetToolTip(
            '仅开启 640x480 RGB；关闭深度、点云和模型推理，供人工驾驶低负载观察')
        self.vision_camera_button = wx.ToggleButton(
            tools_panel, label='自动视觉', size=(112, 48))
        self.vision_camera_button.SetToolTip(
            '开启 RGB、深度、环境点云和远端优先的柑橘识别；AUTO 会自动选择此模式')
        self.cockpit_button = wx.Button(
            tools_panel, label='驾驶舱', size=(108, 48))
        self.rviz_button = wx.ToggleButton(
            tools_panel, label='启动 RViz', size=(108, 48))
        self.egl_check = wx.CheckBox(tools_panel, label='EGL')
        self.egl_check.SetValue(self.rviz_egl_default)
        self.experiment_button = wx.Button(
            tools_panel, label='参数中心', size=(108, 48))
        self.experiment_button.SetToolTip(
            '打开全部采摘运行参数与中文说明，包括末端相对法兰 XYZ、'
            '等待时间、视角策略、批量颗数、路径门限和速度')
        self.capture_button = wx.Button(
            tools_panel, label='保存 RGB-D', size=(108, 48))
        self.capture_button.SetToolTip('保存 RGB、原始深度、标注图和元数据')
        self.approach_button = wx.Button(
            tools_panel, label='伸入方向', size=(108, 48))
        self.approach_button.SetToolTip(
            '打开球面光照与工具通道结果；只规划后可复用缓存轨迹执行抓取周期')
        self.probe_button = wx.Button(
            tools_panel, label='探针实测', size=(108, 48))
        self.probe_button.SetToolTip(
            '打开 48 次柔性探针盲评流程：F5 一键自动完成十视角扫描、F6 伸入、'
            'F7 保存评分；窗口不显示方法编号')
        for control in (self.camera_status_label, self.rgb_camera_button,
                        self.vision_camera_button,
                        self.cockpit_button, self.rviz_button,
                        self.egl_check, self.experiment_button,
                        self.capture_button, self.approach_button,
                        self.probe_button):
            if control is not self.camera_status_label:
                control_row.Add(control, 0,
                                wx.ALL | wx.ALIGN_CENTER_VERTICAL, 6)
        tools_box.Add(control_row, 0, wx.EXPAND)
        action_row = wx.BoxSizer(wx.HORIZONTAL)
        self.action_text = wx.StaticText(tools_panel, label='')
        action_row.Add(self.action_text, 1,
                       wx.ALL | wx.ALIGN_CENTER_VERTICAL, 8)
        tools_box.Add(action_row, 0, wx.EXPAND)
        outer.Add(tools_box, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 12)

        # Keep the operator controls in one compact row and give the status
        # text its own full-width row.  At the frame's minimum size this avoids
        # squeezing the x/10, hardware-readiness and sample-recommendation
        # text into an unreadable sliver.
        paper_box = wx.StaticBoxSizer(wx.VERTICAL, root, '论文采集 / D455 标定')
        paper_panel = paper_box.GetStaticBox()
        paper_controls = wx.BoxSizer(wx.HORIZONTAL)
        self.nbv_mode_button = wx.ToggleButton(
            paper_panel, label='论文采集 OFF', size=(132, 48))
        self.nbv_mode_button.SetToolTip(
            '开启后建立/恢复一个十视角批次；驾驶舱内按 Enter 或直接按机械臂实体 POINT 记录一次有效快照')
        paper_controls.Add(self.nbv_mode_button, 0,
                           wx.ALL | wx.ALIGN_CENTER_VERTICAL, 6)
        paper_controls.Add(wx.StaticText(paper_panel, label='场景编号'), 0,
                           wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 4)
        self.nbv_scene_text = wx.TextCtrl(
            paper_panel, value=ros_bridge.nbv_scene_default, size=(145, -1))
        self.nbv_scene_text.SetToolTip('例如 scene_01_left_occlusion；不要把同一布局拆成独立场景')
        paper_controls.Add(self.nbv_scene_text, 0,
                           wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        paper_controls.Add(wx.StaticText(paper_panel, label='备注'), 0,
                           wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 4)
        self.nbv_notes_text = wx.TextCtrl(paper_panel, value='', size=(180, -1))
        self.nbv_notes_text.SetToolTip('记录遮挡方向、果实数量、是否重整场景等审计信息')
        paper_controls.Add(self.nbv_notes_text, 0,
                           wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.latest_map_button = wx.Button(
            paper_panel, label='载入最近10视角', size=(145, 48))
        self.latest_map_button.SetToolTip(
            '读取最近一个完整 staging 的十视角 RGB-D/语义数据并重建方向图；不重新拍摄')
        paper_controls.Add(self.latest_map_button, 0,
                           wx.ALL | wx.ALIGN_CENTER_VERTICAL, 6)
        paper_controls.AddStretchSpacer(1)
        self.use_calibration_button = wx.Button(
            paper_panel, label='使用当前/最新标定', size=(160, 48))
        self.use_calibration_button.SetToolTip(
            '直接刷新当前已安装外参；若有更新且质量通过、序列号匹配的候选则备份后安装')
        paper_controls.Add(self.use_calibration_button, 0,
                           wx.ALL | wx.ALIGN_CENTER_VERTICAL, 6)
        self.calibration_button = wx.Button(
            paper_panel, label='D455 重新标定', size=(145, 48))
        self.calibration_button.SetToolTip(
            '只读 Charuco eye-in-hand 采样；质量通过后备份旧 YAML 并原子安装新外参')
        paper_controls.Add(self.calibration_button, 0,
                           wx.ALL | wx.ALIGN_CENTER_VERTICAL, 6)
        paper_box.Add(paper_controls, 0, wx.EXPAND)
        self.nbv_status_label = wx.StaticText(
            paper_panel, label='等待开启；建议 2 批试拍 / 18 批工程最低 / 24 批趋势最低',
            size=(-1, 122), style=wx.ST_NO_AUTORESIZE)
        self.nbv_status_label.Wrap(640)
        paper_box.Add(self.nbv_status_label, 0,
                      wx.ALL | wx.EXPAND, 6)
        outer.Add(paper_box, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 12)

        root.SetSizer(outer)
        self.execute_button.Bind(wx.EVT_TOGGLEBUTTON, self.on_execute)
        self.plan_button.Bind(wx.EVT_BUTTON, self.on_plan)
        self.run_button.Bind(wx.EVT_BUTTON, self.on_run)
        self.auto_button.Bind(wx.EVT_TOGGLEBUTTON, self.on_auto)
        self.tool_wait_apply_button.Bind(
            wx.EVT_BUTTON, self.on_tool_wait_apply)
        self.batch_limit_apply_button.Bind(
            wx.EVT_BUTTON, self.on_batch_limit_apply)
        self.speed_slider.Bind(wx.EVT_SLIDER, self.on_global_speed)
        self.recovery_speed_slider.Bind(
            wx.EVT_SLIDER, self.on_recovery_speed)
        self.stop_button.Bind(wx.EVT_BUTTON, self.on_stop)
        self.experiment_button.Bind(wx.EVT_BUTTON, self.on_experiment)
        self.cockpit_button.Bind(wx.EVT_BUTTON, self.on_cockpit)
        self.rgb_camera_button.Bind(
            wx.EVT_TOGGLEBUTTON, self.on_rgb_camera)
        self.vision_camera_button.Bind(
            wx.EVT_TOGGLEBUTTON, self.on_full_vision)
        self.rviz_button.Bind(wx.EVT_TOGGLEBUTTON, self.on_rviz)
        self.target_choice.Bind(wx.EVT_CHOICE, self.on_target_choice)
        self.capture_button.Bind(wx.EVT_BUTTON, self.on_capture)
        self.approach_button.Bind(wx.EVT_BUTTON, self.on_approach)
        self.probe_button.Bind(wx.EVT_BUTTON, self.on_probe_trial)
        self.nbv_mode_button.Bind(wx.EVT_TOGGLEBUTTON, self.on_nbv_mode)
        self.latest_map_button.Bind(wx.EVT_BUTTON, self.on_load_latest_map)
        self.use_calibration_button.Bind(
            wx.EVT_BUTTON, self.on_use_calibration)
        self.calibration_button.Bind(wx.EVT_BUTTON, self.on_calibration)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Bind(wx.EVT_CHAR_HOOK, self.on_char_hook)
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)
        self.timer.Start(250)
        self.camera_runtime.probe_async()
        threading.Thread(
            target=self._load_quick_runtime_values, daemon=True).start()

    def _load_quick_runtime_values(self):
        result = self.ros_bridge.request_runtime_config('get')
        if result.get('success'):
            values = result.get('values', {})
            value = float(values.get(
                'tool_timeout_s', 10.0))
            wx.CallAfter(self.tool_wait_spin.SetValue, value)
            speed = int(round(100.0 * float(values.get(
                'velocity_scaling', self.global_speed_percent / 100.0))))
            wx.CallAfter(self.apply_global_speed, speed, False)
            recovery_speed = int(round(100.0 * float(values.get(
                'active_recovery_velocity_scaling',
                self.recovery_speed_percent / 100.0))))
            wx.CallAfter(
                self.apply_recovery_speed, recovery_speed, False)
            batch_limit = int(values.get(
                'automatic_maximum_cycles_per_run', 3))
            wx.CallAfter(self.batch_limit_spin.SetValue, batch_limit)

    def on_tool_wait_apply(self, _event):
        if self.tool_wait_apply_inflight:
            return
        value = max(0.01, min(120.0, float(
            self.tool_wait_spin.GetValue())))
        self.tool_wait_apply_inflight = True
        self.tool_wait_apply_button.Enable(False)

        def worker():
            result = self.ros_bridge.request_runtime_config(
                'apply', {'tool_timeout_s': value})
            wx.CallAfter(self._tool_wait_applied, result)

        threading.Thread(target=worker, daemon=True).start()

    def _tool_wait_applied(self, result):
        self.tool_wait_apply_inflight = False
        if result.get('success'):
            value = float(result.get('values', {}).get(
                'tool_timeout_s', self.tool_wait_spin.GetValue()))
            self.tool_wait_spin.SetValue(value)
        self.tool_wait_apply_button.Enable(True)

    def on_batch_limit_apply(self, _event):
        if self.batch_limit_apply_inflight:
            return
        value = max(1, min(50, int(self.batch_limit_spin.GetValue())))
        self.batch_limit_apply_inflight = True
        self.batch_limit_apply_button.Enable(False)

        def worker():
            result = self.ros_bridge.request_runtime_config('apply', {
                'automatic_maximum_cycles_per_run': value,
                'automatic_batch_lock_maximum_targets': value,
            })
            wx.CallAfter(self._batch_limit_applied, result)

        threading.Thread(target=worker, daemon=True).start()

    def _batch_limit_applied(self, result):
        self.batch_limit_apply_inflight = False
        if result.get('success'):
            values = result.get('values', {})
            value = int(values.get(
                'automatic_maximum_cycles_per_run',
                self.batch_limit_spin.GetValue()))
            self.batch_limit_spin.SetValue(value)
            self.state.set_action_status(
                '连续夹取数已设为 %d 颗；下次 AUTO 生效' % value)
        else:
            self.state.set_action_status(
                result.get('message') or '连续夹取数未修改')
        self.batch_limit_apply_button.Enable(True)

    def on_rgb_camera(self, _event):
        snapshot = self.camera_runtime.snapshot()
        desired = bool(self.rgb_camera_button.GetValue())
        if desired:
            self.camera_runtime.start_async(mode='rgb')
        elif snapshot['running'] and snapshot['active_mode'] == 'rgb':
            self.camera_runtime.stop_async()

    def on_full_vision(self, _event):
        snapshot = self.camera_runtime.snapshot()
        desired = bool(self.vision_camera_button.GetValue())
        if desired:
            self.camera_runtime.start_async(mode='full')
        elif snapshot['running'] and snapshot['active_mode'] == 'full':
            self.camera_runtime.stop_async()

    def on_execute(self, _event):
        desired = bool(self.execute_button.GetValue())
        if desired:
            answer = wx.MessageBox(
                '确认扫掠区清空、末端与线缆固定、急停和电闸可立即触达，并有专人值守？',
                '二次清场确认',
                wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING)
            if answer != wx.YES:
                self.execute_button.SetValue(False)
                return
        self.execute_request_inflight = True
        self.execute_button.Enable(False)
        command = 'execute_enable' if desired else 'execute_disable'

        def worker():
            result = self.ros_bridge.request_harvest_command(command, -1)
            wx.CallAfter(self._execute_finished, desired, result)

        threading.Thread(target=worker, daemon=True).start()

    def _execute_finished(self, desired, result):
        self.execute_request_inflight = False
        success, _message = result
        if not success:
            self.execute_button.SetValue(not desired)
        self.execute_button.Enable(True)

    def on_global_speed(self, _event):
        if self.speed_slider is None:
            return
        value = int(self.speed_slider.GetValue())
        self.global_speed_percent = value
        self.speed_label.SetLabel('%d%%' % value)
        if self.speed_apply_later is not None:
            self.speed_apply_later.Stop()
        self.speed_apply_later = wx.CallLater(
            350, self.apply_global_speed, value)

    def apply_global_speed(self, value, write_runtime=True):
        value = int(max(1, min(100, value)))
        self.global_speed_percent = value
        if self.speed_slider is not None:
            self.speed_slider.SetValue(value)
        if self.speed_label is not None:
            self.speed_label.SetLabel('%d%%' % value)
        if self.cockpit_dialog is not None and \
                not self.cockpit_dialog.closed:
            self.cockpit_dialog.plan_speed_slider.SetValue(value)
            self.cockpit_dialog.plan_speed_label.SetLabel('%d%%' % value)
        if write_runtime:
            threading.Thread(
                target=self.ros_bridge.request_runtime_config,
                args=('apply', {'velocity_scaling': value / 100.0,
                                'acceleration_scaling': value / 100.0}),
                daemon=True).start()

    def on_recovery_speed(self, _event):
        if self.recovery_speed_slider is None:
            return
        value = int(self.recovery_speed_slider.GetValue())
        self.recovery_speed_percent = value
        self.recovery_speed_label.SetLabel('%d%%' % value)
        if self.recovery_speed_apply_later is not None:
            self.recovery_speed_apply_later.Stop()
        self.recovery_speed_apply_later = wx.CallLater(
            350, self.apply_recovery_speed, value)

    def apply_recovery_speed(self, value, write_runtime=True):
        value = int(max(1, min(100, value)))
        self.recovery_speed_percent = value
        if self.recovery_speed_slider is not None:
            self.recovery_speed_slider.SetValue(value)
        if self.recovery_speed_label is not None:
            self.recovery_speed_label.SetLabel('%d%%' % value)
        if self.cockpit_dialog is not None and \
                not self.cockpit_dialog.closed:
            self.cockpit_dialog.recovery_speed_slider.SetValue(value)
            self.cockpit_dialog.recovery_speed_label.SetLabel('%d%%' % value)
        if write_runtime:
            threading.Thread(
                target=self.ros_bridge.request_runtime_config,
                args=('apply', {
                    'active_recovery_velocity_scaling': value / 100.0}),
                daemon=True).start()

    def on_cockpit(self, _event):
        if self.cockpit_dialog is not None:
            self.cockpit_dialog.Show()
            self.cockpit_dialog.Raise()
            self.cockpit_dialog.SetFocus()
            return
        self.cockpit_dialog = CockpitDialog(self)
        self.cockpit_dialog.Show()
        self.cockpit_dialog.Raise()
        self.cockpit_dialog.SetFocus()
        if self.ros_bridge.nbv_auto_enable_on_cockpit and \
                not self.ros_bridge.nbv_recorder.enabled:
            self.nbv_mode_button.SetValue(True)
            self.on_nbv_mode(None)

    def on_rviz(self, _event):
        if self.rviz_button.GetValue():
            self.start_rviz()
        else:
            self.stop_rviz()

    def start_rviz(self):
        if self.rviz_process is not None and self.rviz_process.poll() is None:
            return
        command = [
            'roslaunch', 'elfin5_moveit_config', 'moveit_rviz.launch',
            'config:=true', 'config_file:=%s' % self.rviz_config,
            'use_egl:=%s' % ('true' if self.egl_check.GetValue() else 'false'),
        ]
        try:
            environment = os.environ.copy()
            # The dashboard node lives below /elfin_vision, while the hardware
            # MoveIt stack and its planning-scene services are global. RViz
            # must not inherit the dashboard's ROS namespace.
            environment.pop('ROS_NAMESPACE', None)
            self.rviz_process = subprocess.Popen(
                command, start_new_session=True, env=environment)
            self.rviz_button.SetValue(True)
            self.rviz_button.SetLabel('关闭 RViz')
            self.egl_check.Enable(False)
            self.state.set_action_status(
                'RViz 已启动%s' % ('（EGL）' if self.egl_check.GetValue() else ''))
        except Exception as error:
            self.rviz_process = None
            self.rviz_button.SetValue(False)
            self.rviz_button.SetLabel('启动 RViz')
            self.egl_check.Enable(True)
            self.state.set_action_status('RViz 启动失败：%s' % error)

    def _stop_rviz_process(self):
        process = self.rviz_process
        self.rviz_process = None
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=4.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                pass

    def stop_rviz(self):
        self._stop_rviz_process()
        self.rviz_button.SetValue(False)
        self.rviz_button.SetLabel('启动 RViz')
        self.egl_check.Enable(True)

    def on_plan(self, _event):
        selected = self._selected_target_request()
        if selected is None:
            self.state.set_action_status('该目标没有有效机器人坐标，不能规划')
            return
        threading.Thread(target=self.ros_bridge.request_plan, args=selected,
                         daemon=True).start()

    def on_target_choice(self, _event):
        selection = self.target_choice.GetStringSelection()
        enabled = selection.isdigit() and self.target_valid.get(int(selection), False)
        if enabled and self.current_targets is not None:
            target = self.current_targets.targets[int(selection)]
            self.selected_target_point = (
                target.target_point.x,
                target.target_point.y,
                target.target_point.z,
            )
        self.plan_button.Enable(enabled)

    def _selected_target_request(self):
        selection = self.target_choice.GetStringSelection()
        if not selection.isdigit() or not self.target_valid.get(int(selection), False):
            return None
        index = int(selection)
        if self.current_targets is None or index >= len(self.current_targets.targets):
            return None
        target = self.current_targets.targets[index]
        point = (target.target_point.x, target.target_point.y,
                 target.target_point.z)
        return index, point, target.target_frame, target.label

    def on_run(self, _event):
        selected = self._selected_target_request()
        if selected is None:
            self.state.set_action_status('该目标没有有效机器人坐标，不能采摘')
            return
        answer = wx.MessageBox(
            '将先把所选目标移入中央视野，并执行预抓取、直线伸入、撤回和'
            '返回视角四段完整预检；所有轨迹点还会检查活动连杆、法兰和夹取 '
            'TCP 的最低高度。通过后才会执行真机路径并触发 DO0。\n\n'
            '确认扫掠区清空、急停和电闸有人值守？',
            '确认采摘所选目标', wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING)
        if answer != wx.YES:
            return
        threading.Thread(
            target=self.ros_bridge.request_harvest_command,
            args=('run',) + selected, daemon=True).start()

    def on_auto(self, _event):
        desired = bool(self.auto_button.GetValue())
        if desired:
            answer = wx.MessageBox(
                '连续模式会一次锁定当前视野内多颗柑橘的基座坐标，按绿框面积从大到小逐颗执行；'
                '进入树冠后即使相机看不见其余目标也不会改写本批坐标。\n\n'
                '每颗仍会独立做完整路径预检，并在末端动作后按两段实际轨迹原路返回'
                '本颗开始前视角。确认扫掠区持续清空、急停和电闸持续有人值守？',
                '确认自动循环', wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING)
            if answer != wx.YES:
                self.auto_button.SetValue(False)
                return
            camera = self.camera_runtime.snapshot()
            if camera['connected'] and (
                    not camera['running'] or
                    camera['active_mode'] != 'full'):
                self.camera_runtime.start_async(mode='full')
        self.auto_request_inflight = True
        command = 'auto_start' if desired else 'auto_stop'
        continuous = bool(self.continuous_check.GetValue())
        patrol = bool(self.patrol_check.GetValue())

        def worker():
            result = self.ros_bridge.request_harvest_command(
                command, -1, continuous=continuous, patrol=patrol)
            wx.CallAfter(self._auto_finished, desired, result)

        threading.Thread(target=worker, daemon=True).start()

    def _auto_finished(self, desired, result):
        self.auto_request_inflight = False
        success, _message = result
        if not success:
            self.auto_button.SetValue(not desired)

    def on_stop(self, _event):
        self.ros_bridge.stop_cockpit_jog()
        threading.Thread(
            target=self.ros_bridge.request_planner_stop,
            daemon=True).start()
        if self.cockpit_dialog is not None:
            self.cockpit_dialog.Close()
        threading.Thread(
            target=self.ros_bridge.request_cockpit_active,
            args=(False,), daemon=True).start()
        threading.Thread(
            target=self.ros_bridge.request_harvest_command,
            args=('stop', -1), daemon=True).start()

    def on_capture(self, _event):
        threading.Thread(target=self.ros_bridge.save_capture, daemon=True).start()

    def on_approach(self, _event):
        if self.approach_dialog is not None:
            self.approach_dialog.Raise()
            self.approach_dialog.SetFocus()
            return
        self.approach_dialog = ApproachDirectionDialog(self)
        self.approach_dialog.Show()
        self.approach_dialog.Raise()

    def on_probe_trial(self, _event):
        if self.probe_dialog is not None:
            self.probe_dialog.Raise()
            self.probe_dialog.SetFocus()
            return
        self.probe_dialog = ProbeTrialDialog(self)
        self.probe_dialog.Show()
        self.probe_dialog.Raise()

    def on_nbv_mode(self, _event):
        desired = bool(self.nbv_mode_button.GetValue())
        if self.nbv_toggle_inflight:
            return
        if desired:
            camera = self.camera_runtime.snapshot()
            if not camera['running'] or camera['active_mode'] != 'full':
                # Starting full RGB-D is asynchronous; the recorder itself
                # still refuses Enter until fresh frames and TF exist.
                self.camera_runtime.start_async(mode='full')
            scene_id = self.nbv_scene_text.GetValue().strip()
            notes = self.nbv_notes_text.GetValue().strip()
        else:
            scene_id, notes = '', ''
        self.nbv_toggle_inflight = True
        self.nbv_mode_button.Enable(False)

        def worker():
            status = self.ros_bridge.set_nbv_mode(desired, scene_id, notes)
            wx.CallAfter(self._nbv_mode_finished, desired, status)

        threading.Thread(target=worker, name='nbv-mode-toggle', daemon=True).start()

    def _nbv_mode_finished(self, desired, status):
        self.nbv_toggle_inflight = False
        self.nbv_mode_button.SetValue(bool(status.get('enabled', False)))
        self.nbv_mode_button.Enable(True)
        self._update_nbv_controls(status)

    def _update_nbv_controls(self, status):
        status = dict(status or {})
        enabled = bool(status.get('enabled', False))
        self.nbv_mode_button.SetValue(enabled)
        self.nbv_mode_button.SetLabel(
            '论文采集 ON' if enabled else '论文采集 OFF')
        count = int(status.get('view_count', 0))
        maximum = int(status.get('max_views', 10))
        group = int(status.get('group_number', 1))
        scene = str(status.get('scene_id', '') or '未命名场景')
        result = str(status.get('last_result', '') or '')
        error = str(status.get('last_error', '') or '')
        camera = status.get('camera') or {}
        readiness = status.get('readiness') or {}
        camera_line = ('相机 %s / %s / USB %s；%s；%s；%s' % (
            camera.get('device_type') or camera.get('name') or '未识别',
            camera.get('serial') or '无序列号',
            camera.get('usb_type') or '未知',
            '标定匹配' if readiness.get('calibration_serial_matches')
            else '标定未匹配',
            'HW READY' if readiness.get('ready') else 'HW NOT READY',
            '语义完整门禁' if readiness.get('semantics_required')
            else '诊断采集'))
        imu = status.get('imu') or {}
        imu_line = ('IMU：gyro/accel 已收到' if
                    imu.get('gyro_seen') and imu.get('accel_seen') else
                    'IMU：当前未收到 gyro/accel；试拍会记录缺口')
        line = ('场景 %s / 批次 %04d / 当前视角 %d/%d；Enter/实体POINT=单次快照' %
                (scene, group, min(count + 1, maximum), maximum))
        hint = ('视角提示：改变距离/偏航/俯仰，覆盖上一视角看不到的果实和枝叶；'
                '2 批仅链路试拍，18 批工程最低，24 批趋势最低。')
        if error:
            hint += ' ' + error
        elif result:
            hint += ' ' + result
        self.nbv_status_label.SetLabel(
            line + '\n' + camera_line + '\n' + imu_line + '\n' + hint)
        self.nbv_status_label.Wrap(max(320, self.nbv_status_label.GetSize().width))

    def on_calibration(self, _event):
        if self.calibration_request_inflight:
            return
        if self.ros_bridge.calibration_running():
            self.ros_bridge.stop_calibration()
            self.calibration_button.SetLabel('D455 重新标定')
            self.state.set_action_status('手眼标定窗口已请求关闭；live 配置未因关闭而改变')
            return
        answer = wx.MessageBox(
            '将启动只读 D455 Charuco 手眼标定。标定窗口不会驱动机械臂；'
            '只有质量通过且当前序列号匹配时，才会备份旧 camera_to_robot.yaml 并安装新外参。'
            '\n\n请确认标定板已固定、机械臂由人工安全保持静止/缓慢换位。',
            '确认启动 D455 手眼标定',
            wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING)
        if answer != wx.YES:
            return
        self.calibration_request_inflight = True
        self.calibration_button.Enable(False)

        def worker():
            result = self.ros_bridge.start_calibration(install_live=True)
            wx.CallAfter(self._calibration_finished, result)

        threading.Thread(target=worker, name='start-eye-hand-calibration',
                         daemon=True).start()

    def on_use_calibration(self, _event):
        if self.calibration_reuse_inflight or \
                self.ros_bridge.calibration_running():
            return
        self.calibration_reuse_inflight = True
        self.use_calibration_button.Enable(False)

        def worker():
            result = self.ros_bridge.use_current_or_latest_calibration()
            wx.CallAfter(self._use_calibration_finished, result)

        threading.Thread(
            target=worker, name='reuse-eye-hand-calibration',
            daemon=True).start()

    def _use_calibration_finished(self, result):
        self.calibration_reuse_inflight = False
        self.use_calibration_button.Enable(True)
        self.state.set_action_status(str(result[1]))

    def on_load_latest_map(self, _event):
        if self.latest_map_load_inflight:
            return
        self.latest_map_load_inflight = True
        self.latest_map_button.Enable(False)

        def worker():
            result = self.ros_bridge.load_latest_complete_approach_map()
            wx.CallAfter(self._load_latest_map_finished, result)

        threading.Thread(
            target=worker, name='load-latest-approach-map',
            daemon=True).start()

    def _load_latest_map_finished(self, result):
        self.latest_map_load_inflight = False
        self.latest_map_button.Enable(True)
        self.state.set_action_status(str(result[1]))

    def _calibration_finished(self, result):
        self.calibration_request_inflight = False
        success, message = result
        self.calibration_button.Enable(True)
        self.calibration_button.SetLabel(
            '停止标定窗口' if success else 'D455 重新标定')
        self.state.set_action_status(message)
        self.ros_bridge.refresh_nbv_status()

    def on_experiment(self, _event):
        if self.experiment_dialog is not None:
            self.experiment_dialog.Show()
            self.experiment_dialog.Raise()
            return
        self.experiment_dialog = ExperimentConsoleDialog(self)
        self.experiment_dialog.Show()

    def start_demo_command(self, command):
        if command == 'stop':
            self.on_stop(None)
            return
        arguments = (command, -1)
        if command in ('demo_lock', 'demo_preview'):
            snapshot = self.state.snapshot()
            harvest = snapshot['harvest_status']
            use_locked = (command == 'demo_preview' and harvest is not None and
                          bool(harvest.target_locked))
            if not use_locked:
                selected = self._selected_target_request()
                if selected is None:
                    self.state.set_action_status(
                        '请先选择一个具有机器人坐标的稳定目标')
                    return
                arguments = (command,) + selected
        threading.Thread(
            target=self.ros_bridge.request_harvest_command,
            args=arguments, daemon=True).start()

    def on_close(self, event):
        if self.closing:
            event.Skip()
            return
        self.closing = True
        self.timer.Stop()
        if self.cockpit_dialog is not None:
            self.cockpit_dialog.Close()
        if self.approach_dialog is not None:
            self.approach_dialog.Close()
        if self.probe_dialog is not None:
            self.probe_dialog.Close()
        self.Hide()
        self.ros_bridge.panel_jog.close()

        def cleanup():
            self.ros_bridge.request_planner_stop()
            if not rospy.is_shutdown() and not self.external_shutdown_requested:
                # Closing the only operator terminal is fail-closed, but all
                # ROS waits run outside wx so the window disappears instantly.
                self.ros_bridge.request_cockpit_active(False)
                self.ros_bridge.request_harvest_command(
                    'execute_disable', -1)
            self.camera_runtime.close()
            self.ros_bridge.stop_calibration()
            # wx controls may already have been destroyed after event.Skip().
            # The bounded background cleanup must only stop the child process;
            # all widget updates remain on the wx event thread.
            self._stop_rviz_process()
            if not rospy.is_shutdown():
                rospy.signal_shutdown('dashboard closed')

        threading.Thread(
            target=cleanup, name='dashboard-bounded-cleanup',
            daemon=False).start()
        event.Skip()

    def on_char_hook(self, event):
        if event.ControlDown() and int(event.GetKeyCode()) in (
                ord('C'), ord('c')):
            self.Close()
            return
        event.Skip()

    def close_from_ros(self):
        if not self.closing:
            self.Close()

    @staticmethod
    def _point_text(point):
        return '%.3f, %.3f, %.3f' % (point.x, point.y, point.z)

    def update_targets(self, message):
        previous_point = self.selected_target_point
        self.target_list.DeleteAllItems()
        self.target_valid = {}
        self.current_targets = message
        if message is None:
            self.target_choice.Set(['--'])
            self.target_choice.SetSelection(0)
            self.selected_target_point = None
            self.plan_button.Enable(False)
            return
        for index, target in enumerate(message.targets):
            row = self.target_list.InsertItem(index, str(index))
            self.target_list.SetItem(row, 1, target.label)
            self.target_list.SetItem(row, 2, '%.2f' % target.confidence)
            self.target_list.SetItem(row, 3,
                                     '%.3f' % target.depth_m if target.depth_m > 0 else '--')
            camera_text = self._point_text(target.camera_point) \
                if target.depth_m > 0.0 else '--'
            self.target_list.SetItem(row, 4, camera_text)
            robot_text = self._point_text(target.target_point) \
                if target.target_point_valid else '未标定/无 TF'
            self.target_list.SetItem(row, 5, robot_text)
            self.target_valid[index] = bool(target.target_point_valid)
        choices = [str(index) for index in range(len(message.targets))] or ['--']
        self.target_choice.Set(choices)
        valid_indices = [index for index, valid in self.target_valid.items() if valid]
        selected = -1
        if previous_point is not None and valid_indices:
            selected = min(
                valid_indices,
                key=lambda index: np.linalg.norm(np.asarray(previous_point) -
                                                  np.asarray((
                                                      message.targets[index].target_point.x,
                                                      message.targets[index].target_point.y,
                                                      message.targets[index].target_point.z,
                                                  ))))
            selected_point = message.targets[selected].target_point
            separation = np.linalg.norm(
                np.asarray(previous_point) - np.asarray((
                    selected_point.x, selected_point.y, selected_point.z)))
            if separation > 0.080:
                selected = -1
        elif valid_indices:
            selected = valid_indices[0]
        self.target_choice.SetSelection(selected)
        if selected >= 0:
            target = message.targets[selected]
            self.selected_target_point = (
                target.target_point.x,
                target.target_point.y,
                target.target_point.z,
            )
        self.plan_button.Enable(selected >= 0)

    def update_joints(self, message):
        if message is None or not message.name:
            self.joint_text.SetLabel('关节状态：等待 /joint_states')
            return
        values = []
        for name, position in zip(message.name, message.position):
            if name.startswith('elfin_joint'):
                values.append('%s %.1f°' % (name.replace('elfin_joint', 'J'),
                                            np.degrees(position)))
        if not values:
            values = ['%s %.1f°' % (name, np.degrees(position))
                      for name, position in zip(message.name[:6], message.position[:6])]
        self.joint_text.SetLabel('关节状态：' + '   '.join(values))
        self.joint_text.Wrap(max(300, self.joint_text.GetSize().width))

    def on_timer(self, _event):
        self.ros_bridge.publish_dashboard_heartbeat()
        self.camera_runtime.poll()
        camera = self.camera_runtime.snapshot()
        self.ros_bridge.update_camera_identity(camera.get('device', {}))
        self.ros_bridge.camera_display_mode = (
            camera['active_mode'] if camera['running'] else 'off')
        if not camera['running'] and self._camera_ui_running is not False:
            for view in (self.color_view, self.depth_view,
                         self.annotated_view):
                view.clear()
            self.seen_versions = {'color': -1, 'depth': -1, 'annotated': -1}
        self._camera_ui_running = camera['running']

        if camera['busy'] in ('starting', 'switching'):
            camera_run_text = '正在开启'
        elif camera['busy'] == 'stopping':
            camera_run_text = '正在关闭'
        elif camera['running']:
            camera_run_text = (
                'RGB 目视运行中' if camera['active_mode'] == 'rgb'
                else '完整视觉运行中')
        else:
            camera_run_text = '已关闭'
        if camera['probing'] and not camera['connected']:
            connection_text = '检测中'
        else:
            connection_text = '已连接' if camera['connected'] else '未连接'
        self.camera_status_label.SetLabel(
            '摄像头：%s / %s / %s %dx%d@%d' % (
                connection_text, camera_run_text, camera['profile']['name'],
                camera['profile']['width'], camera['profile']['height'],
                camera['profile']['fps']))
        self.camera_status_label.Wrap(
            max(360, self.camera_status_label.GetSize().width))
        rgb_active = bool(
            camera['running'] and camera['active_mode'] == 'rgb')
        vision_active = bool(
            camera['running'] and camera['active_mode'] == 'full')
        self.rgb_camera_button.SetValue(rgb_active)
        self.vision_camera_button.SetValue(vision_active)
        self.rgb_camera_button.SetLabel(
            '关闭 RGB' if rgb_active else
            ('切换中...' if camera['busy'] else '睁眼 RGB'))
        self.vision_camera_button.SetLabel(
            '关闭自动视觉' if vision_active else
            ('切换中...' if camera['busy'] else '自动视觉'))
        camera_controls_enabled = bool(
            not camera['busy'] and
            (camera['running'] or camera['connected']))
        self.rgb_camera_button.Enable(camera_controls_enabled)
        self.vision_camera_button.Enable(camera_controls_enabled)

        if self.rviz_process is not None and self.rviz_process.poll() is not None:
            self.rviz_process = None
            self.rviz_button.SetValue(False)
            self.rviz_button.SetLabel('启动 RViz')
            self.egl_check.Enable(True)
            self.state.set_action_status('RViz 已关闭；不会自动重新启动')
        snapshot = self.state.snapshot()
        self._update_nbv_controls(snapshot.get('nbv_status', {}))
        if self.ros_bridge.calibration_running():
            self.calibration_button.SetLabel('停止标定窗口')
        elif not self.calibration_request_inflight:
            self.calibration_button.SetLabel('D455 重新标定')
        self.use_calibration_button.Enable(
            not self.calibration_reuse_inflight and
            not self.ros_bridge.calibration_running())
        approach_busy = bool(
            (snapshot.get('approach_status') or {}).get('computing', False))
        self.latest_map_button.Enable(
            not self.latest_map_load_inflight and not approach_busy)
        cockpit_visible = self.cockpit_dialog is not None and \
            self.cockpit_dialog.IsShown()
        if not cockpit_visible:
            views = {'color': self.color_view, 'depth': self.depth_view,
                     'annotated': self.annotated_view}
            for key, view in views.items():
                version = snapshot['versions'][key]
                if version != self.seen_versions[key] and \
                        snapshot['images'][key] is not None:
                    view.set_bgr(snapshot['images'][key])
                    self.seen_versions[key] = version

        targets = snapshot['targets']
        target_version = (-1 if targets is None else
                          (targets.header.seq, targets.header.stamp.to_nsec()))
        if getattr(self, '_target_version', None) != target_version:
            self.update_targets(targets)
            self._target_version = target_version
        joints = snapshot['joints']
        joint_version = (-1 if joints is None else
                         (joints.header.seq, joints.header.stamp.to_nsec()))
        if getattr(self, '_joint_version', None) != joint_version:
            self.update_joints(joints)
            self._joint_version = joint_version

        count = 0 if targets is None else len(targets.targets)
        valid = 0 if targets is None else sum(
            1 for item in targets.targets if item.target_point_valid)
        harvest = snapshot['harvest_status']
        if harvest is None:
            harvest_line = '状态机：未启动（视觉只规划模式）'
            blockers_line = ''
        else:
            harvest_line = ('状态机：%s / %s；演示 %s；%s；完成 %d，失败 %d' %
                            (harvest.mode, harvest.phase,
                             harvest.manual_stage, harvest.message,
                             harvest.completed_cycles,
                             harvest.failed_cycles))
            blocker_label = '自动采摘等待' if cockpit_visible else '门禁'
            blockers_line = ('\n%s：' % blocker_label +
                             '；'.join(harvest.blockers)) \
                if harvest.blockers else ''
        device = camera['device']
        vision_status = (
            'RGB 驾驶低负载：深度、点云与模型推理均未启动'
            if camera['running'] and camera['active_mode'] == 'rgb'
            else snapshot['vision_status'])
        status = (
            '相机：%s / %s；%s；序列号 %s；接口 USB %s @ %s\n'
            '数据：RGB %.1f FPS，深度 %.1f FPS\n'
            '识别：%s（目标 %d，机器人坐标有效 %d）\n'
            '规划：%s\n'
            '轨迹显示：%d 段，%d 个轨迹点\n'
            '%s%s'
        ) % (connection_text, camera_run_text,
             camera['message'] or device.get('name', 'RealSense'),
             device.get('serial') or self.ros_bridge.camera_serial or '自动',
             device.get('usb_type') or '自动',
             device.get('usb_port_id') or '自动',
             snapshot['fps']['color'], snapshot['fps']['depth'],
             vision_status, count, valid,
             snapshot['planner_status'], snapshot['trajectory_count'],
             snapshot['trajectory_points'], harvest_line, blockers_line)
        self.status_text.SetValue(status)

        if getattr(self, '_event_version', -1) != snapshot['event_version']:
            self.event_log.update_lines(snapshot['events'])
            self._event_version = snapshot['event_version']

        selection = self.target_choice.GetStringSelection()
        selected_valid = (selection.isdigit() and
                          self.target_valid.get(int(selection), False))
        busy = bool(harvest.busy) if harvest is not None else False
        target_locked = bool(harvest.target_locked) \
            if harvest is not None else False
        execution_ready = bool(harvest.execution_ready) \
            if harvest is not None else False
        execution_capable = bool(harvest.execution_capable) \
            if harvest is not None else False
        execution_requested = bool(harvest.execution_requested) \
            if harvest is not None else False
        auto_active = bool(harvest is not None and harvest.mode == 'AUTO')
        cockpit = snapshot['cockpit_status']
        cockpit_active = bool(cockpit.active) if cockpit is not None else False
        mode_colour = wx.Colour(38, 155, 76)
        if cockpit_active:
            self.mode_badge.SetLabel('人工驾驶舱 ACTIVE')
            self.mode_badge.SetBackgroundColour(mode_colour)
            self.mode_badge.SetForegroundColour(wx.WHITE)
        elif auto_active:
            self.mode_badge.SetLabel('自动识别采摘 ACTIVE')
            self.mode_badge.SetBackgroundColour(mode_colour)
            self.mode_badge.SetForegroundColour(wx.WHITE)
        elif vision_active:
            self.mode_badge.SetLabel('自动视觉待命')
            self.mode_badge.SetBackgroundColour(wx.Colour(51, 122, 183))
            self.mode_badge.SetForegroundColour(wx.WHITE)
        elif rgb_active:
            self.mode_badge.SetLabel('RGB 人工目视')
            self.mode_badge.SetBackgroundColour(wx.Colour(51, 122, 183))
            self.mode_badge.SetForegroundColour(wx.WHITE)
        else:
            self.mode_badge.SetLabel('系统待命')
            self.mode_badge.SetBackgroundColour(wx.Colour(205, 209, 214))
            self.mode_badge.SetForegroundColour(wx.Colour(55, 60, 65))

        if not self.execute_request_inflight:
            self.execute_button.SetValue(execution_requested)
        self.execute_button.SetLabel(
            'EXECUTE ON' if execution_requested else 'EXECUTE OFF')
        self.execute_button.SetBackgroundColour(
            wx.Colour(42, 145, 72) if execution_requested else wx.NullColour)
        self.execute_button.SetToolTip(
            '计划模式：自动轨迹和 DO0 固定关闭；手动 Panel 驾驶舱独立可用'
            if not execution_capable else
            '开启时在本窗口完成二次清场确认；关闭即撤销自动真机授权')
        self.execute_button.Enable(
            execution_capable and
            (execution_requested or (not busy and not target_locked)))

        if not self.auto_request_inflight:
            self.auto_button.SetValue(auto_active)
        self.auto_button.SetLabel('AUTO ON' if auto_active else 'AUTO OFF')
        self.auto_button.SetBackgroundColour(
            wx.Colour(42, 145, 72) if auto_active else wx.NullColour)
        if harvest is not None and auto_active:
            self.continuous_check.SetValue(bool(harvest.continuous_enabled))
            self.patrol_check.SetValue(bool(harvest.patrol_enabled))
        self.continuous_check.Enable(not auto_active and not busy)
        self.patrol_check.Enable(not auto_active and not busy)
        self.batch_limit_spin.Enable(not auto_active and not busy)
        if not self.batch_limit_apply_inflight:
            self.batch_limit_apply_button.Enable(not auto_active and not busy)
        self.tool_wait_spin.Enable(not busy)
        if not self.tool_wait_apply_inflight:
            self.tool_wait_apply_button.Enable(not busy)

        self.plan_button.Enable(
            selected_valid and not busy and not target_locked and
            not cockpit_active)
        self.run_button.Enable(
            selected_valid and execution_ready and not busy and
            not target_locked and not cockpit_active)
        self.auto_button.Enable(
            auto_active or (execution_ready and not busy and
                            not target_locked and not cockpit_active))
        self.stop_button.Enable(busy or cockpit_active)
        if self.experiment_dialog is not None:
            self.experiment_dialog.update_status(snapshot, selected_valid)
        if cockpit_visible:
            if cockpit_active:
                self.safety.SetLabel('驾驶舱 ACTIVE；自动采摘环境状态不限制人工驾驶')
                self.safety.SetForegroundColour(wx.Colour(20, 110, 45))
            else:
                self.safety.SetLabel('驾驶舱正在自动接管；等待实时机器人状态')
                self.safety.SetForegroundColour(wx.Colour(154, 62, 0))
        elif harvest is None or not execution_requested:
            self.safety.SetLabel('只规划；真机轨迹与 DO0 关闭')
            self.safety.SetForegroundColour(wx.Colour(154, 62, 0))
        elif harvest.execution_ready:
            self.safety.SetLabel('真机执行门禁已通过；每段仍会重新核验')
            self.safety.SetForegroundColour(wx.Colour(20, 110, 45))
        else:
            self.safety.SetLabel('真机执行已请求，但门禁未通过')
            self.safety.SetForegroundColour(wx.Colour(170, 35, 35))
        self.action_text.SetLabel(snapshot['action_status'])


def main():
    rospy.init_node('elfin_vision_dashboard', disable_signals=True)
    if not os.environ.get('DISPLAY'):
        rospy.logerr('DISPLAY is not set; the wx dashboard needs a local desktop or X forwarding')
        return
    state = SharedState()
    ros_bridge = RosBridge(state)
    app = wx.App(False)
    frame = DashboardFrame(state, ros_bridge)
    def close_from_ros():
        # rospy can finish after wx has already destroyed its global App.
        # Calling wx.CallAfter in that state raises "No wx.App created yet".
        try:
            if wx.GetApp() is not None:
                wx.CallAfter(frame.close_from_ros)
        except (AssertionError, RuntimeError):
            pass
    rospy.on_shutdown(close_from_ros)
    def shutdown_signal(signum, _frame):
        # Never run rospy shutdown synchronously inside the wx main thread's
        # POSIX signal handler. That can prevent the queued Close event from
        # being dispatched and leave an orphaned GUI after roslaunch exits.
        frame.external_shutdown_requested = True
        wx.CallAfter(frame.close_from_ros)
        def request_shutdown():
            if not rospy.is_shutdown():
                rospy.signal_shutdown('dashboard received signal %d' % signum)
        threading.Thread(
            target=request_shutdown, name='dashboard-signal-shutdown',
            daemon=True).start()
    signal.signal(signal.SIGINT, shutdown_signal)
    signal.signal(signal.SIGTERM, shutdown_signal)
    frame.Show()
    if frame.rviz_auto_start:
        wx.CallAfter(frame.start_rviz)
    app.MainLoop()


if __name__ == '__main__':
    main()
