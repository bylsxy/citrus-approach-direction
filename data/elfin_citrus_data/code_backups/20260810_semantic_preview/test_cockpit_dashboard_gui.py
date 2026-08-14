#!/usr/bin/env python3

import os
import json
import importlib.util
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import wx


SCRIPT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                      'scripts', 'elfin_vision_dashboard.py')
SPEC = importlib.util.spec_from_file_location(
    'elfin_vision_dashboard_script', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CockpitDialog = MODULE.CockpitDialog
ApproachDirectionDialog = MODULE.ApproachDirectionDialog
ProtectedLogView = MODULE.ProtectedLogView
RosBridge = MODULE.RosBridge
SharedState = MODULE.SharedState


class _FakeBridge(object):

    def __init__(self):
        self.lock = threading.Lock()
        self.action_states = []
        self.speeds = []
        self.message = 'fake Panel jog'

    def set_cockpit_actions(self, actions):
        with self.lock:
            self.action_states.append(tuple(actions))
        return True

    def set_panel_jog_speed(self, percent):
        with self.lock:
            self.speeds.append(int(percent))

    def cockpit_jog_message(self):
        return self.message

    @staticmethod
    def request_cockpit_active(_enabled):
        return True, 'fake'

    @staticmethod
    def request_runtime_config(_operation, _values=None):
        return {'success': True}

    @staticmethod
    def run_flange_snake_maneuver(reverse=False, base_duration_s=0.15):
        return True, 'fake flange snake %s %.2f' % (
            reverse, base_duration_s)

    @staticmethod
    def request_planner_stop():
        return True, 'fake'

    @staticmethod
    def request_harvest_command(_command, _target_index, **_kwargs):
        return True, 'fake'


class _FakeOwner(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None)
        self.state = SharedState()
        self.ros_bridge = _FakeBridge()
        self.global_speed_percent = 5
        self.recovery_speed_percent = 25
        self.cockpit_speed_percent = 50
        self.cockpit_dialog = None
        self.applied_global_speeds = []

    def apply_global_speed(self, value):
        self.applied_global_speeds.append(int(value))

    def apply_recovery_speed(self, value):
        self.recovery_speed_percent = int(value)


class _KeyEvent(object):

    def __init__(self, code):
        self.code = code
        self.skipped = False

    def GetKeyCode(self):
        return self.code

    @staticmethod
    def ControlDown():
        return False

    def Skip(self):
        self.skipped = True


class _ActivateEvent(object):

    def __init__(self, active):
        self.active = active

    def GetActive(self):
        return self.active

    @staticmethod
    def Skip():
        return None


class _CloseEvent(object):

    @staticmethod
    def Skip():
        return None


class CockpitBridgeCombinationTest(unittest.TestCase):

    def test_latest_complete_staging_ignores_newer_empty_batch(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            complete_dir = os.path.join(directory, '.nbv_staging_g0010')
            empty_dir = os.path.join(directory, '.nbv_staging_g0011')
            os.makedirs(complete_dir)
            os.makedirs(empty_dir)
            with open(os.path.join(complete_dir, 'manifest.json'), 'w',
                      encoding='utf-8') as stream:
                json.dump({
                    'complete': True, 'group_number': 10,
                    'view_count': 10,
                    'views': [{'index': index} for index in range(1, 11)],
                }, stream)
            with open(os.path.join(empty_dir, 'manifest.json'), 'w',
                      encoding='utf-8') as stream:
                json.dump({
                    'complete': False, 'group_number': 11,
                    'view_count': 0, 'views': [],
                }, stream)
            bridge = object.__new__(RosBridge)
            bridge.nbv_output_dir = directory
            bridge.approach_minimum_views = 10

            selected = bridge._latest_complete_staging()

            self.assertEqual(selected['group_number'], 10)

    def test_moveit_handoff_releases_controller_and_planner_claim(self):
        bridge = object.__new__(RosBridge)
        bridge.state = SharedState()
        with bridge.state.lock:
            bridge.state.cockpit_status = SimpleNamespace(active=True)
        bridge.set_cockpit_actions = mock.Mock()
        bridge.request_cockpit_active = mock.Mock(
            return_value=(True, 'controller released'))
        bridge.planner_cockpit_claim_service = '/fake/planner_claim'
        response = SimpleNamespace(success=True, message='planner released')
        client = mock.Mock(return_value=response)

        with mock.patch.object(MODULE.rospy, 'wait_for_service'), \
                mock.patch.object(MODULE.rospy, 'ServiceProxy',
                                  return_value=client):
            released = bridge.release_cockpit_for_moveit()

        self.assertTrue(released[0])
        bridge.set_cockpit_actions.assert_called_once_with(())
        bridge.request_cockpit_active.assert_called_once_with(False)
        client.assert_called_once_with(False)

    def test_cached_grasp_cycle_uses_runtime_dwell_value(self):
        bridge = object.__new__(RosBridge)
        bridge.approach_compute_lock = threading.Lock()
        bridge.approach_plan_cache = {
            'plan_id': 'approach-runtime-dwell',
            'result': {'safe': True, 'moveit_plan_id': 'approach-runtime-dwell'},
        }
        bridge.request_runtime_config = mock.Mock(return_value={
            'success': True, 'message': 'ok',
            'values': {'tool_timeout_s': 0.01},
        })
        bridge.approach_session = mock.Mock()
        bridge.approach_session.status.return_value = {'view_count': 10}
        bridge.approach_minimum_views = 10
        bridge.approach_discovered_targets = []
        bridge.state = SharedState()
        bridge.release_cockpit_for_moveit = mock.Mock(return_value=(True, 'ok'))
        response = SimpleNamespace(
            success=True, executed=True, message='ok', timings_json='{}')
        bridge._call_cached_approach_stage = mock.Mock(return_value=response)
        bridge._run_cached_tool_action = mock.Mock(return_value='done')
        bridge._persist_approach_result = mock.Mock(return_value='')
        bridge.approach_result_pub = mock.Mock()

        success, message = bridge.execute_cached_approach()

        self.assertTrue(success, message)
        result = bridge.state.snapshot()['approach_status']['result']
        self.assertAlmostEqual(result['grasp_dwell_requested_s'], 0.01)
        bridge.request_runtime_config.assert_called_once_with('get')
        self.assertEqual(
            [call.args[1] for call in
             bridge._call_cached_approach_stage.call_args_list],
            ['execute_outbound', 'execute_return'])
        self.assertEqual(
            [call.args[0] for call in bridge._run_cached_tool_action.call_args_list],
            ['cockpit_close', 'cockpit_open'])

    def test_approach_status_distinguishes_outward_and_insertion_vectors(self):
        dialog = SimpleNamespace(
            _vector_text=ApproachDirectionDialog._vector_text)
        status = {
            'message': '只规划通过',
            'view_count': 10,
            'minimum_views': 10,
            'map_summary': {
                'occupied_voxel_count': 100,
                'semantic_voxel_count': 40,
            },
            'result': {
                'target_center': [0.5, 0.1, 0.4],
                'target_radius_m': 0.04,
                'target_voxel_count': 80,
                'minimum_required_clearance_m': 0.21,
                'minimum_patch_angular_radius_deg': 25.0,
                'direction_candidate_ready': True,
                'geometry_candidate_ready': True,
                'moveit_message': 'MoveIt plan-only passed',
                'best': {
                    'outward_direction': [1.0, 0.0, 0.0],
                    'insertion_direction': [-1.0, 0.0, 0.0],
                    'clearance_m': 0.23,
                    'light_transmission': 0.9,
                    'local_light': 0.8,
                    'known_free_fraction': 0.7,
                    'unknown_fraction': 0.3,
                    'patch_angular_radius_deg': 28.0,
                },
            },
        }

        text = ApproachDirectionDialog._status_message(dialog, status)

        self.assertIn('果实 -> 外部亮斑：1.000, 0.000, 0.000', text)
        self.assertIn('机械臂实际伸入：-1.000, 0.000, 0.000', text)
        self.assertIn('候选已通过，可执行', text)

    def test_dashboard_nbv_metadata_has_math_dependency(self):
        self.assertTrue(MODULE.math.isfinite(0.0))
        self.assertFalse(MODULE.math.isfinite(float('inf')))

    def test_dashboard_selects_first_moveit_validated_direction_candidate(self):
        bridge = object.__new__(RosBridge)
        bridge.approach_plan_service = '/fake/plan_approach_direction'
        bridge.target_frame = 'elfin_base_link'
        candidates = [
            {'outward_direction': [1.0, 0.0, 0.0]},
            {'outward_direction': [0.0, 1.0, 0.0]},
        ]
        result = {'best': candidates[0], 'candidates': candidates}
        client = mock.Mock(side_effect=[
            SimpleNamespace(success=False, message='IK failed'),
            SimpleNamespace(success=True, message='plan-only passed',
                            endpoint_condition_number=5.5,
                            plan_id='approach-2', planning_duration_s=0.4,
                            timings_json='{}'),
        ])

        with mock.patch.object(MODULE.rospy, 'wait_for_service'), \
                mock.patch.object(MODULE.rospy, 'ServiceProxy',
                                  return_value=client):
            selected = bridge._validate_approach_with_moveit(
                result, (0.5, 0.1, 0.4), 'elfin_base_link')

        self.assertTrue(selected['moveit_validated'])
        self.assertEqual(selected['moveit_candidate_rank'], 2)
        self.assertEqual(selected['best']['outward_direction'], [0.0, 1.0, 0.0])
        self.assertAlmostEqual(
            selected['moveit_endpoint_condition_number'], 5.5)
        self.assertEqual(client.call_count, 2)
        self.assertFalse(client.call_args.kwargs['execute'])
        self.assertEqual(client.call_args.kwargs['command'], 'plan')
        self.assertEqual(selected['moveit_plan_id'], 'approach-2')

    def test_dashboard_selects_fastest_eta_and_labels_farther_backup(self):
        bridge = object.__new__(RosBridge)
        bridge.approach_plan_service = '/fake/plan_approach_direction'
        bridge.target_frame = 'elfin_base_link'
        candidates = [
            {'outward_direction': [1.0, 0.0, 0.0], 'utility': 0.95},
            {'outward_direction': [0.0, 1.0, 0.0], 'utility': 0.70},
            {'outward_direction': [0.0, 0.0, 1.0], 'utility': 0.80},
        ]
        result = {'best': candidates[0], 'candidates': candidates}

        def response(success, plan_id, distance, eta):
            return SimpleNamespace(
                success=success, message='plan-only passed',
                endpoint_condition_number=4.0, plan_id=plan_id,
                planning_duration_s=0.2,
                timings_json=json.dumps({
                    'selected_strategy': 'roll',
                    'entry_distance_m': distance,
                    'distance_score_exp_neg_d': MODULE.math.exp(-distance),
                    'max_speed_entry_eta_s': eta,
                }))

        client = mock.Mock(side_effect=[
            response(True, 'approach-1', 0.20, 2.0),
            response(True, 'approach-2', 0.30, 1.0),
            response(True, 'approach-3', 0.60, 1.5),
            SimpleNamespace(success=True, message='selected'),
        ])

        with mock.patch.object(MODULE.rospy, 'wait_for_service'), \
                mock.patch.object(MODULE.rospy, 'ServiceProxy',
                                  return_value=client):
            selected = bridge._validate_approach_with_moveit(
                result, (0.5, 0.1, 0.4), 'elfin_base_link')

        self.assertEqual(selected['moveit_plan_id'], 'approach-2')
        self.assertAlmostEqual(selected['moveit_selected_eta_s'], 1.0)
        self.assertEqual(selected['best']['display_label'], 'SELECTED: FASTEST')
        self.assertEqual(
            selected['candidates'][2]['display_label'], 'FEASIBLE: FARTHER')
        self.assertEqual(client.call_count, 4)
        self.assertEqual(client.call_args_list[-1].kwargs['command'], 'select')

    def test_reconstructed_target_uses_entry_facing_fitted_surface(self):
        bridge = object.__new__(RosBridge)
        bridge.approach_plan_service = '/fake/plan_approach_direction'
        bridge.target_frame = 'elfin_base_link'
        result = {
            'target_center': [0.50, 0.10, 0.40],
            'target_radius_m': 0.04,
            'best': {'outward_direction': [1.0, 0.0, 0.0]},
            'candidates': [{'outward_direction': [1.0, 0.0, 0.0]}],
        }
        client = mock.Mock(return_value=SimpleNamespace(
            success=True, message='plan-only passed',
            endpoint_condition_number=4.0, plan_id='approach-fit',
            planning_duration_s=0.2, timings_json='{}'))

        with mock.patch.object(MODULE.rospy, 'wait_for_service'), \
                mock.patch.object(MODULE.rospy, 'ServiceProxy',
                                  return_value=client):
            selected = bridge._validate_approach_with_moveit(
                result, (0.50, 0.10, 0.40), 'elfin_base_link',
                use_fitted_surface=True)

        request = client.call_args.kwargs
        self.assertAlmostEqual(request['target_point'].x, 0.54)
        self.assertAlmostEqual(request['target_point'].y, 0.10)
        self.assertEqual(
            selected['moveit_target_source'], 'fitted_entry_surface')

    def test_dashboard_executes_only_selected_validated_direction(self):
        bridge = object.__new__(RosBridge)
        bridge.approach_plan_service = '/fake/plan_approach_direction'
        bridge.target_frame = 'elfin_base_link'
        result = {
            'target_center': [0.50, 0.10, 0.40],
            'target_radius_m': 0.04,
            'best': {'outward_direction': [1.0, 0.0, 0.0]},
            'moveit_validated': True,
            'moveit_plan_id': 'approach-cached',
        }
        client = mock.Mock(return_value=SimpleNamespace(
            success=True, execution_attempted=True, executed=True,
            message='已执行缓存出站'))

        with mock.patch.object(MODULE.rospy, 'wait_for_service'), \
                mock.patch.object(MODULE.rospy, 'ServiceProxy',
                                  return_value=client):
            selected = bridge._execute_approach_with_moveit(
                result, (0.50, 0.10, 0.40), 'elfin_base_link',
                use_fitted_surface=True)

        request = client.call_args.kwargs
        self.assertTrue(request['execute'])
        self.assertEqual(request['command'], 'execute_outbound')
        self.assertEqual(request['plan_id'], 'approach-cached')
        self.assertTrue(selected['execution_attempted'])
        self.assertTrue(selected['executed'])

    def test_direction_dialog_can_select_reconstructed_citrus_without_live_depth(self):
        choice = mock.Mock()
        choice.GetSelection.return_value = 1
        dialog = SimpleNamespace(
            target_source_choice=choice,
            reconstructed_targets=[{
                'center': [0.45, -0.10, 0.55],
                'voxel_count': 80,
            }],
            ros_bridge=SimpleNamespace(target_frame='elfin_base_link'),
            owner=mock.Mock(),
        )

        selected = ApproachDirectionDialog._selected_target(dialog)

        self.assertEqual(selected[1], (0.45, -0.10, 0.55))
        self.assertEqual(selected[2], 'elfin_base_link')
        self.assertEqual(selected[3], 'reconstructed_citrus_1')
        dialog.owner._selected_target_request.assert_not_called()

    def test_j_and_k_keydowns_send_distinct_commands_without_repeat_lockout(self):
        # wxPython forbids object.__new__ for C-extension window subclasses on
        # some Noetic builds. This test exercises only the pure key handlers.
        dialog = SimpleNamespace()
        dialog.input_state = MODULE.FocusedCockpitInput()
        dialog.input_state.set_focused(True)
        dialog.tool_keys_held = set()
        dialog.ros_bridge = mock.Mock()
        dialog.on_tool = mock.Mock()
        dialog.on_key_down = CockpitDialog.on_key_down.__get__(dialog)
        dialog.on_key_up = CockpitDialog.on_key_up.__get__(dialog)

        dialog.on_key_down(_KeyEvent(ord('J')))
        dialog.on_key_down(_KeyEvent(ord('J')))  # OS repeat while held
        dialog.on_key_up(_KeyEvent(ord('J')))
        dialog.on_key_down(_KeyEvent(ord('K')))
        dialog.on_key_up(_KeyEvent(ord('K')))
        dialog.on_key_down(_KeyEvent(ord('J')))  # allowed after key-up

        self.assertEqual(
            [call.args[1] for call in dialog.on_tool.call_args_list],
            ['close', 'open', 'close'])

    def test_combined_keys_become_one_normalized_semantic_service_request(self):
        bridge = object.__new__(RosBridge)
        bridge.panel_cockpit_service = '/fake/cockpit'
        bridge._prepare_panel_end = mock.Mock(
            return_value=(True, 'prepared'))
        response = SimpleNamespace(success=True, message='started')
        client = mock.Mock(return_value=response)

        with mock.patch.object(
                MODULE.rospy, 'wait_for_service') as wait_for_service, \
                mock.patch.object(
                    MODULE.rospy, 'ServiceProxy',
                    return_value=client):
            result = bridge._start_panel_jog((
                'forward', 'left', 'up', 'yaw_right',
                'pitch_up', 'roll_left'), ())

        self.assertEqual(result, (True, 'started'))
        wait_for_service.assert_called_once_with(
            '/fake/cockpit', timeout=1.0)
        kwargs = client.call_args.kwargs
        self.assertEqual(kwargs['forward'], 1.0)
        self.assertEqual(kwargs['strafe'], -1.0)
        self.assertEqual(kwargs['vertical'], 1.0)
        self.assertEqual(kwargs['yaw'], -1.0)
        self.assertEqual(kwargs['pitch'], 1.0)
        self.assertEqual(kwargs['roll'], -1.0)
        self.assertEqual(
            kwargs['camera_frame'], 'camera_cockpit_optical_frame')

    def test_enter_capture_is_one_edge_until_keyup(self):
        # Exercise the cockpit key handler without creating a wx window.  The
        # recorder call is deliberately edge-triggered so desktop key repeat
        # cannot create duplicate paper views.
        dialog = SimpleNamespace()
        dialog.input_state = MODULE.FocusedCockpitInput()
        dialog.input_state.set_focused(True)
        dialog.nbv_enter_gate = MODULE.SinglePressGate()
        dialog.ros_bridge = mock.Mock()
        dialog.tool_keys_held = set()
        dialog.owner = mock.Mock()
        dialog.closed = False
        dialog.on_key_down = CockpitDialog.on_key_down.__get__(dialog)
        dialog.on_key_up = CockpitDialog.on_key_up.__get__(dialog)

        for _repeat in range(50):
            dialog.on_key_down(_KeyEvent(13))
        self.assertEqual(dialog.ros_bridge.capture_nbv_view_async.call_count, 1)
        dialog.on_key_up(_KeyEvent(13))
        dialog.on_key_down(_KeyEvent(13))
        self.assertEqual(dialog.ros_bridge.capture_nbv_view_async.call_count, 2)

    def test_physical_point_rising_edge_matches_one_enter_capture(self):
        bridge = object.__new__(RosBridge)
        bridge.nbv_point_button_bit = 4
        bridge.nbv_point_button_gate = MODULE.SinglePressGate()
        bridge.nbv_point_button_initialized = False
        bridge.nbv_point_button_init_lock = threading.Lock()
        bridge.nbv_recorder = SimpleNamespace(enabled=True)
        bridge.state = SharedState()
        bridge.capture_nbv_view_async = mock.Mock()

        # The first latched raw_di value only establishes a baseline. A held
        # POINT button during Dashboard startup must never create a view.
        bridge.point_button_callback(SimpleNamespace(data=1 << 4))
        self.assertEqual(bridge.capture_nbv_view_async.call_count, 0)
        bridge.point_button_callback(SimpleNamespace(data=1 << 4))
        self.assertEqual(bridge.capture_nbv_view_async.call_count, 0)

        bridge.point_button_callback(SimpleNamespace(data=0))
        for _repeat in range(20):
            bridge.point_button_callback(SimpleNamespace(data=1 << 4))
        self.assertEqual(bridge.capture_nbv_view_async.call_count, 1)
        bridge.point_button_callback(SimpleNamespace(data=0))
        bridge.point_button_callback(SimpleNamespace(data=1 << 4))
        self.assertEqual(bridge.capture_nbv_view_async.call_count, 2)
        self.assertEqual(
            bridge.capture_nbv_view_async.call_args.kwargs,
            {'capture_trigger': 'tool_point_di4'})

    def test_physical_point_does_not_capture_when_paper_mode_is_off(self):
        bridge = object.__new__(RosBridge)
        bridge.nbv_point_button_bit = 4
        bridge.nbv_point_button_gate = MODULE.SinglePressGate()
        bridge.nbv_point_button_initialized = False
        bridge.nbv_point_button_init_lock = threading.Lock()
        bridge.nbv_recorder = SimpleNamespace(enabled=False)
        bridge.state = SharedState()
        bridge.capture_nbv_view_async = mock.Mock()

        bridge.point_button_callback(SimpleNamespace(data=0))
        bridge.point_button_callback(SimpleNamespace(data=1 << 4))

        bridge.capture_nbv_view_async.assert_not_called()
        self.assertIn('论文采集未开启', bridge.state.snapshot()['action_status'])

    def test_cockpit_snake_uses_shared_flange_steps_and_stops_each_step(self):
        bridge = object.__new__(RosBridge)
        bridge.panel_cockpit_service = '/fake/cockpit'
        bridge.panel_jog = mock.Mock()
        bridge.state = mock.Mock()
        bridge._prepare_panel_end = mock.Mock(
            return_value=(True, 'prepared'))
        bridge._stop_panel_jog = mock.Mock(return_value=(True, 'stopped'))
        client = mock.Mock(return_value=SimpleNamespace(
            success=True, message='started'))

        with mock.patch.object(MODULE.rospy, 'wait_for_service'), \
                mock.patch.object(MODULE.rospy, 'ServiceProxy',
                                  return_value=client), \
                mock.patch.object(MODULE.rospy, 'is_shutdown',
                                  return_value=False):
            result = bridge.run_flange_snake_maneuver(
                reverse=False, base_duration_s=0.001)

        self.assertTrue(result[0], result[1])
        self.assertEqual(client.call_count, 2)
        self.assertEqual(bridge._stop_panel_jog.call_count, 2)
        first = client.call_args_list[0].kwargs
        self.assertEqual(first['forward'], -1.0)
        self.assertEqual(first['vertical'], 0.70)
        self.assertEqual(first['pitch'], -0.70)

    def test_panel_recovery_mapping_preserves_combined_direction(self):
        bridge = object.__new__(RosBridge)

        values = bridge._panel_recovery_values((
            'forward', 'left', 'up', 'yaw_right',
            'pitch_up', 'roll_left'))

        self.assertEqual(values['forward'], 1.0)
        self.assertEqual(values['strafe'], -1.0)
        self.assertEqual(values['vertical'], 1.0)
        self.assertEqual(values['view_yaw'], -1.0)
        self.assertEqual(values['view_pitch'], 1.0)
        self.assertEqual(values['view_roll'], -1.0)
        self.assertFalse(values['automatic'])

    def test_nbv_readiness_requires_installed_quality_passed_calibration(self):
        bridge = object.__new__(RosBridge)
        bridge.calibration_config_file = '/tmp/test-camera.yaml'
        bridge.camera_name = 'Intel RealSense D455'
        bridge.camera_serial = 'd455-test'
        bridge.camera_device_type = 'D455'
        bridge.camera_usb_type = '3.2'
        bridge.camera_usb_port_id = 'usb-test'
        bridge.nbv_expected_camera_type = 'd455'
        bridge.nbv_allow_non_expected_camera = False
        bridge.nbv_require_usb3 = True
        bridge.nbv_require_semantics = True
        bridge.nbv_require_instance_ids = False
        bridge.state = mock.Mock()

        base = {
            'configured': True,
            'quality_passed': True,
            'camera_serial': 'd455-test',
            'parent_frame': 'elfin_end_link',
            'child_frame': 'camera_link',
        }
        with mock.patch.object(MODULE, 'read_install_status',
                               return_value=dict(base, installed=False)):
            bridge._publish_nbv_status({'view_count': 0, 'max_views': 10})
        self.assertFalse(bridge.nbv_status()['readiness']['ready'])

        with mock.patch.object(MODULE, 'read_install_status',
                               return_value=dict(base, installed=True)):
            bridge._publish_nbv_status({'view_count': 0, 'max_views': 10})
        self.assertTrue(bridge.nbv_status()['readiness']['ready'])

        with mock.patch.object(MODULE, 'read_install_status',
                               return_value=dict(base, installed=True,
                                                  quality_passed=False)):
            bridge._publish_nbv_status({'view_count': 0, 'max_views': 10})
        self.assertFalse(bridge.nbv_status()['readiness']['ready'])


@unittest.skipUnless(
    os.environ.get('DISPLAY') and wx.App.IsDisplayAvailable(),
    'requires an accessible wx display')
class CockpitDashboardGuiTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = wx.App(False)

    def test_window_local_key_focus_and_bounded_close(self):
        owner = _FakeOwner()
        dialog = CockpitDialog(owner)
        owner.cockpit_dialog = dialog
        dialog.Show()
        dialog.Raise()
        dialog.SetFocus()
        wx.Yield()
        dialog.on_activate(_ActivateEvent(True))

        event = _KeyEvent(ord('W'))
        dialog.on_key_down(event)
        self.assertFalse(event.skipped)
        # Operating-system key repeat must never create repeated ROS work.
        for _repeat in range(100):
            dialog.on_key_down(_KeyEvent(ord('W')))
        with owner.ros_bridge.lock:
            self.assertEqual(
                owner.ros_bridge.action_states.count(('forward',)), 1)

        dialog.on_key_down(_KeyEvent(ord('A')))
        dialog.on_key_down(_KeyEvent(wx.WXK_SHIFT))
        dialog.on_key_down(_KeyEvent(wx.WXK_LEFT))
        with owner.ros_bridge.lock:
            self.assertEqual(
                owner.ros_bridge.action_states[-1],
                ('forward', 'left', 'up', 'yaw_left'))

        dialog.on_key_up(_KeyEvent(ord('A')))
        with owner.ros_bridge.lock:
            self.assertEqual(
                owner.ros_bridge.action_states[-1],
                ('forward', 'up', 'yaw_left'))
        dialog.on_key_up(_KeyEvent(wx.WXK_SHIFT))
        dialog.on_key_up(_KeyEvent(wx.WXK_LEFT))
        dialog.on_key_up(_KeyEvent(ord('W')))
        with owner.ros_bridge.lock:
            self.assertEqual(owner.ros_bridge.action_states[-1], ())

        dialog.on_key_down(_KeyEvent(ord('X')))
        with owner.ros_bridge.lock:
            self.assertEqual(
                owner.ros_bridge.action_states[-1], ('base_left',))
        dialog.on_key_up(_KeyEvent(ord('X')))
        dialog.on_key_down(_KeyEvent(ord('Z')))
        with owner.ros_bridge.lock:
            self.assertEqual(
                owner.ros_bridge.action_states[-1], ('base_right',))
        dialog.on_key_up(_KeyEvent(ord('Z')))

        dialog.speed_slider.SetValue(80)
        dialog.on_speed(None)
        self.assertEqual(owner.cockpit_speed_percent, 80)
        self.assertEqual(owner.global_speed_percent, 5)
        with owner.ros_bridge.lock:
            self.assertEqual(owner.ros_bridge.speeds[-1], 80)

        dialog.plan_speed_slider.SetValue(22)
        dialog.on_plan_speed(None)
        self.assertEqual(owner.global_speed_percent, 22)
        self.assertEqual(owner.cockpit_speed_percent, 80)

        dialog.recovery_speed_slider.SetValue(61)
        dialog.on_recovery_speed(None)
        self.assertEqual(owner.recovery_speed_percent, 61)

        dialog.on_activate(_ActivateEvent(False))
        with owner.ros_bridge.lock:
            self.assertEqual(owner.ros_bridge.action_states[-1], ())

        dialog.on_close(_CloseEvent())
        with owner.ros_bridge.lock:
            self.assertEqual(owner.ros_bridge.action_states[-1], ())
        dialog.Destroy()
        owner.Destroy()

    def test_log_wrap_selection_and_latest_error_are_stable(self):
        frame = wx.Frame(None, title='log test', size=(520, 330))
        view = ProtectedLogView(frame, minimum_size=(420, 220))
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(view, 1, wx.EXPAND)
        frame.SetSizer(sizer)
        frame.Show()
        wx.Yield()

        long_error = (
            '[12:34:56] ERROR · AUTO：' +
            '没有空格的超长错误路径' * 30)
        view.update_lines(['[12:34:55] INFO · ready', long_error])
        self.assertFalse(bool(view.text.GetWindowStyle() & wx.HSCROLL))
        self.assertIn(long_error, view.issue_text.GetValue())
        self.assertTrue(view.locate_button.IsEnabled())

        view._pause_follow()
        view.text.SetSelection(0, 18)
        selection = view.text.GetSelection()
        view.update_lines([
            '[12:34:55] INFO · ready', long_error,
            '[12:34:57] INFO · a new message must not steal selection'])
        self.assertEqual(view.text.GetSelection(), selection)
        self.assertFalse(view.following)
        self.assertGreaterEqual(view.GetClientSize().height, 200)

        frame.Destroy()


if __name__ == '__main__':
    unittest.main()
