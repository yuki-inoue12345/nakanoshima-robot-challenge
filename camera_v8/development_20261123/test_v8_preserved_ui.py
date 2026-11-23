"""Hardware-free V7-layout V8 UI routing tests."""
import ast
from pathlib import Path
import unittest

from v8_ui_routes import (
    V8RoutingError, build_live_search_command, choose_photo_preview_mode,
)

HERE = Path(__file__).resolve().parent


class V8UISafetyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            'input_mode': 'camera',
            'person_arrival_topic': '/person_area_arrived',
            'person_done_topic': '/person_search_done',
            'person_detail_topic': '/person_search_done_detail',
            'targets': [{'target_id': 'target_A', 'target_upper_color': 'white'}],
        }

    def test_ros_engine_only(self):
        cmd = build_live_search_command('/python', '/v8_engine.py', self.cfg,
                                        ['--search_duration_sec', '20'],
                                        ['--process_every', '5'])
        self.assertEqual(cmd[:3], ['/python', '/v8_engine.py', '--person-search-server'])
        self.assertIn('--raw_image_topic', cmd)
        self.assertIn('/koko/person_search/raw_image', cmd)
        self.assertNotIn('--record_run_video', cmd)
        self.assertNotIn('--camera_device', cmd)
        self.assertNotIn('--camera_index', cmd)

    def test_video_mode_cannot_enter_live_server(self):
        self.cfg['input_mode'] = 'video'
        with self.assertRaises(V8RoutingError):
            build_live_search_command('python', 'engine', self.cfg, [], [])

    def test_bad_targets_fail_closed(self):
        self.cfg['targets'] = []
        with self.assertRaises(V8RoutingError):
            build_live_search_command('python', 'engine', self.cfg, [], [])

    def test_direct_camera_args_fail_closed(self):
        with self.assertRaises(V8RoutingError):
            build_live_search_command('python', 'engine', self.cfg,
                                      ['--camera_device', '/dev/video0'], [])

    def test_shared_owner_prevents_double_open(self):
        self.assertEqual(choose_photo_preview_mode(True, True), 'ros_shared')
        self.assertEqual(choose_photo_preview_mode(True, False), 'ros_shared')
        with self.assertRaises(V8RoutingError):
            choose_photo_preview_mode(False, True)
        self.assertEqual(choose_photo_preview_mode(False, False), 'standalone')

    def test_shared_preview_does_not_open_camera_or_drive(self):
        src = (HERE / 'v8_shared_photo_preview.py').read_text(encoding='utf-8')
        ast.parse(src)
        for forbidden in ('VideoCapture(', 'VideoWriter(', '/cmd_vel', '/navigate_to_pose'):
            self.assertNotIn(forbidden, src)

    def test_preserved_ui_is_loaded_read_only(self):
        src = (HERE / 'ui_v7_compat.py').read_text(encoding='utf-8')
        ast.parse(src)
        self.assertIn("EXPECTED_V7_SHA256 = '720a2968563cf58b0a972f9034dc10747b430582ef12c6f9af2b41c7cfaf2ee9'", src)
        self.assertIn('class CameraV8PreservedUI(v7.Milestone2TargetUI)', src)
        self.assertNotIn('VideoCapture(', src)
        self.assertNotIn('VideoWriter(', src)


if __name__ == '__main__':
    unittest.main()
