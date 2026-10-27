"""Regression tests for Camera V7 arrival/worker terminal observability."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parent
UI_PATH = ROOT / "daikoudai_v7.py.new"


def load_ui_module():
    loader = importlib.machinery.SourceFileLoader(
        "v7_arrival_observability_ui", str(UI_PATH)
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


UI = load_ui_module()


class FakeLogger:
    def __init__(self):
        self.messages = []

    def _append(self, level, message):
        self.messages.append((level, str(message)))

    def info(self, message):
        self._append("info", message)

    def warning(self, message):
        self._append("warning", message)

    def error(self, message):
        self._append("error", message)


class FakePublisher:
    def __init__(self, topic, fail_topics):
        self.topic = topic
        self.fail_topics = fail_topics
        self.messages = []

    def publish(self, message):
        if self.topic in self.fail_topics:
            raise RuntimeError(f"injected publish failure: {self.topic}")
        self.messages.append(message)


class FakeNode:
    instances = []
    fail_topics = set()

    def __init__(self, _name):
        self.logger = FakeLogger()
        self.publishers = {}
        self.subscriptions = {}
        self.publisher_qos = {}
        self.subscription_qos = {}
        type(self).instances.append(self)

    def create_publisher(self, _message_type, topic, qos):
        publisher = FakePublisher(topic, type(self).fail_topics)
        self.publishers[topic] = publisher
        self.publisher_qos[topic] = qos
        return publisher

    def create_subscription(self, _message_type, topic, callback, qos):
        self.subscriptions[topic] = callback
        self.subscription_qos[topic] = qos
        return SimpleNamespace(topic=topic)

    def get_logger(self):
        return self.logger

    def destroy_node(self):
        return None


class Bool:
    def __init__(self, data=False):
        self.data = bool(data)


class String:
    def __init__(self):
        self.data = ""


class QoSProfile:
    def __init__(self, *, depth, reliability, durability):
        self.depth = depth
        self.reliability = reliability
        self.durability = durability


class FakeManifest:
    def __init__(self, path, logger=None):
        self.path = Path(path)

    def snapshot(self):
        return {}


class FakePreviewManager:
    def __init__(self, **_kwargs):
        pass

    def shutdown(self):
        return True


class FakeOwnership:
    pause_exception = None

    def __init__(self, *_args, **_kwargs):
        pass

    def start_drive(self):
        return None

    def pause_for_search(self):
        if type(self).pause_exception is not None:
            raise type(self).pause_exception
        return True

    def resume_after_search(self):
        return True

    def shutdown(self):
        return None


class FakeParser:
    def parse_args(self, _argv):
        return SimpleNamespace()


class ArrivalObservabilityTests(unittest.TestCase):
    def setUp(self):
        FakeNode.instances = []
        FakeNode.fail_topics = set()
        FakeOwnership.pause_exception = None
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)

    @staticmethod
    def success_result():
        return {
            "status": "FOUND_FRONT_PHOTO",
            "found": True,
            "person_found": True,
            "ms1_success": True,
            "ms2_success": True,
            "front_photo_saved": True,
            "final_score": 0.9,
        }

    def run_server(self, scenario, *, search_result=None, search_exception=None):
        reliability = SimpleNamespace(RELIABLE="reliable")
        durability = SimpleNamespace(
            VOLATILE="volatile", TRANSIENT_LOCAL="transient_local"
        )
        signal_options = SimpleNamespace(NO="no")
        rclpy = types.ModuleType("rclpy")
        rclpy.init = lambda **_kwargs: None
        rclpy.ok = lambda: True
        rclpy.shutdown = lambda: None
        executors = types.ModuleType("rclpy.executors")
        executors.ExternalShutdownException = RuntimeError
        node_module = types.ModuleType("rclpy.node")
        node_module.Node = FakeNode
        qos_module = types.ModuleType("rclpy.qos")
        qos_module.DurabilityPolicy = durability
        qos_module.QoSProfile = QoSProfile
        qos_module.ReliabilityPolicy = reliability
        signals = types.ModuleType("rclpy.signals")
        signals.SignalHandlerOptions = signal_options
        std_msgs = types.ModuleType("std_msgs")
        std_msgs_msg = types.ModuleType("std_msgs.msg")
        std_msgs_msg.Bool = Bool
        std_msgs_msg.String = String
        ms2 = types.ModuleType("person_search_vlm_milestone2_v7")
        ms2.build_parser = lambda: FakeParser()
        ms2.open_camera_capture_with_fallback = lambda _source: None
        ms2.request_stop = lambda: None

        def run_search(*_args, **_kwargs):
            if search_exception is not None:
                raise search_exception
            return dict(search_result or self.success_result())

        ms2.run_milestone2_search = run_search
        modules = {
            "rclpy": rclpy,
            "rclpy.executors": executors,
            "rclpy.node": node_module,
            "rclpy.qos": qos_module,
            "rclpy.signals": signals,
            "std_msgs": std_msgs,
            "std_msgs.msg": std_msgs_msg,
            "person_search_vlm_milestone2_v7": ms2,
        }

        def spin(node, _rclpy, _shutdown_requested):
            scenario(node)

        old_cwd = Path.cwd()
        try:
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(UI, "block_server_shutdown_signals"),
                mock.patch.object(UI, "start_server_shutdown_signal_waiter"),
                mock.patch.object(UI, "spin_node_with_signal_poll", side_effect=spin),
                mock.patch.object(UI, "CameraVideoManifest", FakeManifest),
                mock.patch.object(UI, "HighGuiPreviewManager", FakePreviewManager),
                mock.patch.object(UI, "CameraOwnershipCoordinator", FakeOwnership),
            ):
                UI.os.chdir(self.tempdir.name)
                UI.run_person_search_server_mode([])
        finally:
            UI.os.chdir(old_cwd)
        self.assertTrue(FakeNode.instances)
        return FakeNode.instances[-1]

    @staticmethod
    def log_text(node):
        return "\n".join(message for _level, message in node.logger.messages)

    @staticmethod
    def send_true(node):
        node.subscriptions["/person_area_arrived"](Bool(True))

    def send_true_and_join(self, node):
        self.send_true(node)
        worker = node._worker_thread
        if worker is not None:
            worker.join(timeout=3.0)
            self.assertFalse(worker.is_alive())

    def test_true_callback_ack_worker_and_success_terminal(self):
        node = self.run_server(self.send_true_and_join)
        self.assertEqual(1, len(node.publishers["/person_search/arrival_received"].messages))
        self.assertEqual(1, len(node.publishers["/person_search_done_detail"].messages))
        self.assertTrue(node.publishers["/person_search_done"].messages[0].data)
        logs = self.log_text(node)
        for stage in (
            "ARRIVAL_CALLBACK_ENTERED",
            "ARRIVAL_ACK_PUBLISH_SUCCEEDED",
            "WORKER_START_REQUESTED",
            "WORKER_ENTERED",
            "TERMINAL_DETAIL_PUBLISH_SUCCEEDED",
            "TERMINAL_DONE_PUBLISH_SUCCEEDED",
            "WORKER_EXIT",
        ):
            self.assertIn(stage, logs)

    def test_worker_start_failure_publishes_fail_closed_terminal(self):
        def scenario(node):
            with mock.patch.object(
                UI.threading, "Thread", side_effect=RuntimeError("thread start unavailable")
            ):
                self.send_true(node)

        node = self.run_server(scenario)
        detail_message = node.publishers["/person_search_done_detail"].messages[0]
        detail = json.loads(detail_message.data)
        self.assertEqual("ERROR", detail["status"])
        self.assertEqual("WORKER_THREAD_CREATE_FAILED", detail["error_reason"])
        self.assertFalse(node.publishers["/person_search_done"].messages[0].data)
        logs = self.log_text(node)
        self.assertIn("WORKER_START_FAILED", logs)
        self.assertIn("phase=thread_object_creation", logs)

    def test_thread_start_failure_publishes_fail_closed_terminal(self):
        class StartFailThread:
            name = "injected-start-failure"

            def __init__(self, **_kwargs):
                pass

            def start(self):
                raise RuntimeError("thread start unavailable")

        def scenario(node):
            with mock.patch.object(UI.threading, "Thread", StartFailThread):
                self.send_true(node)

        node = self.run_server(scenario)
        detail = json.loads(
            node.publishers["/person_search_done_detail"].messages[0].data
        )
        self.assertEqual("WORKER_START_FAILED", detail["error_reason"])
        self.assertFalse(node.publishers["/person_search_done"].messages[0].data)
        self.assertIn("phase=thread_start", self.log_text(node))

    def test_worker_python_exception_publishes_fail_closed_terminal(self):
        node = self.run_server(
            self.send_true_and_join,
            search_exception=RuntimeError("injected VLM failure"),
        )
        detail = json.loads(
            node.publishers["/person_search_done_detail"].messages[0].data
        )
        self.assertEqual("ERROR", detail["status"])
        self.assertFalse(detail["safe_to_advance"])
        self.assertFalse(node.publishers["/person_search_done"].messages[0].data)
        self.assertIn("SEARCH_LOOP_FAILED", self.log_text(node))

    def test_shutdown_true_is_acknowledged_and_reason_is_logged(self):
        def scenario(node):
            node._shutdown_requested.set()
            self.send_true(node)

        node = self.run_server(scenario)
        self.assertEqual(1, len(node.publishers["/person_search/arrival_received"].messages))
        self.assertIsNone(node._worker_thread)
        logs = self.log_text(node)
        self.assertIn("ARRIVAL_REJECTED", logs)
        self.assertIn("reason=shutdown_requested", logs)

    def test_false_callback_logs_state_and_resets_rising_edge(self):
        def scenario(node):
            node._last_arrival_state = True
            node.subscriptions["/person_area_arrived"](Bool(False))

        node = self.run_server(scenario)
        self.assertFalse(node._last_arrival_state)
        self.assertEqual(0, len(node.publishers["/person_search/arrival_received"].messages))
        logs = self.log_text(node)
        self.assertIn("ARRIVAL_CALLBACK_ENTERED", logs)
        self.assertIn("received_value=False", logs)
        self.assertIn("previous_arrival_state=True", logs)
        self.assertIn("ARRIVAL_FALSE_RESET", logs)

    def test_duplicate_and_busy_true_have_reason_logs(self):
        def scenario(node):
            node._last_arrival_state = True
            self.send_true(node)
            node._last_arrival_state = False
            node._busy = True
            self.send_true(node)

        node = self.run_server(scenario)
        logs = self.log_text(node)
        self.assertIn("reason=duplicate_or_stale_true", logs)
        self.assertIn("reason=already_searching", logs)

    def test_ack_failure_is_logged_and_worker_still_runs(self):
        FakeNode.fail_topics = {"/person_search/arrival_received"}
        node = self.run_server(self.send_true_and_join)
        logs = self.log_text(node)
        self.assertIn("ARRIVAL_ACK_PUBLISH_FAILED", logs)
        self.assertIn("WORKER_ENTERED", logs)
        self.assertTrue(node.publishers["/person_search_done"].messages[0].data)

    def test_terminal_detail_success_done_failure_are_distinguishable(self):
        FakeNode.fail_topics = {"/person_search_done"}
        node = self.run_server(self.send_true_and_join)
        self.assertEqual(1, len(node.publishers["/person_search_done_detail"].messages))
        self.assertEqual(0, len(node.publishers["/person_search_done"].messages))
        logs = self.log_text(node)
        self.assertIn("TERMINAL_DETAIL_PUBLISH_SUCCEEDED", logs)
        self.assertIn("TERMINAL_DONE_PUBLISH_FAILED", logs)

    def test_ros_interface_and_qos_contract_are_unchanged(self):
        def scenario(_node):
            return None

        node = self.run_server(scenario)
        self.assertEqual(
            {
                "/person_search_done",
                "/person_search_done_detail",
                "/person_search/arrival_received",
            },
            set(node.publishers),
        )
        self.assertEqual({"/person_area_arrived"}, set(node.subscriptions))
        arrival = node.subscription_qos["/person_area_arrived"]
        self.assertEqual(1, arrival.depth)
        self.assertEqual("reliable", arrival.reliability)
        self.assertEqual("transient_local", arrival.durability)
        ack = node.publisher_qos["/person_search/arrival_received"]
        self.assertEqual(10, ack.depth)
        self.assertEqual("reliable", ack.reliability)
        self.assertEqual("volatile", ack.durability)


if __name__ == "__main__":
    unittest.main()
