"""Hardware-free regression checks for LM Studio preflight routing.

Avoid importing Tk/YOLO/ROS or starting LM Studio. Extract only the V8
worker method and exercise it against a fake subprocess.
"""
import ast
from pathlib import Path
from queue import Queue
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main, mock

HERE = Path(__file__).resolve().parent
UI = HERE / 'ui_v7_compat.py'


def extract_worker():
    syntax = ast.parse(UI.read_text(encoding='utf-8'))
    cls = next(n for n in syntax.body
               if isinstance(n, ast.ClassDef) and n.name == 'CameraV8PreservedUI')
    method = next(n for n in cls.body
                  if isinstance(n, ast.FunctionDef) and n.name == '_prepare_lm_studio_worker')
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    return compile(module, str(UI), 'exec')


class LMPreflightTests(TestCase):
    def execute(self, *, helper_present=True, returncode=0, output='V7 startup preflight: PASS', raises=None):
        with TemporaryDirectory() as folder:
            v7_dir = Path(folder) / 'v7'
            v8_dir = Path(folder) / 'v8'
            v7_dir.mkdir()
            v8_dir.mkdir()
            helper = v7_dir / 'v7_startup_preflight.py'
            if helper_present:
                helper.write_text('pass\n', encoding='utf-8')
            runner = mock.Mock()
            runner.run.return_value = SimpleNamespace(returncode=returncode, stdout=output)
            if raises is not None:
                runner.run.side_effect = raises
            subprocess_proxy = SimpleNamespace(
                run=runner.run,
                PIPE=subprocess.PIPE,
                STDOUT=subprocess.STDOUT,
                TimeoutExpired=subprocess.TimeoutExpired,
            )
            namespace = {
                'V7_DIR': v7_dir,
                'HERE': v8_dir,
                'subprocess': subprocess_proxy,
                'sys': sys,
                'v7': SimpleNamespace(LM_STUDIO_PREPARE_TIMEOUT_SEC=420.0),
            }
            exec(extract_worker(), namespace)
            queue = Queue()
            namespace['_prepare_lm_studio_worker'](
                SimpleNamespace(log_queue=queue), 'http://127.0.0.1:1234/v1'
            )
            out = []
            while not queue.empty():
                out.append(queue.get_nowait())
            return out, runner.run.call_args, helper, v8_dir

    def test_uses_v7_helper_but_v8_workdir(self):
        out, args, helper, v8_dir = self.execute()
        self.assertIn('__UI_LM_STATUS_OK__', out)
        self.assertEqual(args.args[0][0:3], [sys.executable, '-B', str(helper)])
        self.assertEqual(args.args[0][3:], ['--base-url', 'http://127.0.0.1:1234/v1', '--prepare-lm'])
        self.assertEqual(args.kwargs['cwd'], str(v8_dir))
        self.assertEqual(args.kwargs['timeout'], 420.0)

    def test_fails_closed_if_helper_missing(self):
        out, args, _, _ = self.execute(helper_present=False)
        self.assertIn('__UI_LM_STATUS_FAIL__', out)
        self.assertIsNone(args)

    def test_model_missing_is_distinguished(self):
        out, _, _, _ = self.execute(returncode=1, output='TARGET_MODEL_NOT_IN_LOCAL_CATALOG: not installed')
        self.assertIn('__UI_LM_STATUS_MODEL_MISSING__', out)

    def test_cli_error_is_not_reported_as_success(self):
        out, _, _, _ = self.execute(returncode=1, output='LMS_NOT_FOUND: CLI missing')
        self.assertIn('__UI_LM_STATUS_FAIL__', out)
        self.assertNotIn('__UI_LM_STATUS_OK__', out)

    def test_timeout_is_reported(self):
        out, _, _, _ = self.execute(raises=subprocess.TimeoutExpired('lms', 420))
        self.assertIn('__UI_LM_STATUS_FAIL__', out)
        self.assertTrue(any('timeout' in x for x in out))

    def test_only_v8_ui_integration_affected(self):
        src = UI.read_text(encoding='utf-8')
        self.assertIn("EXPECTED_V7_SHA256 = '720a2968563cf58b0a972f9034dc10747b430582ef12c6f9af2b41c7cfaf2ee9'", src)
        self.assertIn("helper = V7_DIR / 'v7_startup_preflight.py'", src)
        self.assertNotIn('VideoCapture(', src)
        self.assertNotIn('VideoWriter(', src)


if __name__ == '__main__':
    main()
