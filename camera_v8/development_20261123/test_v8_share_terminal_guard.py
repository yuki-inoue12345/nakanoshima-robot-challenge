"""No-ROS pure callback regression for read-only V8 share termination.

Extraction tests the EXACT committed callback implementations without needing
rclpy, C920, DDS discovery, Nav2, or a moving robot.
"""
import ast
import json
from pathlib import Path
import time
import types
import unittest

SOURCE = Path(__file__).with_name('v8_person_frame_share.py')


def load_callbacks():
    source = SOURCE.read_text(encoding='utf-8')
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'V8PersonFrameShare')
    names = {'_on_arrival','_on_done','_on_result'}
    body = [n for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(body) != len(names):
        raise AssertionError('Missing terminal callback(s)')
    isolated = ast.Module(body=[ast.ClassDef(name='Callbacks',bases=[],keywords=[],
                      body=body,decorator_list=[])],type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace={'json':json,'time':time}
    exec(compile(isolated,str(SOURCE),'exec'),namespace)
    return namespace['Callbacks'], cls


class Logger:
    def __init__(self):self.messages=[]
    def info(self,text):self.messages.append(text)
    def error(self,text):self.messages.append(text)


class V8ShareStopTests(unittest.TestCase):
    def setUp(self):
        self.callback_cls,self.source_class=load_callbacks()
        self.logger=Logger()
        self.fake=types.SimpleNamespace(
            _active=False,_frame_count=0,_started=0.0,
            get_logger=lambda:self.logger,
        )

    def test_subscription_registers_done_boolean(self):
        constructor=next(n for n in self.source_class.body
                         if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        text=ast.unparse(constructor)
        self.assertIn("'/person_search_done', self._on_done, detail_qos",text)
        self.assertIn("'/person_search_done_detail', self._on_result, detail_qos",text)

    def test_true_done_stops_stream(self):
        self.callback_cls._on_arrival(self.fake,types.SimpleNamespace(data=True))
        self.fake._frame_count=17
        self.callback_cls._on_done(self.fake,types.SimpleNamespace(data=True))
        self.assertFalse(self.fake._active)
        self.assertTrue(any('sent=17' in msg and 'safe_to_advance=True' in msg
                            for msg in self.logger.messages))

    def test_false_done_also_stops_stream(self):
        self.callback_cls._on_arrival(self.fake,types.SimpleNamespace(data=True))
        self.callback_cls._on_done(self.fake,types.SimpleNamespace(data=False))
        self.assertFalse(self.fake._active)
        self.assertTrue(any('safe_to_advance=False' in msg for msg in self.logger.messages))

    def test_false_arrival_is_logged_and_stops(self):
        self.callback_cls._on_arrival(self.fake,types.SimpleNamespace(data=True))
        self.fake._frame_count=27
        self.callback_cls._on_arrival(self.fake,types.SimpleNamespace(data=False))
        self.assertFalse(self.fake._active)
        self.assertTrue(any('false arrival; sent=27' in msg for msg in self.logger.messages))

    def test_nonterminal_diagnostic_keeps_stream(self):
        self.callback_cls._on_arrival(self.fake,types.SimpleNamespace(data=True))
        self.callback_cls._on_result(self.fake,types.SimpleNamespace(data=json.dumps(
            {'terminal':False,'status':'BUSY'})))
        self.assertTrue(self.fake._active)

    def test_terminal_detail_remains_supported(self):
        self.callback_cls._on_arrival(self.fake,types.SimpleNamespace(data=True))
        self.callback_cls._on_result(self.fake,types.SimpleNamespace(data=json.dumps(
            {'terminal':True,'status':'NOT_FOUND'})))
        self.assertFalse(self.fake._active)
        self.assertTrue(any('terminal result' in msg for msg in self.logger.messages))

    def test_duplicate_done_after_detail_does_nothing(self):
        self.callback_cls._on_arrival(self.fake,types.SimpleNamespace(data=True))
        self.callback_cls._on_result(self.fake,types.SimpleNamespace(data='{"terminal":true}'))
        size=len(self.logger.messages)
        self.callback_cls._on_done(self.fake,types.SimpleNamespace(data=True))
        self.assertEqual(len(self.logger.messages),size)
        self.assertFalse(self.fake._active)


if __name__=='__main__':unittest.main()
