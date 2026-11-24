"""No ROS or camera required: flock concurrency regression tests."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import v8_search_singleton as lockmod


class V8SingletonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='v8singleton_')
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.root_patch = patch.object(lockmod, 'LOCK_ROOT', self.dir)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.domain_patch = patch.dict(os.environ, {'ROS_DOMAIN_ID': '215'})
        self.domain_patch.start()
        self.addCleanup(self.domain_patch.stop)

    def test_same_domain_second_server_rejected_and_close_releases(self):
        fd = lockmod.acquire_v8_person_search_lock()
        try:
            with self.assertRaises(lockmod.V8ServerAlreadyRunning):
                lockmod.acquire_v8_person_search_lock()
        finally:
            os.close(fd)
        fd2 = lockmod.acquire_v8_person_search_lock()
        os.close(fd2)

    def test_distinct_domain_allowed(self):
        fd=lockmod.acquire_v8_person_search_lock()
        try:
            with patch.dict(os.environ, {'ROS_DOMAIN_ID': '216'}):
                fd2=lockmod.acquire_v8_person_search_lock()
                os.close(fd2)
        finally:
            os.close(fd)

    def test_bad_domain_fails_closed(self):
        with patch.dict(os.environ, {'ROS_DOMAIN_ID': '../54'}):
            with self.assertRaises(RuntimeError):
                lockmod.acquire_v8_person_search_lock()

    def test_child_process_lock_and_os_cleanup(self):
        script=(
            'import os,sys,time;from pathlib import Path; '
            'import v8_search_singleton as s; s.LOCK_ROOT=Path(sys.argv[1]); '
            'fd=s.acquire_v8_person_search_lock(); '
            'print("READY",flush=True);time.sleep(5)'
        )
        env=os.environ.copy()
        env['PYTHONPATH']=str(Path(lockmod.__file__).parent)
        child=subprocess.Popen([sys.executable,'-c',script,str(self.dir)],
                               env=env,stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(),'READY')
            with self.assertRaises(lockmod.V8ServerAlreadyRunning):
                lockmod.acquire_v8_person_search_lock()
        finally:
            child.kill()
            child.communicate(timeout=5)
        fd=lockmod.acquire_v8_person_search_lock()
        os.close(fd)


if __name__=='__main__':
    unittest.main()
