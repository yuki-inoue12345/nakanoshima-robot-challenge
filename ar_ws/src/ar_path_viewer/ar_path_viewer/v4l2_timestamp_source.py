"""Optional Linux V4L2 single-plane MJPEG camera with per-buffer timestamps.

This source is opt-in.  It never silently substitutes OpenCV receive time for
kernel buffer time, and requires a monotonic timestamp flag from VIDIOC_DQBUF.
No timestamp is a guarantee of precise exposure time; V4L2 EOF timestamps in
particular may trail exposure.  The Viewer applies further age/motion gates.
"""

import ctypes as C
import fcntl
import mmap
import os
import select
import sys
import time

import cv2
import numpy as np


class _Timeval(C.Structure):
    _fields_ = [("tv_sec", C.c_long), ("tv_usec", C.c_long)]


class _Timecode(C.Structure):
    _fields_ = [
        ("type", C.c_uint32), ("flags", C.c_uint32),
        ("frames", C.c_uint8), ("seconds", C.c_uint8),
        ("minutes", C.c_uint8), ("hours", C.c_uint8),
        ("userbits", C.c_uint8 * 4),
    ]


class _BufferMemory(C.Union):
    _fields_ = [("offset", C.c_uint32), ("userptr", C.c_ulong),
                ("planes", C.c_void_p), ("fd", C.c_int32)]


class _Buffer(C.Structure):
    _fields_ = [
        ("index", C.c_uint32), ("type", C.c_uint32),
        ("bytesused", C.c_uint32), ("flags", C.c_uint32),
        ("field", C.c_uint32), ("timestamp", _Timeval),
        ("timecode", _Timecode), ("sequence", C.c_uint32),
        ("memory", C.c_uint32), ("m", _BufferMemory),
        ("length", C.c_uint32), ("reserved2", C.c_uint32),
        ("request_fd", C.c_int32),
    ]


class _PixFormat(C.Structure):
    _fields_ = [
        ("width", C.c_uint32), ("height", C.c_uint32),
        ("pixelformat", C.c_uint32), ("field", C.c_uint32),
        ("bytesperline", C.c_uint32), ("sizeimage", C.c_uint32),
        ("colorspace", C.c_uint32), ("priv", C.c_uint32),
        ("flags", C.c_uint32), ("ycbcr_enc", C.c_uint32),
        ("quantization", C.c_uint32), ("xfer_func", C.c_uint32),
    ]


class _FormatUnion(C.Union):
    _fields_ = [("pix", _PixFormat), ("raw_data", C.c_uint8 * 200),
                ("alignment", C.c_uint64)]


class _Format(C.Structure):
    _fields_ = [("type", C.c_uint32), ("fmt", _FormatUnion)]


class _RequestBuffers(C.Structure):
    _fields_ = [
        ("count", C.c_uint32), ("type", C.c_uint32),
        ("memory", C.c_uint32), ("capabilities", C.c_uint32),
        ("flags", C.c_uint8), ("reserved", C.c_uint8 * 3),
    ]


def _ioctl(direction, number, struct_type):
    # Linux generic _IOC layout, supported here on x86_64 Linux (NUC).
    return ((direction << 30) | (C.sizeof(struct_type) << 16) |
            (ord('V') << 8) | number)


_VIDEO_CAPTURE = 1
_MEMORY_MMAP = 1
_MJPEG = int.from_bytes(b'MJPG', 'little')
_S_FMT = _ioctl(3, 5, _Format)
_REQBUFS = _ioctl(3, 8, _RequestBuffers)
_QUERYBUF = _ioctl(3, 9, _Buffer)
_QBUF = _ioctl(3, 15, _Buffer)
_DQBUF = _ioctl(3, 17, _Buffer)
_STREAMON = _ioctl(1, 18, C.c_int)
_STREAMOFF = _ioctl(1, 19, C.c_int)
_TIMESTAMP_MASK = 0x0000E000
_TIMESTAMP_MONOTONIC = 0x00002000
_TIMESTAMP_SOURCE_MASK = 0x00070000
_TIMESTAMP_SOURCE_SOE = 0x00010000
_BUF_ERROR = 0x00000040


def timestamp_from_v4l2_buffer(buf, *, monotonic_now):
    """Return (monotonic timestamp, kind) or raise for untrusted metadata."""
    if buf.flags & _BUF_ERROR:
        raise ValueError('v4l2_buffer_error')
    if (buf.flags & _TIMESTAMP_MASK) != _TIMESTAMP_MONOTONIC:
        raise ValueError('v4l2_timestamp_not_monotonic')
    seconds = float(buf.timestamp.tv_sec) + float(buf.timestamp.tv_usec) * 1e-6
    if seconds <= 0.0 or not np.isfinite(seconds):
        raise ValueError('v4l2_invalid_timestamp')
    if seconds > monotonic_now + 0.005:
        raise ValueError('v4l2_future_timestamp')
    timestamp_source = buf.flags & _TIMESTAMP_SOURCE_MASK
    if timestamp_source not in (0, _TIMESTAMP_SOURCE_SOE):
        raise ValueError('v4l2_unknown_timestamp_source')
    source = ('v4l2_monotonic_soe' if
              timestamp_source == _TIMESTAMP_SOURCE_SOE
              else 'v4l2_monotonic_eof')
    return seconds, source


class V4L2TimestampCameraSource:
    """Read MJPEG frames and matching V4L2 DQBUF timestamps from one fd.

    No file, socket or ROS publishers are created.  An unsupported camera or
    kernel is an explicit startup error: do not fall back to stale OpenCV input.
    """

    def __init__(self, device, width, height):
        if sys.platform != 'linux' or C.sizeof(C.c_void_p) != 8:
            raise OSError('V4L2 timestamp capture requires 64-bit Linux')
        if not isinstance(device, str) or not device.startswith('/dev/'):
            raise ValueError('V4L2 capture requires an absolute /dev video device')
        self._fd = -1
        self._mappings = []
        self._running = False
        self.frame_timestamp_monotonic = None
        self.frame_timestamp_kind = ''
        self._last_sequence = None
        try:
            self._fd = os.open(device, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
            fmt = _Format()
            fmt.type = _VIDEO_CAPTURE
            fmt.fmt.pix.width = int(width)
            fmt.fmt.pix.height = int(height)
            fmt.fmt.pix.pixelformat = _MJPEG
            fmt.fmt.pix.field = 1  # V4L2_FIELD_NONE
            fcntl.ioctl(self._fd, _S_FMT, fmt, True)
            if (fmt.fmt.pix.width, fmt.fmt.pix.height) != (width, height):
                raise OSError('V4L2 camera did not accept requested resolution')
            if fmt.fmt.pix.pixelformat != _MJPEG:
                raise OSError('V4L2 camera does not support MJPEG at this resolution')
            req = _RequestBuffers()
            req.count = 4
            req.type = _VIDEO_CAPTURE
            req.memory = _MEMORY_MMAP
            fcntl.ioctl(self._fd, _REQBUFS, req, True)
            if req.count < 2 or req.count > 8:
                raise OSError('V4L2 camera returned invalid mmap buffer count')
            for i in range(req.count):
                buf = self._new_buffer(i)
                fcntl.ioctl(self._fd, _QUERYBUF, buf, True)
                if buf.length <= 0 or buf.length > 16 * 1024 * 1024:
                    raise OSError('V4L2 camera returned invalid mmap buffer size')
                self._mappings.append(mmap.mmap(
                    self._fd, buf.length, flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=buf.m.offset))
                fcntl.ioctl(self._fd, _QBUF, self._new_buffer(i), True)
            data_type = C.c_int(_VIDEO_CAPTURE)
            fcntl.ioctl(self._fd, _STREAMON, data_type, True)
            self._running = True
        except Exception:
            self.close()
            raise

    @staticmethod
    def _new_buffer(index=0):
        buf = _Buffer()
        buf.type = _VIDEO_CAPTURE
        buf.memory = _MEMORY_MMAP
        buf.index = index
        return buf

    def read(self):
        self.frame_timestamp_monotonic = None
        self.frame_timestamp_kind = ''
        if self._fd < 0:
            return None
        try:
            ready, _, _ = select.select([self._fd], [], [], 0.12)
        except OSError:
            return None
        if not ready:
            return None
        buf = self._new_buffer()
        try:
            fcntl.ioctl(self._fd, _DQBUF, buf, True)
        except (BlockingIOError, OSError):
            return None
        try:
            if buf.index >= len(self._mappings):
                raise ValueError('V4L2 returned out-of-bounds buffer index')
            if not 0 < buf.bytesused <= len(self._mappings[buf.index]):
                raise ValueError('V4L2 returned invalid frame size')
            stamp, kind = timestamp_from_v4l2_buffer(
                buf, monotonic_now=time.monotonic())
            if self._last_sequence is not None and buf.sequence <= self._last_sequence:
                raise ValueError('V4L2 frame sequence did not advance')
            raw = self._mappings[buf.index][:buf.bytesused]
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('V4L2 MJPEG frame could not be decoded')
            self._last_sequence = buf.sequence
            self.frame_timestamp_monotonic = stamp
            self.frame_timestamp_kind = kind
            return image
        except ValueError:
            return None
        finally:
            try:
                fcntl.ioctl(self._fd, _QBUF, self._new_buffer(buf.index), True)
            except OSError:
                # Do not expose a timestamp from a failed video stream.
                self.frame_timestamp_monotonic = None
                self.frame_timestamp_kind = ''

    def close(self):
        if getattr(self, '_fd', -1) >= 0:
            if getattr(self, '_running', False):
                try:
                    fcntl.ioctl(self._fd, _STREAMOFF,
                                C.c_int(_VIDEO_CAPTURE), True)
                except OSError:
                    pass
                self._running = False
            for region in getattr(self, '_mappings', []):
                region.close()
            self._mappings = []
            os.close(self._fd)
            self._fd = -1
