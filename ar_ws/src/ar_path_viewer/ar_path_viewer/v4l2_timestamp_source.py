"""Optional Linux V4L2 single-plane MJPEG camera with per-buffer timestamps.

This source is opt-in.  It never silently substitutes OpenCV receive time for
kernel buffer time, and requires a monotonic timestamp flag from VIDIOC_DQBUF.
No timestamp is a guarantee of precise exposure time; V4L2 EOF timestamps in
particular may trail exposure.  The Viewer applies further age/motion gates.
"""

import ctypes as C
from dataclasses import dataclass
import errno
import fcntl
import mmap
import os
import select
import sys
import time
from fractions import Fraction

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


class _Fract(C.Structure):
    _fields_ = [("numerator", C.c_uint32), ("denominator", C.c_uint32)]


class _CaptureParm(C.Structure):
    _fields_ = [
        ("capability", C.c_uint32), ("capturemode", C.c_uint32),
        ("timeperframe", _Fract), ("extendedmode", C.c_uint32),
        ("readbuffers", C.c_uint32), ("reserved", C.c_uint32 * 4),
    ]


class _StreamParmUnion(C.Union):
    _fields_ = [("capture", _CaptureParm), ("raw_data", C.c_uint8 * 200),
                ("alignment", C.c_uint64)]


class _StreamParm(C.Structure):
    _fields_ = [("type", C.c_uint32), ("parm", _StreamParmUnion)]


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
_G_PARM = _ioctl(3, 21, _StreamParm)
_S_PARM = _ioctl(3, 22, _StreamParm)
_TIMESTAMP_MASK = 0x0000E000
_TIMESTAMP_MONOTONIC = 0x00002000
_TIMESTAMP_SOURCE_MASK = 0x00070000
_TIMESTAMP_SOURCE_SOE = 0x00010000
_BUF_ERROR = 0x00000040


@dataclass(frozen=True)
class V4L2ReadMetrics:
    """Bounded per-read evidence; values never authorize an AR overlay."""

    frame_timestamp_monotonic_sec: float | None = None
    dequeue_monotonic_sec: float | None = None
    transport_age_sec: float | None = None
    read_duration_sec: float = 0.0
    dequeue_duration_sec: float = 0.0
    decode_duration_sec: float = 0.0
    sequence: int | None = None
    sequence_gap: int = 0
    dequeued_buffers: int = 0
    discarded_buffers: int = 0
    total_discarded_buffers: int = 0
    requested_capture_fps: float | None = None
    effective_capture_fps: float | None = None
    capture_rate_reason: str = ""
    error_reason: str = ""


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

    def __init__(self, device, width, height, fps=None):
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
        self._discarded_buffer_total = 0
        self.requested_capture_fps = self._validate_requested_fps(fps)
        self.effective_capture_fps = None
        self.capture_rate_reason = 'not_requested'
        self.last_read_metrics = V4L2ReadMetrics()
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
            self._configure_capture_rate()
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
    def _validate_requested_fps(value):
        if value is None:
            return None
        numeric = float(value)
        if not np.isfinite(numeric) or not 1.0 <= numeric <= 120.0:
            raise ValueError('V4L2 capture FPS must be finite and in [1, 120]')
        return numeric

    @staticmethod
    def _fps_from_streamparm(parm):
        numerator = int(parm.parm.capture.timeperframe.numerator)
        denominator = int(parm.parm.capture.timeperframe.denominator)
        if numerator <= 0 or denominator <= 0:
            return None
        value = denominator / numerator
        return value if np.isfinite(value) and value > 0.0 else None

    def _configure_capture_rate(self):
        """Request and read back the kernel capture rate before STREAMON.

        Failure to negotiate a rate is diagnostic, not a timestamp fallback.
        The Viewer still measures actual SOE intervals and reports a mismatch.
        """
        requested = self.requested_capture_fps
        if requested is None:
            self.capture_rate_reason = 'not_requested'
        else:
            fraction = Fraction(1.0 / requested).limit_denominator(1_000_000)
            parm = _StreamParm()
            parm.type = _VIDEO_CAPTURE
            parm.parm.capture.timeperframe.numerator = fraction.numerator
            parm.parm.capture.timeperframe.denominator = fraction.denominator
            try:
                fcntl.ioctl(self._fd, _S_PARM, parm, True)
            except OSError as exc:
                self.capture_rate_reason = (
                    f'set_failed_errno_{exc.errno}'
                )
            else:
                self.capture_rate_reason = 'requested'

        observed = _StreamParm()
        observed.type = _VIDEO_CAPTURE
        try:
            fcntl.ioctl(self._fd, _G_PARM, observed, True)
        except OSError as exc:
            suffix = f'get_failed_errno_{exc.errno}'
            self.capture_rate_reason = (
                suffix if self.capture_rate_reason == 'not_requested'
                else f'{self.capture_rate_reason},{suffix}'
            )
            return
        self.effective_capture_fps = self._fps_from_streamparm(observed)
        if self.effective_capture_fps is None:
            self.capture_rate_reason = f'{self.capture_rate_reason},not_reported'
        elif requested is None:
            self.capture_rate_reason = 'driver_reported'
        elif abs(self.effective_capture_fps - requested) <= max(
            0.1, requested * 0.01
        ):
            self.capture_rate_reason = 'matched'
        else:
            self.capture_rate_reason = 'negotiated_mismatch'

    @staticmethod
    def _new_buffer(index=0):
        buf = _Buffer()
        buf.type = _VIDEO_CAPTURE
        buf.memory = _MEMORY_MMAP
        buf.index = index
        return buf

    def read(self):
        read_started = time.monotonic()
        self.frame_timestamp_monotonic = None
        self.frame_timestamp_kind = ''
        self.last_read_metrics = V4L2ReadMetrics(
            total_discarded_buffers=self._discarded_buffer_total,
            requested_capture_fps=getattr(
                self, 'requested_capture_fps', None
            ),
            effective_capture_fps=getattr(
                self, 'effective_capture_fps', None
            ),
            capture_rate_reason=getattr(
                self, 'capture_rate_reason', 'unavailable'
            ),
        )
        if self._fd < 0:
            self._record_read_metrics(
                read_started=read_started, error_reason='camera_closed'
            )
            return None
        try:
            ready, _, _ = select.select([self._fd], [], [], 0.12)
        except OSError as exc:
            self._record_read_metrics(
                read_started=read_started,
                error_reason=f'v4l2_select_failed_errno_{exc.errno}',
            )
            return None
        if not ready:
            self._record_read_metrics(
                read_started=read_started, error_reason='v4l2_select_timeout'
            )
            return None

        # V4L2 returns completed buffers in capture order.  Processing only
        # one buffer per 20 Hz Viewer tick can permanently trail a 20 Hz
        # camera by the other completed MMAP buffers.  Drain the finite ready
        # set without blocking, requeue older buffers, and retain exactly the
        # newest buffer with its own image bytes, sequence and kernel stamp.
        candidate = None
        dequeued = 0
        discarded = 0
        dequeue_started = time.monotonic()
        dequeue_error = ''
        for _ in range(len(self._mappings)):
            buf = self._new_buffer()
            try:
                fcntl.ioctl(self._fd, _DQBUF, buf, True)
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break
                dequeue_error = f'v4l2_dequeue_failed_errno_{exc.errno}'
                break
            try:
                self._validate_dequeued_buffer(buf)
            except ValueError as exc:
                dequeue_error = str(exc)
                if buf.index < len(self._mappings):
                    try:
                        self._queue_buffer(buf.index)
                    except OSError:
                        pass
                break
            dequeued += 1
            if candidate is not None:
                try:
                    self._queue_buffer(candidate.index)
                except OSError as exc:
                    dequeue_error = f'v4l2_qbuf_failed_errno_{exc.errno}'
                    try:
                        self._queue_buffer(buf.index)
                    except OSError:
                        pass
                    candidate = None
                    break
                discarded += 1
                self._discarded_buffer_total += 1
            candidate = buf
        dequeue_finished = time.monotonic()
        if dequeue_error:
            if candidate is not None:
                try:
                    self._queue_buffer(candidate.index)
                except OSError:
                    pass
            self._record_read_metrics(
                read_started=read_started,
                dequeue_duration_sec=dequeue_finished - dequeue_started,
                dequeued_buffers=dequeued,
                discarded_buffers=discarded,
                error_reason=dequeue_error,
            )
            return None
        if candidate is None:
            self._record_read_metrics(
                read_started=read_started,
                dequeue_duration_sec=dequeue_finished - dequeue_started,
                dequeued_buffers=dequeued,
                discarded_buffers=discarded,
                error_reason=dequeue_error or 'v4l2_dequeue_empty',
            )
            return None

        image = None
        stamp = None
        kind = ''
        sequence_gap = 0
        decode_duration = 0.0
        error_reason = dequeue_error
        try:
            stamp, kind = timestamp_from_v4l2_buffer(
                candidate, monotonic_now=time.monotonic())
            if (
                self._last_sequence is not None
                and candidate.sequence <= self._last_sequence
            ):
                raise ValueError('V4L2 frame sequence did not advance')
            if self._last_sequence is not None:
                sequence_gap = max(
                    0, int(candidate.sequence) - int(self._last_sequence) - 1
                )
            raw = self._mappings[candidate.index][:candidate.bytesused]
            decode_started = time.monotonic()
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            decode_duration = time.monotonic() - decode_started
            if image is None:
                raise ValueError('V4L2 MJPEG frame could not be decoded')
            self._last_sequence = candidate.sequence
            self.frame_timestamp_monotonic = stamp
            self.frame_timestamp_kind = kind
        except ValueError as exc:
            error_reason = str(exc)
            image = None
        finally:
            try:
                self._queue_buffer(candidate.index)
            except OSError as exc:
                # Do not expose a timestamp from a failed video stream.
                self.frame_timestamp_monotonic = None
                self.frame_timestamp_kind = ''
                error_reason = f'v4l2_qbuf_failed_errno_{exc.errno}'
        self._record_read_metrics(
            read_started=read_started,
            dequeue_duration_sec=dequeue_finished - dequeue_started,
            decode_duration_sec=decode_duration,
            frame_timestamp_monotonic_sec=stamp,
            dequeue_monotonic_sec=dequeue_finished,
            sequence=int(candidate.sequence),
            sequence_gap=sequence_gap,
            dequeued_buffers=dequeued,
            discarded_buffers=discarded,
            error_reason=error_reason,
        )
        return image

    def _validate_dequeued_buffer(self, buf):
        if buf.index >= len(self._mappings):
            raise ValueError('V4L2 returned out-of-bounds buffer index')
        if not 0 < buf.bytesused <= len(self._mappings[buf.index]):
            raise ValueError('V4L2 returned invalid frame size')

    def _queue_buffer(self, index):
        fcntl.ioctl(self._fd, _QBUF, self._new_buffer(index), True)

    def _record_read_metrics(
        self,
        *,
        read_started,
        dequeue_duration_sec=0.0,
        decode_duration_sec=0.0,
        frame_timestamp_monotonic_sec=None,
        dequeue_monotonic_sec=None,
        sequence=None,
        sequence_gap=0,
        dequeued_buffers=0,
        discarded_buffers=0,
        error_reason='',
    ):
        finished = time.monotonic()
        transport_age = None
        if frame_timestamp_monotonic_sec is not None:
            transport_age = finished - float(frame_timestamp_monotonic_sec)
        self.last_read_metrics = V4L2ReadMetrics(
            frame_timestamp_monotonic_sec=frame_timestamp_monotonic_sec,
            dequeue_monotonic_sec=dequeue_monotonic_sec,
            transport_age_sec=transport_age,
            read_duration_sec=max(0.0, finished - float(read_started)),
            dequeue_duration_sec=max(0.0, float(dequeue_duration_sec)),
            decode_duration_sec=max(0.0, float(decode_duration_sec)),
            sequence=sequence,
            sequence_gap=max(0, int(sequence_gap)),
            dequeued_buffers=max(0, int(dequeued_buffers)),
            discarded_buffers=max(0, int(discarded_buffers)),
            total_discarded_buffers=self._discarded_buffer_total,
            requested_capture_fps=getattr(
                self, 'requested_capture_fps', None
            ),
            effective_capture_fps=getattr(
                self, 'effective_capture_fps', None
            ),
            capture_rate_reason=getattr(
                self, 'capture_rate_reason', 'unavailable'
            ),
            error_reason=str(error_reason),
        )

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
