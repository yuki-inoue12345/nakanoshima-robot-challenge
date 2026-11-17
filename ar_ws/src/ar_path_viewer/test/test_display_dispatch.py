import threading
import time

import numpy as np

from ar_path_viewer.display_dispatch import LatestDisplayFrame


def _frame(value: int) -> np.ndarray:
    return np.full((12, 16, 3), value, dtype=np.uint8)


def test_latest_frame_exchange_drops_superseded_frames_without_backlog():
    exchange = LatestDisplayFrame()
    for value in range(20):
        assert exchange.publish(_frame(value))

    sequence, frame = exchange.wait_for_new(0, 0.0)

    assert sequence == 20
    assert frame is not None
    assert np.all(frame == 19)


def test_worker_can_publish_at_20fps_while_main_thread_consumes():
    exchange = LatestDisplayFrame()
    producer_thread_id = []

    def produce():
        producer_thread_id.append(threading.get_ident())
        for value in range(10):
            assert exchange.publish(_frame(value))
            time.sleep(0.005)  # accelerated equivalent of a paced producer

    worker = threading.Thread(target=produce)
    worker.start()
    main_thread_id = threading.get_ident()
    sequence = 0
    observed = []
    while worker.is_alive() or sequence < 10:
        sequence, frame = exchange.wait_for_new(sequence, 0.05)
        if frame is not None:
            observed.append(int(frame[0, 0, 0]))
    worker.join()

    assert producer_thread_id == [worker.ident]
    assert worker.ident != main_thread_id
    assert observed
    assert observed[-1] == 9
    assert sequence == 10


def test_close_wakes_waiter_and_rejects_future_frames():
    exchange = LatestDisplayFrame()
    result = []

    def wait():
        result.append(exchange.wait_for_new(0, 1.0))

    waiter = threading.Thread(target=wait)
    waiter.start()
    exchange.close()
    waiter.join(timeout=0.5)

    assert not waiter.is_alive()
    assert result == [(0, None)]
    assert not exchange.publish(_frame(1))


def test_publish_copies_frame_before_handoff():
    exchange = LatestDisplayFrame()
    original = _frame(4)
    exchange.publish(original)
    original[:] = 99

    _, received = exchange.wait_for_new(0, 0.0)

    assert received is not None
    assert np.all(received == 4)


def test_display_metadata_stays_with_the_exact_latest_pixels():
    exchange = LatestDisplayFrame()
    exchange.publish(_frame(3), 100.01)
    exchange.publish(_frame(8), 100.04)

    sequence, frame, metadata = exchange.wait_for_new_with_metadata(0, 0.0)

    assert sequence == 2
    assert np.all(frame == 8)
    assert metadata.source_timestamp_monotonic_sec == 100.04
