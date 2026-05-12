"""Tests for InputsWatcher (pipeline/streaming.py)."""

import asyncio
import threading
import time
import types
from unittest.mock import AsyncMock, MagicMock

from emblase.pipeline.streaming import InputsWatcher


def _make_watcher(submit_mock):
    """Return an InputsWatcher wired to a mock backend."""
    inputs_node = MagicMock()
    inputs_node.uri = "https://tiled.example.com/api/v1/metadata/smi/inputs_copy"

    backend = MagicMock()
    backend.submit_streaming = submit_mock

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    return InputsWatcher(
        inputs_node=inputs_node,
        output_root="smi/results",
        backend=backend,
        model_name="noop",
        loop=loop,
    ), loop


def _make_event(key):
    e = types.SimpleNamespace(key=key)
    return e


def test_duplicate_child_created_submits_once():
    """Two child_created events for the same key submit only one job."""
    submit_mock = AsyncMock(return_value=42)
    watcher, loop = _make_watcher(submit_mock)

    watcher._on_child_created(_make_event("run_abc"))
    watcher._on_child_created(_make_event("run_abc"))

    time.sleep(0.2)

    assert submit_mock.call_count == 1


def test_concurrent_duplicate_child_created_submits_once():
    """Concurrent child_created events from multiple threads submit only one job.

    This is the race condition that caused double-submission in production:
    two WS callback threads both pass the `if run_key in _seen_runs` check
    before either adds the key. The fix uses a threading.Lock.
    """
    submit_mock = AsyncMock(return_value=42)
    watcher, loop = _make_watcher(submit_mock)

    barrier = threading.Barrier(10)
    threads = []
    for _ in range(10):

        def _fire(b=barrier, w=watcher):
            b.wait()  # all threads start at the same time
            w._on_child_created(_make_event("run_abc"))

        threads.append(threading.Thread(target=_fire))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    time.sleep(0.3)

    assert submit_mock.call_count == 1


def test_different_runs_each_submit_once():
    """Two different run keys each get exactly one job."""
    submit_mock = AsyncMock(return_value=42)
    watcher, loop = _make_watcher(submit_mock)

    watcher._on_child_created(_make_event("run_a"))
    watcher._on_child_created(_make_event("run_b"))

    time.sleep(0.2)

    assert submit_mock.call_count == 2


def test_stop_disconnects_subscription():
    """stop() calls disconnect on the active subscription if one exists."""
    submit_mock = AsyncMock(return_value=42)
    watcher, loop = _make_watcher(submit_mock)

    mock_sub = MagicMock()
    watcher._sub = mock_sub

    watcher.stop()

    mock_sub.disconnect.assert_called_once()
