"""Tests for tiled_io — read_images and write_output."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from emblase.tiled_io import read_images, write_output


def _make_array_node(arr: np.ndarray, sliced: dict | None = None) -> MagicMock:
    """Return a mock tiled array node.

    ``sliced`` maps slice strings to the expected sub-array; if None, .read()
    always returns ``arr`` regardless of the argument.
    """
    node = MagicMock()
    if sliced:
        def _read(s=None):
            return sliced[s] if s is not None else arr
        node.read.side_effect = _read
    else:
        node.read.return_value = arr
    return node


def _make_client(nodes: dict) -> MagicMock:
    """Return a mock tiled root client where client[path] returns nodes[path]."""
    root = MagicMock()
    root.__getitem__ = MagicMock(side_effect=lambda key: nodes[key])
    return root


# ---------------------------------------------------------------------------
# read_images — bare path (str entries)
# ---------------------------------------------------------------------------

def test_read_images_2d():
    """A single 2-D array (bare path) yields one frame."""
    arr = np.ones((64, 128), dtype=np.float32)
    client = _make_client({"scan": _make_array_node(arr)})

    frames = read_images(client, ["scan"])

    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0], arr)
    assert frames[0].dtype == np.float32


def test_read_images_3d_stack():
    """A (N, H, W) array (bare path) yields N frames."""
    arr = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    client = _make_client({"stack": _make_array_node(arr)})

    frames = read_images(client, ["stack"])

    assert len(frames) == 3
    for i, frame in enumerate(frames):
        assert frame.shape == (4, 5)
        np.testing.assert_array_equal(frame, arr[i])


def test_read_images_4d_stack():
    """A (A, B, H, W) array (bare path) yields A*B frames."""
    arr = np.zeros((2, 3, 8, 8), dtype=np.float32)
    client = _make_client({"nd": _make_array_node(arr)})

    frames = read_images(client, ["nd"])

    assert len(frames) == 6
    assert all(f.shape == (8, 8) for f in frames)


def test_read_images_mixed_paths():
    """Multiple bare paths with different shapes combine into one flat list."""
    arr_2d = np.ones((10, 20), dtype=np.float32)
    arr_3d = np.zeros((4, 6, 7), dtype=np.float32)
    client = _make_client({"a": _make_array_node(arr_2d), "b": _make_array_node(arr_3d)})

    frames = read_images(client, ["a", "b"])

    assert len(frames) == 5  # 1 + 4
    assert frames[0].shape == (10, 20)
    assert frames[1].shape == (6, 7)


def test_read_images_1d_raises():
    """A 1-D array should raise ValueError."""
    arr = np.ones((10,), dtype=np.float32)
    client = _make_client({"bad": _make_array_node(arr)})

    with pytest.raises(ValueError, match="fewer than 2 dimensions"):
        read_images(client, ["bad"])


# ---------------------------------------------------------------------------
# read_images — (path, slice) tuple entries
# ---------------------------------------------------------------------------

def test_read_images_with_slice():
    """(path, slice) tuple passes the slice string to node.read()."""
    full = np.arange(5 * 32 * 32, dtype=np.float32).reshape(5, 32, 32)
    sliced = full[1:3]  # shape (2, 32, 32)
    node = _make_array_node(full, sliced={"1:3": sliced})
    client = _make_client({"scan": node})

    frames = read_images(client, [("scan", "1:3")])

    node.read.assert_called_once_with("1:3")
    assert len(frames) == 2
    assert frames[0].shape == (32, 32)


def test_read_images_mixed_str_and_tuple():
    """A list mixing bare paths and (path, slice) tuples is handled correctly."""
    arr_full = np.ones((4, 8, 8), dtype=np.float32)
    arr_slice = arr_full[0:1]  # shape (1, 8, 8)
    node_a = _make_array_node(arr_full)
    node_b = _make_array_node(arr_full, sliced={"0:1": arr_slice})
    client = _make_client({"a": node_a, "b": node_b})

    frames = read_images(client, ["a", ("b", "0:1")])

    assert len(frames) == 5  # 4 from "a", 1 from ("b", "0:1")
    node_a.read.assert_called_once_with()
    node_b.read.assert_called_once_with("0:1")


# ---------------------------------------------------------------------------
# write_output
# ---------------------------------------------------------------------------

def test_write_output():
    """write_output calls container.write_array with the correct array."""
    output = np.zeros((5, 512), dtype=np.float32)
    container = MagicMock()
    client = _make_client({"results": container})

    write_output(client, "results", output, key="job_42")

    container.write_array.assert_called_once()
    call_args = container.write_array.call_args
    np.testing.assert_array_equal(call_args.args[0], output)
    assert call_args.kwargs.get("key") == "job_42"
