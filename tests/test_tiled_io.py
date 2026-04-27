"""Tests for tiled_io — read_images and write_output."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_array_node(arr: np.ndarray) -> MagicMock:
    """Return a mock tiled array node whose .read() returns arr."""
    node = MagicMock()
    node.read.return_value = arr
    return node


def _make_client(nodes: dict) -> MagicMock:
    """Return a mock tiled root client that resolves single path parts via __getitem__."""
    root = MagicMock()
    root.__getitem__ = MagicMock(side_effect=lambda key: nodes[key])
    return root


@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    monkeypatch.setenv("EMBLASE_TILED_SERVER_URL", "http://fake-tiled")
    monkeypatch.setenv("EMBLASE_TILED_API_KEY", "testkey")


def test_read_images_2d(monkeypatch):
    """A single 2-D array yields one frame."""
    arr = np.ones((64, 128), dtype=np.float32)
    client = _make_client({"scan": _make_array_node(arr)})

    with patch("emblase.tiled_io._client", return_value=client):
        from emblase.tiled_io import read_images
        frames = read_images(["scan"])

    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0], arr)
    assert frames[0].dtype == np.float32


def test_read_images_3d_stack(monkeypatch):
    """A (N, H, W) array yields N frames."""
    arr = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    client = _make_client({"stack": _make_array_node(arr)})

    with patch("emblase.tiled_io._client", return_value=client):
        from emblase.tiled_io import read_images
        frames = read_images(["stack"])

    assert len(frames) == 3
    for i, frame in enumerate(frames):
        assert frame.shape == (4, 5)
        np.testing.assert_array_equal(frame, arr[i])


def test_read_images_4d_stack(monkeypatch):
    """A (A, B, H, W) array yields A*B frames."""
    arr = np.zeros((2, 3, 8, 8), dtype=np.float32)
    client = _make_client({"nd": _make_array_node(arr)})

    with patch("emblase.tiled_io._client", return_value=client):
        from emblase.tiled_io import read_images
        frames = read_images(["nd"])

    assert len(frames) == 6
    assert all(f.shape == (8, 8) for f in frames)


def test_read_images_mixed_uris(monkeypatch):
    """Multiple URIs with different shapes are combined into one flat list."""
    arr_2d = np.ones((10, 20), dtype=np.float32)
    arr_3d = np.zeros((4, 6, 7), dtype=np.float32)
    client = _make_client({"a": _make_array_node(arr_2d), "b": _make_array_node(arr_3d)})

    with patch("emblase.tiled_io._client", return_value=client):
        from emblase.tiled_io import read_images
        frames = read_images(["a", "b"])

    assert len(frames) == 5          # 1 + 4
    assert frames[0].shape == (10, 20)
    assert frames[1].shape == (6, 7)


def test_read_images_1d_raises(monkeypatch):
    """A 1-D array should raise ValueError."""
    arr = np.ones((10,), dtype=np.float32)
    client = _make_client({"bad": _make_array_node(arr)})

    with patch("emblase.tiled_io._client", return_value=client):
        from emblase.tiled_io import read_images
        with pytest.raises(ValueError, match="fewer than 2 dimensions"):
            read_images(["bad"])


def test_write_output(monkeypatch):
    """write_output calls container.write_array with the correct array."""
    output = np.zeros((5, 512), dtype=np.float32)
    container = MagicMock()
    client = _make_client({"results": container})

    with patch("emblase.tiled_io._client", return_value=client):
        from emblase.tiled_io import write_output
        write_output("results", output, key="job_42")

    container.write_array.assert_called_once()
    call_args = container.write_array.call_args
    np.testing.assert_array_equal(call_args.args[0], output)
    assert call_args.kwargs.get("key") == "job_42"
