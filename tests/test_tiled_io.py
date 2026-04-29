"""Tests for emblase.tiled.client — read_images and write_output."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from emblase.tiled.client import read_images, write_output


def _make_array_node(arr: np.ndarray, sliced_arr: np.ndarray | None = None) -> MagicMock:
    """Return a mock tiled array node.

    If ``sliced_arr`` is provided, ``node.read(anything_truthy)`` returns it;
    ``node.read()`` / ``node.read(None)`` returns ``arr``.
    """
    node = MagicMock()

    def _read(s=None):
        if s is not None and sliced_arr is not None:
            return sliced_arr
        return arr

    node.read.side_effect = _read
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
    """A single 2-D array (bare path) yields one frame and one provenance entry."""
    arr = np.ones((64, 128), dtype=np.float32)
    client = _make_client({"scan": _make_array_node(arr)})

    frames, frame_entries = read_images(client, ["scan"])

    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0], arr)
    assert frames[0].dtype == np.float32
    assert frame_entries == ["scan"]


def test_read_images_3d_stack():
    """A (N, H, W) array yields N frames, each pointing to the same entry."""
    arr = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    client = _make_client({"stack": _make_array_node(arr)})

    frames, frame_entries = read_images(client, ["stack"])

    assert len(frames) == 3
    for i, frame in enumerate(frames):
        assert frame.shape == (4, 5)
        np.testing.assert_array_equal(frame, arr[i])
    assert frame_entries == ["stack"] * 3


def test_read_images_4d_stack():
    """A (A, B, H, W) array yields A*B frames all pointing to the same entry."""
    arr = np.zeros((2, 3, 8, 8), dtype=np.float32)
    client = _make_client({"nd": _make_array_node(arr)})

    frames, frame_entries = read_images(client, ["nd"])

    assert len(frames) == 6
    assert all(f.shape == (8, 8) for f in frames)
    assert frame_entries == ["nd"] * 6


def test_read_images_mixed_paths():
    """frame_entries tracks which entry each frame came from."""
    arr_2d = np.ones((10, 20), dtype=np.float32)
    arr_3d = np.zeros((4, 6, 7), dtype=np.float32)
    client = _make_client({"a": _make_array_node(arr_2d), "b": _make_array_node(arr_3d)})

    frames, frame_entries = read_images(client, ["a", "b"])

    assert len(frames) == 5  # 1 + 4
    assert frames[0].shape == (10, 20)
    assert frames[1].shape == (6, 7)
    assert frame_entries == ["a"] + ["b"] * 4


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
    """(path, slice) tuple passes a slice object to node.read()."""
    full = np.arange(5 * 32 * 32, dtype=np.float32).reshape(5, 32, 32)
    sliced = full[1:3]  # shape (2, 32, 32)
    node = _make_array_node(full, sliced_arr=sliced)
    client = _make_client({"scan": node})

    frames, frame_entries = read_images(client, [("scan", "1:3")])

    node.read.assert_called_once()  # called with an NDSlice, not bare string
    assert len(frames) == 2
    assert frames[0].shape == (32, 32)
    assert frame_entries == [("scan", "1:3")] * 2


def test_read_images_mixed_str_and_tuple():
    """A list mixing bare paths and (path, slice) tuples is handled correctly."""
    arr_full = np.ones((4, 8, 8), dtype=np.float32)
    arr_slice = arr_full[0:1]  # shape (1, 8, 8)
    node_a = _make_array_node(arr_full)
    node_b = _make_array_node(arr_full, sliced_arr=arr_slice)
    client = _make_client({"a": node_a, "b": node_b})

    frames, frame_entries = read_images(client, ["a", ("b", "0:1")])

    assert len(frames) == 5  # 4 from "a", 1 from ("b", "0:1")
    node_a.read.assert_called_once()
    node_b.read.assert_called_once()
    assert frame_entries == ["a"] * 4 + [("b", "0:1")]


# ---------------------------------------------------------------------------
# write_output
# ---------------------------------------------------------------------------

@patch("emblase.tiled.client.create_embedding_container")
@patch("emblase.tiled.client.LatentSpaceEmbedding")
def test_write_output_creates_container_when_missing(mock_lse_cls, mock_create):
    """write_output creates a new container if the key is missing."""
    embeddings = np.zeros((3, 4), dtype=np.float32)
    container = MagicMock()
    container.metadata = {"embedding_dim": 4, "thumb_shape": [64, 64]}
    mock_create.return_value = container

    root = MagicMock()
    root.__getitem__ = MagicMock(side_effect=KeyError("results"))

    write_output(root, "results", embeddings)

    mock_create.assert_called_once()
    container.append.assert_called_once()
    call = container.append.call_args
    np.testing.assert_array_equal(call.args[0], embeddings)  # embeddings
    assert call.args[1].shape == (3, 64, 64)  # default thumb_shape zeros


@patch("emblase.tiled.client.create_embedding_container")
def test_write_output_appends_to_existing_container(mock_create):
    """write_output appends to existing container without re-creating it."""
    from emblase.tiled.client import LatentSpaceEmbedding
    from unittest.mock import create_autospec

    embeddings = np.ones((2, 4), dtype=np.float32)
    container = create_autospec(LatentSpaceEmbedding, instance=True)
    container.metadata = {"embedding_dim": 4, "thumb_shape": [64, 64]}

    root = MagicMock()
    root.__getitem__ = MagicMock(return_value=container)

    write_output(root, "results", embeddings)

    mock_create.assert_not_called()
    container.append.assert_called_once()


@patch("emblase.tiled.client.create_embedding_container")
@patch("emblase.tiled.client.LatentSpaceEmbedding")
def test_write_output_passes_provenance(mock_lse_cls, mock_create):
    """tiled_entries are split into paths and slices in the append call."""
    embeddings = np.zeros((2, 4), dtype=np.float32)
    container = MagicMock()
    container.metadata = {"embedding_dim": 4, "thumb_shape": [64, 64]}
    mock_create.return_value = container

    root = MagicMock()
    root.__getitem__ = MagicMock(side_effect=KeyError)

    entries = ["scan/001", ("scan/002", "3:5")]
    write_output(root, "results", embeddings, source_entries=entries)

    call = container.append.call_args
    assert call.kwargs["paths"] == ["scan/001", "scan/002"]
    assert call.kwargs["slices"] == [None, "3:5"]


@patch("emblase.tiled.client.create_embedding_container")
@patch("emblase.tiled.client.LatentSpaceEmbedding")
def test_write_output_custom_thumb_fn(mock_lse_cls, mock_create):
    """A custom thumb_fn is called on the whole batch when shapes are uniform."""
    embeddings = np.zeros((2, 4), dtype=np.float32)
    images = [np.ones((32, 32), dtype=np.float32) * i for i in range(2)]
    container = MagicMock()
    container.metadata = {"embedding_dim": 4, "thumb_shape": [64, 64]}
    mock_create.return_value = container

    root = MagicMock()
    root.__getitem__ = MagicMock(side_effect=KeyError)

    called_with = []

    def my_thumb(frames):
        called_with.append(frames.shape)
        return np.zeros((frames.shape[0], 4, 4), dtype=np.float32)

    write_output(root, "results", embeddings, images=images, thumb_fn=my_thumb)

    # Uniform shapes → called once on the full (N, H, W) batch
    assert len(called_with) == 1
    assert called_with[0] == (2, 32, 32)
    thumbnails = container.append.call_args.args[1]
    assert thumbnails.shape == (2, 4, 4)


@patch("emblase.tiled.client.create_embedding_container")
@patch("emblase.tiled.client.LatentSpaceEmbedding")
def test_write_output_custom_thumb_fn_ragged(mock_lse_cls, mock_create):
    """A custom thumb_fn is called once per frame for ragged (varying-shape) inputs."""
    embeddings = np.zeros((2, 4), dtype=np.float32)
    # Two frames with different spatial shapes → ragged
    images = [np.ones((32, 32), dtype=np.float32), np.ones((64, 64), dtype=np.float32)]
    container = MagicMock()
    container.metadata = {"embedding_dim": 4, "thumb_shape": [4, 4]}
    mock_create.return_value = container

    root = MagicMock()
    root.__getitem__ = MagicMock(side_effect=KeyError)

    called_with = []

    def my_thumb(frames):
        called_with.append(frames.shape)
        return np.zeros((frames.shape[0], 4, 4), dtype=np.float32)

    write_output(root, "results", embeddings, images=images, thumb_fn=my_thumb)

    # Ragged shapes → called once per frame as (1, H, W)
    assert len(called_with) == 2
    assert called_with[0] == (1, 32, 32)
    assert called_with[1] == (1, 64, 64)
    thumbnails = container.append.call_args.args[1]
    assert thumbnails.shape == (2, 4, 4)


@patch("emblase.tiled.client.create_embedding_container")
@patch("emblase.tiled.client.LatentSpaceEmbedding")
def test_write_output_passes_access_tags_to_create(mock_lse_cls, mock_create):
    """access_tags is forwarded to create_embedding_container."""
    embeddings = np.zeros((2, 4), dtype=np.float32)
    container = MagicMock()
    container.metadata = {"embedding_dim": 4, "thumb_shape": [64, 64]}
    mock_create.return_value = container

    root = MagicMock()
    root.__getitem__ = MagicMock(side_effect=KeyError)

    write_output(root, "results", embeddings, access_tags=["nsls2", "staff"])

    _, kwargs = mock_create.call_args
    assert kwargs["access_tags"] == ["nsls2", "staff"]


@patch("emblase.tiled.client.create_embedding_container")
@patch("emblase.tiled.client.LatentSpaceEmbedding")
def test_write_output_passes_access_tags_to_append(mock_lse_cls, mock_create):
    """access_tags is forwarded to container.append."""
    embeddings = np.zeros((2, 4), dtype=np.float32)
    container = MagicMock()
    container.metadata = {"embedding_dim": 4, "thumb_shape": [64, 64]}
    mock_create.return_value = container

    root = MagicMock()
    root.__getitem__ = MagicMock(side_effect=KeyError)

    write_output(root, "results", embeddings, access_tags=["nsls2"])

    call = container.append.call_args
    assert call.kwargs["access_tags"] == ["nsls2"]
