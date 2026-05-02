"""Tests for emblase.tiled.client — read_images and write_output."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from emblase.tiled.client import (
    read_images,
    write_output,
    _log_thumb_fn,
    _LOG_THUMB_ROI,
    _infer_param_specs,
    _make_index_schema,
    THUMB_MODES,
    LatentSpaceEmbedding,
    ParamSpec,
)


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


# ---------------------------------------------------------------------------
# _log_thumb_fn and thumb_mode tests
# ---------------------------------------------------------------------------

class TestLogThumbFn:
    """Unit tests for the log-normalised thumbnail function."""

    def test_output_shape(self):
        """Output shape is (N, 64, 64) for any (N, H, W) input that fits the ROI."""
        frames = np.ones((5, 300, 500), dtype=np.float32) * 100
        out = _log_thumb_fn(frames)
        assert out.shape == (5, 64, 64)

    def test_clips_negatives(self):
        """Negative pixel values are clipped to 0 before log1p."""
        frames = np.full((1, 300, 500), -50.0, dtype=np.float32)
        out = _log_thumb_fn(frames)
        assert out.min() == 0.0

    def test_log1p_applied(self):
        """Output equals log1p(clip(roi, 0)) after resize — spot-check with known value."""
        val = 100.0
        frames = np.full((1, 300, 500), val, dtype=np.float32)
        out = _log_thumb_fn(frames)
        expected = np.log1p(val)
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_custom_roi(self):
        """A custom ROI can override the default."""
        frames = np.zeros((2, 300, 500), dtype=np.float32)
        frames[:, 50:100, 100:200] = 1000.0  # only non-zero in a non-default region
        roi = (slice(50, 100), slice(100, 200))
        out = _log_thumb_fn(frames, roi=roi)
        assert out.max() > 0  # non-zero region was captured

    def test_log_thumb_roi_default_constant(self):
        """The default ROI covers rows 0:180, cols 220:400."""
        assert _LOG_THUMB_ROI == (slice(0, 180), slice(220, 400))

    def test_in_thumb_modes_registry(self):
        """Both 'default' and 'log' are registered in THUMB_MODES."""
        assert "default" in THUMB_MODES
        assert "logroi" in THUMB_MODES


@patch("emblase.tiled.client.create_embedding_container")
@patch("emblase.tiled.client.LatentSpaceEmbedding")
def test_write_output_thumb_mode_log(mock_lse_cls, mock_create):
    """thumb_mode='log' routes through _log_thumb_fn."""
    embeddings = np.zeros((3, 4), dtype=np.float32)
    # Frames large enough for the default ROI (rows 0:180, cols 220:400)
    images = [np.ones((300, 500), dtype=np.float32) * 200.0 for _ in range(3)]
    container = MagicMock()
    container.metadata = {"embedding_dim": 4, "thumb_shape": [64, 64]}
    mock_create.return_value = container
    root = MagicMock()
    root.__getitem__ = MagicMock(side_effect=KeyError)

    write_output(root, "results", embeddings, images=images, thumb_mode="logroi")

    thumbnails = container.append.call_args.args[1]
    assert thumbnails.shape == (3, 64, 64)
    expected_val = np.log1p(200.0)
    np.testing.assert_allclose(thumbnails, expected_val, rtol=1e-5)


@patch("emblase.tiled.client.create_embedding_container")
@patch("emblase.tiled.client.LatentSpaceEmbedding")
def test_write_output_thumb_mode_unknown_raises(mock_lse_cls, mock_create):
    """An unknown thumb_mode raises ValueError."""
    embeddings = np.zeros((2, 4), dtype=np.float32)
    images = [np.ones((300, 500), dtype=np.float32) for _ in range(2)]
    container = MagicMock()
    container.metadata = {"embedding_dim": 4, "thumb_shape": [64, 64]}
    mock_create.return_value = container
    root = MagicMock()
    root.__getitem__ = MagicMock(side_effect=KeyError)

    with pytest.raises(ValueError, match="Unknown thumb_mode"):
        write_output(root, "results", embeddings, images=images, thumb_mode="bogus")


@patch("emblase.tiled.client.create_embedding_container")
@patch("emblase.tiled.client.LatentSpaceEmbedding")
def test_write_output_thumb_fn_overrides_mode(mock_lse_cls, mock_create):
    """Explicit thumb_fn takes priority over thumb_mode."""
    embeddings = np.zeros((2, 4), dtype=np.float32)
    images = [np.ones((300, 500), dtype=np.float32) for _ in range(2)]
    container = MagicMock()
    container.metadata = {"embedding_dim": 4, "thumb_shape": [64, 64]}
    mock_create.return_value = container
    root = MagicMock()
    root.__getitem__ = MagicMock(side_effect=KeyError)

    sentinel = object()
    call_log = []

    def custom_fn(frames):
        call_log.append(sentinel)
        return np.zeros((frames.shape[0], 4, 4), dtype=np.float32)

    # thumb_fn supplied alongside thumb_mode="logroi" — custom_fn must win
    write_output(root, "results", embeddings, images=images,
                 thumb_fn=custom_fn, thumb_mode="logroi")

    assert len(call_log) == 1 and call_log[0] is sentinel


@patch("emblase.tiled.client.create_embedding_container")
@patch("emblase.tiled.client.LatentSpaceEmbedding")
def test_write_output_no_images_produces_zero_thumbnails(mock_lse_cls, mock_create):
    """When images=None, thumbnails are zeros of the expected shape."""
    embeddings = np.zeros((3, 4), dtype=np.float32)
    container = MagicMock()
    container.metadata = {"embedding_dim": 4, "thumb_shape": [8, 8]}
    mock_create.return_value = container
    root = MagicMock()
    root.__getitem__ = MagicMock(side_effect=KeyError)

    write_output(root, "results", embeddings, images=None, thumb_shape=(8, 8))

    thumbnails = container.append.call_args.args[1]
    assert thumbnails.shape == (3, 8, 8)
    np.testing.assert_array_equal(thumbnails, 0.0)


@patch("emblase.tiled.client.create_embedding_container")
@patch("emblase.tiled.client.LatentSpaceEmbedding")
def test_write_output_default_thumb_mode_resizes(mock_lse_cls, mock_create):
    """thumb_mode='default' resizes frames to thumb_shape without normalisation."""
    embeddings = np.zeros((2, 4), dtype=np.float32)
    pixel_val = 42.0
    images = [np.full((128, 256), pixel_val, dtype=np.float32) for _ in range(2)]
    container = MagicMock()
    container.metadata = {"embedding_dim": 4, "thumb_shape": [64, 64]}
    mock_create.return_value = container
    root = MagicMock()
    root.__getitem__ = MagicMock(side_effect=KeyError)

    write_output(root, "results", embeddings, images=images, thumb_mode="default")

    thumbnails = container.append.call_args.args[1]
    assert thumbnails.shape == (2, 64, 64)
    # Constant-value image should survive resize unchanged.
    np.testing.assert_allclose(thumbnails, pixel_val, rtol=1e-5)


# ---------------------------------------------------------------------------
# LatentSpaceEmbedding._write_arrays — projections always present
# ---------------------------------------------------------------------------

class _FakeLSE:
    """Minimal stand-in for LatentSpaceEmbedding for _write_arrays testing."""

    _write_arrays = LatentSpaceEmbedding._write_arrays

    def __init__(self, proj_dim: int = 2):
        self._arrays_initialised: bool | None = False
        self._num_embeddings = 0
        self.metadata = {"projection_dim": proj_dim}
        self.access_blob = {}
        self._written: dict = {}

    def write_array(self, data, key, **kwargs):
        arr = MagicMock()
        arr.data = data.copy()
        self._written[key] = arr
        return arr

    def __contains__(self, key):
        return key in self._written

    def __getitem__(self, key):
        return self._written[key]


def test_write_arrays_projections_nan_when_not_supplied():
    """When projections=None, a NaN-filled (B, 2) array must be written."""
    lse = _FakeLSE(proj_dim=2)
    embeddings = np.zeros((3, 8), dtype=np.float32)
    thumbnails = np.zeros((3, 16, 16), dtype=np.float32)

    LatentSpaceEmbedding._write_arrays(lse, embeddings, thumbnails)

    assert "projections" in lse._written
    proj = lse._written["projections"].data
    assert proj.shape == (3, 2)
    assert proj.dtype == np.float32
    assert np.all(np.isnan(proj))


def test_write_arrays_projections_stored_when_supplied():
    """When projections are provided they must be stored as-is."""
    lse = _FakeLSE(proj_dim=2)
    embeddings = np.zeros((3, 8), dtype=np.float32)
    thumbnails = np.zeros((3, 16, 16), dtype=np.float32)
    projections = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    LatentSpaceEmbedding._write_arrays(lse, embeddings, thumbnails, projections=projections)

    proj = lse._written["projections"].data
    np.testing.assert_array_equal(proj, projections)


def test_write_arrays_projections_shape_matches_batch():
    """Shape of the stored projections must equal (batch_size, projection_dim)."""
    for n in (1, 5, 16):
        lse = _FakeLSE(proj_dim=2)
        embeddings = np.zeros((n, 4), dtype=np.float32)
        thumbnails = np.zeros((n, 8, 8), dtype=np.float32)
        LatentSpaceEmbedding._write_arrays(lse, embeddings, thumbnails)
        assert lse._written["projections"].data.shape == (n, 2), f"failed for n={n}"


# ---------------------------------------------------------------------------
# _make_index_schema — param column generation
# ---------------------------------------------------------------------------

class TestMakeIndexSchema:
    def test_no_params_returns_base_schema(self):
        schema = _make_index_schema()
        names = schema.names
        assert "path" in names
        assert "timestamp" in names
        assert not any(n.startswith("param_") for n in names)

    def test_float_param_is_float64(self):
        import pyarrow as pa
        schema = _make_index_schema({"temp": {"dtype": "float", "units": "°C"}})
        field = schema.field("param_temp")
        assert field.type == pa.float64()
        assert field.nullable

    def test_integer_param_is_int64(self):
        import pyarrow as pa
        schema = _make_index_schema({"scan_id": {"dtype": "integer"}})
        assert schema.field("param_scan_id").type == pa.int64()

    def test_string_param_is_string(self):
        import pyarrow as pa
        schema = _make_index_schema({"sample": {"dtype": "string"}})
        assert schema.field("param_sample").type == pa.string()

    def test_boolean_param_is_bool(self):
        import pyarrow as pa
        schema = _make_index_schema({"is_cal": {"dtype": "boolean"}})
        assert schema.field("param_is_cal").type == pa.bool_()

    def test_multiple_params_all_present(self):
        specs = {
            "temperature": {"dtype": "float"},
            "scan_id": {"dtype": "integer"},
            "sample": {"dtype": "string"},
        }
        schema = _make_index_schema(specs)
        assert "param_temperature" in schema.names
        assert "param_scan_id" in schema.names
        assert "param_sample" in schema.names

    def test_unknown_dtype_raises(self):
        with pytest.raises(ValueError, match="Unknown param dtype"):
            _make_index_schema({"x": {"dtype": "array"}})  # type: ignore


# ---------------------------------------------------------------------------
# _infer_param_specs — dtype inference from values
# ---------------------------------------------------------------------------

class TestInferParamSpecs:
    def test_float_values(self):
        specs = _infer_param_specs({"temp": [25.1, 25.3]})
        assert specs["temp"]["dtype"] == "float"

    def test_int_values(self):
        specs = _infer_param_specs({"scan_id": [1, 2, 3]})
        assert specs["scan_id"]["dtype"] == "integer"

    def test_bool_values(self):
        specs = _infer_param_specs({"flag": [True, False]})
        assert specs["flag"]["dtype"] == "boolean"

    def test_string_values(self):
        specs = _infer_param_specs({"sample": ["CsPbBr3", "MAPbI3"]})
        assert specs["sample"]["dtype"] == "string"

    def test_none_only_defaults_to_float(self):
        specs = _infer_param_specs({"unknown": [None, None]})
        assert specs["unknown"]["dtype"] == "float"

    def test_none_skipped_to_find_sample(self):
        specs = _infer_param_specs({"scan_id": [None, 42]})
        assert specs["scan_id"]["dtype"] == "integer"

    def test_multiple_params(self):
        specs = _infer_param_specs({
            "temp": [25.0], "scan_id": [1], "sample": ["A"]
        })
        assert specs["temp"]["dtype"] == "float"
        assert specs["scan_id"]["dtype"] == "integer"
        assert specs["sample"]["dtype"] == "string"


# ---------------------------------------------------------------------------
# LatentSpaceEmbedding.append — params validation
# ---------------------------------------------------------------------------

class _FakeLSEWithParams:
    """Stand-in for LatentSpaceEmbedding with param support for append() testing."""

    append = LatentSpaceEmbedding.append
    _write_arrays = MagicMock()
    num_embeddings = 0
    _num_embeddings = 0

    def __init__(self, param_specs=None):
        import pyarrow as pa
        self._param_specs = param_specs or {}
        self._index_schema = _make_index_schema(self._param_specs)
        self.metadata = {"embedding_dim": 4, "thumb_shape": [8, 8]}
        self.embedding_dim = 4

        # Track appended tables
        self._appended = []

        class FakeTable:
            def append_partition(inner_self, partition, table):
                self._appended.append(table)

        self._index_table = FakeTable()
        self._write_arrays = MagicMock()

    def _write_arrays(self, *a, **kw):
        pass


def test_append_params_stored_in_index_table():
    """Param values end up as columns in the PyArrow table passed to append_partition."""
    lse = _FakeLSEWithParams({"temp": {"dtype": "float"}, "scan_id": {"dtype": "integer"}})
    embeddings = np.zeros((2, 4), dtype=np.float32)
    thumbnails = np.zeros((2, 8, 8), dtype=np.float32)

    LatentSpaceEmbedding.append(
        lse, embeddings, thumbnails,
        paths=["a", "b"],
        params={"temp": [25.1, 25.3], "scan_id": [1, 2]},
    )

    assert len(lse._appended) == 1
    table = lse._appended[0]
    assert "param_temp" in table.schema.names
    assert "param_scan_id" in table.schema.names
    assert table.column("param_temp").to_pylist() == [25.1, 25.3]
    assert table.column("param_scan_id").to_pylist() == [1, 2]


def test_append_missing_param_stored_as_null():
    """A declared param absent from the params dict is stored as all-None."""
    lse = _FakeLSEWithParams({"temp": {"dtype": "float"}, "sample": {"dtype": "string"}})
    embeddings = np.zeros((2, 4), dtype=np.float32)
    thumbnails = np.zeros((2, 8, 8), dtype=np.float32)

    # Only supply "temp", omit "sample"
    LatentSpaceEmbedding.append(
        lse, embeddings, thumbnails,
        paths=["a", "b"],
        params={"temp": [25.0, 26.0]},
    )

    table = lse._appended[0]
    assert table.column("param_sample").to_pylist() == [None, None]


def test_append_unknown_param_raises():
    """Supplying a param not in param_specs raises ValueError."""
    lse = _FakeLSEWithParams({"temp": {"dtype": "float"}})
    embeddings = np.zeros((2, 4), dtype=np.float32)
    thumbnails = np.zeros((2, 8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="Unknown parameter"):
        LatentSpaceEmbedding.append(
            lse, embeddings, thumbnails,
            paths=["a", "b"],
            params={"temp": [25.0, 26.0], "typo": [1, 2]},
        )


def test_append_param_wrong_length_raises():
    """A param list with wrong length raises ValueError."""
    lse = _FakeLSEWithParams({"temp": {"dtype": "float"}})
    embeddings = np.zeros((3, 4), dtype=np.float32)
    thumbnails = np.zeros((3, 8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="expected 3 values"):
        LatentSpaceEmbedding.append(
            lse, embeddings, thumbnails,
            paths=["a", "b", "c"],
            params={"temp": [25.0, 26.0]},  # only 2, batch_size=3
        )


def test_append_no_params_ok_when_none_declared():
    """append() with no params works fine on containers with no param_specs."""
    lse = _FakeLSEWithParams({})
    embeddings = np.zeros((2, 4), dtype=np.float32)
    thumbnails = np.zeros((2, 8, 8), dtype=np.float32)

    LatentSpaceEmbedding.append(
        lse, embeddings, thumbnails, paths=["a", "b"],
    )

    table = lse._appended[0]
    assert not any(n.startswith("param_") for n in table.schema.names)


# ---------------------------------------------------------------------------
# write_output — params forwarded correctly
# ---------------------------------------------------------------------------

@patch("emblase.tiled.client.create_embedding_container")
@patch("emblase.tiled.client.LatentSpaceEmbedding")
def test_write_output_passes_params_to_append(mock_lse_cls, mock_create):
    """params dict is forwarded to container.append()."""
    embeddings = np.zeros((2, 4), dtype=np.float32)
    container = MagicMock()
    container.metadata = {"embedding_dim": 4, "thumb_shape": [64, 64]}
    mock_create.return_value = container

    root = MagicMock()
    root.__getitem__ = MagicMock(side_effect=KeyError)

    params = {"temperature": [25.1, 25.3], "scan_id": [1, 2]}
    write_output(root, "results", embeddings, params=params)

    call = container.append.call_args
    assert call.kwargs["params"] == params


@patch("emblase.tiled.client.create_embedding_container")
@patch("emblase.tiled.client.LatentSpaceEmbedding")
def test_write_output_infers_param_specs_when_not_given(mock_lse_cls, mock_create):
    """When param_specs is omitted, specs are inferred and passed to create_embedding_container."""
    embeddings = np.zeros((2, 4), dtype=np.float32)
    container = MagicMock()
    container.metadata = {"embedding_dim": 4, "thumb_shape": [64, 64]}
    mock_create.return_value = container

    root = MagicMock()
    root.__getitem__ = MagicMock(side_effect=KeyError)

    write_output(root, "results", embeddings,
                 params={"temp": [25.0, 26.0], "scan_id": [1, 2]})

    _, kwargs = mock_create.call_args
    specs = kwargs["params"]
    assert specs["temp"]["dtype"] == "float"
    assert specs["scan_id"]["dtype"] == "integer"


@patch("emblase.tiled.client.create_embedding_container")
@patch("emblase.tiled.client.LatentSpaceEmbedding")
def test_write_output_uses_explicit_param_specs(mock_lse_cls, mock_create):
    """Explicit param_specs override inference and are passed to create_embedding_container."""
    embeddings = np.zeros((2, 4), dtype=np.float32)
    container = MagicMock()
    container.metadata = {"embedding_dim": 4, "thumb_shape": [64, 64]}
    mock_create.return_value = container

    root = MagicMock()
    root.__getitem__ = MagicMock(side_effect=KeyError)

    explicit_specs = {
        "temp": {"dtype": "float", "units": "°C", "display_name": "Temperature"},
    }
    write_output(root, "results", embeddings,
                 params={"temp": [25.0, 26.0]},
                 param_specs=explicit_specs)

    _, kwargs = mock_create.call_args
    assert kwargs["params"] == explicit_specs
