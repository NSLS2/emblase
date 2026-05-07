"""Tests for copy_embedding (pipeline/copy_tiled.py)."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import pyarrow as pa

from emblase.pipeline.copy_tiled import copy_embedding
from emblase.tiled.client import LatentSpaceEmbedding, _make_index_schema


def _make_src(
    n: int = 4,
    dim: int = 8,
    thumb_shape=(4, 4),
    param_specs=None,
    labels=None,
    shuffle_indx=False,
):
    """Build a minimal mock LatentSpaceEmbedding source."""
    param_specs = param_specs or {}

    embeddings = np.arange(n * dim, dtype=np.float32).reshape(n, dim)
    thumbnails = np.zeros((n, *thumb_shape), dtype=np.float32)
    projections = np.column_stack([np.arange(n, dtype=np.float32),
                                   np.arange(n, dtype=np.float32)])

    # Build _index table.
    # indx[i] == i always (indx IS the array row offset).
    # shuffle_indx=True simulates the SQL table returning rows in reverse order,
    # so idx_df row 0 has indx=n-1, row 1 has indx=n-2, etc.
    sql_row_order = list(reversed(range(n))) if shuffle_indx else list(range(n))

    schema = _make_index_schema(param_specs)
    table_data = {
        "indx": sql_row_order,                              # SQL rows in shuffled order
        "path": [f"scan/{i}" for i in sql_row_order],      # path matches indx value
        "slice": [str(i) for i in sql_row_order],
        "label": ([labels[i] for i in sql_row_order] if labels is not None
                  else [None] * n),
        "model_version": ["v1"] * n,
        "mlflow_run_id": [""] * n,
        "timestamp": [float(i) for i in sql_row_order],
    }
    for name in sorted(param_specs):
        table_data[f"param_{name}"] = [float(i) for i in sql_row_order]
    index_table = pa.table(table_data, schema=schema)

    src = MagicMock(spec=LatentSpaceEmbedding)
    src.metadata = {
        "embedding_dim": dim,
        "thumb_shape": list(thumb_shape),
        "model_name": "test_model",
        "model_version": "v1",
        "param_specs": param_specs,
    }
    src.item = {"id": "run_test"}
    src.__getitem__ = MagicMock(side_effect=lambda key: {
        "embeddings": MagicMock(read=lambda: embeddings),
        "thumbnails": MagicMock(read=lambda: thumbnails),
        "projections": MagicMock(read=lambda: projections),
    }[key])
    src.base = MagicMock()
    src.base.__getitem__ = MagicMock(return_value=MagicMock(read=lambda: index_table.to_pandas()))

    return src, embeddings, thumbnails, projections


@patch("emblase.tiled.client.create_embedding_container")
def test_copy_embedding_basic(mock_create):
    """copy_embedding creates a container and calls append once per batch."""
    src, embeddings, thumbnails, projections = _make_src(n=4, dim=8)

    dst = MagicMock()
    dst.append = MagicMock(return_value=4)
    mock_create.return_value = dst

    dst_parent = MagicMock()

    result = copy_embedding(src, dst_parent, batch_size=4)

    mock_create.assert_called_once()
    assert dst.append.call_count == 1


@patch("emblase.tiled.client.create_embedding_container")
def test_copy_embedding_preserves_labels(mock_create):
    """Labels from _index.label are forwarded to each append() call."""
    labels = ["cls_a", "cls_b", "cls_a", "cls_b"]
    src, _, _, _ = _make_src(n=4, dim=8, labels=labels)

    dst = MagicMock()
    dst.append = MagicMock(return_value=4)
    mock_create.return_value = dst

    copy_embedding(src, MagicMock(), batch_size=4)

    call = dst.append.call_args
    assert call.kwargs["labels"] == labels


@patch("emblase.tiled.client.create_embedding_container")
def test_copy_embedding_null_labels_passed_as_none(mock_create):
    """When all labels are null, None is passed to append()."""
    src, _, _, _ = _make_src(n=3, dim=8, labels=None)

    dst = MagicMock()
    dst.append = MagicMock(return_value=3)
    mock_create.return_value = dst

    copy_embedding(src, MagicMock(), batch_size=3)

    call = dst.append.call_args
    assert call.kwargs["labels"] is None


@patch("emblase.tiled.client.create_embedding_container")
def test_copy_embedding_reorders_by_indx(mock_create):
    """_index metadata rows are aligned with the correct array rows after indx sort.

    The SQL table has no guaranteed row order.  copy_embedding must sort by
    'indx' before batching so that idx_df.iloc[i] (metadata) matches
    embeddings[i] (array row), regardless of the SQL return order.
    """
    n = 4
    src, embeddings, thumbnails, projections = _make_src(n=n, dim=4, shuffle_indx=True)

    captured = []

    def capture_append(emb, thu, **kwargs):
        captured.append({"embeddings": emb.copy(), "paths": kwargs["paths"]})
        return len(captured)

    dst = MagicMock()
    dst.append = MagicMock(side_effect=capture_append)
    mock_create.return_value = dst

    copy_embedding(src, MagicMock(), batch_size=n)

    result = captured[0]
    # After sort by indx, idx_df row 0 has indx=0, row 1 has indx=1, etc.
    # Since indx == array row offset, embeddings[indx] is the correct embedding.
    # So destination row i should equal embeddings[i] regardless of SQL order.
    for i in range(n):
        np.testing.assert_array_equal(result["embeddings"][i], embeddings[i])
    # Paths should also be in indx order (path "scan/0" → indx=0, etc.)
    assert result["paths"] == [f"scan/{i}" for i in range(n)]


@patch("emblase.tiled.client.create_embedding_container")
def test_copy_embedding_multi_batch(mock_create):
    """Multiple batches are written when n_total > batch_size."""
    src, _, _, _ = _make_src(n=6, dim=4)

    dst = MagicMock()
    dst.append = MagicMock(return_value=0)
    mock_create.return_value = dst

    copy_embedding(src, MagicMock(), batch_size=2)

    assert dst.append.call_count == 3


def test_copy_embedding_raises_on_wrong_type():
    """Passing a non-LatentSpaceEmbedding as src raises TypeError."""
    with pytest.raises(TypeError, match="LatentSpaceEmbedding"):
        copy_embedding(MagicMock(), MagicMock())


@patch("emblase.tiled.client.create_embedding_container")
def test_copy_embedding_pandas_na_in_slice_and_label(mock_create):
    """pd.NA values in 'slice' and 'label' columns must not raise ArrowTypeError.

    Tiled returns nullable-string columns where missing values are pd.NA, not
    Python None.  PyArrow rejects pd.NA for string fields → the copy must
    normalise these to None before building the Arrow table.
    """
    n = 3
    src, embeddings, thumbnails, projections = _make_src(n=n, dim=4)

    # Patch the _index table to include pd.NA in slice and label
    df = src.base["_index"].read()
    df = df.copy()
    df["slice"] = pd.array([pd.NA, "1", pd.NA], dtype=pd.StringDtype())
    df["label"] = pd.array([pd.NA, pd.NA, pd.NA], dtype=pd.StringDtype())
    src.base.__getitem__ = MagicMock(return_value=MagicMock(read=lambda: df))

    captured = []

    def capture_append(emb, thu, **kwargs):
        captured.append(kwargs)
        return len(captured)

    dst = MagicMock()
    dst.append = MagicMock(side_effect=capture_append)
    mock_create.return_value = dst

    # Should not raise ArrowTypeError
    copy_embedding(src, MagicMock(), batch_size=n)

    assert dst.append.call_count == 1
    call = captured[0]
    # pd.NA slice values must become None
    assert call["slices"] == [None, "1", None]
    # All-null label column → labels=None passed to append
    assert call["labels"] is None
