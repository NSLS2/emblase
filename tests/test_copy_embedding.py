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

    # Build _index table
    indx = list(range(n))
    if shuffle_indx:
        indx = list(reversed(indx))  # simulate out-of-order storage

    schema = _make_index_schema(param_specs)
    table_data = {
        "indx": indx,
        "path": [f"scan/{i}" for i in range(n)],
        "slice": [str(i) for i in range(n)],
        "label": labels if labels is not None else [None] * n,
        "model_version": ["v1"] * n,
        "mlflow_run_id": [""] * n,
        "timestamp": [float(i) for i in range(n)],
    }
    for name in sorted(param_specs):
        table_data[f"param_{name}"] = [float(i) for i in range(n)]
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
    """Arrays are reordered by indx so row 0 of the destination matches indx=0 in the source."""
    n = 4
    src, embeddings, thumbnails, projections = _make_src(n=n, dim=4, shuffle_indx=True)

    appended_embeddings = []

    def capture_append(emb, thu, **kwargs):
        appended_embeddings.append(emb.copy())
        return len(appended_embeddings)

    dst = MagicMock()
    dst.append = MagicMock(side_effect=capture_append)
    mock_create.return_value = dst

    copy_embedding(src, MagicMock(), batch_size=n)

    # With shuffle_indx=True, indx column is [3,2,1,0] for storage rows [0,1,2,3].
    # After sort_values("indx"), the order becomes: storage row3 (indx=0), row2 (indx=1), ...
    # order = [0,1,2,3] (the sorted indx values).
    # embeddings[order] = embeddings[[0,1,2,3]] — so destination row 0 = embeddings[0].
    # This correctly places the embedding with indx=0 at destination row 0.
    result = appended_embeddings[0]
    assert result.shape == (n, 4)
    np.testing.assert_array_equal(result[0], embeddings[0])  # indx=0 → array row 0
    np.testing.assert_array_equal(result[1], embeddings[1])  # indx=1 → array row 1


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
