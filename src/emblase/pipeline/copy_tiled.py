"""Tiled copy helpers.

The ``deepcopy`` function copies a Tiled node into a destination container,
mirroring its structure.  When *batch_delay* > 0, image arrays are written
in batches with a sleep between each batch, simulating a live acquisition.

``batch_size`` controls how many rows are written per Tiled PATCH call:
- ``batch_size=1`` (default) — one row at a time, maximum granularity for
  streaming consumers.  **Required for the streaming pipeline** — see below.
- ``batch_size=None`` or ``batch_size=0`` — write the entire array in one
  call (no incremental patching).

.. warning::
    **Do not use** ``batch_size=None`` with the streaming pipeline.  When the
    entire array is written in one call, Tiled stores it using its own internal
    spatial chunking (e.g. ``(294, 294, 31)`` along H).  The WS subscription
    replays one event *per physical chunk*, so ``event.data()`` contains a
    partial spatial tile (e.g. ``(288, 1, 31, 5)``) rather than a full frame.
    The Orion job will receive malformed frames and fail.  Always use
    ``batch_size=1`` when copying data for streaming inference.

The stop document is always written *after* all frames, so a streaming
Orion job subscribing with ``start=1`` will see each frame arrive and will
only flush + exit once the stop doc lands.
"""

from __future__ import annotations

import time
from tiled.structures.core import StructureFamily
from tiled.client.container import Container


def _walk_readables(node):
    """Walk node and all its descendants in depth-first order.

    Yields relative paths (lists of keys) from *node* to each readable,
    including the node itself as the empty path ``[]``.
    """
    base = node.base if any(spec.name == "composite" for spec in node.specs) else node
    if node.structure_family == StructureFamily.container:
        yield []  # the node itself
        for k, v in base.items():
            for sub_path in _walk_readables(v):
                yield [k] + sub_path
    else:
        yield []


def deepcopy(
    src,
    dst,
    rename: str | None = None,
    access_tags=None,
    batch_size: int | None = 1,
    batch_delay: float = 0.0,
    image_key: str = "pil900KW_image",
):
    """Copy src into dst, cp-style.

    Creates a container named after src inside dst, then recursively copies
    all of src's contents into it.  Equivalent to ``cp -r src dst``.

    Parameters
    ----------
    rename:
        If given, the root container is created under this name in dst instead
        of src's own key.  All children keep their original names.
    batch_size:
        Number of rows written per PATCH call for the image array.
        ``None`` or ``0`` writes the entire array in one shot (no incremental
        patching).  Default is 1 (one frame at a time).
    batch_delay:
        Seconds to sleep between consecutive batches of the image array
        identified by *image_key*.  0 means copy as fast as possible.
    image_key:
        Name of the array that should be written incrementally.
        All other arrays are copied in one shot.
    """
    src = src.new_variation(structure_clients="numpy")
    src_key = rename or src.item["id"]

    for path in _walk_readables(src):
        s = Container.__getitem__(src, tuple(path)) if path else src
        # The destination parent is dst for the root, dst[src_key] for everything else.
        if not path:
            dst_parent = dst
        else:
            dst_parent = Container.__getitem__(dst, (src_key, *path[:-1]))
        copy_func = _registry[s.structure_family]
        key = path[-1] if path else src_key
        is_image = key == image_key and s.structure_family == StructureFamily.array
        copy_func(
            s, dst_parent, key,
            access_tags=access_tags,
            batch_size=batch_size if is_image else None,
            batch_delay=batch_delay if is_image else 0.0,
        )

    if "stop" in src.metadata:
        dst[src_key].update_metadata({"stop": src.metadata["stop"]}, drop_revision=True)


_registry = {}


def _register(structure_family: StructureFamily):
    def f(func):
        _registry[structure_family] = func
        return func

    return f


@_register(StructureFamily.array)
def copy_array(
    src, dst, key,
    access_tags=None,
    batch_size: int | None = 1,
    batch_delay: float = 0.0,
):
    n_rows = src.shape[0]
    # batch_size=None/0 → write entire array at once
    if not batch_size:
        dst.write_array(
            src.read(),
            key=key,
            metadata=dict(src.metadata),
            specs=src.specs,
            access_tags=access_tags or src.access_blob.get("tags", None),
        )
        return

    # Incremental: write first batch, then patch the rest batch_size rows at a time
    bs = batch_size
    arr = dst.write_array(
        src[0:bs, ...],
        key=key,
        metadata=dict(src.metadata),
        specs=src.specs,
        access_tags=access_tags or src.access_blob.get("tags", None),
    )
    row = bs
    while row < n_rows:
        if batch_delay > 0:
            time.sleep(batch_delay)
        end = min(row + bs, n_rows)
        arr.patch(src[row:end, ...], offset=row, extend=True)
        row = end


@_register(StructureFamily.container)
def copy_container(src, dst, key, access_tags=None, batch_size=None, batch_delay: float = 0.0):
    dst.create_container(
        key=key,
        metadata={k: v for k, v in src.metadata.items() if k != "stop"},
        specs=src.specs,
        access_tags=access_tags or src.access_blob.get("tags", None),
    )


@_register(StructureFamily.table)
def copy_table(src, dst, key, access_tags=None, batch_size=None, batch_delay: float = 0.0):
    if key in dst.base:
        dst.base.delete_contents(key, external_only=False, recursive=True)
    dst.write_table(
        src.read(),
        key=key,
        metadata=dict(src.metadata),
        specs=src.specs,
        access_tags=access_tags or src.access_blob.get("tags", None),
    )


# ---------------------------------------------------------------------------
# LatentSpaceEmbedding copy
# ---------------------------------------------------------------------------

def copy_embedding(
    src,
    dst_parent,
    rename: str | None = None,
    batch_size: int = 16,
    batch_delay: float = 0.0,
    access_tags: list[str] | None = None,
) -> "LatentSpaceEmbedding":
    """Copy a ``LatentSpaceEmbedding`` container into *dst_parent*, batch by batch.

    *src* and *dst_parent* may point to different Tiled servers — the function
    only calls the public Python API on each client.

    All arrays (embeddings, thumbnails, projections) and the ``_index`` table
    are read from *src* in one shot.  The ``_index`` table is sorted by ``indx``
    (the stable join key that equals the row offset in the arrays), then the
    arrays are reordered accordingly.  This ensures row *i* of the destination
    container always corresponds to ``indx=i``, even if the source wrote rows
    in a non-sequential order.

    ``notes`` and ``user_labels`` are intentionally skipped — they are
    user-edited fields that do not exist during a live acquisition.
    ``label`` (classifier-assigned) is preserved if present.

    Parameters
    ----------
    src:
        A ``LatentSpaceEmbedding`` client (source).
    dst_parent:
        The destination parent Tiled container (may be on a different server).
    rename:
        Base name for the new container key.  A Unix timestamp is appended
        automatically (``<rename>_<ts>``).  Defaults to the source key.
    batch_size:
        Number of embeddings written per ``append()`` call.  Default 16.
    batch_delay:
        Seconds to sleep between batches.  0 means copy as fast as possible.
    access_tags:
        Tiled access tags applied to the new container and its children.

    Returns
    -------
    LatentSpaceEmbedding
        Client for the newly created destination container.
    """
    import numpy as np
    from ..tiled.client import LatentSpaceEmbedding, create_embedding_container

    if not isinstance(src, LatentSpaceEmbedding):
        raise TypeError(
            f"src must be a LatentSpaceEmbedding, got {type(src).__name__!r}"
        )

    src_meta = src.metadata
    param_specs = src_meta.get("param_specs") or {}
    embedding_dim = src_meta.get("embedding_dim", 512)
    thumb_shape = tuple(src_meta.get("thumb_shape", [64, 64]))
    model_name = src_meta.get("model_name", "")
    model_version = src_meta.get("model_version", "")

    base_name = rename or src.item["id"]
    dst_key = f"{base_name}_{int(time.time())}"

    # Read all source data up front
    embeddings = src["embeddings"].read()    # (N, D)
    thumbnails = src["thumbnails"].read()    # (N, *thumb_shape)
    projections = src["projections"].read()  # (N, P)
    idx_df = src.base["_index"].read()

    # Sort _index by indx (= row offset into the Zarr arrays) so that
    # idx_df.iloc[i] always corresponds to embeddings[i].
    # The SQL table has no guaranteed row order, so this sort is required.
    # Older containers without an 'indx' column fall back to 'slice'.
    if "indx" in idx_df.columns:
        idx_df = idx_df.sort_values("indx", key=lambda s: s.astype(int)).reset_index(drop=True)
        # After sorting by indx, idx_df.iloc[i] has indx==i, which is exactly
        # the row offset into the arrays — no array reordering needed.
    else:
        # Legacy containers: 'slice' holds the per-source frame index, not the
        # array offset, so we can only sort by it as a best-effort ordering.
        idx_df = idx_df.sort_values("slice", key=lambda s: s.astype(int)).reset_index(drop=True)

    n_total = len(embeddings)
    param_names = list(param_specs.keys())

    dst = create_embedding_container(
        dst_parent,
        dst_key,
        embedding_dim=embedding_dim,
        thumb_shape=thumb_shape,
        model_name=model_name,
        model_version=model_version,
        params=param_specs,
        access_tags=access_tags,
    )

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        rows = idx_df.iloc[start:end]

        params = None
        if param_names:
            params = {
                name: rows[f"param_{name}"].tolist()
                for name in param_names
                if f"param_{name}" in rows.columns
            }

        labels = None
        if "label" in rows.columns:
            raw = rows["label"].tolist()
            # Only pass labels if at least one is non-null
            if any(v is not None and str(v) != "" for v in raw):
                labels = [v if (v is not None and str(v) != "") else "" for v in raw]

        dst.append(
            embeddings[start:end].astype(np.float32),
            thumbnails[start:end].astype(np.float32),
            paths=rows["path"].tolist(),
            slices=rows["slice"].tolist(),
            model_version=model_version or None,
            timestamps=rows["timestamp"].tolist(),
            projections=projections[start:end].astype(np.float32),
            params=params,
            labels=labels,
            access_tags=access_tags,
        )

        if batch_delay > 0 and end < n_total:
            time.sleep(batch_delay)

    return dst

