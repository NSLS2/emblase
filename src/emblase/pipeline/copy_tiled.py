"""Tiled deepcopy helpers.

The ``deepcopy`` function copies a Tiled node into a destination container,
mirroring its structure.  When *batch_delay* > 0, image arrays are written
in batches with a sleep between each batch, simulating a live acquisition.

``batch_size`` controls how many rows are written per Tiled PATCH call:
- ``batch_size=1`` (default) — one row at a time, maximum granularity for
  streaming consumers.
- ``batch_size=None`` or ``batch_size=0`` — write the entire array in one
  call (no incremental patching).

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

