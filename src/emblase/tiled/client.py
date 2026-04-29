import time
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import pyarrow as pa
from tiled.client.composite import CompositeClient
from tiled.client.container import Container
from tiled.ndslice import NDSlice
from tiled.structures.core import Spec, StructureFamily

# ---------------------------------------------------------------------------
# I/O helpers — public API for reading source images and writing embeddings
# ---------------------------------------------------------------------------

# Type alias: a tiled entry is either a bare path string or a (path, slice) tuple.
TiledEntry = str | tuple[str, str]

_DEFAULT_THUMB_SHAPE: tuple[int, int] = (64, 64)


def _default_thumb_fn(frames: np.ndarray) -> np.ndarray:
    """Resize (N, H, W) frames to _DEFAULT_THUMB_SHAPE using nearest-neighbour."""
    n, h, w = frames.shape
    th, tw = _DEFAULT_THUMB_SHAPE
    row_idx = (np.arange(th) * h / th).astype(int)
    col_idx = (np.arange(tw) * w / tw).astype(int)
    return frames[:, row_idx[:, None], col_idx[None, :]].astype(np.float32)


def read_images(
    client,
    entries: Sequence[TiledEntry],
) -> tuple[list[np.ndarray], list[TiledEntry]]:
    """Read tiled nodes and return frames with per-frame provenance.

    Parameters
    ----------
    client:
        An already-initialised tiled client pointing at the catalog root
        (e.g. ``tiled.client.from_uri(url, api_key=key)``).
    entries:
        Each item is either:
        - a ``str`` tiled path  →  ``client[path].read()``
        - a ``(path, slice)`` tuple  →  ``client[path].read(NDSlice)``

        The tiled client resolves slash-separated paths natively.

    Each node may be:
    - A 2-D array ``(H, W)``       → one frame
    - An N-D array ``(..., H, W)`` → all frames (product of leading dims)

    Returns
    -------
    frames : list[np.ndarray]
        Flat list of 2-D float32 arrays, one per frame across all entries.
    frame_entries : list[TiledEntry]
        Parallel list of the source entry for each frame — same length as
        ``frames``. Pass directly to ``write_output`` as ``source_entries``.
    """
    frames: list[np.ndarray] = []
    frame_entries: list[TiledEntry] = []
    for entry in entries:
        if isinstance(entry, str):
            path, slc = entry, None
        else:
            path, slc = entry

        slc = NDSlice.from_numpy_str(slc) if slc is not None else None
        arr = client[path].read(slc)

        if arr.ndim < 2:
            raise ValueError(
                f"Array at {path!r} has fewer than 2 dimensions: {arr.shape}"
            )
        for frame in arr.reshape(-1, arr.shape[-2], arr.shape[-1]):
            frames.append(frame)
            frame_entries.append(entry)
    return frames, frame_entries


_embedding_container_cache: dict[tuple, "LatentSpaceEmbedding"] = {}


def write_output(
    client,
    path: str,
    embeddings: np.ndarray,
    images: Optional[list[np.ndarray]] = None,
    *,
    source_entries: Optional[Sequence[TiledEntry]] = None,
    thumb_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    thumb_shape: tuple[int, int] = _DEFAULT_THUMB_SHAPE,
    model_name: str = "",
    model_version: str = "",
    embedding_dim: Optional[int] = None,
    metadata: Optional[dict] = None,
    access_tags: Optional[list[str]] = None,
) -> None:
    """Write embeddings into a LatentSpaceEmbedding container at ``path``.

    Creates the container if it does not exist.

    Parameters
    ----------
    client:
        An already-initialised tiled client pointing at the catalog root.
    path:
        Slash-separated path to the target LatentSpaceEmbedding container.
        The final component is the key; everything before it is the parent path.
    embeddings:
        Shape ``(N, D)`` float32 array of embedding vectors.
    images:
        List of N 2-D float32 numpy arrays (one per embedding) used to generate
        thumbnails. If ``None``, zero arrays are stored.
    source_entries:
        Original tiled entries used as input — stored as provenance in the
        ``_index`` table. Length must equal ``len(embeddings)`` if provided.
    thumb_fn:
        ``(frames: ndarray (N,H,W)) -> ndarray (N,th,tw)``.
        Defaults to nearest-neighbour resize to ``thumb_shape``.
    thumb_shape:
        ``(height, width)`` passed to the default ``thumb_fn``.
    model_name, model_version:
        Stored in container metadata on creation.
    embedding_dim:
        Inferred from ``embeddings`` if omitted.
    metadata:
        Extra key/value pairs merged into container metadata on creation.
    access_tags:
        Tiled access tags applied to every node written (container, arrays,
        index table).  When ``None`` the server's default policy applies.
        Typically sourced from ``settings.tiled_access_tags``.
    """
    n = len(embeddings)
    d = embedding_dim or embeddings.shape[1]

    # Resolve parent container and leaf key from slash-separated path.
    parts = path.rstrip("/").split("/")
    key = parts[-1]
    parent_path = "/".join(parts[:-1])

    # Cache the LatentSpaceEmbedding client object across calls so that the
    # local state (_num_embeddings, _arrays_initialised, _index_table) is
    # preserved and we avoid redundant GETs on every append.
    base_url = str(client.context.base_url)
    _cache_key = (base_url, path)
    container = _embedding_container_cache.get(_cache_key)
    if container is None:
        parent = client[parent_path] if parent_path else client
        # Get or create the LatentSpaceEmbedding container.
        try:
            container = parent[key]
            if not isinstance(container, LatentSpaceEmbedding):
                raise TypeError(
                    f"Node at {path!r} exists but is not a LatentSpaceEmbedding "
                    f"(got {type(container).__name__})."
                )
        except KeyError:
            container = create_embedding_container(
                parent,
                key,
                embedding_dim=d,
                thumb_shape=thumb_shape,
                model_name=model_name,
                model_version=model_version,
                metadata=metadata,
                access_tags=access_tags,
            )
        _embedding_container_cache[_cache_key] = container

    # Build thumbnails — apply thumb_fn per image (handles ragged shapes).
    if images is not None:
        _fn = thumb_fn or _default_thumb_fn
        thumbs_list = [_fn(img[np.newaxis].astype(np.float32))[0] for img in images]
        thumbnails = np.stack(thumbs_list, axis=0)
    else:
        thumbnails = np.zeros((n, *thumb_shape), dtype=np.float32)

    # Unpack provenance.
    if source_entries:
        paths = [e if isinstance(e, str) else e[0] for e in source_entries]
        slices = [None if isinstance(e, str) else e[1] for e in source_entries]
    else:
        paths = [""] * n
        slices = [None] * n

    container.append(
        embeddings.astype(np.float32),
        thumbnails,
        paths=paths,
        slices=slices,
        model_version=model_version or None,
        timestamps=list(time.time() + np.arange(n) * 1e-6),
        access_tags=access_tags,
    )

# Maximum character widths for zarr string arrays
NOTES_MAX_LEN = 1024
USER_LABEL_MAX_LEN = 64

# PyArrow schema for the _index table (append-only, immutable columns)
INDEX_SCHEMA = pa.schema(
    [
        pa.field("path", pa.string()),
        pa.field("slice", pa.string(), nullable=True),
        pa.field("label", pa.string(), nullable=True),
        pa.field("model_version", pa.string(), nullable=True),
        pa.field("mlflow_run_id", pa.string(), nullable=True),
        pa.field("timestamp", pa.float64()),
    ]
)

REQUIRED_METADATA_KEYS = set()


def _make_string_array(values: list[str], max_len: int) -> np.ndarray:
    """Create a fixed-width unicode numpy array, truncating if needed."""
    return np.array(values, dtype=f"<U{max_len}")


def create_embedding_container(
    parent: Container,
    key: str,
    *,
    embedding_dim: int,
    thumb_shape: tuple[int, ...],
    projection_dim: int = 2,
    model_name: str = "",
    model_version: str = "",
    mlflow_model_uri: str = "",
    description: str = "",
    metadata: Optional[dict[str, Any]] = None,
    access_tags: Optional[list[str]] = None,
) -> "LatentSpaceEmbedding":
    """Create a new LatentSpaceEmbedding container with the required internal structure.

    Parameters
    ----------
    parent : Container
        The parent Tiled container (e.g. the catalog root or a sub-container).
    key : str
        Name/key for this embedding container.
    embedding_dim : int
        Dimensionality D of the embedding vectors.
    thumb_shape : tuple[int, ...]
        Shape of each thumbnail, e.g. (64, 64) for grayscale or (64, 64, 3) for RGB.
    projection_dim : int
        Dimensionality of the visualization projections (2 or 3). Default 2.
    model_name : str
        Name of the ML model producing embeddings.
    model_version : str
        Version of the ML model.
    mlflow_model_uri : str
        MLFlow model URI (e.g. ``models:/my_encoder/3``).
    description : str
        Human-readable description of this embedding collection.
    metadata : dict, optional
        Additional user metadata merged into the container metadata.
        Use this for experiment-specific context (beamline, sample info, etc.).

    Returns
    -------
    LatentSpaceEmbedding
        The client for the newly created container.
    """
    container_metadata = {
        "embedding_dim": embedding_dim,
        "thumb_shape": list(thumb_shape),
        "projection_dim": projection_dim,
        "model_name": model_name,
        "model_version": model_version,
        "mlflow_model_uri": mlflow_model_uri,
        "created_at": time.time(),
        "description": description,
    }
    if metadata:
        container_metadata.update(metadata)

    node = parent.create_container(
        key=key,
        metadata=container_metadata,
        specs=["LatentSpaceEmbedding", "composite"],
        access_tags=access_tags,
    )

    # create_container returns the node via spec dispatch — it should already
    # be a LatentSpaceEmbedding (CompositeClient subclass).  Return it directly
    # to avoid re-fetching the container from the server (which would cost an
    # extra GET + a second __init__ that loses the eagerly-set caches).
    if isinstance(node, LatentSpaceEmbedding):
        return node

    # Fallback: server did not dispatch to LatentSpaceEmbedding (e.g. spec
    # validator not registered).  Re-fetch so callers always get the right type.
    return parent[key]


class LatentSpaceEmbedding(CompositeClient):
    """Composite client for latent (feature) space embedding containers.

    Data layout (children of the composite node):

    Zarr arrays (mutable via patch):
        embeddings   : (N, D) float32 — embedding vectors
        thumbnails   : (N, *thumb_shape) — downsampled source images
        projections  : (N, P) float32 — 2D/3D visualization vectors
        notes        : (N,) <U1024 — freeform mutable annotations
        user_labels  : (N,) <U64 — mutable user-assigned labels

    SQL table (append-only, immutable after write):
        _index : [path, slice, label, model_version, mlflow_run_id, timestamp]
            ``label`` is the immutable model-assigned label.
            Rows ordered by timestamp on read.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Use the structure.count already present in the fetched item to decide
        # whether child arrays exist, avoiding a separate search GET.
        # count=0 means no children → arrays not yet initialised.
        # count>0 means children exist → treat as initialised (True).
        # Fall back to None (lazy check) if count is unavailable.
        _structure = self.item.get("attributes", {}).get("structure", {})
        _count = _structure.get("count") if isinstance(_structure, dict) else getattr(_structure, "count", None)
        if _count == 0:
            self._arrays_initialised: bool | None = False
        elif _count is not None and _count > 0:
            self._arrays_initialised = True
        else:
            self._arrays_initialised = None

        # Local embedding count cache.  None = unknown (resolved lazily from
        # the _index table on first access).  Set to 0 immediately on fresh
        # containers so we never need a remote read just to find the offset.
        self._num_embeddings: int | None = None

        # Create the _index table if not exists (appendable, SQL-backed, immutable after write)
        try:
            self._index_table = self.base["_index"]
        except KeyError:
            self._index_table = self.create_appendable_table(
                INDEX_SCHEMA,
                key="_index",
                metadata={"description": "Per-embedding metadata index (append-only)"},
                access_tags=self.access_blob.get("tags", None)
            )
            # We just created both the table and the container — nothing has
            # been written yet.  Initialise the caches eagerly so the first
            # append() doesn't need any extra GET requests to discover these
            # facts remotely.
            self._arrays_initialised = False
            self._num_embeddings = 0

    def __repr__(self) -> str:
        n = self.num_embeddings
        dim = self.metadata.get("embedding_dim", "?")
        model = self.metadata.get("model_name", "")
        parts = [f"name='{self.item['id']}'", f"n={n}", f"dim={dim}"]
        if model:
            parts.append(f"model='{model}'")
        return f"LatentSpaceEmbedding({', '.join(parts)})"

    @property
    def embedding_dim(self) -> int:
        return self.metadata["embedding_dim"]

    @property
    def num_embeddings(self) -> int:
        """Return the number of embeddings currently stored.

        Uses a local counter when available to avoid a remote table read.
        The counter is populated eagerly on fresh containers and updated on
        every ``append()`` call.  On containers that already existed when this
        client object was created the counter starts as ``None`` and is
        populated from the remote _index table on first access.
        """
        if self._num_embeddings is None:
            self._num_embeddings = len(self._index_table.read(columns=["timestamp"]))
        return self._num_embeddings

    def _write_arrays(self, embeddings, thumbnails, projections=None, offset=0, access_tags=None):
        batch_size = len(embeddings)
        empty_notes = _make_string_array([""] * batch_size, NOTES_MAX_LEN)
        empty_user_labels = _make_string_array([""] * batch_size, USER_LABEL_MAX_LEN)

        # Resolve effective tags: explicit argument takes priority over the
        # tags already stored in the container's access_blob.
        tags = access_tags if access_tags is not None else self.access_blob.get("tags", None)

        # Check once whether arrays have been initialised; cache the result to
        # avoid a Tiled listing request on every subsequent append call.
        if self._arrays_initialised is None:
            self._arrays_initialised = "embeddings" in self

        # Arrays: create on first insert, extend on subsequent.
        if not self._arrays_initialised:
            self._arr_embeddings = self.write_array(
                embeddings.astype(np.float32),
                key="embeddings",
                metadata={"description": "Embedding vectors"},
                dims=["sample", "feature"],
                access_tags=tags,
            )
            self._arr_thumbnails = self.write_array(
                thumbnails,
                key="thumbnails",
                metadata={"description": "Thumbnail images"},
                access_tags=tags,
            )
            self._arr_notes = self.write_array(
                empty_notes,
                key="notes",
                metadata={"description": "Freeform mutable annotations"},
                access_tags=tags,
            )
            self._arr_user_labels = self.write_array(
                empty_user_labels,
                key="user_labels",
                metadata={"description": "Mutable user-assigned labels"},
                access_tags=tags,
            )
            if projections is not None:
                self.write_array(
                    projections.astype(np.float32),
                    key="projections",
                    metadata={"description": "Visualization projections"},
                    access_tags=tags,
                )
            self._arrays_initialised = True
        else:
            # Use cached array clients when available to avoid GET /metadata/…/key
            # on each patch call (self[key] would re-fetch from the server).
            emb = getattr(self, "_arr_embeddings", None) or self["embeddings"]
            thu = getattr(self, "_arr_thumbnails", None) or self["thumbnails"]
            nts = getattr(self, "_arr_notes", None) or self["notes"]
            ulb = getattr(self, "_arr_user_labels", None) or self["user_labels"]
            emb.patch(embeddings.astype(np.float32), offset=(offset,), extend=True)
            thu.patch(thumbnails, offset=(offset,), extend=True)
            nts.patch(empty_notes, offset=(offset,), extend=True)
            ulb.patch(empty_user_labels, offset=(offset,), extend=True)
            if projections is not None:
                if "projections" in self:
                    self["projections"].patch(
                        projections.astype(np.float32),
                        offset=(offset,),
                        extend=True,
                    )
                else:
                    self.write_array(
                        projections.astype(np.float32),
                        key="projections",
                        metadata={"description": "Visualization projections"},
                        access_tags=tags,
                    )

    def append(
        self,
        embeddings: np.ndarray,
        thumbnails: np.ndarray,
        *,
        paths: list[str],
        slices: Optional[list[str]] = None,
        labels: Optional[list[str]] = None,
        model_version: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
        timestamps: Optional[list[float]] = None,
        projections: Optional[np.ndarray] = None,
        access_tags: Optional[list[str]] = None,
    ) -> int:
        """Append one or more embeddings with their associated data.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (B, D) array of embedding vectors to append.
        thumbnails : np.ndarray
            Shape (B, *thumb_shape) array of thumbnail images.
        paths : list[str]
            Tiled paths to the original source data, one per embedding.
        slices : list[str], optional
            Slice strings for each embedding (e.g. "5", "3:10").
        labels : list[str], optional
            Immutable model-assigned labels.
        model_version : str, optional
            Version of the model that produced these embeddings.
            Defaults to the container's ``model_version`` metadata.
        mlflow_run_id : str, optional
            MLFlow run ID that produced these embeddings.
        timestamps : list[float], optional
            Epoch timestamps. Defaults to current time for each.
        projections : np.ndarray, optional
            Shape (B, P) pre-computed projection vectors.
        access_tags : list[str], optional
            Tiled access tags applied to newly created child arrays.
            Falls back to the container's own access_blob tags when ``None``.

        Returns
        -------
        int
            The new total number of embeddings after appending.
        """
        batch_size = len(embeddings)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            msg = (
                f"Expected embeddings shape (B, {self.embedding_dim}), "
                f"got {embeddings.shape}"
            )
            raise ValueError(msg)

        meta = self.metadata
        expected_thumb_shape = (batch_size, *meta.get("thumb_shape", ()))
        if thumbnails.shape != expected_thumb_shape:
            msg = (
                f"Expected thumbnails shape {expected_thumb_shape}, "
                f"got {thumbnails.shape}"
            )
            raise ValueError(msg)

        if len(paths) != batch_size:
            msg = f"Expected {batch_size} paths, got {len(paths)}"
            raise ValueError(msg)

        current_n = self.num_embeddings

        # Write arrays first, then _index table last.
        # The UI subscribes to both streams; writing the table last ensures
        # projections are committed before the table WS event arrives.
        self._write_arrays(embeddings, thumbnails, projections, offset=current_n, access_tags=access_tags)

        table = pa.table(
            {
                "path": paths,
                "slice": [None] * batch_size if slices is None else slices,
                "label": [None] * batch_size if labels is None else labels,
                "model_version": [model_version or self.metadata.get("model_version")]
                * batch_size,
                "mlflow_run_id": [
                    mlflow_run_id or self.metadata.get("mlflow_run_id", "")
                ]
                * batch_size,
                "timestamp": timestamps
                or np.array([time.time()] * batch_size) + np.arange(batch_size) * 1e-6,
            },
            schema=INDEX_SCHEMA,
        )
        self._index_table.append_partition(0, table)

        # Keep the local counter in sync so subsequent append() / num_embeddings
        # calls don't need a remote table read to find the new offset.
        self._num_embeddings = current_n + batch_size

        return self._num_embeddings

    def update_note(self, index: int, note: str) -> None:
        """Update the note for a specific embedding by array index."""
        if index == -1:
            index = self.num_embeddings - 1
        if index < 0 or index >= self.num_embeddings:
            raise IndexError(f"Index {index} out of range [0, {self.num_embeddings})")
        self["notes"].patch(_make_string_array([note], NOTES_MAX_LEN), offset=(index,))

    def update_user_label(self, index: int, label: str) -> None:
        """Update the user-assigned label for a specific embedding."""
        if index == -1:
            index = self.num_embeddings - 1
        if index < 0 or index >= self.num_embeddings:
            raise IndexError(f"Index {index} out of range [0, {self.num_embeddings})")
        self["user_labels"].patch(
            _make_string_array([label], USER_LABEL_MAX_LEN), offset=(index,)
        )

    def update_projections(self, projections: np.ndarray, access_tags: Optional[list[str]] = None) -> None:
        """Replace all projection vectors (e.g. after re-running UMAP/t-SNE).

        Parameters
        ----------
        projections : np.ndarray
            Shape (N, P) array of projection vectors.
        access_tags : list[str], optional
            Tiled access tags applied when the projections array is created for
            the first time. Falls back to the container's own access_blob tags.
        """
        n = self.num_embeddings
        if projections.shape[0] != n:
            raise ValueError(f"Expected {n} projections, got {projections.shape[0]}")

        tags = access_tags if access_tags is not None else self.access_blob.get("tags", None)
        data = projections.astype(np.float32)
        if "projections" not in self:
            self.write_array(
                data,
                key="projections",
                metadata={"description": "Visualization projections"},
                access_tags=tags,
            )
        elif self["projections"].shape == data.shape:
            self["projections"].patch(data, offset=(0,))
        else:
            self.delete_contents(["projections"], external_only=False)
            self.write_array(
                data,
                key="projections",
                metadata={"description": "Visualization projections"},
                access_tags=tags,
            )

    def read_embeddings(self, indices=None) -> np.ndarray:
        "Read embedding vectors, optionally sliced."
        return self["embeddings"][indices]

    def read_thumbnails(self, indices=None) -> np.ndarray:
        "Read thumbnail images, optionally sliced."
        return self["thumbnails"][indices]

    def read_projections(self, indices=None) -> Optional[np.ndarray]:
        "Read projection vectors. Returns None if not yet computed"
        if "projections" not in self:
            return None
        return self["projections"][indices]

    def read_notes(self, indices=None) -> np.ndarray:
        "Read notes array, optionally sliced"
        return self["notes"][indices]

    def read_user_labels(self, indices=None) -> np.ndarray:
        "Read user labels array, optionally sliced"
        return self["user_labels"][indices]

    def read_index(self):
        "Read the _index table as a pandas DataFrame; sort rows by timestamp"
        df = self._index_table.read()
        return df.sort_values("timestamp").reset_index(drop=True)


async def validate_embedding(spec, metadata, entry, structure_family, structure):
    """Spec validator for LatentSpaceEmbedding containers.

    On creation (entry is None):
        - Verifies structure_family is 'container'
        - Verifies required metadata keys are present (embedding_dim, thumb_shape)
        - Defaults optional metadata fields (model_name, model_version, etc.)

    On update (entry exists):
        - Verifies the container has an '_index' table child
    """
    from tiled.validation_registration import ValidationError

    if entry is None:
        if structure_family != StructureFamily.container:
            raise ValidationError(
                f"LatentSpaceEmbedding spec requires structure_family 'container', "
                f"got '{structure_family}'."
            )
        if metadata is None:
            metadata = {}
        missing = REQUIRED_METADATA_KEYS - set(metadata.keys())
        if missing:
            raise ValidationError(
                f"LatentSpaceEmbedding metadata is missing required keys: {missing}"
            )
        defaults = {
            "projection_dim": 2,
            "model_name": "",
            "model_version": "",
            "mlflow_model_uri": "",
            "description": "",
        }
        changed = False
        for k, v in defaults.items():
            if k not in metadata:
                metadata[k] = v
                changed = True
        if "created_at" not in metadata:
            metadata["created_at"] = time.time()
            changed = True
        if changed:
            return metadata
    else:
        has_index = False
        async for key, _item in entry.items_range(offset=0, limit=None):
            if key == "_index":
                has_index = True
                break
        if not has_index:
            raise ValidationError(
                "LatentSpaceEmbedding container must have an '_index' table."
            )
