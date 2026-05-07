import time
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
import pyarrow as pa
from tiled.client.composite import CompositeClient
from tiled.client.container import Container
from tiled.ndslice import NDSlice
from tiled.structures.core import StructureFamily
from typing_extensions import NotRequired, TypedDict

# ---------------------------------------------------------------------------
# Parameter descriptors
# ---------------------------------------------------------------------------

ParamDtype = Literal["float", "integer", "string", "boolean"]


class ParamSpec(TypedDict):
    """Descriptor for a single continuous or categorical parameter stored in _index.

    Modelled on the Bluesky DataKey vocabulary (event-model), restricted to the
    scalar case (shape=[]) and with ``dtype`` using plain names instead of
    JSON-Schema ``"number"``/``"integer"``:

    dtype
        ``"float"``   → pa.float64()   (temperature, pressure, …)
        ``"integer"`` → pa.int64()     (scan_id, frame_number, …)
        ``"string"``  → pa.string()    (sample_name, phase, …)
        ``"boolean"`` → pa.bool_()     (flag, is_calibration, …)
    units
        Engineering units, e.g. ``"°C"``, ``"bar"``, ``"mm"``.
    source
        Where the value was extracted from, e.g. ``"start.sample.temperature"``.
        Informational — not used for data access.
    precision
        Number of decimal places to show in the UI (floats only).
    display_name
        Human-readable label for the UI.  Falls back to the param name when absent.
    """

    dtype: ParamDtype
    units: NotRequired[str]
    source: NotRequired[str]
    precision: NotRequired[int]
    display_name: NotRequired[str]


# Mapping from ParamDtype → PyArrow field type (always nullable)
_PARAM_DTYPE_TO_PA: dict[ParamDtype, pa.DataType] = {
    "float": pa.float64(),
    "integer": pa.int64(),
    "string": pa.string(),
    "boolean": pa.bool_(),
}

# ---------------------------------------------------------------------------
# I/O helpers — public API for reading source images and writing embeddings
# ---------------------------------------------------------------------------

# Type alias: a tiled entry is either a bare path string or a (path, slice) tuple.
TiledEntry = str | tuple[str, str]

_DEFAULT_THUMB_SHAPE: tuple[int, int] = (64, 64)

# ROI used by the log-normalised thumbnail function (rows, cols).
# Chosen to cover the Bragg-peak region in SMI resting-state diffraction images.
_LOG_THUMB_ROI: tuple[slice, slice] = (slice(0, 180), slice(220, 400))


def _resize_nn(frames: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbour resize (N, H, W) → (N, out_h, out_w)."""
    n, h, w = frames.shape
    row_idx = (np.arange(out_h) * h / out_h).astype(int)
    col_idx = (np.arange(out_w) * w / out_w).astype(int)
    return frames[:, row_idx[:, None], col_idx[None, :]].astype(np.float32)


def _default_thumb_fn(frames: np.ndarray) -> np.ndarray:
    """Resize (N, H, W) frames to _DEFAULT_THUMB_SHAPE using nearest-neighbour.

    No intensity normalisation — pixel values are preserved as-is (float32).
    Suitable when frames have already been preprocessed or when a raw look is
    preferred.
    """
    th, tw = _DEFAULT_THUMB_SHAPE
    return _resize_nn(frames, th, tw)


def _log_thumb_fn(
    frames: np.ndarray,
    roi: tuple[slice, slice] = _LOG_THUMB_ROI,
    out_shape: tuple[int, int] = _DEFAULT_THUMB_SHAPE,
) -> np.ndarray:
    """Crop a ROI, log-normalise, then resize to *out_shape*.

    Processing pipeline per frame
    -----------------------------
    1. Crop ``frames[:, roi[0], roi[1]]`` — isolates the region of interest.
    2. Clip negative values to 0 (dark-current / detector artefacts are tiny
       and negative; treating them as zero preserves log domain).
    3. Apply ``log1p`` — maps photon-count data (Poisson-distributed, range
       [0, ~10^6]) to a perceptually uniform [0, ~14] scale that compresses
       bright Bragg peaks without saturating background.
    4. Nearest-neighbour resize to *out_shape*.

    Parameters
    ----------
    frames:
        ``(N, H, W)`` float32 array.  H and W must be large enough to contain
        the ROI.
    roi:
        ``(row_slice, col_slice)`` defining the crop region.
        Default ``_LOG_THUMB_ROI`` = ``(slice(0,180), slice(220,400))``.
    out_shape:
        ``(height, width)`` of the output thumbnails.  Default 64×64.
    """
    cropped = frames[:, roi[0], roi[1]].astype(np.float32)
    cropped = np.clip(cropped, 0.0, None)
    log_normed = np.log1p(cropped)
    return _resize_nn(log_normed, out_shape[0], out_shape[1])


# Registry of named thumbnail functions.
# Extend this dict to add new presets; the name is what callers pass as
# ``thumb_mode`` in ``write_output``.
THUMB_MODES: dict[str, "Callable[[np.ndarray], np.ndarray]"] = {
    "default": _default_thumb_fn,
    "logroi": _log_thumb_fn,
}


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
            raise ValueError(f"Array at {path!r} has fewer than 2 dimensions: {arr.shape}")
        flat = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
        for i, frame in enumerate(flat):
            frames.append(frame)
            frame_entries.append((path, str(i)))
    return frames, frame_entries


def _infer_param_specs(params: dict[str, list]) -> dict[str, "ParamSpec"]:
    """Infer minimal ParamSpec dicts from param value lists.

    Used by ``write_output`` when ``param_specs`` is not provided explicitly.
    Looks at the first non-None value in each list to guess the dtype.
    """
    specs: dict[str, ParamSpec] = {}
    for name, values in params.items():
        # Find first non-None value to determine type
        sample = next((v for v in values if v is not None), None)
        if sample is None or isinstance(sample, float):
            dtype: ParamDtype = "float"
        elif isinstance(sample, bool):
            dtype = "boolean"
        elif isinstance(sample, int):
            dtype = "integer"
        else:
            dtype = "string"
        specs[name] = {"dtype": dtype}
    return specs


_embedding_container_cache: dict[tuple, "LatentSpaceEmbedding"] = {}


def _clear_embedding_container_cache() -> None:
    """Clear the module-level embedding container cache.

    Call this in tests that create containers with the same path across
    multiple test functions, to prevent stale container objects leaking
    between tests.
    """
    _embedding_container_cache.clear()


def write_output(
    client,
    path: str,
    embeddings: np.ndarray,
    images: Optional[list[np.ndarray]] = None,
    *,
    source_entries: Optional[Sequence[TiledEntry]] = None,
    thumb_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    thumb_mode: str = "default",
    thumb_shape: tuple[int, int] = _DEFAULT_THUMB_SHAPE,
    model_name: str = "",
    model_version: str = "",
    embedding_dim: Optional[int] = None,
    params: Optional[dict[str, list]] = None,
    param_specs: Optional[dict[str, "ParamSpec"]] = None,
    projections: Optional[np.ndarray] = None,
    labels: Optional[list[str]] = None,
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
    embeddings:
        Shape ``(N, D)`` float32 array of embedding vectors.
    images:
        List of N 2-D float32 numpy arrays used to generate thumbnails.
        If ``None``, zero arrays are stored.
    source_entries:
        Original tiled entries used as input — stored as provenance in the
        ``_index`` table. Length must equal ``len(embeddings)`` if provided.
    thumb_fn:
        Custom thumbnail function ``(N, H, W) → (N, th, tw)``.
        Takes priority over ``thumb_mode``.
    thumb_mode:
        Built-in thumbnail preset name (``"default"`` or ``"logroi"``).
    thumb_shape:
        ``(height, width)`` of thumbnails when using default resize.
    model_name, model_version:
        Stored in container metadata on creation.
    embedding_dim:
        Inferred from ``embeddings`` if omitted.
    params : dict[str, list], optional
        Per-embedding parameter values to store alongside each embedding in
        the _index table.  Keys are parameter names; values are lists of
        length N.  Must match the ``param_specs`` declared at container
        creation — or, on first write, ``param_specs`` is used to initialise
        the container schema.
    param_specs : dict[str, ParamSpec], optional
        Parameter descriptors used when *creating* the container for the first
        time.  Ignored if the container already exists.  If omitted but
        ``params`` is provided, minimal specs (``{"dtype": "float"}``) are
        inferred from the value types.
    projections : np.ndarray, optional
        Shape (N, P) pre-computed projection coordinates (e.g. from the projector
        approximator or UMAP).  When provided, stored directly instead of NaN placeholders.
    labels : list[str], optional
        Classifier-assigned label strings, one per embedding.  Stored in the
        ``label`` column of ``_index``.  When omitted, the column is NULL.
    metadata:
        Extra key/value pairs merged into container metadata on creation.
    access_tags:
        Tiled access tags applied to every node written.
    """
    n = len(embeddings)
    d = embedding_dim or embeddings.shape[1]

    parts = path.rstrip("/").split("/")
    key = parts[-1]
    parent_path = "/".join(parts[:-1])

    base_url = str(client.context.base_url)
    _cache_key = (base_url, path)
    container = _embedding_container_cache.get(_cache_key)
    if container is None:
        parent = client[parent_path] if parent_path else client
        try:
            container = parent[key]
            if not isinstance(container, LatentSpaceEmbedding):
                raise TypeError(
                    f"Node at {path!r} exists but is not a LatentSpaceEmbedding "
                    f"(got {type(container).__name__})."
                )
        except KeyError:
            # Infer param_specs from params values if not provided explicitly
            resolved_specs = param_specs or {}
            if params and not resolved_specs:
                resolved_specs = _infer_param_specs(params)
            container = create_embedding_container(
                parent,
                key,
                embedding_dim=d,
                thumb_shape=thumb_shape,
                model_name=model_name,
                model_version=model_version,
                params=resolved_specs,
                metadata=metadata,
                access_tags=access_tags,
            )
        _embedding_container_cache[_cache_key] = container

    # Build thumbnails
    if images is not None:
        if thumb_fn is None:
            if thumb_mode not in THUMB_MODES:
                raise ValueError(
                    f"Unknown thumb_mode {thumb_mode!r}. Available: {sorted(THUMB_MODES)}"
                )
            thumb_fn = THUMB_MODES[thumb_mode]
        shapes = {img.shape for img in images}
        if len(shapes) == 1:
            thumbnails = thumb_fn(np.stack(images).astype(np.float32))
        else:
            thumbnails = np.stack(
                [thumb_fn(img[np.newaxis].astype(np.float32))[0] for img in images]
            )
    else:
        thumbnails = np.zeros((n, *thumb_shape), dtype=np.float32)

    # Unpack provenance
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
        projections=projections,
        labels=labels,
        params=params,
        access_tags=access_tags,
    )


NOTES_MAX_LEN = 1024
USER_LABEL_MAX_LEN = 64

# Base fields present in every _index table (param columns are appended after)
_INDEX_BASE_FIELDS = [
    pa.field("indx", pa.int64()),
    pa.field("path", pa.string()),
    pa.field("slice", pa.string(), nullable=True),
    pa.field("label", pa.string(), nullable=True),
    pa.field("model_version", pa.string(), nullable=True),
    pa.field("mlflow_run_id", pa.string(), nullable=True),
    pa.field("timestamp", pa.float64()),
]


def _make_index_schema(
    param_specs: Optional[dict[str, "ParamSpec"]] = None,
) -> pa.Schema:
    """Build the PyArrow schema for the _index table.

    Base fields are always present.  One nullable column is appended for each
    declared parameter, named ``param_<name>`` and typed according to its
    ``dtype``.
    """
    fields = list(_INDEX_BASE_FIELDS)
    for name, spec in sorted((param_specs or {}).items()):
        pa_type = _PARAM_DTYPE_TO_PA.get(spec["dtype"])
        if pa_type is None:
            raise ValueError(
                f"Unknown param dtype {spec['dtype']!r} for parameter {name!r}. "
                f"Valid dtypes: {list(_PARAM_DTYPE_TO_PA)}"
            )
        fields.append(pa.field(f"param_{name}", pa_type, nullable=True))
    return pa.schema(fields)


# Keep backward-compat name for containers created without params
INDEX_SCHEMA = _make_index_schema()

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
    params: Optional[dict[str, "ParamSpec"]] = None,
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
    params : dict[str, ParamSpec], optional
        Declares the continuous/categorical parameters stored alongside each
        embedding in the _index table.  Keys are parameter names (column names
        will be ``param_<name>``); values are ``ParamSpec`` dicts describing
        dtype, units, source, display_name, and precision.

        Example::

            params={
                "temperature": {"dtype": "float", "units": "°C",
                                "source": "start.sample.temperature"},
                "scan_id":     {"dtype": "integer", "source": "start.scan_id"},
                "sample":      {"dtype": "string",  "source": "start.sample_name",
                                "display_name": "Sample"},
            }

    metadata : dict, optional
        Additional user metadata merged into the container metadata.
    access_tags : list[str], optional
        Tiled access tags applied to every node written.

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
        "param_specs": params or {},
    }
    if metadata:
        container_metadata.update(metadata)

    node = parent.create_container(
        key=key,
        metadata=container_metadata,
        specs=["LatentSpaceEmbedding", "composite"],
        access_tags=access_tags,
    )

    if isinstance(node, LatentSpaceEmbedding):
        return node
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
        _index : [indx, path, slice, label, model_version, mlflow_run_id, timestamp,
                  param_<name>, ...]
            ``indx`` is the 0-based row offset in the ``embeddings`` and
            ``projections`` arrays, making it the stable join key between the
            table and the arrays regardless of SQL row ordering.
            ``param_*`` columns hold per-embedding parameter values declared
            in ``param_specs`` container metadata.
            Rows ordered by timestamp on read.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Use the structure.count already present in the fetched item to decide
        # whether child arrays exist, avoiding a separate search GET.
        _structure = self.item.get("attributes", {}).get("structure", {})
        _count = (
            _structure.get("count")
            if isinstance(_structure, dict)
            else getattr(_structure, "count", None)
        )
        if _count == 0:
            self._arrays_initialised: bool | None = False
        elif _count is not None and _count > 0:
            self._arrays_initialised = True
        else:
            self._arrays_initialised = None

        # Local embedding count cache.
        self._num_embeddings: int | None = None

        # Build the _index schema from param_specs in container metadata.
        # Containers created before params were introduced will have an empty dict.
        self._param_specs: dict[str, ParamSpec] = self.metadata.get("param_specs", {})
        self._index_schema = _make_index_schema(self._param_specs)

        # Create the _index table if not exists
        try:
            self._index_table = self.base["_index"]
        except KeyError:
            self._index_table = self.create_appendable_table(
                self._index_schema,
                key="_index",
                metadata={"description": "Per-embedding metadata index (append-only)"},
                access_tags=self.access_blob.get("tags", None),
            )
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
        tags = access_tags if access_tags is not None else self.access_blob.get("tags", None)

        proj_dim = self.metadata.get("projection_dim", 2)
        if projections is not None:
            proj_data = projections.astype(np.float32)
        else:
            proj_data = np.full((batch_size, proj_dim), np.nan, dtype=np.float32)

        if self._arrays_initialised is None:
            self._arrays_initialised = "embeddings" in self

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
            self._arr_projections = self.write_array(
                proj_data,
                key="projections",
                metadata={"description": "Visualization projections"},
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
            self._arrays_initialised = True
        else:
            emb = getattr(self, "_arr_embeddings", None) or self["embeddings"]
            thu = getattr(self, "_arr_thumbnails", None) or self["thumbnails"]
            prj = getattr(self, "_arr_projections", None) or self["projections"]
            nts = getattr(self, "_arr_notes", None) or self["notes"]
            ulb = getattr(self, "_arr_user_labels", None) or self["user_labels"]
            emb.patch(embeddings.astype(np.float32), offset=(offset,), extend=True)
            thu.patch(thumbnails, offset=(offset,), extend=True)
            prj.patch(proj_data, offset=(offset,), extend=True)
            nts.patch(empty_notes, offset=(offset,), extend=True)
            ulb.patch(empty_user_labels, offset=(offset,), extend=True)

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
        params: Optional[dict[str, list]] = None,
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
        mlflow_run_id : str, optional
            MLFlow run ID that produced these embeddings.
        timestamps : list[float], optional
            Epoch timestamps. Defaults to current time for each.
        projections : np.ndarray, optional
            Shape (B, P) pre-computed projection vectors.
        params : dict[str, list], optional
            Per-embedding parameter values.  Keys must match the ``param_specs``
            declared at container creation.  Each value is a list of length B.

            Example::

                params={
                    "temperature": [25.1, 25.3, 24.9],
                    "scan_id":     [1042, 1042, 1043],
                    "sample":      ["CsPbBr3", "CsPbBr3", "MAPbI3"],
                }

            Unknown keys raise ``ValueError``.  Declared params that are absent
            from ``params`` are stored as ``None`` (null) for all rows in the batch.
        access_tags : list[str], optional
            Tiled access tags applied to newly created child arrays.

        Returns
        -------
        int
            The new total number of embeddings after appending.
        """
        batch_size = len(embeddings)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            msg = f"Expected embeddings shape (B, {self.embedding_dim}), got {embeddings.shape}"
            raise ValueError(msg)

        meta = self.metadata
        expected_thumb_shape = (batch_size, *meta.get("thumb_shape", ()))
        if thumbnails.shape != expected_thumb_shape:
            msg = f"Expected thumbnails shape {expected_thumb_shape}, got {thumbnails.shape}"
            raise ValueError(msg)

        if len(paths) != batch_size:
            msg = f"Expected {batch_size} paths, got {len(paths)}"
            raise ValueError(msg)

        # Validate param keys against declared param_specs
        params = params or {}
        unknown = set(params) - set(self._param_specs)
        if unknown:
            raise ValueError(
                f"Unknown parameter(s) {sorted(unknown)}. "
                f"Declared params: {sorted(self._param_specs)}"
            )
        for name, values in params.items():
            if len(values) != batch_size:
                raise ValueError(
                    f"Parameter {name!r}: expected {batch_size} values, got {len(values)}"
                )

        current_n = self.num_embeddings

        self._write_arrays(
            embeddings,
            thumbnails,
            projections,
            offset=current_n,
            access_tags=access_tags,
        )

        # Build the _index table row, including param columns
        table_data: dict[str, Any] = {
            "indx": list(range(current_n, current_n + batch_size)),
            "path": paths,
            "slice": [None] * batch_size if slices is None else slices,
            "label": [None] * batch_size if labels is None else labels,
            "model_version": [model_version or self.metadata.get("model_version")] * batch_size,
            "mlflow_run_id": [mlflow_run_id or self.metadata.get("mlflow_run_id", "")] * batch_size,
            "timestamp": timestamps or list(time.time() + np.arange(batch_size) * 1e-6),
        }
        for name in sorted(self._param_specs):
            col = f"param_{name}"
            table_data[col] = params.get(name, [None] * batch_size)

        table = pa.table(table_data, schema=self._index_schema)
        self._index_table.append_partition(0, table)

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
        self["user_labels"].patch(_make_string_array([label], USER_LABEL_MAX_LEN), offset=(index,))

    def update_projections(
        self, projections: np.ndarray, access_tags: Optional[list[str]] = None
    ) -> None:
        """Replace all projection vectors (e.g. after re-running the projector approximator).

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

    def read_projections(self, indices=None) -> np.ndarray:
        "Read projection vectors. Returns NaN-filled array if not yet computed."
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

    def read_params(
        self,
        names: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Read parameter columns from the _index table.

        Parameters
        ----------
        names : list[str], optional
            Parameter names to read (without the ``param_`` prefix).
            When ``None`` (default), all declared parameters are returned.

        Returns
        -------
        dict[str, pandas.Series]
            Mapping from parameter name → Series (sorted by timestamp,
            index-aligned with ``read_index()``).

        Raises
        ------
        ValueError
            If any requested name is not a declared parameter.
        """
        if names is None:
            names = list(self._param_specs)
        else:
            unknown = set(names) - set(self._param_specs)
            if unknown:
                raise ValueError(
                    f"Unknown parameter(s) {sorted(unknown)}. "
                    f"Declared params: {sorted(self._param_specs)}"
                )

        if not names:
            return {}

        columns = [f"param_{n}" for n in names] + ["timestamp"]
        df = self._index_table.read(columns=columns)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return {n: df[f"param_{n}"] for n in names}


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
            "param_specs": {},
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
            raise ValidationError("LatentSpaceEmbedding container must have an '_index' table.")
