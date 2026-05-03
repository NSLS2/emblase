import React from "react";

declare global {
  interface Window {
    __TILED_ACCESS_TOKEN__?: string | null;
  }
}

interface WsConn {
  ws: WebSocket | null;
  timer: ReturnType<typeof setTimeout> | null;
  connected: boolean;
  schemaReceived: boolean;
}

/** Return auth headers if a token is available (works with or without auth). */
function authHeaders(extra?: Record<string, string>): Record<string, string> {
  const h: Record<string, string> = { ...extra };
  const token = window.__TILED_ACCESS_TOKEN__;
  if (token) h["Authorization"] = `Bearer ${token}`;
  return h;
}

/** Fetch a URL with auth and return an object URL for use as img src. */
function useAuthBlobUrl(url: string | null): string | null {
  const [blobUrl, setBlobUrl] = React.useState<string | null>(null);
  React.useEffect(() => {
    if (!url) { setBlobUrl((prev) => { if (prev) URL.revokeObjectURL(prev); return null; }); return; }
    let objectUrl: string | null = null;
    let cancelled = false;
    fetch(url, { headers: authHeaders() })
      .then((r) => (r.ok ? r.blob() : Promise.reject(r.statusText)))
      .then((blob) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setBlobUrl((prev) => { if (prev) URL.revokeObjectURL(prev); return objectUrl; });
      })
      .catch(() => { if (!cancelled) setBlobUrl((prev) => { if (prev) URL.revokeObjectURL(prev); return null; }); });
    return () => {
      cancelled = true;
      // objectUrl may not be assigned yet if fetch is still in-flight;
      // the setBlobUrl functional updater above handles that case on resolve.
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [url]);
  return blobUrl;
}

/** <img> that fetches via auth headers (works with and without auth). */
function AuthImg(props: React.ImgHTMLAttributes<HTMLImageElement>) {
  const { src, style, ...rest } = props;
  const blobUrl = useAuthBlobUrl(src ?? null);
  if (!blobUrl)
    return React.createElement("div", {
      style: { background: "#e0e0e0", ...style },
      ...rest,
    });
  return React.createElement("img", { ...rest, src: blobUrl, style });
}

interface EmbeddingPoint {
  x: number;
  y: number;
  index: number;
  label?: string;
  path?: string;
  params?: Record<string, number | null>;
  metaReceived: boolean; // false until _index row has been applied
}

interface ParamSpec {
  dtype: "float" | "integer" | "string" | "boolean";
  units?: string;
  display_name?: string;
  precision?: number;
}

interface TooltipInfo {
  point: EmbeddingPoint;
  screenX: number;
  screenY: number;
  thumbnailUrl: string;
}

interface SelectedPointInfo {
  point: EmbeddingPoint;
  thumbnailUrl: string;
  note: string;
  userLabel: string;
  originalNote: string;
  originalUserLabel: string;
  saving: boolean;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}

interface ViewState {
  offsetX: number;
  offsetY: number;
  scale: number;
}

type WsStatus = "disconnected" | "connecting" | "connected";
type ToolMode = "pan" | "lasso";

const POINT_RADIUS = 4;
const HOVER_RADIUS = 8;
const NOTES_MAX_LEN = 1024;
const USER_LABEL_MAX_LEN = 64;
const CANVAS_HEIGHT = 500;
const PANEL_WIDTH = 280;
const COLOR_PALETTE = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
  "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
];

/** Assign a palette color to a label deterministically by its rank in the
 *  sorted list of all known labels.  Passing the sorted list ensures the
 *  same label always gets the same color regardless of arrival order. */
function getLabelColor(label: string, sortedLabels: string[]): string {
  if (!label) return "#888888";
  const idx = sortedLabels.indexOf(label);
  return COLOR_PALETTE[(idx < 0 ? 0 : idx) % COLOR_PALETTE.length];
}

// Viridis colormap (sampled at 10 stops) — t ∈ [0, 1]
const VIRIDIS: [number, number, number][] = [
  [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
  [38, 130, 142], [31, 158, 137], [53, 183, 121], [110, 206, 88],
  [181, 222, 43], [253, 231, 37],
];
function viridis(t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  const scaled = clamped * (VIRIDIS.length - 1);
  const lo = Math.floor(scaled);
  const hi = Math.min(lo + 1, VIRIDIS.length - 1);
  const frac = scaled - lo;
  const r = Math.round(VIRIDIS[lo][0] + (VIRIDIS[hi][0] - VIRIDIS[lo][0]) * frac);
  const g = Math.round(VIRIDIS[lo][1] + (VIRIDIS[hi][1] - VIRIDIS[lo][1]) * frac);
  const b = Math.round(VIRIDIS[lo][2] + (VIRIDIS[hi][2] - VIRIDIS[lo][2]) * frac);
  return `rgb(${r},${g},${b})`;
}

const GREY_OUT = "#cccccc";

/**
 * Determine a point's fill color and whether it is "active" (within filter).
 * Returns { color, active } where active=false means grey it out.
 */
function resolvePointColor(
  p: EmbeddingPoint,
  colorBy: string,
  hiddenLabels: Set<string>,
  paramRange: [number, number] | null,
  paramMin: number,
  paramMax: number,
  sortedLabels: string[],
): { color: string; active: boolean } {
  // Point arrived from projection stream but _index metadata not yet received
  if (!p.metaReceived) return { color: GREY_OUT, active: false };
  if (colorBy === "label") {
    const lbl = p.label || "";
    const active = lbl === "" || !hiddenLabels.has(lbl);
    return { color: getLabelColor(lbl, sortedLabels), active };
  }
  // Continuous param
  const val = p.params?.[colorBy] ?? null;
  if (val === null) return { color: GREY_OUT, active: false };
  const range = paramRange ?? [paramMin, paramMax];
  const active = val >= range[0] && val <= range[1];
  const span = paramMax - paramMin || 1;
  const t = (val - paramMin) / span;
  return { color: viridis(t), active };
}

function encodeUtf32LE(str: string, maxLen: number): ArrayBuffer {
  const buf = new ArrayBuffer(maxLen * 4);
  const view = new Uint32Array(buf);
  let out = 0;
  for (let i = 0; i < str.length && out < maxLen; ) {
    const cp = str.codePointAt(i) ?? 0;
    view[out++] = cp;
    i += cp > 0xffff ? 2 : 1; // surrogate pairs occupy 2 UTF-16 code units
  }
  return buf;
}

async function patchStringArray(
  apiUrl: string,
  nodePath: string,
  arrayName: string,
  index: number,
  value: string,
  maxLen: number,
): Promise<Response> {
  const body = encodeUtf32LE(value, maxLen);
  return fetch(
    `${apiUrl}/array/full/${nodePath}/${arrayName}` +
      `?offset=${index}&shape=1`,
    { method: "PATCH", headers: authHeaders({ "Content-Type": "application/octet-stream" }), body },
  );
}

async function fetchStringValue(
  apiUrl: string,
  nodePath: string,
  arrayName: string,
  index: number,
): Promise<string> {
  const res = await fetch(
    `${apiUrl}/array/full/${nodePath}/${arrayName}` +
      `?format=application/json&slice=${index}`,
    { headers: authHeaders() },
  );
  if (!res.ok) return "";
  const data = await res.json();
  return typeof data === "string" ? data : String(data ?? "");
}

function fitViewToPoints(
  pts: EmbeddingPoint[],
  width: number,
  height: number,
): ViewState {
  if (pts.length === 0) return { offsetX: 0, offsetY: 0, scale: 1 };
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const { x, y } of pts) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const padding = 0.1;
  const scaleX = width / (rangeX * (1 + 2 * padding));
  const scaleY = height / (rangeY * (1 + 2 * padding));
  const scale = Math.min(scaleX, scaleY);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  return {
    offsetX: width / 2 - cx * scale,
    offsetY: height / 2 + cy * scale,
    scale,
  };
}

// Ray-casting point-in-polygon test (screen coordinates)
function pointInPolygon(px: number, py: number, poly: { x: number; y: number }[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x, yi = poly[i].y;
    const xj = poly[j].x, yj = poly[j].y;
    if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

const STATUS_COLORS: Record<WsStatus, string> = {
  disconnected: "#999",
  connecting: "#f0ad4e",
  connected: "#5cb85c",
};

const THUMB_R = 7; // thumb radius in px — half of the visual handle diameter

/**
 * Dual-handle range slider drawn directly on top of the viridis colorbar.
 * Faded overlay covers the out-of-range portions.
 * Labels sit below the bar; thumbs sit on the bar.
 */
function RangeSlider({
  min, max, value, onChange, units, precision,
}: {
  min: number;
  max: number;
  value: [number, number];
  onChange: (lo: number, hi: number) => void;
  units?: string;
  precision?: number;
}) {
  const trackRef = React.useRef<HTMLDivElement>(null);
  const dragging = React.useRef<"lo" | "hi" | null>(null);
  const fmt = (v: number) => v.toFixed(precision ?? 2);
  const span = max - min || 1;
  // Keep mutable refs so the stable effect closure always sees fresh values
  const valueRef = React.useRef(value);
  valueRef.current = value;
  const onChangeRef = React.useRef(onChange);
  onChangeRef.current = onChange;
  const spanRef = React.useRef(span);
  spanRef.current = span;

  const loFrac = (value[0] - min) / span;
  const hiFrac = (value[1] - min) / span;

  function fracFromEvent(clientX: number): number {
    const track = trackRef.current;
    if (!track) return 0;
    const rect = track.getBoundingClientRect();
    return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  }

  React.useEffect(() => {
    function onMove(e: MouseEvent) {
      if (!dragging.current) return;
      const frac = fracFromEvent(e.clientX);
      const v = min + frac * spanRef.current;
      const cur = valueRef.current;
      const sp = spanRef.current;
      if (dragging.current === "lo") {
        onChangeRef.current(Math.min(v, cur[1] - sp / 200), cur[1]);
      } else {
        onChangeRef.current(cur[0], Math.max(v, cur[0] + sp / 200));
      }
    }
    function onUp() { dragging.current = null; }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [min]); // only re-attach if min changes (needed for fracFromEvent calc)

  const BAR_HEIGHT = 14;
  const gradientStops = Array.from({ length: 10 }, (_, i) => viridis(i / 9)).join(", ");

  return React.createElement(
    "div",
    { style: { fontSize: 11, userSelect: "none" as const } },
    // ── Combined colorbar + thumb track ─────────────────────────────────────
    React.createElement(
      "div",
      {
        ref: trackRef,
        style: {
          position: "relative" as const,
          height: BAR_HEIGHT,
          borderRadius: BAR_HEIGHT / 2,
          background: `linear-gradient(to right, ${gradientStops})`,
          cursor: "default",
        },
      },
      // Left fade overlay (out-of-range)
      loFrac > 0.001
        ? React.createElement("div", {
            style: {
              position: "absolute" as const,
              top: 0, left: 0, bottom: 0,
              width: `${loFrac * 100}%`,
              borderRadius: `${BAR_HEIGHT / 2}px 0 0 ${BAR_HEIGHT / 2}px`,
              background: "rgba(250,250,250,0.72)",
              pointerEvents: "none" as const,
            },
          })
        : null,
      // Right fade overlay (out-of-range)
      hiFrac < 0.999
        ? React.createElement("div", {
            style: {
              position: "absolute" as const,
              top: 0, right: 0, bottom: 0,
              width: `${(1 - hiFrac) * 100}%`,
              borderRadius: `0 ${BAR_HEIGHT / 2}px ${BAR_HEIGHT / 2}px 0`,
              background: "rgba(250,250,250,0.72)",
              pointerEvents: "none" as const,
            },
          })
        : null,
      // Lo thumb
      React.createElement("div", {
        onMouseDown: (e: React.MouseEvent) => { e.preventDefault(); dragging.current = "lo"; },
        style: {
          position: "absolute" as const,
          top: "50%",
          left: `${loFrac * 100}%`,
          transform: "translate(-50%, -50%)",
          width: THUMB_R * 2,
          height: THUMB_R * 2,
          borderRadius: "50%",
          background: "white",
          border: "2px solid rgba(0,0,0,0.35)",
          boxShadow: "0 1px 3px rgba(0,0,0,0.25)",
          cursor: "ew-resize",
          boxSizing: "border-box" as const,
          zIndex: 3,
        },
      }),
      // Hi thumb
      React.createElement("div", {
        onMouseDown: (e: React.MouseEvent) => { e.preventDefault(); dragging.current = "hi"; },
        style: {
          position: "absolute" as const,
          top: "50%",
          left: `${hiFrac * 100}%`,
          transform: "translate(-50%, -50%)",
          width: THUMB_R * 2,
          height: THUMB_R * 2,
          borderRadius: "50%",
          background: "white",
          border: "2px solid rgba(0,0,0,0.35)",
          boxShadow: "0 1px 3px rgba(0,0,0,0.25)",
          cursor: "ew-resize",
          boxSizing: "border-box" as const,
          zIndex: 3,
        },
      }),
    ),
    // ── Labels row (below the bar) ──────────────────────────────────────────
    React.createElement(
      "div",
      { style: { display: "flex", justifyContent: "space-between", marginTop: 3, color: "#888" } },
      React.createElement("span", null, fmt(value[0])),
      React.createElement(
        "span",
        null,
        fmt(value[1]) + (units ? `\u2009${units}` : ""),
      ),
    ),
  );
}

function EmbeddingScatter({
  segments,
  item,
}: {
  segments: string[];
  item: any;
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const containerRef = React.useRef<HTMLDivElement>(null);
  const [points, setPoints] = React.useState<EmbeddingPoint[]>([]);
  const [tooltip, setTooltip] = React.useState<TooltipInfo | null>(null);
  const [dragging, setDragging] = React.useState(false);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [view, setView] = React.useState<ViewState>({
    offsetX: 0,
    offsetY: 0,
    scale: 1,
  });
  const [wsStatus, setWsStatus] = React.useState<WsStatus>("disconnected");
  const [liveEnabled, setLiveEnabled] = React.useState(true);
  const [selected, setSelected] = React.useState<SelectedPointInfo | null>(
    null,
  );
  const [saveError, setSaveError] = React.useState<string | null>(null);
  const [chatOpen, setChatOpen] = React.useState(false);
  const [chatMessages, setChatMessages] = React.useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = React.useState("");
  const [chatSending, setChatSending] = React.useState(false);
  const chatListRef = React.useRef<HTMLDivElement>(null);
  const [toolMode, setToolMode] = React.useState<ToolMode>("pan");
  const [lassoPath, setLassoPath] = React.useState<{ x: number; y: number }[]>([]);
  const [lassoSelected, setLassoSelected] = React.useState<Set<number>>(new Set());
  const lassoDrawing = React.useRef(false);
  const dragRef = React.useRef<{
    startX: number;
    startY: number;
    startOffsetX: number;
    startOffsetY: number;
  } | null>(null);
  const pointCountRef = React.useRef(0);
  // Skip the catch-up refreshAll on the first live-effect run after initial load
  const skipCatchupRef = React.useRef(false);

  // Coloring state — paramSpecs/paramNames derived from item prop (stable after mount)
  const paramSpecs: Record<string, ParamSpec> = React.useMemo(
    () => item?.data?.attributes?.metadata?.param_specs || {},
    [item],
  );
  const paramNames = React.useMemo(() => Object.keys(paramSpecs), [paramSpecs]);
  const [colorBy, setColorBy] = React.useState<string>("label");
  // Categorical filter: set of labels hidden from view
  const [hiddenLabels, setHiddenLabels] = React.useState<Set<string>>(new Set());
  // Continuous filter: [lo, hi] in data units (null = full range)
  const [paramRange, setParamRange] = React.useState<[number, number] | null>(null);

  // Reset filters when color dimension changes
  React.useEffect(() => {
    setHiddenLabels(new Set());
    setParamRange(null);
  }, [colorBy]);

  const apiUrl = `${window.location.origin}/api/v1`;

  const nodePath = segments.join("/");
  const customUrl = `${window.location.origin}/custom/emblase`;

  // canvasWidth tracks the canvas element's actual CSS pixel width (updated by ResizeObserver)
  const [canvasWidth, setCanvasWidth] = React.useState(800);

  // Fetch all current data (projections + index) and merge into state
  const refreshAll = React.useCallback(async () => {
    try {
      const [projRes, indexRes] = await Promise.all([
        fetch(
          `${apiUrl}/array/full/${nodePath}/projections?format=application/json`,
          { headers: authHeaders() },
        ),
        fetch(
          `${apiUrl}/table/full/${nodePath}/_index?format=application/json`,
          { headers: authHeaders() },
        ),
      ]);
      if (!projRes.ok || !indexRes.ok) return;

      const projData: number[][] = await projRes.json();
      const indexData = await indexRes.json();
      if (projData.length === 0) return;

      // Sort _index rows by `indx` column to guarantee alignment with projections array
      const indxCol: number[] = indexData.indx || indexData.slice || [];
      const order = indxCol.map((v, i) => ({ v, i }))
        .sort((a, b) => a.v - b.v)
        .map((x) => x.i);

      const labels: string[] = indexData.label || [];
      const paths: string[] = indexData.path || [];

      const pts: EmbeddingPoint[] = projData.map(
        (coords: number[], rowIdx: number) => {
          const src = order[rowIdx] ?? rowIdx; // sorted row → original column position
          const params: Record<string, number | null> = {};
          for (const name of paramNames) {
            const col: (number | null)[] = indexData[`param_${name}`] || [];
            params[name] = col[src] ?? null;
          }
          return {
            x: coords[0],
            y: coords[1],
            index: rowIdx,
            label: labels[src] || "",
            path: paths[src] || "",
            params,
            metaReceived: true,
          };
        },
      );
      pointCountRef.current = pts.length;
      setPoints(pts);
      return pts;
    } catch {
      return undefined;
    }
  }, [apiUrl, nodePath, paramNames]);

  const fitView = React.useCallback(() => {
    if (points.length === 0) return;
    const w = canvasRef.current?.clientWidth || canvasWidth;
    setView(fitViewToPoints(points, w, CANVAS_HEIGHT));
  }, [points, canvasWidth]);

  // Initial data load — runs once on mount
  const refreshAllRef = React.useRef(refreshAll);
  refreshAllRef.current = refreshAll;
  React.useEffect(() => {
    let cancelled = false;

    async function fetchData() {
      try {
        setLoading(true);
        const pts = await refreshAllRef.current();
        if (cancelled) return;
        setError(null);
        if (pts && pts.length > 0) {
          const w = canvasRef.current?.clientWidth || canvasWidth;
          setView(fitViewToPoints(pts, w, CANVAS_HEIGHT));
        }
      } catch (err: any) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) {
          skipCatchupRef.current = true;
          setLoading(false);
        }
      }
    }

    fetchData();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // intentionally empty — runs once on mount; refreshAllRef stays current

  // Auto-refit view when new points arrive via live updates,
  // unless the user has already panned or zoomed manually.
  const prevPointCountRef = React.useRef(0);

  // Live updates via dual WebSocket subscriptions.
  //
  // We subscribe to both the projections array and the _index table.
  // The write order on the server is: arrays first, _index table last.
  // So the projection event (with x/y coords) typically arrives before the
  // table event (with label/path). Points render immediately from the
  // projection payload; when the table event arrives it fills in metadata.
  //
  // A pendingMeta buffer handles the rare case where the table event
  // arrives before the projection event: metadata is buffered and applied
  // once the projection event creates the points.
  //
  // On (re-)connect and on live re-enable we do a full refreshAll() to
  // catch any updates that arrived while disconnected or paused.
  React.useEffect(() => {
    if (loading || !liveEnabled) return;

    const wsScheme = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsBase = `${wsScheme}//${window.location.host}/api/v1/stream/single/${nodePath}`;

    const proj: WsConn = { ws: null, timer: null, connected: false, schemaReceived: false };
    const idx: WsConn = { ws: null, timer: null, connected: false, schemaReceived: false };
    let disposed = false;

    // Buffer for table metadata that arrived before projections
    const pendingMeta = new Map<number, { labels: string[]; paths: string[]; params: Record<string, (number | null)[]> }>();

    function updateStatus() {
      if (proj.connected && idx.connected) setWsStatus("connected");
      else if (proj.connected || idx.connected) setWsStatus("connecting");
      else setWsStatus("disconnected");
    }

    function connect(conn: WsConn, subpath: string, handler: (msg: any) => void) {
      function doConnect() {
        if (disposed) return;
        conn.schemaReceived = false;
        conn.ws = new WebSocket(`${wsBase}/${subpath}?envelope_format=json`);
        conn.ws.onopen = () => {
          if (disposed) { conn.ws?.close(); return; }
          // Send credentials via first message (avoids token in URL/logs).
          const token = window.__TILED_ACCESS_TOKEN__;
          if (token) {
            conn.ws!.send(JSON.stringify({ type: "auth", access_token: token }));
          }
          conn.connected = true;
          updateStatus();
        };
        conn.ws.onmessage = (event) => {
          if (disposed) return;
          try {
            const msg = JSON.parse(event.data);
            if (!conn.schemaReceived && msg.type?.endsWith("-schema")) {
              conn.schemaReceived = true;
              return;
            }
            handler(msg);
          } catch { /* ignore parse errors */ }
        };
        conn.ws.onclose = () => {
          conn.connected = false;
          updateStatus();
          if (!disposed) conn.timer = setTimeout(doConnect, 3000);
        };
        conn.ws.onerror = () => {};
      }
      doConnect();
    }

    // Catch up on anything missed while live was off or WS was disconnected.
    // Skip on the first run right after initial load (data is already fresh).
    if (skipCatchupRef.current) {
      skipCatchupRef.current = false;
    } else {
      refreshAll();
    }

    function applyMeta(startIdx: number, count: number) {
      const meta = pendingMeta.get(startIdx);
      if (!meta) return;
      if (meta.labels.length !== count) return;
      pendingMeta.delete(startIdx);
      setPoints((prev) =>
        prev.map((p) => {
          const rel = p.index - startIdx;
          if (rel >= 0 && rel < count) {
            const newParams: Record<string, number | null> = { ...(p.params || {}) };
            for (const name of Object.keys(meta.params)) {
              newParams[name] = meta.params[name][rel] ?? null;
            }
            return {
              ...p,
              label: meta.labels[rel] || p.label,
              path: meta.paths[rel] || p.path,
              params: newParams,
              metaReceived: true,
            };
          }
          return p;
        }),
      );
    }

    function handleProjectionEvent(msg: any) {
      if (msg.type !== "array-data" && msg.type !== "array-ref") return;
      const offset = msg.offset;
      const payload = msg.payload;
      if (!offset || !Array.isArray(offset) || offset.length === 0) {
        refreshAll();
        return;
      }
      const startIdx = offset[0];
      if (!payload || !Array.isArray(payload) || payload.length === 0) {
        refreshAll();
        return;
      }
      const newPts: EmbeddingPoint[] = payload.map(
        (coords: number[], i: number) => ({
          x: coords[0],
          y: coords[1],
          index: startIdx + i,
          label: "",
          path: "",
          metaReceived: false,
        }),
      );
      pointCountRef.current = startIdx + payload.length;
      setPoints((prev) => prev.slice(0, startIdx).concat(newPts));

      // Check if metadata already arrived for this range
      applyMeta(startIdx, payload.length);
    }

    function handleTableEvent(msg: any) {
      if (msg.type !== "table-data") return;
      if (!msg.append) {
        refreshAll();
        return;
      }
      const payload = msg.payload;
      if (!payload) return;
      const labels: string[] = payload.label || [];
      const paths: string[] = payload.path || [];
      const count = labels.length;
      if (count === 0) return;

      // Extract param columns from payload
      const params: Record<string, (number | null)[]> = {};
      for (const name of paramNames) {
        params[name] = payload[`param_${name}`] || [];
      }

      // Determine which indices these rows correspond to.
      // If projections already arrived, pointCountRef is updated and the
      // points exist — apply metadata directly. Otherwise buffer it.
      const startIdx = pointCountRef.current - count;
      if (startIdx >= 0) {
        setPoints((prev) => {
          // Verify the points at this range exist
          if (prev.length >= startIdx + count) {
            return prev.map((p) => {
              const rel = p.index - startIdx;
              if (rel >= 0 && rel < count) {
                const newParams: Record<string, number | null> = { ...(p.params || {}) };
                for (const name of paramNames) {
                  newParams[name] = params[name]?.[rel] ?? null;
                }
                return {
                  ...p,
                  label: labels[rel] || p.label,
                  path: paths[rel] || p.path,
                  params: newParams,
                  metaReceived: true,
                };
              }
              return p;
            });
          }
          // Points don't exist yet; buffer under startIdx so applyMeta can find it
          pendingMeta.set(startIdx, { labels, paths, params });
          return prev;
        });
      } else {
        // Projection event hasn't arrived yet; buffer under startIdx (will be >= 0 when it does)
        pendingMeta.set(Math.max(0, startIdx), { labels, paths, params });
      }
    }

    connect(proj, "projections", handleProjectionEvent);
    connect(idx, "_index", handleTableEvent);

    return () => {
      disposed = true;
      for (const c of [proj, idx]) {
        if (c.timer) clearTimeout(c.timer);
        if (c.ws) { c.ws.onclose = null; c.ws.close(); }
      }
      setWsStatus("disconnected");
    };
  }, [loading, liveEnabled, nodePath, apiUrl, refreshAll]);

  // ResizeObserver — tracks canvas element width so canvasWidth stays current
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = Math.floor(entry.contentRect.width);
        if (w > 0) setCanvasWidth(w);
      }
    });
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  // Compute min/max for the currently selected continuous param (must be before canvas draw effect)
  const uniqueLabels = React.useMemo(() => {
    const labels = new Set<string>();
    for (const p of points) {
      if (p.label) labels.add(p.label);
    }
    return Array.from(labels).sort();
  }, [points]);

  const { paramMin, paramMax } = React.useMemo(() => {
    if (colorBy === "label" || !paramNames.includes(colorBy)) return { paramMin: 0, paramMax: 1 };
    let mn = Infinity, mx = -Infinity;
    for (const p of points) {
      const v = p.params?.[colorBy];
      if (v !== null && v !== undefined) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    }
    if (!isFinite(mn)) { mn = 0; mx = 1; }
    if (mn === mx) { mn -= 0.5; mx += 0.5; }
    return { paramMin: mn, paramMax: mx };
  }, [points, colorBy, paramNames]);

  // When new live data extends the param range past the current filter, clear it.
  React.useEffect(() => {
    setParamRange((prev) =>
      prev && prev[0] <= paramMin && prev[1] >= paramMax ? null : prev,
    );
  }, [paramMin, paramMax]);

  // Draw canvas
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || canvasWidth;
    canvas.width = w * dpr;
    canvas.height = CANVAS_HEIGHT * dpr;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = "#fafafa";
    ctx.fillRect(0, 0, w, CANVAS_HEIGHT);

    // Grid
    ctx.strokeStyle = "#e0e0e0";
    ctx.lineWidth = 1;
    const gridStep = 50 * view.scale;
    if (gridStep > 10) {
      const startX = ((view.offsetX % gridStep) + gridStep) % gridStep; // always positive
      for (let x = startX; x < w; x += gridStep) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, CANVAS_HEIGHT); ctx.stroke();
      }
      const startY = ((view.offsetY % gridStep) + gridStep) % gridStep;
      for (let y = startY; y < CANVAS_HEIGHT; y += gridStep) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }
    }

    // Points
    const hasLasso = lassoSelected.size > 0;
    for (const p of points) {
      const sx = p.x * view.scale + view.offsetX;
      const sy = -p.y * view.scale + view.offsetY;
      if (
        sx < -10 ||
        sx > w + 10 ||
        sy < -10 ||
        sy > CANVAS_HEIGHT + 10
      )
        continue;
      const isHovered = tooltip?.point.index === p.index;
      const isSelected = selected?.point.index === p.index;
      const isLassoed = hasLasso && lassoSelected.has(p.index);
      const { color, active } = resolvePointColor(p, colorBy, hiddenLabels, paramRange, paramMin, paramMax, uniqueLabels);
      const radius = isHovered || isSelected ? HOVER_RADIUS : POINT_RADIUS;
      ctx.beginPath();
      ctx.arc(sx, sy, radius, 0, Math.PI * 2);
      // Grey out inactive points; dim lasso-unselected points
      if (!active && !isHovered && !isSelected) {
        ctx.fillStyle = GREY_OUT;
        ctx.globalAlpha = hasLasso && !isLassoed ? 0.08 : 0.25;
      } else {
        ctx.fillStyle = color;
        ctx.globalAlpha = hasLasso && !isLassoed && !isHovered && !isSelected ? 0.15 : (isHovered || isSelected ? 1.0 : 0.75);
      }
      ctx.fill();
      if (isSelected) {
        ctx.globalAlpha = 1.0;
        ctx.strokeStyle = "#1976d2";
        ctx.lineWidth = 2.5;
        ctx.stroke();
      } else if (isLassoed) {
        ctx.globalAlpha = 1.0;
        ctx.strokeStyle = "#ff6f00";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      } else if (isHovered) {
        ctx.globalAlpha = 1.0;
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1.0;

    // Draw lasso path (stored in data coords, convert to screen)
    if (lassoPath.length > 1) {
      ctx.beginPath();
      const lp0x = lassoPath[0].x * view.scale + view.offsetX;
      const lp0y = -lassoPath[0].y * view.scale + view.offsetY;
      ctx.moveTo(lp0x, lp0y);
      for (let i = 1; i < lassoPath.length; i++) {
        const lpx = lassoPath[i].x * view.scale + view.offsetX;
        const lpy = -lassoPath[i].y * view.scale + view.offsetY;
        ctx.lineTo(lpx, lpy);
      }
      if (!lassoDrawing.current && lassoPath.length > 2) {
        ctx.closePath();
      }
      ctx.strokeStyle = "#ff6f00";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
      if (!lassoDrawing.current && lassoPath.length > 2) {
        ctx.fillStyle = "rgba(255, 111, 0, 0.06)";
        ctx.fill();
      }
    }
  }, [points, view, canvasWidth, tooltip, selected, lassoPath, lassoSelected, colorBy, hiddenLabels, paramRange, paramMin, paramMax, uniqueLabels]);
  // Mouse handlers
  const toDataCoords = React.useCallback(
    (clientX: number, clientY: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      return { mx: clientX - rect.left, my: clientY - rect.top };
    },
    [],
  );

  // Convert screen coords to data-space coords (inverse of the view transform)
  const screenToData = React.useCallback(
    (sx: number, sy: number, v: ViewState) => ({
      x: (sx - v.offsetX) / v.scale,
      y: -(sy - v.offsetY) / v.scale,
    }),
    [],
  );

  const findPoint = React.useCallback(
    (mx: number, my: number): EmbeddingPoint | null => {
      let closest: EmbeddingPoint | null = null;
      let minDist2 = (HOVER_RADIUS * 2) ** 2;
      for (const p of points) {
        const sx = p.x * view.scale + view.offsetX;
        const sy = -p.y * view.scale + view.offsetY;
        const d2 = (mx - sx) ** 2 + (my - sy) ** 2;
        if (d2 < minDist2) {
          minDist2 = d2;
          closest = p;
        }
      }
      return closest;
    },
    [points, view],
  );

  const handleMouseMove = React.useCallback(
    (e: React.MouseEvent) => {
      const coords = toDataCoords(e.clientX, e.clientY);
      if (!coords) return;

      // Lasso drawing
      if (toolMode === "lasso" && lassoDrawing.current) {
        const dp = screenToData(coords.mx, coords.my, viewRef.current);
        setLassoPath((prev) => [...prev, dp]);
        return;
      }

      // Pan dragging
      if (dragRef.current) {
        const dx = e.clientX - dragRef.current.startX;
        const dy = e.clientY - dragRef.current.startY;
        setView((v) => ({
          ...v,
          offsetX: dragRef.current!.startOffsetX + dx,
          offsetY: dragRef.current!.startOffsetY + dy,
        }));
        return;
      }

      const p = findPoint(coords.mx, coords.my);
      if (p) {
        const thumbUrl = `${apiUrl}/array/full/${nodePath}/thumbnails?format=image/png&slice=${p.index}`;
        setTooltip({
          point: p,
          screenX: coords.mx,
          screenY: coords.my,
          thumbnailUrl: thumbUrl,
        });
      } else {
        setTooltip(null);
      }
    },
    [toDataCoords, findPoint, apiUrl, nodePath, toolMode],
  );

  const viewRef = React.useRef(view);
  viewRef.current = view;

  const pointsRef = React.useRef(points);
  pointsRef.current = points;

  // Refs for lasso filter — keep current values accessible in callbacks
  const colorByRef = React.useRef(colorBy);
  colorByRef.current = colorBy;
  const hiddenLabelsRef = React.useRef(hiddenLabels);
  hiddenLabelsRef.current = hiddenLabels;
  const paramRangeRef = React.useRef(paramRange);
  paramRangeRef.current = paramRange;
  const paramMinRef = React.useRef(paramMin);
  paramMinRef.current = paramMin;
  const paramMaxRef = React.useRef(paramMax);
  paramMaxRef.current = paramMax;
  const uniqueLabelsRef = React.useRef(uniqueLabels);
  uniqueLabelsRef.current = uniqueLabels;

  const handleMouseDown = React.useCallback(
    (e: React.MouseEvent) => {
      if (toolMode === "lasso") {
        const coords = toDataCoords(e.clientX, e.clientY);
        if (!coords) return;
        lassoDrawing.current = true;
        const dp = screenToData(coords.mx, coords.my, viewRef.current);
        setLassoPath([dp]);
        setLassoSelected(new Set());
        return;
      }
      dragRef.current = {
        startX: e.clientX,
        startY: e.clientY,
        startOffsetX: viewRef.current.offsetX,
        startOffsetY: viewRef.current.offsetY,
      };
      setDragging(true);
    },
    [toolMode, toDataCoords],
  );

  const handleMouseUp = React.useCallback(() => {
    if (toolMode === "lasso" && lassoDrawing.current) {
      lassoDrawing.current = false;
      // Compute which points are inside the lasso polygon
      const path = lassoPath;
      if (path.length < 3) {
        setLassoPath([]);
        return;
      }
      const sel = new Set<number>();
      for (const p of pointsRef.current) {
        if (!pointInPolygon(p.x, p.y, path)) continue;
        // Only select points that are currently active (not greyed out by filter)
        const { active } = resolvePointColor(
          p,
          colorByRef.current,
          hiddenLabelsRef.current,
          paramRangeRef.current,
          paramMinRef.current,
          paramMaxRef.current,
          uniqueLabelsRef.current,
        );
        if (active) sel.add(p.index);
      }
      if (sel.size === 0) {
        setLassoPath([]);
      } else {
        setLassoSelected(sel);
        // Close detail panel and chat when lasso selects points
        setSelected(null);
        setChatOpen(false);
      }
      return;
    }
    dragRef.current = null;
    setDragging(false);
  }, [toolMode, lassoPath]);

  // Attach mouseup to window so drag/lasso release is always caught outside canvas
  React.useEffect(() => {
    const onUp = () => {
      if (dragRef.current) {
        dragRef.current = null;
        setDragging(false);
      }
      // Ensure lassoDrawing is reset even if mouseup fires outside the canvas
      lassoDrawing.current = false;
    };
    window.addEventListener("mouseup", onUp);
    return () => window.removeEventListener("mouseup", onUp);
  }, []);

  /** Open the detail panel for a point, fetching its note and user_label. */
  const openPoint = React.useCallback(
    (p: EmbeddingPoint) => {
      setChatOpen(false);
      setSaveError(null);
      const thumbUrl = `${apiUrl}/array/full/${nodePath}/thumbnails?format=image/png&slice=${p.index}`;
      setSelected({
        point: p,
        thumbnailUrl: thumbUrl,
        note: "",
        userLabel: "",
        originalNote: "",
        originalUserLabel: "",
        saving: false,
      });
      Promise.all([
        fetchStringValue(apiUrl, nodePath, "notes", p.index),
        fetchStringValue(apiUrl, nodePath, "user_labels", p.index),
      ]).then(([note, userLabel]) => {
        setSelected((prev) =>
          prev && prev.point.index === p.index
            ? { ...prev, note, userLabel, originalNote: note, originalUserLabel: userLabel }
            : prev,
        );
      });
    },
    [apiUrl, nodePath],
  );

  const handleClick = React.useCallback(
    (e: React.MouseEvent) => {
      // In lasso mode, clicks are handled by mousedown/mouseup
      if (toolMode === "lasso") return;
      const coords = toDataCoords(e.clientX, e.clientY);
      if (!coords) return;
      const p = findPoint(coords.mx, coords.my);
      if (!p) {
        setSelected(null);
        // Don't clear lasso on empty click — user must explicitly clear
        return;
      }
      openPoint(p);
    },
    [toDataCoords, findPoint, openPoint, toolMode],
  );

  const handleWheel = React.useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      const coords = toDataCoords(e.clientX, e.clientY);
      if (!coords) return;
      const factor = e.deltaY > 0 ? 0.9 : 1.1;
      setView((v) => ({
        scale: v.scale * factor,
        offsetX: coords.mx - (coords.mx - v.offsetX) * factor,
        offsetY: coords.my - (coords.my - v.offsetY) * factor,
      }));
    },
    [toDataCoords],
  );

  const handleSave = React.useCallback(async () => {
    if (!selected) return;
    setSaveError(null);
    setSelected((prev) => (prev ? { ...prev, saving: true } : prev));
    const [noteRes, labelRes] = await Promise.all([
      patchStringArray(
        apiUrl, nodePath, "notes",
        selected.point.index, selected.note, NOTES_MAX_LEN,
      ),
      patchStringArray(
        apiUrl, nodePath, "user_labels",
        selected.point.index, selected.userLabel, USER_LABEL_MAX_LEN,
      ),
    ]);
    setSelected((prev) => (prev ? { ...prev, saving: false } : prev));
    if (noteRes.status === 401 || labelRes.status === 401 ||
        noteRes.status === 403 || labelRes.status === 403) {
      setSaveError(
        window.__TILED_ACCESS_TOKEN__
          ? "You don't have write permission for this entry."
          : "Authentication required. Please log in to save changes.",
      );
    } else if (!noteRes.ok || !labelRes.ok) {
      setSaveError("Save failed");
    } else {
      // Save succeeded — update originals so button goes back to gray
      setSelected((prev) =>
        prev
          ? { ...prev, originalNote: prev.note, originalUserLabel: prev.userLabel }
          : prev,
      );
    }
  }, [selected, apiUrl, nodePath]);

  const loadChatHistory = React.useCallback(async () => {
    try {
      const res = await fetch(`${customUrl}/chat/history/${nodePath}`, {
        headers: authHeaders(),
      });
      if (res.ok) {
        const history: ChatMessage[] = await res.json();
        setChatMessages(history);
      }
    } catch { /* ignore */ }
  }, [customUrl, nodePath]);

  const sendChatMessage = React.useCallback(async () => {
    const text = chatInput.trim();
    if (!text || chatSending) return;
    setChatInput("");
    setChatSending(true);

    // Optimistically add user message
    const userMsg: ChatMessage = {
      id: Date.now().toString(36),
      role: "user",
      content: text,
      timestamp: Date.now() / 1000,
    };
    setChatMessages((prev) => [...prev, userMsg]);

    try {
      const res = await fetch(`${customUrl}/chat`, {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ message: text, node_path: nodePath }),
      });
      if (res.ok) {
        const assistantMsg: ChatMessage = await res.json();
        setChatMessages((prev) => [...prev, assistantMsg]);
      }
    } catch { /* ignore */ }
    setChatSending(false);
  }, [chatInput, chatSending, customUrl, nodePath]);

  const clearChatHistory = React.useCallback(async () => {
    try {
      await fetch(`${customUrl}/chat/history/${nodePath}`, {
        method: "DELETE",
        headers: authHeaders(),
      });
      setChatMessages([]);
    } catch { /* ignore */ }
  }, [customUrl, nodePath]);

  // Load chat history when chat panel opens
  React.useEffect(() => {
    if (chatOpen) loadChatHistory();
  }, [chatOpen, loadChatHistory]);

  // Auto-scroll chat to bottom when new messages arrive
  React.useEffect(() => {
    if (chatListRef.current) {
      chatListRef.current.scrollTop = chatListRef.current.scrollHeight;
    }
  }, [chatMessages]);

  // Escape key clears lasso selection
  React.useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape" && lassoSelected.size > 0) {
        setLassoSelected(new Set());
        setLassoPath([]);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [lassoSelected]);

  // Lasso selection summary
  const lassoSummary = React.useMemo(() => {
    if (lassoSelected.size === 0) return null;
    const selectedPts = points.filter((p) => lassoSelected.has(p.index));
    const labelCounts: Record<string, number> = {};
    for (const p of selectedPts) {
      const lbl = p.label || "(unlabeled)";
      labelCounts[lbl] = (labelCounts[lbl] || 0) + 1;
    }
    return { count: selectedPts.length, labelCounts, points: selectedPts };
  }, [lassoSelected, points]);

  if (error) {
    return React.createElement(
      "div",
      { style: { padding: 24, color: "#c00" } },
      `Error: ${error}`,
    );
  }

  const isDirty =
    selected != null &&
    (selected.note !== selected.originalNote ||
      selected.userLabel !== selected.originalUserLabel);

  return React.createElement(
    "div",
    { ref: containerRef },
    // Header (full width, always above canvas + panel)
    React.createElement(
      "div",
      {
        style: {
          marginBottom: 12,
          fontSize: 14,
          color: "#555",
          display: "flex",
          alignItems: "center",
          gap: 16,
        },
      },
      React.createElement("span", null,
        loading ? "Loading…" : `${points.length} embeddings`,
      ),
      // Color-by dropdown
      React.createElement(
        "div",
        { style: { display: "flex", alignItems: "center", gap: 6, fontSize: 12 } },
        React.createElement("span", { style: { color: "#777" } }, "Color by:"),
        React.createElement(
          "select",
          {
            value: colorBy,
            onChange: (e: React.ChangeEvent<HTMLSelectElement>) => setColorBy(e.target.value),
            style: {
              fontSize: 12,
              padding: "2px 6px",
              border: "1px solid #ccc",
              borderRadius: 6,
              background: "#fff",
              cursor: "pointer",
            },
          },
          React.createElement("option", { value: "label" }, "label"),
          ...paramNames.map((name) => {
            const spec = paramSpecs[name];
            const display = spec.display_name || name;
            const units = spec.units ? ` (${spec.units})` : "";
            return React.createElement("option", { key: name, value: name }, `${display}${units}`);
          }),
        ),
      ),
      // Home (fit-all) button
      React.createElement(
        "button",
        {
          onClick: fitView,
          title: "Fit all points",
          style: {
            display: "flex",
            alignItems: "center",
            gap: 3,
            fontSize: 12,
            background: "none",
            border: "1px solid #ccc",
            borderRadius: 12,
            padding: "2px 10px",
            cursor: "pointer",
            color: "#555",
            marginLeft: "auto",
          },
        },
        React.createElement(
          "svg",
          { width: 12, height: 12, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round" },
          React.createElement("path", { d: "M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" }),
          React.createElement("polyline", { points: "9 22 9 12 15 12 15 22" }),
        ),
        React.createElement("span", null, "Home"),
      ),
      // Tool mode toggle (Pan / Lasso)
      React.createElement(
        "div",
        {
          style: {
            display: "flex",
            border: "1px solid #ccc",
            borderRadius: 12,
            overflow: "hidden",
          },
        },
        React.createElement(
          "button",
          {
            onClick: () => { setToolMode("pan"); },
            title: "Pan & zoom (drag to pan)",
            style: {
              display: "flex",
              alignItems: "center",
              gap: 3,
              fontSize: 12,
              background: toolMode === "pan" ? "#e3f2fd" : "none",
              border: "none",
              borderRight: "1px solid #ccc",
              padding: "2px 10px",
              cursor: "pointer",
              color: toolMode === "pan" ? "#1976d2" : "#555",
            },
          },
          // Move/pan icon
          React.createElement(
            "svg",
            { width: 12, height: 12, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round" },
            React.createElement("polyline", { points: "5 9 2 12 5 15" }),
            React.createElement("polyline", { points: "9 5 12 2 15 5" }),
            React.createElement("polyline", { points: "15 19 12 22 9 19" }),
            React.createElement("polyline", { points: "19 9 22 12 19 15" }),
            React.createElement("line", { x1: 2, y1: 12, x2: 22, y2: 12 }),
            React.createElement("line", { x1: 12, y1: 2, x2: 12, y2: 22 }),
          ),
          React.createElement("span", null, "Pan"),
        ),
        React.createElement(
          "button",
          {
            onClick: () => { setToolMode("lasso"); },
            title: "Lasso select (draw to select points)",
            style: {
              display: "flex",
              alignItems: "center",
              gap: 3,
              fontSize: 12,
              background: toolMode === "lasso" ? "#fff3e0" : "none",
              border: "none",
              padding: "2px 10px",
              cursor: "pointer",
              color: toolMode === "lasso" ? "#ff6f00" : "#555",
            },
          },
          // Lasso icon
          React.createElement(
            "svg",
            { width: 12, height: 12, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round" },
            React.createElement("path", { d: "M7 22a5 5 0 0 1-2-4c0-2 1-3 2-4l8-8c2-2 5-2 7 0s2 5 0 7l-8 8c-1 1-2 2-4 2s-3-1-3-1" }),
          ),
          React.createElement("span", null, "Lasso"),
        ),
      ),
      // Live status toggle
      React.createElement(
        "button",
        {
          onClick: () => setLiveEnabled((v) => !v),
          style: {
            display: "flex",
            alignItems: "center",
            gap: 4,
            fontSize: 12,
            background: "none",
            border: "1px solid #ccc",
            borderRadius: 12,
            padding: "2px 10px",
            cursor: "pointer",
            color: liveEnabled ? STATUS_COLORS[wsStatus] : "#999",
          },
        },
        React.createElement("span", {
          style: {
            width: 8,
            height: 8,
            borderRadius: "50%",
            backgroundColor: liveEnabled
              ? STATUS_COLORS[wsStatus]
              : "#999",
            display: "inline-block",
          },
        }),
        React.createElement(
          "span",
          null,
          !liveEnabled
            ? "Off"
            : wsStatus === "connected"
              ? "Live"
              : wsStatus === "connecting"
                ? "Connecting..."
                : "Reconnecting...",
        ),
      ),
      // Chat toggle button
      React.createElement(
        "button",
        {
          onClick: () => {
            setChatOpen((v) => {
              if (!v) setSelected(null); // close detail panel when opening chat
              return !v;
            });
            setSaveError(null);
          },
          title: chatOpen ? "Close chat" : "Chat with data",
          style: {
            display: "flex",
            alignItems: "center",
            gap: 4,
            fontSize: 12,
            background: chatOpen ? "#1976d2" : "none",
            border: chatOpen ? "1px solid #1976d2" : "1px solid #ccc",
            borderRadius: 12,
            padding: "2px 10px",
            cursor: "pointer",
            color: chatOpen ? "#fff" : "#555",
            transition: "background 0.2s, color 0.2s, border-color 0.2s",
          },
        },
        // Chat bubble icon (SVG)
        React.createElement(
          "svg",
          {
            width: 14,
            height: 14,
            viewBox: "0 0 24 24",
            fill: "none",
            stroke: "currentColor",
            strokeWidth: 2,
            strokeLinecap: "round",
            strokeLinejoin: "round",
          },
          React.createElement("path", {
            d: "M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z",
          }),
        ),
        React.createElement("span", null, "Chat"),
      ),
    ),
    React.createElement(
      "div",
      { style: { display: "flex", alignItems: "flex-start", gap: 0 } },
      // Canvas + tooltip wrapper (flex: 1, shrinks when panel opens)
      React.createElement(
        "div",
        { style: { flex: 1, minWidth: 0, position: "relative" } },
      // Canvas
      React.createElement("canvas", {
        ref: canvasRef,
        height: CANVAS_HEIGHT,
        style: {
          width: "100%",
          height: CANVAS_HEIGHT,
          cursor: toolMode === "lasso" ? "crosshair" : dragging ? "grabbing" : tooltip ? "pointer" : "grab",
          border: "1px solid #ddd",
          borderRadius: selected || chatOpen || (lassoSummary && !selected && !chatOpen) ? "4px 0 0 4px" : 4,
          display: "block",
        },
        onMouseMove: handleMouseMove,
        onMouseDown: handleMouseDown,
        onMouseUp: handleMouseUp,
        onMouseLeave: () => {
          dragRef.current = null;
          setDragging(false);
          setTooltip(null);
        },
        onWheel: handleWheel,
        onClick: handleClick,
      }),
      // Tooltip
      tooltip
        ? React.createElement(
            "div",
            {
              style: {
                position: "absolute",
                left: tooltip.screenX + 16,
                top: tooltip.screenY - 16,
                background: "white",
                border: "1px solid #ccc",
                borderRadius: 6,
                padding: 8,
                boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
                pointerEvents: "none",
                zIndex: 10,
                maxWidth: 240,
                fontSize: 12,
              },
            },
            React.createElement(AuthImg, {
              src: tooltip.thumbnailUrl,
              alt: "thumbnail",
              style: {
                width: 96,
                height: 96,
                objectFit: "contain",
                display: "block",
                marginBottom: 4,
                imageRendering: "pixelated",
              },
            }),
            React.createElement("div", null, `#${tooltip.point.index}`),
            tooltip.point.label
              ? React.createElement(
                  "div",
                  null,
                  `Label: ${tooltip.point.label}`,
                )
              : null,

          )
        : null,
      ),
      // Detail panel — flex sibling of canvas
      selected
        ? React.createElement(
            "div",
            {
              style: {
                width: PANEL_WIDTH,
                flexShrink: 0,
                height: CANVAS_HEIGHT,
                background: "white",
                border: "1px solid #ccc",
                borderLeft: "none",
                borderRadius: "0 4px 4px 0",
                padding: "12px 16px",
                boxShadow: "2px 0 8px rgba(0,0,0,0.08)",
                overflowY: "auto" as const,
                fontSize: 13,
                boxSizing: "border-box" as const,
                position: "relative" as const,
              },
            },
            // Close button
            React.createElement(
              "button",
              {
                onClick: () => { setSelected(null); setSaveError(null); },
                style: {
                  position: "absolute" as const,
                  top: 6,
                  right: 8,
                  background: "none",
                  border: "none",
                  fontSize: 18,
                  cursor: "pointer",
                  color: "#999",
                  lineHeight: 1,
                },
              },
              "\u00D7",
            ),
            // Thumbnail
            React.createElement(AuthImg, {
              src: selected.thumbnailUrl,
              alt: "thumbnail",
              style: {
                width: "100%",
                maxHeight: 180,
                objectFit: "contain",
                display: "block",
                marginBottom: 10,
                imageRendering: "pixelated" as const,
                background: "#f5f5f5",
                borderRadius: 4,
              },
            }),
            // Point info row
            React.createElement(
              "div",
              {
                style: {
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "baseline",
                  marginBottom: 4,
                  color: "#333",
                },
              },
              React.createElement("span", null, `#${selected.point.index}`),
              selected.point.label
                ? React.createElement(
                    "span",
                    { style: { color: getLabelColor(selected.point.label, uniqueLabels), fontWeight: 500 } },
                    selected.point.label,
                  )
                : null,
            ),
            // Source link
            selected.point.path
              ? React.createElement(
                  "a",
                  {
                    href: `/ui/browse/${selected.point.path}`,
                    target: "_blank",
                    rel: "noopener",
                    style: {
                      display: "block",
                      color: "#1976d2",
                      fontSize: 11,
                      marginBottom: 12,
                      wordBreak: "break-all" as const,
                    },
                  },
                  "View source \u2192",
                )
              : React.createElement("div", { style: { marginBottom: 12 } }),
            // User label
            React.createElement(
              "label",
              { style: { display: "block", fontSize: 11, color: "#777", marginBottom: 2 } },
              "User label",
            ),
            React.createElement("input", {
              type: "text",
              value: selected.userLabel,
              maxLength: USER_LABEL_MAX_LEN,
              onChange: (e: React.ChangeEvent<HTMLInputElement>) => {
                const val = e.target.value;
                setSelected((prev) => (prev ? { ...prev, userLabel: val } : prev));
              },
              onKeyDown: (e: React.KeyboardEvent) => {
                if (e.key === "Enter") handleSave();
              },
              style: {
                width: "100%",
                padding: "4px 8px",
                border: "1px solid #ccc",
                borderRadius: 4,
                fontSize: 13,
                marginBottom: 10,
                boxSizing: "border-box" as const,
              },
            }),
            // Note
            React.createElement(
              "label",
              { style: { display: "block", fontSize: 11, color: "#777", marginBottom: 2 } },
              "Note",
            ),
            React.createElement("textarea", {
              value: selected.note,
              maxLength: NOTES_MAX_LEN,
              onChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => {
                const val = e.target.value;
                setSelected((prev) => (prev ? { ...prev, note: val } : prev));
              },
              style: {
                width: "100%",
                minHeight: 72,
                padding: "4px 8px",
                border: "1px solid #ccc",
                borderRadius: 4,
                fontSize: 13,
                resize: "vertical" as const,
                boxSizing: "border-box" as const,
              },
            }),
            // Save button (highlighted when dirty)
            React.createElement(
              "button",
              {
                onClick: handleSave,
                disabled: selected.saving || !isDirty,
                style: {
                  marginTop: 8,
                  width: "100%",
                  padding: "6px 0",
                  fontSize: 13,
                  border: isDirty ? "1px solid #1976d2" : "1px solid #ccc",
                  borderRadius: 4,
                  background: selected.saving
                    ? "#eee"
                    : isDirty
                      ? "#1976d2"
                      : "#f5f5f5",
                  color: isDirty && !selected.saving ? "#fff" : "#333",
                  cursor: selected.saving || !isDirty ? "default" : "pointer",
                  transition: "background 0.2s, border-color 0.2s, color 0.2s",
                },
              },
              selected.saving ? "Saving..." : "Save",
            ),
            // Save error
            saveError
              ? React.createElement(
                  "div",
                  { style: { marginTop: 6, fontSize: 11, color: "#c00" } },
                  saveError,
                )
              : null,
          )
        : null,
      // Chat panel — flex sibling of canvas (mutual exclusion with detail panel)
      chatOpen && !selected
        ? React.createElement(
            "div",
            {
              style: {
                width: PANEL_WIDTH,
                flexShrink: 0,
                height: CANVAS_HEIGHT,
                background: "white",
                border: "1px solid #ccc",
                borderLeft: "none",
                borderRadius: "0 4px 4px 0",
                boxShadow: "2px 0 8px rgba(0,0,0,0.08)",
                fontSize: 13,
                boxSizing: "border-box" as const,
                display: "flex",
                flexDirection: "column" as const,
              },
            },
            // Chat header
            React.createElement(
              "div",
              {
                style: {
                  padding: "10px 16px",
                  borderBottom: "1px solid #eee",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  flexShrink: 0,
                },
              },
              React.createElement(
                "span",
                { style: { fontWeight: 500, color: "#333" } },
                "Chat with data",
              ),
              React.createElement(
                "div",
                { style: { display: "flex", gap: 8, alignItems: "center" } },
                // Clear history button
                chatMessages.length > 0
                  ? React.createElement(
                      "button",
                      {
                        onClick: clearChatHistory,
                        title: "Clear conversation",
                        style: {
                          background: "none",
                          border: "none",
                          fontSize: 12,
                          color: "#999",
                          cursor: "pointer",
                          padding: 0,
                        },
                      },
                      "Clear",
                    )
                  : null,
                // Close button
                React.createElement(
                  "button",
                  {
                    onClick: () => setChatOpen(false),
                    style: {
                      background: "none",
                      border: "none",
                      fontSize: 18,
                      cursor: "pointer",
                      color: "#999",
                      lineHeight: 1,
                      padding: 0,
                    },
                  },
                  "\u00D7",
                ),
              ),
            ),
            // Message list
            React.createElement(
              "div",
              {
                ref: chatListRef,
                style: {
                  flex: 1,
                  overflowY: "auto" as const,
                  padding: "8px 16px",
                },
              },
              chatMessages.length === 0
                ? React.createElement(
                    "div",
                    {
                      style: {
                        color: "#aaa",
                        fontSize: 12,
                        textAlign: "center" as const,
                        marginTop: 40,
                      },
                    },
                    "Ask a question about your dataset",
                  )
                : null,
              ...chatMessages.map((msg) =>
                React.createElement(
                  "div",
                  {
                    key: msg.id,
                    style: {
                      marginBottom: 10,
                      display: "flex",
                      flexDirection: "column" as const,
                      alignItems: msg.role === "user" ? "flex-end" : "flex-start",
                    },
                  },
                  React.createElement(
                    "div",
                    {
                      style: {
                        background: msg.role === "user" ? "#1976d2" : "#f0f0f0",
                        color: msg.role === "user" ? "#fff" : "#333",
                        padding: "6px 10px",
                        borderRadius: msg.role === "user" ? "12px 12px 2px 12px" : "12px 12px 12px 2px",
                        maxWidth: "85%",
                        fontSize: 12,
                        lineHeight: "1.4",
                        wordBreak: "break-word" as const,
                      },
                    },
                    msg.content,
                  ),
                ),
              ),
              // Typing indicator while sending
              chatSending
                ? React.createElement(
                    "div",
                    {
                      style: {
                        display: "flex",
                        alignItems: "flex-start",
                        marginBottom: 10,
                      },
                    },
                    React.createElement(
                      "div",
                      {
                        style: {
                          background: "#f0f0f0",
                          padding: "6px 10px",
                          borderRadius: "12px 12px 12px 2px",
                          fontSize: 12,
                          color: "#999",
                        },
                      },
                      "Thinking...",
                    ),
                  )
                : null,
            ),
            // Input area
            React.createElement(
              "div",
              {
                style: {
                  padding: "8px 12px",
                  borderTop: "1px solid #eee",
                  display: "flex",
                  gap: 6,
                  flexShrink: 0,
                },
              },
              React.createElement("input", {
                type: "text",
                value: chatInput,
                placeholder: "Ask about the data...",
                onChange: (e: React.ChangeEvent<HTMLInputElement>) =>
                  setChatInput(e.target.value),
                onKeyDown: (e: React.KeyboardEvent) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendChatMessage();
                  }
                },
                disabled: chatSending,
                style: {
                  flex: 1,
                  padding: "6px 10px",
                  border: "1px solid #ccc",
                  borderRadius: 16,
                  fontSize: 12,
                  outline: "none",
                  boxSizing: "border-box" as const,
                },
              }),
              React.createElement(
                "button",
                {
                  onClick: sendChatMessage,
                  disabled: chatSending || !chatInput.trim(),
                  style: {
                    background: chatInput.trim() && !chatSending ? "#1976d2" : "#e0e0e0",
                    color: chatInput.trim() && !chatSending ? "#fff" : "#999",
                    border: "none",
                    borderRadius: "50%",
                    width: 30,
                    height: 30,
                    cursor: chatInput.trim() && !chatSending ? "pointer" : "default",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexShrink: 0,
                    transition: "background 0.2s",
                  },
                },
                // Send arrow icon
                React.createElement(
                  "svg",
                  {
                    width: 14,
                    height: 14,
                    viewBox: "0 0 24 24",
                    fill: "none",
                    stroke: "currentColor",
                    strokeWidth: 2,
                    strokeLinecap: "round",
                    strokeLinejoin: "round",
                  },
                  React.createElement("line", { x1: 22, y1: 2, x2: 11, y2: 13 }),
                  React.createElement("polygon", { points: "22 2 15 22 11 13 2 9 22 2" }),
                ),
              ),
            ),
          )
        : null,
      // Lasso selection panel — same position as detail/chat panels
      lassoSummary && !selected && !chatOpen
        ? React.createElement(
            "div",
            {
              style: {
                width: PANEL_WIDTH,
                flexShrink: 0,
                height: CANVAS_HEIGHT,
                background: "white",
                border: "1px solid #ccc",
                borderLeft: "none",
                borderRadius: "0 4px 4px 0",
                boxShadow: "2px 0 8px rgba(0,0,0,0.08)",
                fontSize: 13,
                boxSizing: "border-box" as const,
                display: "flex",
                flexDirection: "column" as const,
              },
            },
            // Header
            React.createElement(
              "div",
              {
                style: {
                  padding: "10px 16px",
                  borderBottom: "1px solid #eee",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  flexShrink: 0,
                },
              },
              React.createElement(
                "span",
                { style: { fontWeight: 500, color: "#333" } },
                `${lassoSummary.count} points selected`,
              ),
              React.createElement(
                "button",
                {
                  onClick: () => { setLassoSelected(new Set()); setLassoPath([]); },
                  style: {
                    background: "none",
                    border: "none",
                    fontSize: 12,
                    color: "#999",
                    cursor: "pointer",
                    padding: 0,
                  },
                },
                "Clear",
              ),
            ),
            // Label breakdown
            React.createElement(
              "div",
              {
                style: {
                  padding: "10px 16px",
                  borderBottom: "1px solid #eee",
                  flexShrink: 0,
                },
              },
              React.createElement(
                "div",
                { style: { fontSize: 11, color: "#777", marginBottom: 6 } },
                "Labels",
              ),
              ...Object.entries(lassoSummary.labelCounts)
                .sort((a, b) => b[1] - a[1])
                .map(([label, count]) =>
                  React.createElement(
                    "div",
                    {
                      key: label,
                      style: {
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        marginBottom: 3,
                        fontSize: 12,
                      },
                    },
                    React.createElement("div", {
                      style: {
                        width: 8,
                        height: 8,
                        borderRadius: "50%",
                        backgroundColor: label === "(unlabeled)" ? "#888" : getLabelColor(label, uniqueLabels),
                        flexShrink: 0,
                      },
                    }),
                    React.createElement("span", { style: { flex: 1 } }, label),
                    React.createElement("span", { style: { color: "#999" } }, String(count)),
                  ),
                ),
            ),
            // Point list (scrollable)
            React.createElement(
              "div",
              {
                style: {
                  flex: 1,
                  overflowY: "auto" as const,
                  padding: "8px 16px",
                },
              },
              React.createElement(
                "div",
                { style: { fontSize: 11, color: "#777", marginBottom: 6 } },
                "Points",
              ),
              ...lassoSummary.points.map((p) =>
                React.createElement(
                  "div",
                  {
                    key: p.index,
                    onClick: () => {
                      setToolMode("pan");
                      openPoint(p);
                    },
                    style: {
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                      padding: "3px 4px",
                      borderRadius: 3,
                      cursor: "pointer",
                      fontSize: 12,
                      marginBottom: 1,
                    },
                    onMouseEnter: (e: React.MouseEvent<HTMLDivElement>) => {
                      (e.currentTarget as HTMLDivElement).style.background = "#f5f5f5";
                    },
                    onMouseLeave: (e: React.MouseEvent<HTMLDivElement>) => {
                      (e.currentTarget as HTMLDivElement).style.background = "none";
                    },
                  },
                  React.createElement("span", { style: { color: "#1976d2" } }, `#${p.index}`),
                  p.label
                    ? React.createElement(
                        "span",
                        { style: { color: getLabelColor(p.label, uniqueLabels), fontSize: 11 } },
                        p.label,
                      )
                    : null,
                ),
              ),
            ),
          )
        : null,
    ),
    // ── Color legend / filter controls ─────────────────────────────────────
    colorBy === "label"
      ? // Categorical legend: clickable swatches
        uniqueLabels.length > 0
          ? React.createElement(
              "div",
              {
                style: {
                  marginTop: 10,
                  display: "flex",
                  flexWrap: "wrap" as const,
                  gap: 8,
                  fontSize: 12,
                  alignItems: "center",
                },
              },
              React.createElement("span", { style: { color: "#777", fontSize: 11 } }, "Toggle:"),
              ...uniqueLabels.map((label) => {
                const hidden = hiddenLabels.has(label);
                return React.createElement(
                  "button",
                  {
                    key: label,
                    onClick: () => {
                      setHiddenLabels((prev) => {
                        const next = new Set(prev);
                        if (next.has(label)) next.delete(label);
                        else next.add(label);
                        return next;
                      });
                    },
                    title: hidden ? `Show ${label}` : `Hide ${label}`,
                    style: {
                      display: "flex",
                      alignItems: "center",
                      gap: 5,
                      padding: "3px 8px",
                      border: "1px solid #ccc",
                      borderRadius: 12,
                      background: hidden ? "#f5f5f5" : "white",
                      cursor: "pointer",
                      fontSize: 12,
                      opacity: hidden ? 0.45 : 1,
                      transition: "opacity 0.15s",
                    },
                  },
                  React.createElement("div", {
                    style: {
                      width: 9,
                      height: 9,
                      borderRadius: "50%",
                      backgroundColor: getLabelColor(label, uniqueLabels),
                      flexShrink: 0,
                    },
                  }),
                  React.createElement("span", null, label),
                );
              }),
              hiddenLabels.size > 0
                ? React.createElement(
                    "button",
                    {
                      onClick: () => setHiddenLabels(new Set()),
                      style: {
                        fontSize: 11,
                        color: "#1976d2",
                        background: "none",
                        border: "none",
                        cursor: "pointer",
                        padding: "0 4px",
                        textDecoration: "underline",
                      },
                    },
                    "Show all",
                  )
                : null,
            )
          : null
      : // Continuous param: colorbar + dual-handle range slider
        React.createElement(
          "div",
          { style: { marginTop: 10 } },
          React.createElement(RangeSlider, {
            min: paramMin,
            max: paramMax,
            value: paramRange ?? [paramMin, paramMax],
            onChange: (lo: number, hi: number) => {
              if (lo <= paramMin + 1e-9 && hi >= paramMax - 1e-9) setParamRange(null);
              else setParamRange([lo, hi]);
            },
            units: paramSpecs[colorBy]?.units,
            precision: paramSpecs[colorBy]?.precision ?? 2,
          }),
        ),
    // Instructions
    React.createElement(
      "div",
      { style: { marginTop: 6, fontSize: 11, color: "#999" } },
      "Scroll to zoom. Pan mode: drag to pan, click point to inspect. Lasso mode: draw to select points.",
    ),
  );
}

export default EmbeddingScatter;
