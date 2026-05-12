import EmbeddingScatter from "./embedding-scatter";

// Register the component in the global spec views registry.
// Tiled's browse.tsx DynamicSpecView loader will pick it up from here.
const w = window as any;
if (!w.__TILED_SPEC_VIEWS__) {
  w.__TILED_SPEC_VIEWS__ = {};
}
w.__TILED_SPEC_VIEWS__["LatentSpaceEmbedding"] = EmbeddingScatter;
