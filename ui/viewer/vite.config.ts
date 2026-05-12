import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  build: {
    lib: {
      entry: "src/index.tsx",
      name: "EmblaseViewer",
      formats: ["iife"],
      fileName: () => "main.js",
    },
    rollupOptions: {
      external: ["react", "react-dom"],
      output: {
        globals: {
          react: "React",
          "react-dom": "ReactDOM",
        },
      },
    },
    outDir: resolve(__dirname, "../../src/emblase/tiled/static"),
    emptyOutDir: false,
    minify: true,
  },
  esbuild: {
    jsxFactory: "React.createElement",
    jsxFragment: "React.Fragment",
  },
  define: {
    "process.env.NODE_ENV": '"production"',
  },
});
