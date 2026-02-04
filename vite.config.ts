import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

const rootDir = resolve(__dirname, "frontend");

export default defineConfig({
  root: rootDir,
  base: "/assets/",
  plugins: [react()],
  build: {
    outDir: resolve(__dirname, "assets"),
    emptyOutDir: false,
    assetsDir: "chunks",
    rollupOptions: {
      input: {
        app: resolve(rootDir, "src/main.tsx"),
        memory: resolve(rootDir, "src/memory.tsx"),
        "scheduler/scheduler": resolve(rootDir, "src/scheduler.tsx")
      },
      output: {
        entryFileNames: "[name].js",
        chunkFileNames: "chunks/[name]-[hash].js",
        assetFileNames: "chunks/[name]-[hash][extname]"
      }
    }
  }
});
