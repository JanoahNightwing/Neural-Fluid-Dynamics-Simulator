import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The production build is written into the Python package, so `python -m nfds.app`
// serves it without Node installed. `npm run dev` proxies to a running backend.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: { outDir: "../src/nfds/app/static", emptyOutDir: true },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
