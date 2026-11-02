import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev: `npm run dev` ở cổng 5173, proxy /v1 về backend để cookie phiên cùng origin (giống nginx khi deploy).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/v1": { target: process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000", changeOrigin: false },
      "/health": { target: process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000" },
    },
  },
  build: { outDir: "dist", sourcemap: false },
});
