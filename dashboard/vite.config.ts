import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The production build is served by the FastAPI app (backend/regressionlens/api/static).
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../backend/regressionlens/api/static",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: {
          charts: ["recharts"],
          vendor: ["react", "react-dom", "react-router-dom", "@tanstack/react-query"],
        },
      },
    },
  },
  server: { proxy: { "/api": process.env.RLENS_API ?? "http://localhost:8080" } },
});
