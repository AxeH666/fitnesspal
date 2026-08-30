import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: "autoUpdate",
      manifest: {
        name: "Barbarik Fitness Pal",
        short_name: "Barbarik",
        display: "standalone",
        theme_color: "#171717",
        background_color: "#ffffff"
      }
    })
  ]
});

