import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

const repoRoot = fileURLToPath(new URL('..', import.meta.url));

export default defineConfig({
  plugins: [react()],
  base: './',
  resolve: {
    alias: {
      // The road network is imported straight from the repo-level single source of
      // truth. No copy, no duplicate list -- change config/segments.json and the UI
      // changes with it.
      '@config': fileURLToPath(new URL('../config', import.meta.url)),
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: 5173,
    // Needed because config/segments.json lives above the Vite root.
    fs: { allow: [repoRoot] },
  },
  build: {
    outDir: 'dist',
    // Leaflet + Recharts are both chunky; splitting keeps the first paint quick on a
    // phone, which matters because evaluators will open this on one.
    rollupOptions: {
      output: {
        manualChunks: {
          mapping: ['leaflet', 'react-leaflet'],
          charts: ['recharts'],
          pdf: ['jspdf', 'html2canvas'],
        },
      },
    },
    chunkSizeWarningLimit: 900,
  },
});
