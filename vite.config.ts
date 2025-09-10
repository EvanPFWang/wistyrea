import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  build: {
    target: 'es2022', // remove minify: 'terser' and terserOptions: { ... }
    rollupOptions: {
      output: {
        manualChunks: {
          'mural-controller': ['./src/CrownMuralController.ts']
        }
      }
    }
  },esbuild: {drop: ['debugger'],pure: ['console.log']},
  server: {
    port: 3000
  }
});
