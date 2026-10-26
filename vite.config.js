import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: '0.0.0.0',
    port: 5173,
    headers: {
      // Headers for WebGPU & SharedArrayBuffer if required by high-perf Wasm runtimes
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless'
    }
  },
  optimizeDeps: {
    exclude: ['@mlc-ai/web-llm']
  }
});
