import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 10 minutes — large synth / eval batches against a local model can
// hold the connection open for several minutes. The default proxy
// timeout used to sever the request and surface as "network error"
// on the frontend while the GPU was still actively working.
const PROXY_TIMEOUT_MS = 10 * 60 * 1000

// Proxy target is env-configurable so an isolated stack (E2E in CI, or a
// second backend locally) can point the dev server at a non-default port.
const API_PROXY_TARGET = process.env.VITE_API_PROXY_TARGET || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: API_PROXY_TARGET,
        changeOrigin: true,
        ws: true,
        timeout: PROXY_TIMEOUT_MS,
        proxyTimeout: PROXY_TIMEOUT_MS,
      },
    },
  },
})
