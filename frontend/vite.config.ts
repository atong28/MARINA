/// <reference types="vitest" />
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backendPort = env.BACKEND_PORT || '5000'

  return {
    plugins: [react()],
    // ES-module workers, so the fragment-drawing worker can import RDKit.js.
    worker: { format: 'es' },
    test: {
      // Node environment: the suite covers pure logic (scales, formatting,
      // store reducers), not component rendering, so no DOM shim is needed.
      environment: 'node',
      include: ['src/**/*.test.ts'],
    },
    server: {
      proxy: {
        '/api': {
          target: `http://localhost:${backendPort}`,
          changeOrigin: true,
        },
      },
    },
  }
})
