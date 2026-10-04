import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The placement API (api/server.py) listens on 8765; proxy it so the app can
// call /api in development and preview.
const api = { '/api': 'http://127.0.0.1:8765' }

export default defineConfig({
  plugins: [react()],
  server: { proxy: api },
  preview: { proxy: api },
})
