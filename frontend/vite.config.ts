import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
      '/verify': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/graph': 'http://localhost:8000',
      '/blockchain': 'http://localhost:8000',
      '/verifications': 'http://localhost:8000',
      '/alerts': 'http://localhost:8000',
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
})
