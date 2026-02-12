import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico', 'apple-touch-icon.png', 'site.webmanifest'],
      manifest: {
        name: 'Connect·Hands',
        short_name: 'CH',
        description: '실시간 수어 AI 통역 서비스',
        theme_color: '#FFF5F8',
        background_color: '#FFF5F8',
        display: 'standalone',
        icons: [
          {/* PWA Generator에서 만든 파일명과 일치시켜야 함*/
            src: 'web-app-manifest-192x192.png', 
            sizes: '192x192',
            type: 'image/png'
          },
          {
            src: 'web-app-manifest-512x512.png',
            sizes: '512x512',
            type: 'image/png'
          },
          {
            src: 'web-app-manifest-512x512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any maskable'
          }
        ]
      }
    })
  ]
});