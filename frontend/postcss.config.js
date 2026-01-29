// frontend/postcss.config.js
export default {
  plugins: {
    "@tailwindcss/postcss": {}, // 기존 'tailwindcss' 대신 이걸로 교체!
    "autoprefixer": {},
  },
}