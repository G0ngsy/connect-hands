// tailwind.config.js
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}", // ts, tsx가 포함되어야 합니다!
  ],
  theme: {
    extend: {
       keyframes: {
        morph: {
          '0%, 100%': { borderRadius: '30% 70% 70% 30% / 30% 30% 70% 70%' },
          '50%': { borderRadius: '70% 30% 30% 70% / 70% 70% 30% 30%' },
        }
      },
      colors: {
        pink: {
          100: '#FFD6E6', 200: '#FFB3D1', 400: '#FF66B2', 500: '#FF3399', 600: '#FF007F',
        },
        mint: {
          100: '#D6F5E6', 400: '#4AD799', 500: '#00CC66',
        },
        lime: {
          400: '#A3D14A',
        }
      }

      


    },
  },
  plugins: [],
}