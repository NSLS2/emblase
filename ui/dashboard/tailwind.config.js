/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bnl: {
          blue: '#003087',
          gold: '#C9A84C',
          lightblue: '#0066CC',
        },
        amsc: {
          blue: '#1B4F8A',
          cyan: '#00AEEF',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
}
