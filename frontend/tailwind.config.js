/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // SentinelFetal clinical colors
        'cat-normal': '#28a745',
        'cat-intermediate': '#fd7e14',
        'cat-pathological': '#dc3545',
        'fhr-blue': '#1E90FF',
        'uc-orange': '#FF8C00',
        // Gray scale adjustments for dark theme
        gray: {
          750: '#2d3748',
          850: '#1a202c',
          950: '#0d1117'
        }
      },
      animation: {
        'pulse-subtle': 'pulse-subtle 2s ease-in-out infinite',
      },
      keyframes: {
        'pulse-subtle': {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.85 },
        },
      },
    },
  },
  plugins: [],
}
