/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: { sans: ['Inter', 'system-ui', 'sans-serif'] },
      colors: {
        gray: {
          750: '#2d3748',   /* between 700 and 800 for dark sidebar hover */
          950: '#0a0f1a',
        },
      },
    },
  },
  safelist: [
    'bg-blue-500/10',   'border-blue-500',   'text-blue-400',   'bg-blue-500/20',   'accent-blue-500',
    'bg-emerald-500/10','border-emerald-500','text-emerald-400','bg-emerald-500/20','accent-emerald-500',
    'bg-violet-500/10', 'border-violet-500', 'text-violet-400', 'bg-violet-500/20', 'accent-violet-500',
    'bg-amber-50',  'border-amber-100',
    'bg-emerald-50','border-emerald-100',
    'bg-violet-50', 'border-violet-100',
  ],
  plugins: [],
}
