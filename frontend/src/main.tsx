import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import './styles/utilities.css' // after index.css so a utility wins a tie on specificity
import App from './App'
import { initTheme } from './stores/themeStore'

// Apply the saved theme before the first paint so the page never flashes.
initTheme()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
