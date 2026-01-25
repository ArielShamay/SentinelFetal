import { create } from 'zustand'
import { persist } from 'zustand/middleware'

type Language = 'he' | 'en'
type Theme = 'light' | 'dark' | 'system'

interface UIState {
  // Preferences
  theme: Theme
  soundEnabled: boolean
  language: Language

  // Layout
  sidebarOpen: boolean
  gridColumns: number
  showMiniCharts: boolean
  showConnectionStatus: boolean

  // God Mode
  godModeEnabled: boolean

  // Actions
  setTheme: (theme: Theme) => void
  toggleSound: () => void
  setLanguage: (lang: Language) => void
  setSidebarOpen: (open: boolean) => void
  setGridColumns: (cols: number) => void
  toggleMiniCharts: () => void
  toggleGodMode: () => void
  toggleConnectionStatus: () => void
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      // Initial state
      theme: 'light',
      soundEnabled: true,
      language: 'he',
      sidebarOpen: true,
      gridColumns: 4,
      showMiniCharts: true,
      showConnectionStatus: true,
      godModeEnabled: true,

      // Actions
      setTheme: (theme) => set({ theme }),
      toggleSound: () => set((s) => ({ soundEnabled: !s.soundEnabled })),
      setLanguage: (language) => set({ language }),
      setSidebarOpen: (sidebarOpen) => set({ sidebarOpen }),
      setGridColumns: (gridColumns) => set({ gridColumns }),
      toggleMiniCharts: () => set((s) => ({ showMiniCharts: !s.showMiniCharts })),
      toggleGodMode: () => set((s) => ({ godModeEnabled: !s.godModeEnabled })),
      toggleConnectionStatus: () => set((s) => ({ showConnectionStatus: !s.showConnectionStatus })),
    }),
    { name: 'sentinel-ui-preferences' }
  )
)

// Selector hooks
export const useTheme = () => useUIStore((s) => s.theme)
export const useLanguage = () => useUIStore((s) => s.language)
export const useGridColumns = () => useUIStore((s) => s.gridColumns)
export const useGodMode = () => useUIStore((s) => s.godModeEnabled)
