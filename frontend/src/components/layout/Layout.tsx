import { useState, useEffect } from 'react'
import { Outlet } from 'react-router-dom'
import { Header } from './Header'
import { GodModePanel } from '../godmode'
import { useGodMode, useUIStore } from '../../stores'

export function Layout() {
  const [panelCollapsed, setPanelCollapsed] = useState(false)
  const godModeEnabled = useGodMode()
  const toggleGodMode = useUIStore(state => state.toggleGodMode)

  useEffect(() => {
    if (!godModeEnabled) {
      toggleGodMode()
    }
  }, [godModeEnabled, toggleGodMode])

  const showPanel = godModeEnabled

  return (
    <div className="min-h-screen bg-white">
      <Header
        onToggleSidebar={() => setPanelCollapsed(prev => !prev)}
        showSidebarToggle={showPanel}
        sidebarOpen={!panelCollapsed}
      />
      <div className="flex">
        {/* Main content */}
        <main className="flex-1">
          <div className="container mx-auto px-4 py-6">
            <Outlet />
          </div>
        </main>
      </div>

      {showPanel && (
        <div className="fixed bottom-6 right-6 z-50 pointer-events-none">
          {panelCollapsed ? (
            <button
              onClick={() => setPanelCollapsed(false)}
              className="pointer-events-auto flex items-center gap-2 px-4 py-2 rounded-full bg-purple-600 text-white shadow-lg hover:bg-purple-700 transition-colors"
              title="Show God Mode"
            >
              ⚡ God Mode
            </button>
          ) : (
            <div className="pointer-events-auto w-[22rem] max-w-[calc(100vw-3rem)]">
              <div className="bg-white border border-gray-300 rounded-2xl shadow-2xl overflow-hidden">
                <div className="flex items-center justify-between px-4 py-3 bg-gray-100 border-b border-gray-200">
                  <h2 className="text-base font-semibold text-gray-900">God Mode</h2>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setPanelCollapsed(true)}
                      className="px-2 py-1 text-xs rounded bg-gray-200 hover:bg-gray-300 text-gray-700"
                      title="Minimize God Mode"
                    >
                      Hide
                    </button>
                  </div>
                </div>
                <div className="max-h-[70vh] overflow-y-auto p-4 bg-white">
                  <GodModePanel />
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
