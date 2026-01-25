import { useState } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { Header } from './Header'
import { GodModePanel } from '../godmode'
import { useGodMode } from '../../stores'

export function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const godModeEnabled = useGodMode()
  const location = useLocation()
  
  // Only show sidebar toggle on ward view
  const showSidebarToggle = location.pathname === '/' && godModeEnabled

  return (
    <div className="min-h-screen bg-white">
      <Header
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
        showSidebarToggle={showSidebarToggle}
        sidebarOpen={sidebarOpen}
      />
      <div className="flex">
        {/* Main content */}
        <main className={`flex-1 transition-all duration-300 ${sidebarOpen && showSidebarToggle ? 'mr-80' : ''}`}>
          <div className="container mx-auto px-4 py-6">
            <Outlet />
          </div>
        </main>

        {/* Sidebar - God Mode Panel */}
        {showSidebarToggle && (
          <aside
            className={`
              fixed right-0 top-16 bottom-0 w-80 bg-gray-100 border-l border-gray-300
              transform transition-transform duration-300 overflow-y-auto shadow-lg
              ${sidebarOpen ? 'translate-x-0' : 'translate-x-full'}
            `}
          >
            <div className="p-4">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">God Mode</h2>
                <button
                  onClick={() => setSidebarOpen(false)}
                  className="p-1 hover:bg-gray-200 rounded text-gray-500 hover:text-gray-900"
                >
                  ✕
                </button>
              </div>
              <GodModePanel />
            </div>
          </aside>
        )}
      </div>
    </div>
  )
}
