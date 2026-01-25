import { Link } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { ConnectionStatus } from '../status/ConnectionStatus'
import { SimulationControls } from '../status/SimulationControls'
import { LanguageToggle } from '../common/LanguageToggle'

interface HeaderProps {
  onToggleSidebar?: () => void
  showSidebarToggle?: boolean
  sidebarOpen?: boolean
}

export function Header({ onToggleSidebar, showSidebarToggle, sidebarOpen }: HeaderProps) {
  const { t } = useTranslation()
  
  return (
    <header className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-50">
      <div className="container mx-auto px-4 py-3 flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2">
          <span className="text-xl font-bold text-blue-600">
            {t('app.title')}
          </span>
          <span className="text-sm text-gray-500">{t('app.version')}</span>
        </Link>

        {/* Controls */}
        <div className="flex items-center gap-4">
          <SimulationControls />
          <div className="h-6 w-px bg-gray-300" />
          <LanguageToggle />
          <ConnectionStatus />

          {/* God Mode Toggle */}
          {showSidebarToggle && onToggleSidebar && (
            <>
              <div className="h-6 w-px bg-gray-300" />
              <button
                onClick={onToggleSidebar}
                className={`
                  px-3 py-1.5 rounded-lg text-sm font-medium transition-colors
                  ${sidebarOpen
                    ? 'bg-purple-600 text-white'
                    : 'bg-purple-100 text-purple-700 hover:bg-purple-200'}
                `}
                title="Toggle God Mode Panel"
              >
                ⚡ {t('simulation.godMode', 'God Mode')}
              </button>
            </>
          )}
        </div>
      </div>
    </header>
  )
}
