import React from 'react'
import { usePatientStore, useUIStore } from '../../stores'

interface ConnectionStatusProps {
  className?: string
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({ className = '' }) => {
  const isConnected = usePatientStore(state => state.connected)
  const patientCount = usePatientStore(state => state.patients.size)
  const showConnectionStatus = useUIStore(state => state.showConnectionStatus)
  
  if (!showConnectionStatus) return null
  
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      {/* Connection indicator */}
      <div className="flex items-center gap-1.5">
        <div
          className={`w-2 h-2 rounded-full ${
            isConnected
              ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]'
              : 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]'
          }`}
        />
        <span className="text-xs font-medium text-gray-700">
          {isConnected ? 'Connected' : 'Offline'}
        </span>
      </div>

      {/* Patient count when connected */}
      {isConnected && patientCount > 0 && (
        <span className="text-xs text-gray-500 border-l border-gray-300 pl-2">
          {patientCount} patient{patientCount !== 1 ? 's' : ''}
        </span>
      )}
    </div>
  )
}

export default ConnectionStatus
