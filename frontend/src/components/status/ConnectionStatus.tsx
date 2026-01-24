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
              ? 'bg-green-400 shadow-[0_0_8px_rgba(74,222,128,0.6)]' 
              : 'bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.6)]'
          }`}
        />
        <span className="text-xs font-medium text-gray-300">
          {isConnected ? 'Connected' : 'Offline'}
        </span>
      </div>
      
      {/* Patient count when connected */}
      {isConnected && patientCount > 0 && (
        <span className="text-xs text-gray-400 border-l border-gray-600 pl-2">
          {patientCount} patient{patientCount !== 1 ? 's' : ''}
        </span>
      )}
    </div>
  )
}

export default ConnectionStatus
