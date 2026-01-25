import React, { useState, useCallback, useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { useSimulation } from '../../hooks'
import { usePatientStore } from '../../stores'

interface SimulationControlsProps {
  className?: string
  compact?: boolean
}

export const SimulationControls: React.FC<SimulationControlsProps> = ({
  className = '',
  compact = false
}) => {
  const { t } = useTranslation()
  const {
    loading,
    start,
    stop,
    pause,
    resume,
    reset,
    setPatientCount: updatePatientCount
  } = useSimulation()

  const simulationRunning = usePatientStore(state => state.simulationRunning)
  const simulationPaused = usePatientStore(state => state.simulationPaused)
  const currentPatientCount = usePatientStore(state => state.patientCount)

  const [patientCount, setLocalPatientCount] = useState(currentPatientCount || 10)
  const debounceTimerRef = useRef<number>()

  const isRunning = simulationRunning && !simulationPaused
  const isPaused = simulationRunning && simulationPaused
  const isStopped = !simulationRunning

  // Handle patient count change with debouncing
  const handlePatientCountChange = useCallback(async (newCount: number) => {
    setLocalPatientCount(newCount)
    
    // Clear existing timer
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current)
    }
    
    // Debounce API call by 500ms (waits for user to stop sliding)
    debounceTimerRef.current = window.setTimeout(async () => {
      await updatePatientCount(newCount)
    }, 500)
  }, [updatePatientCount])
  
  // Cleanup debounce timer on unmount
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
      }
    }
  }, [])

  // Start simulation with selected patient count
  const handleStart = useCallback(async () => {
    // First set the patient count, then start
    await updatePatientCount(patientCount)
    await start()
  }, [patientCount, updatePatientCount, start])
  
  const buttonBase = compact 
    ? 'px-2 py-1 text-xs rounded'
    : 'px-3 py-1.5 text-sm rounded-md'
  
  const disabledClass = 'opacity-50 cursor-not-allowed'
  
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      {/* Patient Count Selector - always visible when not compact */}
      {!compact && (
        <div className="flex items-center gap-2 bg-gray-100 border border-gray-300 rounded-md px-3 py-1.5">
          <label className="text-xs text-gray-600 whitespace-nowrap">
            Patients:
          </label>
          <input
            type="range"
            min="1"
            max="20"
            value={patientCount}
            onChange={(e) => handlePatientCountChange(parseInt(e.target.value))}
            className="w-20 h-1.5 bg-gray-300 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
          <span className="text-sm font-medium text-gray-900 w-6 text-center">
            {patientCount}
          </span>
        </div>
      )}

      {/* Start/Resume Button */}
      {(isStopped || isPaused) && (
        <button
          onClick={() => isPaused ? resume() : handleStart()}
          disabled={loading}
          className={`${buttonBase} bg-green-600 hover:bg-green-700 text-white font-medium transition-colors ${
            loading ? disabledClass : ''
          }`}
        >
          {loading ? (
            <span className="flex items-center gap-1">
              <LoadingSpinner size={compact ? 12 : 14} />
              {!compact && t('common.loading')}
            </span>
          ) : (
            isPaused ? `▶ ${t('simulation.resume')}` : `▶ ${t('simulation.start')}`
          )}
        </button>
      )}
      
      {/* Pause Button */}
      {isRunning && (
        <button
          onClick={() => pause()}
          disabled={loading}
          className={`${buttonBase} bg-yellow-600 hover:bg-yellow-700 text-white font-medium transition-colors ${
            loading ? disabledClass : ''
          }`}
        >
          ⏸ {t('simulation.pause')}
        </button>
      )}
      
      {/* Stop Button */}
      {(isRunning || isPaused) && (
        <button
          onClick={() => stop()}
          disabled={loading}
          className={`${buttonBase} bg-red-600 hover:bg-red-700 text-white font-medium transition-colors ${
            loading ? disabledClass : ''
          }`}
        >
          ⏹ {t('simulation.reset')}
        </button>
      )}
      
      {/* Reset Button */}
      {!compact && (
        <button
          onClick={() => reset()}
          disabled={loading || isRunning}
          className={`${buttonBase} bg-gray-600 hover:bg-gray-700 text-white font-medium transition-colors ${
            loading || isRunning ? disabledClass : ''
          }`}
          title="Reset simulation to initial state"
        >
          ↻ {t('simulation.reset')}
        </button>
      )}
      
      {/* Status indicator */}
      {!compact && (
        <span className={`text-xs font-medium px-2 py-1 rounded ${
          isRunning ? 'bg-green-100 text-green-700 border border-green-300' :
          isPaused ? 'bg-yellow-100 text-yellow-700 border border-yellow-300' :
          'bg-gray-100 text-gray-600 border border-gray-300'
        }`}>
          {isRunning ? t('simulation.running').toUpperCase() :
           isPaused ? t('simulation.paused').toUpperCase() :
           t('simulation.stopped').toUpperCase()}
        </span>
      )}
    </div>
  )
}

// Loading spinner component
const LoadingSpinner: React.FC<{ size?: number }> = ({ size = 14 }) => (
  <svg 
    className="animate-spin" 
    width={size} 
    height={size} 
    viewBox="0 0 24 24" 
    fill="none"
  >
    <circle 
      className="opacity-25" 
      cx="12" 
      cy="12" 
      r="10" 
      stroke="currentColor" 
      strokeWidth="4"
    />
    <path 
      className="opacity-75" 
      fill="currentColor" 
      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
    />
  </svg>
)

export default SimulationControls
