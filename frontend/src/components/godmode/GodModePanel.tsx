/**
 * God Mode Panel - Event injection interface for demonstrations
 */

import React, { useState, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import toast from 'react-hot-toast'
import { usePatientStore } from '../../stores'
import { api } from '../../services'
import type { EventType, Severity } from '../../types'

// Event types with detection times
const EVENT_TYPES: { id: EventType; detectionTime: number }[] = [
  { id: 'LATE_DECEL', detectionTime: 15 },
  { id: 'VARIABLE_DECEL', detectionTime: 10 },
  { id: 'PROLONGED_DECEL', detectionTime: 8 },
  { id: 'TACHYCARDIA', detectionTime: 20 },
  { id: 'BRADYCARDIA', detectionTime: 5 },
  { id: 'MINIMAL_VARIABILITY', detectionTime: 30 },
  { id: 'SINUSOIDAL', detectionTime: 25 },
  { id: 'HYPERSTIM', detectionTime: 12 },
]

interface GodModePanelProps {
  className?: string
  collapsed?: boolean
  onToggle?: () => void
}

export const GodModePanel: React.FC<GodModePanelProps> = ({
  className = '',
  collapsed = false,
  onToggle,
}) => {
  const { t } = useTranslation()
  
  const patients = usePatientStore(state => Array.from(state.patients.keys()))
  
  const [eventType, setEventType] = useState<EventType | ''>('')
  const [severity, setSeverity] = useState<Severity>('moderate')
  const [duration, setDuration] = useState(120) // seconds
  const [targetPatient, setTargetPatient] = useState<string>('')
  const [isInjecting, setIsInjecting] = useState(false)

  // Get detection time for selected event
  const detectionTime = EVENT_TYPES.find(e => e.id === eventType)?.detectionTime ?? 0

  // Handle event injection
  const handleInject = useCallback(async () => {
    if (!eventType || !targetPatient) {
      toast.error('Please select event type and patient')
      return
    }

    setIsInjecting(true)
    
    try {
      await api.injectEvent(targetPatient, {
        event_type: eventType,
        severity,
        duration_seconds: duration,
      })
      
      toast.success(t('godmode.success'), {
        icon: '⚡',
        duration: 3000,
      })
      
      // Reset form
      setEventType('')
      setTargetPatient('')
      
    } catch (error) {
      console.error('Event injection failed:', error)
      toast.error(t('godmode.error'))
    } finally {
      setIsInjecting(false)
    }
  }, [eventType, severity, duration, targetPatient, t])

  if (collapsed) {
    return (
      <button
        onClick={onToggle}
        className={`
          flex items-center gap-2 p-3 bg-purple-50 border border-purple-300
          rounded-lg text-purple-700 hover:bg-purple-100 transition-colors
          ${className}
        `}
      >
        <span className="text-lg">⚡</span>
        <span className="text-sm font-medium">{t('godmode.title')}</span>
        <span className="ml-auto">▼</span>
      </button>
    )
  }

  return (
    <div className={`bg-white rounded-xl border border-purple-200 shadow-sm overflow-hidden ${className}`}>
      {/* Header */}
      <div
        className="flex items-center justify-between p-4 bg-purple-50 cursor-pointer"
        onClick={onToggle}
      >
        <div className="flex items-center gap-2">
          <span className="text-xl">⚡</span>
          <h3 className="text-lg font-semibold text-purple-800">
            {t('godmode.title')}
          </h3>
        </div>
        {onToggle && (
          <button className="text-purple-500 hover:text-purple-700">▲</button>
        )}
      </div>

      {/* Content */}
      <div className="p-4 space-y-4">
        {/* Event Type */}
        <div>
          <label className="block text-sm text-gray-600 mb-1">
            {t('godmode.eventType')}
          </label>
          <select
            value={eventType}
            onChange={(e) => setEventType(e.target.value as EventType | '')}
            className="w-full p-2 bg-gray-50 border border-gray-300 rounded-lg text-gray-900 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          >
            <option value="">{t('godmode.selectEvent')}</option>
            {EVENT_TYPES.map(event => (
              <option key={event.id} value={event.id}>
                {t(`events.${event.id}`)}
              </option>
            ))}
          </select>
        </div>

        {/* Severity */}
        <div>
          <label className="block text-sm text-gray-600 mb-2">
            {t('godmode.severity')}
          </label>
          <div className="flex gap-4">
            {(['mild', 'moderate', 'severe'] as Severity[]).map(sev => (
              <label key={sev} className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="severity"
                  value={sev}
                  checked={severity === sev}
                  onChange={() => setSeverity(sev)}
                  className="text-purple-500 focus:ring-purple-500"
                />
                <span className={`text-sm font-medium ${
                  sev === 'mild' ? 'text-green-600' :
                  sev === 'moderate' ? 'text-yellow-600' :
                  'text-red-600'
                }`}>
                  {t(`godmode.${sev}`)}
                </span>
              </label>
            ))}
          </div>
        </div>

        {/* Duration */}
        <div>
          <label className="block text-sm text-gray-600 mb-1">
            {t('godmode.duration')}: {Math.floor(duration / 60)}:{(duration % 60).toString().padStart(2, '0')} min
          </label>
          <input
            type="range"
            min={30}
            max={600}
            step={30}
            value={duration}
            onChange={(e) => setDuration(Number(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-500"
          />
          <div className="flex justify-between text-xs text-gray-400 mt-1">
            <span>30s</span>
            <span>10min</span>
          </div>
        </div>

        {/* Target Patient */}
        <div>
          <label className="block text-sm text-gray-600 mb-1">
            {t('godmode.target')}
          </label>
          <select
            value={targetPatient}
            onChange={(e) => setTargetPatient(e.target.value)}
            className="w-full p-2 bg-gray-50 border border-gray-300 rounded-lg text-gray-900 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          >
            <option value="">{t('godmode.selectPatient')}</option>
            {patients.map(id => (
              <option key={id} value={id}>{id}</option>
            ))}
          </select>
        </div>

        {/* Detection Time */}
        {eventType && (
          <div className="p-2 bg-purple-50 rounded-lg text-center">
            <span className="text-sm text-gray-600">{t('godmode.detection')}: </span>
            <span className="text-purple-700 font-mono font-medium">~{detectionTime}s</span>
          </div>
        )}

        {/* Inject Button */}
        <button
          onClick={handleInject}
          disabled={isInjecting || !eventType || !targetPatient}
          className={`
            w-full py-3 rounded-lg font-semibold text-white
            flex items-center justify-center gap-2
            transition-all duration-200
            ${isInjecting || !eventType || !targetPatient
              ? 'bg-gray-300 cursor-not-allowed opacity-50'
              : 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 shadow-lg hover:shadow-purple-500/25'
            }
          `}
        >
          {isInjecting ? (
            <>
              <span className="animate-spin">⏳</span>
              {t('godmode.injecting')}
            </>
          ) : (
            <>
              <span>💉</span>
              {t('godmode.inject')}
            </>
          )}
        </button>
      </div>
    </div>
  )
}

export default GodModePanel
