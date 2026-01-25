/**
 * God Mode Panel - Event injection interface for demonstrations
 */

import { useState, useCallback, useEffect, useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import toast from 'react-hot-toast'
import { usePatientStore } from '../../stores'
import { api } from '../../services'
import type { EventType, Severity } from '../../types'

const EVENT_DETECTION_META: Record<EventType, { minMinutes: number; label: string; maxMinutes: number }> = {
  LATE_DECEL: { minMinutes: 3, label: '3 Minutes (Recurrence Rule)', maxMinutes: 60 },
  VARIABLE_DECEL: { minMinutes: 3, label: '3 Minutes (Recurrence Rule)', maxMinutes: 60 },
  PROLONGED_DECEL: { minMinutes: 2, label: '2 Minutes', maxMinutes: 60 },
  TACHYCARDIA: { minMinutes: 10, label: '10 Minutes', maxMinutes: 120 },
  BRADYCARDIA: { minMinutes: 3, label: '3 Minutes', maxMinutes: 120 },
  MINIMAL_VARIABILITY: { minMinutes: 10, label: '10 Minutes', maxMinutes: 120 },
  SINUSOIDAL: { minMinutes: 10, label: '10 Minutes', maxMinutes: 60 },
  HYPERSTIM: { minMinutes: 10, label: '10 Minutes', maxMinutes: 60 },
  RECOVERY: { minMinutes: 2, label: '2 Minutes', maxMinutes: 60 },
}

const EVENT_TYPES: EventType[] = [
  'LATE_DECEL',
  'VARIABLE_DECEL',
  'PROLONGED_DECEL',
  'TACHYCARDIA',
  'BRADYCARDIA',
  'MINIMAL_VARIABILITY',
  'SINUSOIDAL',
  'HYPERSTIM',
  'RECOVERY',
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
  const [eventType, setEventType] = useState<EventType | ''>('')
  const [severity, setSeverity] = useState<Severity>('moderate')
  const [durationMinutes, setDurationMinutes] = useState(5)
  const [targetPatient, setTargetPatient] = useState<string>('')
  const [isInjecting, setIsInjecting] = useState(false)

  const snapshotIds = usePatientStore(state => Array.from(state.patients.keys()))
  const summaryList = usePatientStore(state => Array.from(state.patientSummaries.values()))
  const liveIds = usePatientStore(state => Array.from(state.liveUpdates.keys()))
  const updateSummaries = usePatientStore(state => state.updatePatientSummaries)

  const patientOptions = useMemo(() => {
    const ids = new Set<string>()
    summaryList.forEach(summary => ids.add(summary.patient_id))
    snapshotIds.forEach(id => ids.add(id))
    liveIds.forEach(id => ids.add(id))
    return Array.from(ids).sort()
  }, [summaryList, snapshotIds, liveIds])

  useEffect(() => {
    if (summaryList.length > 0) {
      return
    }

    let cancelled = false

    api.getPatientsSummary()
      .then(response => {
        if (cancelled) return
        updateSummaries(response.patients)
      })
      .catch(error => {
        console.warn('Failed to fetch patient summaries for God Mode:', error)
      })

    return () => {
      cancelled = true
    }
  }, [summaryList.length, updateSummaries])

  useEffect(() => {
    if (!targetPatient && patientOptions.length > 0) {
      setTargetPatient(patientOptions[0])
    } else if (targetPatient && !patientOptions.includes(targetPatient)) {
      setTargetPatient(patientOptions[0] ?? '')
    }
  }, [patientOptions, targetPatient])
  
  const detectionMeta = eventType ? EVENT_DETECTION_META[eventType] : null

  useEffect(() => {
    if (!eventType) {
      return
    }

    const minMinutes = EVENT_DETECTION_META[eventType].minMinutes
    setDurationMinutes(minMinutes + 2)
  }, [eventType])

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
        duration_minutes: durationMinutes,
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
      const message = error instanceof Error ? error.message : t('godmode.error')
      toast.error(message)
    } finally {
      setIsInjecting(false)
    }
  }, [eventType, severity, durationMinutes, targetPatient, t])

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
              <option key={event} value={event}>
                {t(`events.${event}`)}
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
            {t('godmode.duration')}: {durationMinutes} min
          </label>
          <input
            type="range"
            min={detectionMeta?.minMinutes ?? 1}
            max={detectionMeta?.maxMinutes ?? 60}
            step={1}
            value={durationMinutes}
            onChange={(e) => setDurationMinutes(Number(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-500"
            disabled={!eventType}
          />
          <div className="flex justify-between text-xs text-gray-400 mt-1">
            <span>{detectionMeta?.minMinutes ?? 1} min</span>
            <span>{detectionMeta?.maxMinutes ?? 60} min</span>
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
            {patientOptions.map(id => (
              <option key={id} value={id}>{id}</option>
            ))}
          </select>
        </div>

        {/* Detection Time */}
        {eventType && detectionMeta && (
          <div className="p-2 bg-purple-50 rounded-lg text-center">
            <span className="text-sm text-gray-600">{t('godmode.detection')}: </span>
            <span className="text-purple-700 font-mono font-medium">{detectionMeta.label}</span>
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
