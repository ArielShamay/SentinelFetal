import React, { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import type { PatientSnapshot, PatientSummary } from '../../types'
import { CategoryBadge } from './CategoryBadge'

interface PatientCardProps {
  patient: PatientSnapshot | PatientSummary
  compact?: boolean
  onClick?: () => void
  className?: string
  /** Optional recent FHR values for sparkline display */
  fhrHistory?: number[]
  /** Optional recent UC values */
  ucHistory?: number[]
  /** MHR detection warning */
  isMHR?: boolean
}

// Type guard to check if patient is a full snapshot
const isFullSnapshot = (patient: PatientSnapshot | PatientSummary): patient is PatientSnapshot => {
  return 'metrics' in patient
}

// Category border colors for medical monitors
const categoryBorderColors: Record<number, string> = {
  1: 'border-l-green-500',
  2: 'border-l-yellow-500',
  3: 'border-l-red-500',
}

export const PatientCard: React.FC<PatientCardProps> = ({
  patient,
  compact = false,
  onClick,
  className = '',
  fhrHistory = [],
  ucHistory = [],
  isMHR = false,
}) => {
  const navigate = useNavigate()

  const handleClick = () => {
    if (onClick) {
      onClick()
    } else {
      navigate(`/patient/${patient.patient_id}`)
    }
  }

  const category = patient.category
  const isPathological = category === 3

  // Extract metrics based on type
  const fhr = isFullSnapshot(patient) ? patient.metrics.current_fhr : patient.current_fhr
  const baseline = isFullSnapshot(patient) ? patient.metrics.baseline_fhr : patient.baseline_fhr
  const variability = isFullSnapshot(patient) ? patient.metrics.variability : undefined
  const toco = isFullSnapshot(patient) ? patient.metrics.current_uc : undefined
  const signalQuality = isFullSnapshot(patient) ? patient.fsqi_score : undefined
  const lastUpdate = patient.last_update

  // Generate SVG path for FHR trace
  const fhrPath = useMemo(() => {
    if (!fhrHistory || fhrHistory.length < 2) return ''
    const width = compact ? 200 : 280
    const height = compact ? 40 : 60
    const padding = 4

    const minVal = Math.min(...fhrHistory, 100)
    const maxVal = Math.max(...fhrHistory, 180)
    const range = maxVal - minVal || 1

    const points = fhrHistory.map((val, i) => {
      const x = padding + (i / (fhrHistory.length - 1)) * (width - 2 * padding)
      const y = height - padding - ((val - minVal) / range) * (height - 2 * padding)
      return `${x},${y}`
    })

    return `M ${points.join(' L ')}`
  }, [fhrHistory, compact])

  // Generate SVG path for UC trace
  const ucPath = useMemo(() => {
    if (!ucHistory || ucHistory.length < 2) return ''
    const width = compact ? 200 : 280
    const height = 30
    const padding = 2

    const minVal = 0
    const maxVal = Math.max(...ucHistory, 100)
    const range = maxVal - minVal || 1

    const points = ucHistory.map((val, i) => {
      const x = padding + (i / (ucHistory.length - 1)) * (width - 2 * padding)
      const y = height - padding - ((val - minVal) / range) * (height - 2 * padding)
      return `${x},${y}`
    })

    return `M ${points.join(' L ')}`
  }, [ucHistory, compact])

  const borderColor = categoryBorderColors[category] || 'border-l-gray-400'

  if (compact) {
    return (
      <div
        onClick={handleClick}
        className={`
          monitor-card bg-white rounded-lg border border-gray-200 border-l-4 ${borderColor}
          p-3 cursor-pointer transition-all duration-200
          hover:shadow-md
          ${isPathological ? 'animate-pulse-subtle' : ''}
          ${isMHR ? 'ring-2 ring-orange-400' : ''}
          ${className}
        `}
      >
        {/* MHR Warning Banner */}
        {isMHR && (
          <div className="bg-orange-100 text-orange-800 text-xs font-semibold px-2 py-1 rounded mb-2 flex items-center gap-1">
            <span>⚠️</span>
            <span>MHR DETECTED</span>
          </div>
        )}

        <div className="flex items-center justify-between mb-2">
          <span className="font-semibold text-gray-900 text-sm">
            {patient.patient_id}
          </span>
          <CategoryBadge category={category} size="sm" showLabel={false} />
        </div>

        {/* Mini FHR Chart */}
        {fhrHistory.length > 1 && (
          <svg width="100%" height="40" viewBox="0 0 200 40" preserveAspectRatio="none" className="mb-2">
            <path
              d={fhrPath}
              fill="none"
              stroke="#1E90FF"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        )}

        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-600">
            <span className="font-medium text-blue-600">{fhr?.toFixed(0) ?? '--'}</span> bpm
          </span>
          <span className="text-gray-600">
            UC: <span className="font-medium text-orange-600">{toco?.toFixed(0) ?? '--'}</span>
          </span>
        </div>
      </div>
    )
  }

  return (
    <div
      onClick={handleClick}
      className={`
        monitor-card bg-white rounded-lg border border-gray-200 border-l-4 ${borderColor}
        cursor-pointer transition-all duration-200
        hover:shadow-lg
        ${isPathological ? 'ring-2 ring-red-200' : ''}
        ${isMHR ? 'ring-2 ring-orange-400' : ''}
        ${className}
      `}
    >
      {/* MHR Warning Banner */}
      {isMHR && (
        <div className="bg-orange-100 border-b border-orange-200 text-orange-800 text-sm font-semibold px-4 py-2 flex items-center gap-2">
          <span className="text-lg">⚠️</span>
          <div>
            <span className="block">MATERNAL PULSE DETECTED</span>
            <span className="text-xs font-normal text-orange-600">Signal may be contaminated with maternal heart rate</span>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-100">
        <div>
          <h3 className="font-bold text-gray-900 text-lg">
            {patient.patient_id}
          </h3>
          <span className="text-xs text-gray-500">Bed #{isFullSnapshot(patient) ? patient.bed_number : '--'}</span>
        </div>
        <CategoryBadge category={category} size="md" />
      </div>

      {/* FHR Chart Section */}
      <div className="p-4 bg-gray-50">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-gray-600">FHR (bpm)</span>
          <span className="text-lg font-bold text-blue-600">{fhr?.toFixed(0) ?? '--'}</span>
        </div>

        {fhrHistory.length > 1 ? (
          <svg width="100%" height="60" viewBox="0 0 280 60" preserveAspectRatio="none" className="bg-white rounded border border-gray-200">
            {/* Normal range band (110-160 bpm) */}
            <rect x="0" y="15" width="280" height="30" fill="#e8f5e9" opacity="0.5" />

            {/* FHR trace */}
            <path
              d={fhrPath}
              fill="none"
              stroke="#1E90FF"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        ) : (
          <div className="h-[60px] bg-white rounded border border-gray-200 flex items-center justify-center text-gray-400 text-sm">
            Waiting for data...
          </div>
        )}
      </div>

      {/* UC Chart Section */}
      <div className="px-4 pb-2">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs font-medium text-gray-600">UC (mmHg)</span>
          <span className="text-sm font-bold text-orange-600">{toco?.toFixed(0) ?? '--'}</span>
        </div>

        {ucHistory.length > 1 ? (
          <svg width="100%" height="30" viewBox="0 0 280 30" preserveAspectRatio="none" className="bg-white rounded border border-gray-200">
            <path
              d={ucPath}
              fill="none"
              stroke="#FF8C00"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        ) : (
          <div className="h-[30px] bg-white rounded border border-gray-200" />
        )}
      </div>

      {/* Vitals Grid */}
      <div className="grid grid-cols-3 gap-2 p-4 pt-2 border-t border-gray-100">
        <VitalDisplay
          label="Baseline"
          value={baseline}
          unit="bpm"
        />
        <VitalDisplay
          label="Variability"
          value={variability}
          unit="bpm"
        />
        <VitalDisplay
          label="Signal"
          value={signalQuality != null ? signalQuality * 100 : undefined}
          unit="%"
          alert={signalQuality != null && signalQuality < 0.7}
        />
      </div>

      {/* Footer */}
      <div className="px-4 pb-3 flex items-center justify-between text-xs text-gray-500">
        <span>{formatTimestamp(lastUpdate)}</span>
        <span className="text-blue-600 hover:text-blue-800">View Details →</span>
      </div>
    </div>
  )
}

// Vital display component - Light theme
interface VitalDisplayProps {
  label: string
  value?: number | null
  unit: string
  alert?: boolean
}

const VitalDisplay: React.FC<VitalDisplayProps> = ({ label, value, unit, alert = false }: VitalDisplayProps) => (
  <div className={`
    p-2 rounded-lg text-center
    ${alert ? 'bg-red-50 border border-red-200' : 'bg-gray-50'}
  `}>
    <div className="text-xs text-gray-500 mb-0.5">{label}</div>
    <div className={`text-sm font-semibold ${alert ? 'text-red-600' : 'text-gray-900'}`}>
      {value != null ? value.toFixed(0) : '--'}
      <span className="text-xs text-gray-400 ml-0.5">{unit}</span>
    </div>
  </div>
)

// Timestamp formatter
const formatTimestamp = (timestamp: number): string => {
  try {
    const date = new Date(timestamp)
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
  } catch {
    return '--:--:--'
  }
}

export default PatientCard
