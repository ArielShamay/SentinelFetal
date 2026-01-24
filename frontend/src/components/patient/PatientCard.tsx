import React from 'react'
import { useNavigate } from 'react-router-dom'
import type { PatientSnapshot, PatientSummary } from '../../types'
import { CategoryBadge } from './CategoryBadge'
import { FHRSparkline } from '../charts'

interface PatientCardProps {
  patient: PatientSnapshot | PatientSummary
  compact?: boolean
  onClick?: () => void
  className?: string
  /** Optional recent FHR values for sparkline display */
  fhrHistory?: number[]
}

// Type guard to check if patient is a full snapshot
const isFullSnapshot = (patient: PatientSnapshot | PatientSummary): patient is PatientSnapshot => {
  return 'metrics' in patient
}

export const PatientCard: React.FC<PatientCardProps> = ({ 
  patient, 
  compact = false,
  onClick,
  className = '',
  fhrHistory = [],
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
  
  // Get border color based on category
  const borderColor = {
    1: 'border-green-700/50 hover:border-green-600',
    2: 'border-yellow-700/50 hover:border-yellow-600',
    3: 'border-red-700/50 hover:border-red-600 ring-1 ring-red-500/30'
  }[category] ?? 'border-gray-700/50 hover:border-gray-600'
  
  if (compact) {
    return (
      <div 
        onClick={handleClick}
        className={`
          bg-gray-800 rounded-lg border ${borderColor}
          p-3 cursor-pointer transition-all duration-200
          hover:bg-gray-750 hover:shadow-lg
          ${isPathological ? 'animate-pulse-subtle' : ''}
          ${className}
        `}
      >
        <div className="flex items-center justify-between">
          <span className="font-medium text-white text-sm">
            {patient.patient_id}
          </span>
          <CategoryBadge category={category} size="sm" showLabel={false} />
        </div>
        <div className="mt-1 flex items-center gap-2 text-xs text-gray-400">
          <span>FHR: {fhr?.toFixed(0) ?? '--'}</span>
          <span>•</span>
          <span>UC: {toco?.toFixed(0) ?? '--'}</span>
        </div>
      </div>
    )
  }
  
  return (
    <div 
      onClick={handleClick}
      className={`
        bg-gray-800 rounded-xl border-2 ${borderColor}
        p-4 cursor-pointer transition-all duration-200
        hover:bg-gray-750 hover:shadow-xl hover:scale-[1.02]
        ${isPathological ? 'animate-pulse-subtle' : ''}
        ${className}
      `}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-white text-lg">
          {patient.patient_id}
        </h3>
        <CategoryBadge category={category} size="md" />
      </div>
      
      {/* Vitals Grid */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        <VitalDisplay 
          label="FHR" 
          value={fhr} 
          unit="bpm"
          alert={fhr != null && (fhr < 110 || fhr > 160)}
        />
        <VitalDisplay 
          label="Contractions" 
          value={toco} 
          unit=""
          alert={toco != null && toco > 80}
        />
        <VitalDisplay 
          label="Baseline" 
          value={baseline} 
          unit="bpm"
        />
        <VitalDisplay 
          label="Variability" 
          value={variability} 
          unit=""
        />
      </div>
      
      {/* Signal Quality */}
      {signalQuality != null && (
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500">Signal Quality</span>
          <SignalQualityBar quality={signalQuality} />
        </div>
      )}
      
      {/* FHR Sparkline */}
      {fhrHistory.length > 1 && (
        <div className="mt-3 pt-3 border-t border-gray-700/50">
          <FHRSparkline 
            data={fhrHistory.slice(-60)} 
            width={160} 
            height={36}
            showTrend={true}
            className="mx-auto"
          />
        </div>
      )}
      
      {/* Timestamp */}
      <div className="mt-2 text-xs text-gray-500 text-right">
        {formatTimestamp(lastUpdate)}
      </div>
    </div>
  )
}

// Vital display component
interface VitalDisplayProps {
  label: string
  value?: number | null
  unit: string
  alert?: boolean
}

const VitalDisplay: React.FC<VitalDisplayProps> = ({ label, value, unit, alert = false }) => (
  <div className={`
    p-2 rounded-lg 
    ${alert ? 'bg-red-900/30 border border-red-700/50' : 'bg-gray-900/50'}
  `}>
    <div className="text-xs text-gray-400 mb-0.5">{label}</div>
    <div className={`text-lg font-semibold ${alert ? 'text-red-300' : 'text-white'}`}>
      {value != null ? value.toFixed(1) : '--'}
      {unit && <span className="text-xs text-gray-500 ml-1">{unit}</span>}
    </div>
  </div>
)

// Signal quality bar
const SignalQualityBar: React.FC<{ quality: number }> = ({ quality }) => {
  const percentage = Math.min(100, Math.max(0, quality * 100))
  const color = percentage >= 70 ? 'bg-green-500' : percentage >= 40 ? 'bg-yellow-500' : 'bg-red-500'
  
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 bg-gray-700 rounded-full overflow-hidden">
        <div 
          className={`h-full ${color} transition-all duration-300`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <span className="text-gray-400 w-8">{percentage.toFixed(0)}%</span>
    </div>
  )
}

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
