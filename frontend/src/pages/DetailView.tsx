import React, { useEffect, useState, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { usePatientStore } from '../stores'
import { CategoryBadge, CTGChart, ChartControls, TrendPanel, ExplanationPanel } from '../components'
import { api } from '../services'
import type { PatientSnapshot, Alert } from '../types'

export const DetailView: React.FC = () => {
  const { patientId } = useParams<{ patientId: string }>()
  const navigate = useNavigate()
  
  const patient = usePatientStore(state => 
    patientId ? state.patients.get(patientId) : undefined
  )
  
  const [detail, setDetail] = useState<PatientSnapshot | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  // Fetch full patient detail
  useEffect(() => {
    if (!patientId) return
    
    setLoading(true)
    setError(null)
    
    api.getPatientSnapshot(patientId)
      .then((data: PatientSnapshot) => {
        setDetail(data)
        setLoading(false)
      })
      .catch((err: Error) => {
        setError(err.message || 'Failed to load patient details')
        setLoading(false)
      })
  }, [patientId])
  
  if (!patientId) {
    return <NotFoundState onBack={() => navigate('/')} />
  }
  
  if (loading) {
    return <LoadingState />
  }
  
  if (error) {
    return <ErrorState error={error} onRetry={() => window.location.reload()} />
  }
  
  // Use real-time data from WebSocket, fallback to fetched detail
  const currentData = patient ?? detail
  
  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Back button & Header */}
      <div className="mb-6">
        <button
          onClick={() => navigate('/')}
          className="text-gray-400 hover:text-white text-sm mb-4 flex items-center gap-1"
        >
          ← Back to Ward View
        </button>
        
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">
              Patient: {patientId}
            </h1>
            <p className="text-gray-400 text-sm mt-1">
              Real-time monitoring data
            </p>
          </div>
          
          {currentData && (
            <CategoryBadge 
              category={currentData.category} 
              size="lg" 
            />
          )}
        </div>
      </div>
      
      {currentData ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main vitals panel */}
          <div className="lg:col-span-2 space-y-6">
            <VitalsPanel patient={currentData} />
            <CTGChartPanel />
          </div>
          
          {/* Sidebar */}
          <div className="space-y-6">
            <TrendPanel 
              data={currentData.trend_data ? {
                deteriorationScore: currentData.trend_data.deterioration_score,
                variabilityTrend: currentData.trend_data.variability_slope > 0 ? 'increasing' 
                  : currentData.trend_data.variability_slope < 0 ? 'decreasing' 
                  : 'stable',
                decelsIn30min: currentData.trend_data.decel_count_30min,
                lateDecelsIn15min: currentData.trend_data.late_decel_count_15min,
                alerts: currentData.trend_data.alerts.map(a => a.message),
              } : undefined}
            />
            <ExplanationPanel 
              data={currentData.explanation ? {
                category: currentData.category,
                primaryReason: currentData.explanation.primary_reason,
                factors: currentData.explanation.contributing_factors,
                confidence: currentData.explanation.confidence,
              } : undefined}
            />
            <EventsPanel alerts={currentData.alerts ?? []} />
            <AlertsPanel patient={currentData} />
          </div>
        </div>
      ) : (
        <NotFoundState onBack={() => navigate('/')} />
      )}
    </div>
  )
}

// Vitals Panel
const VitalsPanel: React.FC<{ patient: PatientSnapshot }> = ({ patient }) => {
  const fhr = patient.metrics.current_fhr
  const isAbnormalFHR = fhr != null && (fhr < 110 || fhr > 160)
  
  return (
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
      <h2 className="text-lg font-semibold text-white mb-4">Current Vitals</h2>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <VitalCard 
          label="FHR" 
          value={patient.metrics.current_fhr} 
          unit="bpm" 
          alert={isAbnormalFHR}
          normalRange="110-160"
        />
        <VitalCard 
          label="Baseline" 
          value={patient.metrics.baseline_fhr} 
          unit="bpm"
        />
        <VitalCard 
          label="Variability" 
          value={patient.metrics.variability} 
          unit=""
        />
        <VitalCard 
          label="Contractions" 
          value={patient.metrics.current_uc} 
          unit=""
        />
      </div>
      
      {/* Additional metrics */}
      <div className="mt-4 pt-4 border-t border-gray-700 grid grid-cols-3 gap-4 text-sm">
        <div>
          <span className="text-gray-500">Signal Quality</span>
          <div className="mt-1 flex items-center gap-2">
            <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
              <div 
                className="h-full bg-blue-500"
                style={{ width: `${(patient.fsqi_score ?? 0) * 100}%` }}
              />
            </div>
            <span className="text-white">
              {((patient.fsqi_score ?? 0) * 100).toFixed(0)}%
            </span>
          </div>
        </div>
        <div>
          <span className="text-gray-500">Accelerations</span>
          <div className="text-white mt-1">{patient.metrics.acceleration_count ?? 0}</div>
        </div>
        <div>
          <span className="text-gray-500">Decelerations</span>
          <div className="text-white mt-1">{patient.metrics.deceleration_count ?? 0}</div>
        </div>
      </div>
      
      {/* Timestamp */}
      <div className="mt-4 text-xs text-gray-500 text-right">
        Last updated: {new Date(patient.last_update).toLocaleTimeString()}
      </div>
    </div>
  )
}

// Vital Card
interface VitalCardProps {
  label: string
  value?: number | null
  unit: string
  alert?: boolean
  normalRange?: string
}

const VitalCard: React.FC<VitalCardProps> = ({ 
  label, 
  value, 
  unit, 
  alert = false,
  normalRange 
}) => (
  <div className={`
    p-4 rounded-lg 
    ${alert ? 'bg-red-900/40 border border-red-700' : 'bg-gray-900/50'}
  `}>
    <div className="text-sm text-gray-400">{label}</div>
    <div className={`text-3xl font-bold ${alert ? 'text-red-300' : 'text-white'}`}>
      {value != null ? value.toFixed(1) : '--'}
    </div>
    <div className="text-xs text-gray-500">
      {unit}
      {normalRange && <span className="ml-1">({normalRange})</span>}
    </div>
  </div>
)

// CTG Chart Panel - Real-time FHR/UC visualization
const CTGChartPanel: React.FC = () => {
  const [timeRange, setTimeRange] = useState<number | null>(10)
  
  const handleZoomIn = useCallback(() => {
    setTimeRange(prev => prev ? Math.max(1, prev - 2) : 5)
  }, [])
  
  const handleZoomOut = useCallback(() => {
    setTimeRange(prev => prev ? prev + 5 : 15)
  }, [])
  
  const handleReset = useCallback(() => {
    setTimeRange(10)
  }, [])
  
  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white">CTG Monitor</h2>
          <ChartControls
            onZoomIn={handleZoomIn}
            onZoomOut={handleZoomOut}
            onReset={handleReset}
            onTimeRangeChange={setTimeRange}
            selectedRange={timeRange ?? undefined}
            className="!bg-transparent !p-0"
          />
        </div>
      </div>
      <CTGChart 
        height={350} 
        showControls={true}
        isLive={true}
      />
    </div>
  )
}

// Events Panel
const EventsPanel: React.FC<{ alerts: Alert[] }> = ({ alerts }) => {
  const recentAlerts = alerts.slice(-10).reverse()
  
  return (
    <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
      <h3 className="text-md font-semibold text-white mb-3">Recent Events</h3>
      
      {recentAlerts.length > 0 ? (
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {recentAlerts.map((alert, idx) => (
            <div 
              key={idx}
              className="p-2 rounded bg-gray-900/50 text-sm"
            >
              <div className="flex items-center justify-between">
                <span className="text-white font-medium">{alert.type}</span>
                <span className="text-xs text-gray-500">
                  {new Date(alert.timestamp).toLocaleTimeString()}
                </span>
              </div>
              <div className="text-xs text-gray-400 mt-1">
                {alert.message}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-gray-500">No events recorded</p>
      )}
    </div>
  )
}

// Alerts Panel
const AlertsPanel: React.FC<{ patient: PatientSnapshot }> = ({ patient }) => {
  const alertMessages: string[] = []
  
  const fhr = patient.metrics.current_fhr
  if (fhr != null && fhr < 110) alertMessages.push('Bradycardia: FHR below 110 bpm')
  if (fhr != null && fhr > 160) alertMessages.push('Tachycardia: FHR above 160 bpm')
  if (patient.metrics.variability != null && patient.metrics.variability < 5) alertMessages.push('Reduced variability')
  if ((patient.fsqi_score ?? 0) < 0.5) alertMessages.push('Poor signal quality')
  if (patient.category === 3) alertMessages.push('Pathological CTG pattern')
  
  return (
    <div className={`
      rounded-xl p-4 border
      ${alertMessages.length > 0 
        ? 'bg-red-900/20 border-red-700/50' 
        : 'bg-gray-800 border-gray-700'}
    `}>
      <h3 className="text-md font-semibold text-white mb-3">
        Alerts {alertMessages.length > 0 && `(${alertMessages.length})`}
      </h3>
      
      {alertMessages.length > 0 ? (
        <ul className="space-y-2">
          {alertMessages.map((alertMsg, idx) => (
            <li key={idx} className="flex items-start gap-2 text-sm">
              <span className="text-red-400">⚠️</span>
              <span className="text-red-200">{alertMsg}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-green-400">✓ No active alerts</p>
      )}
    </div>
  )
}

// Loading State
const LoadingState: React.FC = () => (
  <div className="flex items-center justify-center min-h-[400px]">
    <div className="text-center text-gray-500">
      <div className="animate-spin text-4xl mb-4">⏳</div>
      <p>Loading patient data...</p>
    </div>
  </div>
)

// Error State
const ErrorState: React.FC<{ error: string; onRetry: () => void }> = ({ error, onRetry }) => (
  <div className="flex flex-col items-center justify-center min-h-[400px] text-gray-500">
    <div className="text-4xl mb-4">❌</div>
    <p className="text-red-400 mb-4">{error}</p>
    <button
      onClick={onRetry}
      className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg"
    >
      Retry
    </button>
  </div>
)

// Not Found State
const NotFoundState: React.FC<{ onBack: () => void }> = ({ onBack }) => (
  <div className="flex flex-col items-center justify-center min-h-[400px] text-gray-500">
    <div className="text-4xl mb-4">🔍</div>
    <p className="mb-4">Patient not found</p>
    <button
      onClick={onBack}
      className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg"
    >
      Back to Ward
    </button>
  </div>
)

export default DetailView
