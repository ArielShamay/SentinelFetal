/**
 * SentinelFetal TypeScript Type Definitions
 * ==========================================
 */

// Patient category (FIGO classification)
export type Category = 1 | 2 | 3

// Category display names and colors
export const CATEGORY_NAMES: Record<Category, string> = {
  1: 'Normal',
  2: 'Suspicious',
  3: 'Pathological',
}

export const CATEGORY_COLORS: Record<Category, string> = {
  1: '#28a745', // Green
  2: '#fd7e14', // Orange
  3: '#dc3545', // Red
}

// Patient metrics
export interface PatientMetrics {
  baseline_fhr: number
  current_fhr: number
  variability: number
  current_uc: number
  acceleration_count?: number
  deceleration_count?: number
}

// Alert structure
export interface Alert {
  type: string
  message: string
  severity: 'info' | 'warning' | 'critical'
  timestamp: number
}

// Highlight region for charts
export interface HighlightRegion {
  start_idx: number
  end_idx: number
  severity: 'info' | 'warning' | 'critical'
  label: string
  color?: string
}

// Trend analysis data
export interface TrendData {
  deterioration_score: number
  variability_slope: number
  decel_count_30min: number
  late_decel_count_15min: number
  alerts: Alert[]
}

// AI explanation data
export interface ExplanationData {
  primary_reason: string
  contributing_factors: string[]
  confidence: number
}

// Full patient snapshot (from REST API)
export interface PatientSnapshot {
  patient_id: string
  bed_number: number
  category: Category
  category_name: string
  metrics: PatientMetrics
  fhr_history: number[]
  uc_history: number[]
  timestamps: number[]
  alerts: Alert[]
  trend_data: TrendData | null
  explanation: ExplanationData | null
  highlight_regions?: HighlightRegion[]
  fsqi_score: number
  has_active_event: boolean
  last_update: number
}

// Patient summary for list views
export interface PatientSummary {
  patient_id: string
  bed_number: number
  category: Category
  category_name: string
  current_fhr: number
  baseline_fhr: number
  has_alerts: boolean
  last_update: number
}

// MHR Detection result
export interface MHRAlert {
  is_mhr: boolean
  confidence: number
  recommended_action: 'NONE' | 'FLAG' | 'WARN' | 'BLOCK_SEGMENT'
  detection_methods: string[]
  message?: string
}

// Clinical Findings interfaces
export interface DecelerationFindings {
  late_count: number
  variable_count: number
  early_count: number
  prolonged_count: number
  total_count: number
  recurrent: boolean
}

export interface VariabilityFindings {
  value_bpm: number
  category: string
  is_concerning: boolean
}

export interface BaselineFindings {
  value_bpm: number
  status: string
  is_stable: boolean
}

export interface ClinicalFindings {
  decelerations: DecelerationFindings
  variability: VariabilityFindings
  baseline: BaselineFindings
  accelerations_present: boolean
  tachysystole: boolean
  sinusoidal: boolean
  contraction_frequency: number
}

// WebSocket update (streaming)
export interface WSPatientUpdate {
  patient_id: string
  category: Category
  baseline: number
  variability: number
  fhr_latest: number[]
  uc_latest: number[]
  fsqi: number
  confidence: number
  findings: ClinicalFindings
  // V2.0 fields
  mhr_alert?: MHRAlert | null
  trend_score?: number
  trend_slope?: number
  explanation?: ExplanationData | null
  highlight_regions?: HighlightRegion[]
}

// Simulation status
export interface SimulationStatus {
  running: boolean
  paused: boolean
  patient_count: number
  tick_count: number
  elapsed_seconds: number
  uptime_seconds: number
}

// Simulation response
export interface SimulationResponse {
  success: boolean
  message: string
  status: SimulationStatus
}

// WebSocket message types
export type WSMessageType = 'connected' | 'patient_update' | 'ping' | 'pong' | 'error'

export interface WSMessage {
  type: WSMessageType
  timestamp: number
  client_id?: string
  patient_id?: string
  patients?: WSPatientUpdate[]
  error?: string
  [key: string]: unknown
}

// Event injection (God Mode)
export type EventType =
  | 'LATE_DECEL'
  | 'VARIABLE_DECEL'
  | 'PROLONGED_DECEL'
  | 'BRADYCARDIA'
  | 'TACHYCARDIA'
  | 'MINIMAL_VARIABILITY'
  | 'HYPERSTIM'
  | 'SINUSOIDAL'
  | 'RECOVERY'

export type Severity = 'mild' | 'moderate' | 'severe'

export interface EventInjection {
  event_type: EventType
  severity: Severity
  duration_seconds?: number
  duration_minutes?: number
  params?: Record<string, unknown>
}

export interface EventInjectionResponse {
  success: boolean
  message: string
  patient_id: string
  event_type: string
}

// Chart data point
export interface ChartDataPoint {
  time: number
  value: number
}
