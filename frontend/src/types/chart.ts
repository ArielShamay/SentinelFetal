/**
 * Chart-related type definitions for CTG visualization
 */

import type { IChartApi, ISeriesApi } from 'lightweight-charts'

// Data point for time series
export interface ChartDataPoint {
  time: number  // Unix timestamp in seconds
  value: number
}

// Red zone/highlight region
export interface ChartHighlightRegion {
  startTime: number
  endTime: number
  color: string
  label?: string
  severity: 'info' | 'warning' | 'critical'
}

// Chart configuration
export interface CTGChartConfig {
  fhrRange: { min: number; max: number }
  ucRange: { min: number; max: number }
  timeWindowMinutes: number
  backgroundColor: string
  fhrColor: string
  ucColor: string
}

// Chart instance refs
export interface ChartRefs {
  chart: IChartApi | null
  fhrSeries: ISeriesApi<'Line'> | null
  ucSeries: ISeriesApi<'Line'> | null
}

// Sparkline props
export interface SparklineProps {
  data: number[]
  color?: string
  width?: number
  height?: number
  animated?: boolean
  showTrend?: boolean
  className?: string
}

// Chart state
export interface ChartState {
  isLive: boolean
  visibleRange: { from: number; to: number } | null
  lastUpdate: number
}

// CTG Chart props
export interface CTGChartProps {
  patientId?: string
  fhrData?: number[]
  ucData?: number[]
  timestamps?: number[]
  highlightRegions?: ChartHighlightRegion[]
  showUC?: boolean
  width?: string | number
  height?: number
  showControls?: boolean
  isLive?: boolean
  className?: string
  onZoom?: (range: { from: number; to: number }) => void
  onTimeRangeChange?: (minutes: number | null) => void
}

// Chart controls props
export interface ChartControlsProps {
  isLive?: boolean
  onGoLive?: () => void
  onZoomIn?: () => void
  onZoomOut?: () => void
  onReset?: () => void
  onTimeRangeChange?: (minutes: number | null) => void
  selectedRange?: number
  disabled?: boolean
  className?: string
}
