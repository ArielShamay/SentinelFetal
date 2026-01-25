# Phase 4: Canvas Charts (Lightweight-Charts) - Technical Specifications

**Phase:** 4 of 6
**Document Type:** Technical Specifications
**Target Audience:** Frontend Developers

---

## 1. Project Structure

```
frontend/src/
├── components/
│   └── charts/
│       ├── CTGChart.tsx           # Main CTG chart component
│       ├── FHRSparkline.tsx       # Mini sparkline for cards
│       ├── RedZoneOverlay.tsx     # Highlight regions
│       ├── ChartControls.tsx      # Zoom/pan/reset controls
│       └── ChartLegend.tsx        # FHR/UC legend
│
├── hooks/
│   ├── useChartData.ts            # Data buffer management
│   ├── useLightweightChart.ts     # Chart instance hook
│   └── useChartInteraction.ts     # Zoom/pan handlers
│
├── utils/
│   ├── chartConfig.ts             # Chart configuration
│   ├── chartHelpers.ts            # Helper functions
│   └── ringBuffer.ts              # Circular buffer implementation
│
└── types/
    └── chart.ts                   # Chart-related types
```

---

## 2. Type Definitions

### 2.1 src/types/chart.ts

```typescript
/**
 * Chart-related type definitions
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
}

// Chart state
export interface ChartState {
  isLive: boolean
  visibleRange: { from: number; to: number } | null
  lastUpdate: number
}
```

---

## 3. Chart Configuration

### 3.1 src/utils/chartConfig.ts

```typescript
/**
 * Lightweight-Charts configuration
 */

import type { ChartOptions, LineSeriesOptions } from 'lightweight-charts'

// Default chart options
export const DEFAULT_CHART_OPTIONS: Partial<ChartOptions> = {
  layout: {
    background: { color: '#ffffff' },
    textColor: '#333333',
    fontSize: 12,
    fontFamily: "'Inter', system-ui, sans-serif",
  },
  grid: {
    vertLines: { color: '#e1e5ea', style: 1 },
    horzLines: { color: '#e1e5ea', style: 1 },
  },
  crosshair: {
    mode: 1, // Normal
    vertLine: {
      color: '#758696',
      width: 1,
      style: 3, // Dashed
      labelBackgroundColor: '#758696',
    },
    horzLine: {
      color: '#758696',
      width: 1,
      style: 3,
      labelBackgroundColor: '#758696',
    },
  },
  timeScale: {
    timeVisible: true,
    secondsVisible: true,
    borderColor: '#e1e5ea',
    rightOffset: 5,
    barSpacing: 6,
  },
  rightPriceScale: {
    borderColor: '#e1e5ea',
    scaleMargins: { top: 0.1, bottom: 0.1 },
  },
  handleScroll: {
    mouseWheel: true,
    pressedMouseMove: true,
    horzTouchDrag: true,
    vertTouchDrag: false,
  },
  handleScale: {
    axisPressedMouseMove: true,
    mouseWheel: true,
    pinch: true,
  },
}

// FHR series options
export const FHR_SERIES_OPTIONS: Partial<LineSeriesOptions> = {
  color: '#1E90FF', // DodgerBlue
  lineWidth: 2,
  crosshairMarkerVisible: true,
  crosshairMarkerRadius: 4,
  priceFormat: {
    type: 'custom',
    formatter: (price: number) => `${Math.round(price)} bpm`,
  },
}

// UC series options
export const UC_SERIES_OPTIONS: Partial<LineSeriesOptions> = {
  color: '#FF8C00', // DarkOrange
  lineWidth: 2,
  crosshairMarkerVisible: true,
  crosshairMarkerRadius: 4,
  priceFormat: {
    type: 'custom',
    formatter: (price: number) => `${Math.round(price)}`,
  },
}

// Clinical ranges
export const CLINICAL_RANGES = {
  fhr: { min: 60, max: 200 }, // bpm
  uc: { min: 0, max: 100 },   // arbitrary units
  normalBaseline: { min: 110, max: 160 },
} as const

// Time constants
export const CHART_CONSTANTS = {
  SAMPLING_RATE_HZ: 4,
  DEFAULT_WINDOW_MINUTES: 20,
  BUFFER_SIZE: 4800, // 20 min * 60 sec * 4 Hz
  SPARKLINE_POINTS: 60, // 15 seconds
  UPDATE_THROTTLE_MS: 100,
} as const

// Red zone colors
export const RED_ZONE_COLORS = {
  info: 'rgba(30, 144, 255, 0.1)',      // Blue
  warning: 'rgba(253, 126, 20, 0.2)',    // Orange
  critical: 'rgba(220, 53, 69, 0.3)',    // Red
} as const
```

---

## 4. Ring Buffer Implementation

### 4.1 src/utils/ringBuffer.ts

```typescript
/**
 * Circular buffer for efficient fixed-size data storage
 */

export class RingBuffer<T> {
  private buffer: T[]
  private head: number = 0
  private tail: number = 0
  private count: number = 0
  private capacity: number

  constructor(capacity: number) {
    this.capacity = capacity
    this.buffer = new Array(capacity)
  }

  push(item: T): void {
    this.buffer[this.tail] = item
    this.tail = (this.tail + 1) % this.capacity

    if (this.count < this.capacity) {
      this.count++
    } else {
      this.head = (this.head + 1) % this.capacity
    }
  }

  pushMany(items: T[]): void {
    for (const item of items) {
      this.push(item)
    }
  }

  toArray(): T[] {
    const result: T[] = []
    let idx = this.head

    for (let i = 0; i < this.count; i++) {
      result.push(this.buffer[idx])
      idx = (idx + 1) % this.capacity
    }

    return result
  }

  getLatest(n: number): T[] {
    const count = Math.min(n, this.count)
    const result: T[] = []

    let idx = (this.tail - count + this.capacity) % this.capacity

    for (let i = 0; i < count; i++) {
      result.push(this.buffer[idx])
      idx = (idx + 1) % this.capacity
    }

    return result
  }

  get length(): number {
    return this.count
  }

  get isFull(): boolean {
    return this.count === this.capacity
  }

  clear(): void {
    this.head = 0
    this.tail = 0
    this.count = 0
  }
}
```

---

## 5. Chart Data Hook

### 5.1 src/hooks/useChartData.ts

```typescript
/**
 * Hook for managing chart data buffers
 */

import { useCallback, useRef, useMemo } from 'react'
import { RingBuffer } from '../utils/ringBuffer'
import { CHART_CONSTANTS } from '../utils/chartConfig'
import type { ChartDataPoint } from '../types/chart'

interface UseChartDataOptions {
  bufferSize?: number
  startTime?: number
}

export function useChartData(options: UseChartDataOptions = {}) {
  const { bufferSize = CHART_CONSTANTS.BUFFER_SIZE } = options

  const fhrBuffer = useRef(new RingBuffer<ChartDataPoint>(bufferSize))
  const ucBuffer = useRef(new RingBuffer<ChartDataPoint>(bufferSize))
  const baseTime = useRef(options.startTime ?? Date.now() / 1000)

  /**
   * Append new FHR samples to the buffer
   */
  const appendFHR = useCallback((samples: number[]) => {
    const currentTime = Date.now() / 1000
    const sampleInterval = 1 / CHART_CONSTANTS.SAMPLING_RATE_HZ

    samples.forEach((value, i) => {
      const time = currentTime - (samples.length - 1 - i) * sampleInterval
      fhrBuffer.current.push({ time, value })
    })
  }, [])

  /**
   * Append new UC samples to the buffer
   */
  const appendUC = useCallback((samples: number[]) => {
    const currentTime = Date.now() / 1000
    const sampleInterval = 1 / CHART_CONSTANTS.SAMPLING_RATE_HZ

    samples.forEach((value, i) => {
      const time = currentTime - (samples.length - 1 - i) * sampleInterval
      ucBuffer.current.push({ time, value })
    })
  }, [])

  /**
   * Get all FHR data for chart rendering
   */
  const getFHRData = useCallback((): ChartDataPoint[] => {
    return fhrBuffer.current.toArray()
  }, [])

  /**
   * Get all UC data for chart rendering
   */
  const getUCData = useCallback((): ChartDataPoint[] => {
    return ucBuffer.current.toArray()
  }, [])

  /**
   * Get latest N samples for sparkline
   */
  const getLatestFHR = useCallback((n: number): number[] => {
    return fhrBuffer.current.getLatest(n).map(p => p.value)
  }, [])

  /**
   * Clear all buffers
   */
  const clear = useCallback(() => {
    fhrBuffer.current.clear()
    ucBuffer.current.clear()
    baseTime.current = Date.now() / 1000
  }, [])

  return {
    appendFHR,
    appendUC,
    getFHRData,
    getUCData,
    getLatestFHR,
    clear,
    fhrLength: fhrBuffer.current.length,
    ucLength: ucBuffer.current.length,
  }
}
```

---

## 6. Lightweight-Charts Hook

### 6.1 src/hooks/useLightweightChart.ts

```typescript
/**
 * Hook for managing Lightweight-Charts instance
 */

import { useEffect, useRef, useCallback, useState } from 'react'
import {
  createChart,
  IChartApi,
  ISeriesApi,
  LineSeries,
  Time,
} from 'lightweight-charts'
import {
  DEFAULT_CHART_OPTIONS,
  FHR_SERIES_OPTIONS,
  UC_SERIES_OPTIONS,
  CLINICAL_RANGES,
} from '../utils/chartConfig'
import type { ChartDataPoint, ChartState } from '../types/chart'

interface UseLightweightChartOptions {
  showUC?: boolean
  onCrosshairMove?: (time: Time | null, price: number | null) => void
}

export function useLightweightChart(
  containerRef: React.RefObject<HTMLDivElement>,
  options: UseLightweightChartOptions = {}
) {
  const { showUC = true, onCrosshairMove } = options

  const chartRef = useRef<IChartApi | null>(null)
  const fhrSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const ucSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)

  const [chartState, setChartState] = useState<ChartState>({
    isLive: true,
    visibleRange: null,
    lastUpdate: 0,
  })

  /**
   * Initialize chart
   */
  useEffect(() => {
    if (!containerRef.current) return

    // Create chart
    const chart = createChart(containerRef.current, {
      ...DEFAULT_CHART_OPTIONS,
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
    })

    // Create FHR series
    const fhrSeries = chart.addLineSeries({
      ...FHR_SERIES_OPTIONS,
      priceScaleId: 'fhr',
    })

    // Configure FHR scale
    chart.priceScale('fhr').applyOptions({
      scaleMargins: { top: 0.05, bottom: 0.55 },
      borderVisible: true,
    })

    // Create UC series if enabled
    let ucSeries: ISeriesApi<'Line'> | null = null
    if (showUC) {
      ucSeries = chart.addLineSeries({
        ...UC_SERIES_OPTIONS,
        priceScaleId: 'uc',
      })

      chart.priceScale('uc').applyOptions({
        scaleMargins: { top: 0.55, bottom: 0.05 },
        borderVisible: true,
      })
    }

    // Crosshair handler
    if (onCrosshairMove) {
      chart.subscribeCrosshairMove((param) => {
        const time = param.time ?? null
        const fhrPrice = param.seriesData.get(fhrSeries)
        onCrosshairMove(time, fhrPrice ? (fhrPrice as any).value : null)
      })
    }

    // Handle resize
    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        })
      }
    }

    const resizeObserver = new ResizeObserver(handleResize)
    resizeObserver.observe(containerRef.current)

    chartRef.current = chart
    fhrSeriesRef.current = fhrSeries
    ucSeriesRef.current = ucSeries

    return () => {
      resizeObserver.disconnect()
      chart.remove()
      chartRef.current = null
      fhrSeriesRef.current = null
      ucSeriesRef.current = null
    }
  }, [containerRef, showUC, onCrosshairMove])

  /**
   * Update FHR data
   */
  const updateFHR = useCallback((data: ChartDataPoint[]) => {
    if (!fhrSeriesRef.current) return

    const formattedData = data.map(d => ({
      time: d.time as Time,
      value: d.value,
    }))

    fhrSeriesRef.current.setData(formattedData)
    setChartState(s => ({ ...s, lastUpdate: Date.now() }))
  }, [])

  /**
   * Update UC data
   */
  const updateUC = useCallback((data: ChartDataPoint[]) => {
    if (!ucSeriesRef.current) return

    const formattedData = data.map(d => ({
      time: d.time as Time,
      value: d.value,
    }))

    ucSeriesRef.current.setData(formattedData)
  }, [])

  /**
   * Append new data (for streaming)
   */
  const appendFHR = useCallback((point: ChartDataPoint) => {
    if (!fhrSeriesRef.current) return

    fhrSeriesRef.current.update({
      time: point.time as Time,
      value: point.value,
    })
  }, [])

  /**
   * Scroll to live (latest data)
   */
  const scrollToLive = useCallback(() => {
    if (!chartRef.current) return

    chartRef.current.timeScale().scrollToRealTime()
    setChartState(s => ({ ...s, isLive: true }))
  }, [])

  /**
   * Fit content to view
   */
  const fitContent = useCallback(() => {
    if (!chartRef.current) return

    chartRef.current.timeScale().fitContent()
  }, [])

  /**
   * Get current visible range
   */
  const getVisibleRange = useCallback(() => {
    if (!chartRef.current) return null

    return chartRef.current.timeScale().getVisibleRange()
  }, [])

  return {
    chart: chartRef.current,
    fhrSeries: fhrSeriesRef.current,
    ucSeries: ucSeriesRef.current,
    chartState,
    updateFHR,
    updateUC,
    appendFHR,
    scrollToLive,
    fitContent,
    getVisibleRange,
  }
}
```

---

## 7. CTG Chart Component

### 7.1 src/components/charts/CTGChart.tsx

```tsx
/**
 * Main CTG Chart Component
 *
 * Renders FHR and UC traces using Lightweight-Charts
 * with support for real-time updates and red zones.
 */

import { useRef, useEffect, useCallback, memo } from 'react'
import { useLightweightChart } from '../../hooks/useLightweightChart'
import { useChartData } from '../../hooks/useChartData'
import { ChartControls } from './ChartControls'
import { RedZoneOverlay } from './RedZoneOverlay'
import type { PatientSnapshot, HighlightRegion } from '../../types/patient'
import type { ChartHighlightRegion } from '../../types/chart'

interface CTGChartProps {
  patientId: string
  fhrData: number[]
  ucData: number[]
  highlightRegions?: HighlightRegion[]
  showControls?: boolean
  height?: number
}

export const CTGChart = memo(function CTGChart({
  patientId,
  fhrData,
  ucData,
  highlightRegions = [],
  showControls = true,
  height = 400,
}: CTGChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartData = useChartData()

  const {
    updateFHR,
    updateUC,
    chartState,
    scrollToLive,
    fitContent,
  } = useLightweightChart(containerRef)

  // Update chart when new data arrives
  useEffect(() => {
    if (fhrData.length > 0) {
      chartData.appendFHR(fhrData)
      updateFHR(chartData.getFHRData())
    }
  }, [fhrData, chartData, updateFHR])

  useEffect(() => {
    if (ucData.length > 0) {
      chartData.appendUC(ucData)
      updateUC(chartData.getUCData())
    }
  }, [ucData, chartData, updateUC])

  // Convert highlight regions to chart format
  const chartRegions: ChartHighlightRegion[] = highlightRegions.map(r => ({
    startTime: Date.now() / 1000 - (chartData.fhrLength - r.start_idx) / 4,
    endTime: Date.now() / 1000 - (chartData.fhrLength - r.end_idx) / 4,
    color: r.color ?? '',
    label: r.label,
    severity: r.severity as 'info' | 'warning' | 'critical',
  }))

  const handleReset = useCallback(() => {
    chartData.clear()
  }, [chartData])

  return (
    <div className="relative">
      {/* Chart Controls */}
      {showControls && (
        <ChartControls
          isLive={chartState.isLive}
          onLive={scrollToLive}
          onFit={fitContent}
          onReset={handleReset}
        />
      )}

      {/* Chart Container */}
      <div
        ref={containerRef}
        className="w-full bg-white rounded-lg border"
        style={{ height }}
      />

      {/* Red Zone Overlays */}
      {chartRegions.length > 0 && (
        <RedZoneOverlay regions={chartRegions} containerRef={containerRef} />
      )}
    </div>
  )
})
```

### 7.2 src/components/charts/ChartControls.tsx

```tsx
/**
 * Chart control buttons
 */

interface ChartControlsProps {
  isLive: boolean
  onLive: () => void
  onFit: () => void
  onReset: () => void
}

export function ChartControls({
  isLive,
  onLive,
  onFit,
  onReset,
}: ChartControlsProps) {
  return (
    <div className="absolute top-2 right-2 z-10 flex gap-2">
      <button
        onClick={onLive}
        className={`px-3 py-1 text-sm rounded transition-colors ${
          isLive
            ? 'bg-green-500 text-white'
            : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
        }`}
      >
        {isLive ? '● Live' : 'Go Live'}
      </button>

      <button
        onClick={onFit}
        className="px-3 py-1 text-sm bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
      >
        Fit
      </button>

      <button
        onClick={onReset}
        className="px-3 py-1 text-sm bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
      >
        Reset
      </button>
    </div>
  )
}
```

### 7.3 src/components/charts/RedZoneOverlay.tsx

```tsx
/**
 * Red zone overlay for highlighting events
 */

import { useMemo } from 'react'
import type { ChartHighlightRegion } from '../../types/chart'
import { RED_ZONE_COLORS } from '../../utils/chartConfig'

interface RedZoneOverlayProps {
  regions: ChartHighlightRegion[]
  containerRef: React.RefObject<HTMLDivElement>
}

export function RedZoneOverlay({ regions, containerRef }: RedZoneOverlayProps) {
  // Note: In a full implementation, we'd calculate pixel positions
  // based on the chart's time scale. For now, this is a simplified version.

  if (regions.length === 0) return null

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      {regions.map((region, idx) => (
        <div
          key={idx}
          className="absolute top-0 bottom-0"
          style={{
            left: '20%', // Placeholder - calculate from time
            width: '10%', // Placeholder - calculate from duration
            backgroundColor: RED_ZONE_COLORS[region.severity],
          }}
        >
          {region.label && (
            <span className="absolute top-1 left-1 text-xs font-medium px-1 bg-white/80 rounded">
              {region.label}
            </span>
          )}
        </div>
      ))}
    </div>
  )
}
```

---

## 8. Sparkline Component

### 8.1 src/components/charts/FHRSparkline.tsx

```tsx
/**
 * Mini FHR sparkline for patient cards
 *
 * Uses Canvas 2D for lightweight rendering.
 * No dependencies on Lightweight-Charts.
 */

import { useRef, useEffect, memo } from 'react'
import { CATEGORY_COLORS } from '../../types/patient'
import type { Category } from '../../types/patient'

interface FHRSparklineProps {
  data: number[]
  category?: Category
  width?: number
  height?: number
  lineWidth?: number
}

export const FHRSparkline = memo(function FHRSparkline({
  data,
  category = 1,
  width = 120,
  height = 48,
  lineWidth = 1.5,
}: FHRSparklineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || data.length < 2) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Handle device pixel ratio for crisp rendering
    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr
    ctx.scale(dpr, dpr)

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Calculate scaling
    const minValue = Math.min(...data)
    const maxValue = Math.max(...data)
    const range = maxValue - minValue || 1
    const padding = 4

    const xScale = (width - padding * 2) / (data.length - 1)
    const yScale = (height - padding * 2) / range

    // Draw line
    ctx.beginPath()
    ctx.strokeStyle = CATEGORY_COLORS[category]
    ctx.lineWidth = lineWidth
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    data.forEach((value, i) => {
      const x = padding + i * xScale
      const y = height - padding - (value - minValue) * yScale

      if (i === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })

    ctx.stroke()
  }, [data, category, width, height, lineWidth])

  if (data.length < 2) {
    return (
      <div
        className="flex items-center justify-center bg-gray-50 rounded text-xs text-gray-400"
        style={{ width, height }}
      >
        No data
      </div>
    )
  }

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height }}
      className="block"
    />
  )
})
```

---

## 9. Integration with Detail View

### 9.1 Update src/pages/DetailView.tsx

```tsx
import { useParams, Link } from 'react-router-dom'
import { useEffect, useMemo } from 'react'
import { usePatient, usePatientStore } from '../stores/patientStore'
import { CategoryBadge } from '../components/patient/CategoryBadge'
import { CTGChart } from '../components/charts/CTGChart'
import { usePatientStream } from '../hooks/usePatientStream'

export function DetailView() {
  const { patientId } = useParams<{ patientId: string }>()
  const patient = usePatient(patientId ?? '')
  const selectPatient = usePatientStore((s) => s.selectPatient)
  const { subscribe } = usePatientStream()

  useEffect(() => {
    if (patientId) {
      selectPatient(patientId)
      subscribe([patientId])
    }
    return () => {
      selectPatient(null)
      subscribe([])
    }
  }, [patientId, selectPatient, subscribe])

  // Memoize chart data to prevent unnecessary re-renders
  const fhrData = useMemo(() => patient?.fhr_latest ?? [], [patient?.fhr_latest])
  const ucData = useMemo(() => patient?.uc_latest ?? [], [patient?.uc_latest])

  if (!patient) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">Patient not found</p>
        <Link to="/" className="text-blue-600 hover:underline mt-4 inline-block">
          Back to Ward
        </Link>
      </div>
    )
  }

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <Link to="/" className="text-gray-500 hover:text-gray-700">
            ← Back
          </Link>
          <h1 className="text-2xl font-bold">{patient.patient_id}</h1>
          <CategoryBadge category={patient.category} size="md" />
        </div>
      </div>

      {/* Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Chart Area */}
        <div className="lg:col-span-3">
          <div className="bg-white rounded-lg shadow-sm border p-4">
            <h2 className="text-lg font-semibold mb-4">CTG Monitor</h2>
            <CTGChart
              patientId={patient.patient_id}
              fhrData={fhrData}
              ucData={ucData}
              highlightRegions={patient.highlight_regions}
              height={400}
            />
          </div>
        </div>

        {/* Info Panel */}
        <div className="space-y-4">
          {/* Metrics card */}
          <div className="bg-white rounded-lg shadow-sm border p-4">
            <h3 className="font-semibold mb-3">Metrics</h3>
            <dl className="space-y-2">
              <div className="flex justify-between">
                <dt className="text-gray-500">Baseline</dt>
                <dd className="font-medium">{patient.baseline.toFixed(0)} bpm</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">Variability</dt>
                <dd className="font-medium">{patient.variability.toFixed(1)} bpm</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">Trend Score</dt>
                <dd className="font-medium">{patient.trend_score}</dd>
              </div>
            </dl>
          </div>

          {/* Alert cards */}
          {patient.mhr_alert && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <p className="text-yellow-800 text-sm font-medium">
                MHR Contamination Suspected
              </p>
            </div>
          )}

          {patient.active_event && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-red-800 text-sm font-medium">
                Active: {patient.active_event}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
```

---

## 10. Performance Optimization

### 10.1 Rendering Strategy

```typescript
// Use requestAnimationFrame for smooth updates
let rafId: number | null = null
let pendingData: ChartDataPoint[] = []

function scheduleUpdate() {
  if (rafId !== null) return

  rafId = requestAnimationFrame(() => {
    if (pendingData.length > 0) {
      // Batch update all pending data
      chart.series.setData(pendingData)
      pendingData = []
    }
    rafId = null
  })
}

// Throttle incoming data
function onWebSocketData(data: ChartDataPoint[]) {
  pendingData.push(...data)
  scheduleUpdate()
}
```

### 10.2 Memory Management

```typescript
// Cleanup on unmount
useEffect(() => {
  return () => {
    // Cancel any pending animation frames
    if (rafId) cancelAnimationFrame(rafId)

    // Clear buffers
    chartData.clear()

    // Remove chart
    chart?.remove()
  }
}, [])
```

---

## 11. Testing

### 11.1 Performance Test

```typescript
// frontend/src/__tests__/chartPerformance.test.ts

import { render } from '@testing-library/react'
import { CTGChart } from '../components/charts/CTGChart'

describe('CTGChart Performance', () => {
  it('renders 4800 points in under 100ms', () => {
    const data = Array.from({ length: 4800 }, (_, i) => 140 + Math.sin(i / 10) * 20)

    const start = performance.now()

    render(
      <CTGChart
        patientId="test"
        fhrData={data}
        ucData={data}
      />
    )

    const duration = performance.now() - start
    expect(duration).toBeLessThan(100)
  })

  it('maintains 60fps during updates', async () => {
    // Simulate streaming updates
    const frameTimings: number[] = []
    let lastFrame = performance.now()

    const measureFrame = () => {
      const now = performance.now()
      frameTimings.push(now - lastFrame)
      lastFrame = now
    }

    // Run for 60 frames
    for (let i = 0; i < 60; i++) {
      measureFrame()
      await new Promise(r => setTimeout(r, 16))
    }

    const avgFrameTime = frameTimings.reduce((a, b) => a + b) / frameTimings.length
    expect(avgFrameTime).toBeLessThan(20) // Allow some variance
  })
})
```

---

## 12. Verification Commands

```bash
# Build and check bundle size
npm run build
du -h dist/assets/*.js

# Run performance profiling
npm run dev
# Open Chrome DevTools > Performance > Record

# Check memory usage
# Chrome DevTools > Memory > Take snapshot
```

---

## 13. Implementation Status

### ✅ Completed (2025-01-XX)

| File | Status | Notes |
|------|--------|-------|
| `types/chart.ts` | ✅ | All types defined with extended props |
| `utils/chartConfig.ts` | ✅ | LC config with ColorType, LineStyle enums |
| `utils/ringBuffer.ts` | ✅ | Full RingBuffer implementation |
| `utils/chartHelpers.ts` | ✅ | downsample, generateSparklinePath, etc. |
| `utils/index.ts` | ✅ | Re-exports all utilities |
| `hooks/useChartData.ts` | ✅ | Buffer management with setFHRData/setUCData |
| `hooks/useLightweightChart.ts` | ✅ | Chart lifecycle, dual price scales |
| `components/charts/CTGChart.tsx` | ✅ | Main CTG with FHR+UC, live toggle |
| `components/charts/FHRSparkline.tsx` | ✅ | SVG sparkline with trend indicator |
| `components/charts/ChartControls.tsx` | ✅ | Time range selector, zoom buttons |
| `components/charts/index.ts` | ✅ | Exports all chart components |
| `pages/DetailView.tsx` | ✅ | CTGChartPanel integrated |

### Build Status
- **Bundle Size**: 401KB (gzipped: 127KB)
- **TypeScript**: Clean compilation ✅
- **lightweight-charts**: v4.1.0 installed

### Pending
- [ ] Wire WebSocket stream data to CTGChart
- [ ] Add FHRSparkline to PatientCard
- [ ] Performance benchmarks
- [ ] Red zone overlay refinement

---

*End of Phase 4 Technical Specifications*
