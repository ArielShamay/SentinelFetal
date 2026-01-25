/**
 * Chart helper functions
 */

import type { ChartDataPoint } from '../types/chart'
import { CHART_CONSTANTS, CLINICAL_RANGES } from './chartConfig'

/**
 * Convert arrays to chart data points with timestamps
 */
export function arrayToChartData(
  values: number[],
  timestamps?: number[]
): ChartDataPoint[] {
  const now = Date.now() / 1000
  const interval = 1 / CHART_CONSTANTS.SAMPLING_RATE_HZ

  return values.map((value, i) => ({
    time: timestamps?.[i] ?? (now - (values.length - 1 - i) * interval),
    value,
  }))
}

/**
 * Downsample data for sparklines
 */
export function downsample(data: number[], targetLength: number): number[] {
  if (data.length <= targetLength) return data

  const result: number[] = []
  const ratio = data.length / targetLength

  for (let i = 0; i < targetLength; i++) {
    const start = Math.floor(i * ratio)
    const end = Math.floor((i + 1) * ratio)
    
    // Use max value in window for peaks visibility
    let max = data[start]
    for (let j = start + 1; j < end && j < data.length; j++) {
      if (data[j] > max) max = data[j]
    }
    result.push(max)
  }

  return result
}

/**
 * Calculate visible time range for chart
 */
export function calculateTimeRange(
  minutes: number = CHART_CONSTANTS.DEFAULT_WINDOW_MINUTES
): { from: number; to: number } {
  const now = Date.now() / 1000
  return {
    from: now - minutes * 60,
    to: now,
  }
}

/**
 * Check if FHR value is in normal range
 */
export function isNormalFHR(value: number): boolean {
  return value >= CLINICAL_RANGES.normalBaseline.min && 
         value <= CLINICAL_RANGES.normalBaseline.max
}

/**
 * Get FHR status color
 */
export function getFHRStatusColor(value: number): string {
  if (value < CLINICAL_RANGES.bradycardia) return '#ef4444' // red - bradycardia
  if (value > CLINICAL_RANGES.tachycardia) return '#ef4444' // red - tachycardia
  return '#22c55e' // green - normal
}

/**
 * Format time for display
 */
export function formatChartTime(timestamp: number): string {
  const date = new Date(timestamp * 1000)
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

/**
 * Throttle function for chart updates
 */
export function throttle<T extends (...args: unknown[]) => unknown>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle = false

  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args)
      inThrottle = true
      setTimeout(() => {
        inThrottle = false
      }, limit)
    }
  }
}

/**
 * Generate SVG path for sparkline
 */
export function generateSparklinePath(
  data: number[],
  width: number,
  height: number,
  padding: number = 2,
  range?: { min: number; max: number }
): string {
  if (data.length < 2) return ''

  const minVal = range ? range.min : Math.min(...data)
  const maxVal = range ? range.max : Math.max(...data)
  const valueRange = maxVal - minVal || 1

  const innerWidth = width - padding * 2
  const innerHeight = height - padding * 2

  const xStep = innerWidth / (data.length - 1)

  const points = data.map((value, i) => {
    const x = padding + i * xStep
    const y = padding + innerHeight - ((value - minVal) / valueRange) * innerHeight
    return `${x},${y}`
  })

  return `M ${points.join(' L ')}`
}
