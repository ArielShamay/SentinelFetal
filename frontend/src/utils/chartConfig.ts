/**
 * Lightweight-Charts configuration for CTG visualization
 */

import { ColorType, LineStyle, CrosshairMode } from 'lightweight-charts'
import type { DeepPartial, ChartOptions, LineSeriesOptions } from 'lightweight-charts'

// Default chart options for dark theme
export const DEFAULT_CHART_OPTIONS: DeepPartial<ChartOptions> = {
  layout: {
    background: { type: ColorType.Solid, color: '#1f2937' }, // gray-800
    textColor: '#9ca3af', // gray-400
    fontSize: 11,
    fontFamily: "'Inter', system-ui, sans-serif",
  },
  grid: {
    vertLines: { color: '#374151', style: LineStyle.Solid, visible: true },
    horzLines: { color: '#374151', style: LineStyle.Solid, visible: true },
  },
  crosshair: {
    mode: CrosshairMode.Normal,
    vertLine: {
      color: '#6b7280',
      width: 1,
      style: LineStyle.Dashed,
      labelBackgroundColor: '#4b5563',
      visible: true,
      labelVisible: true,
    },
    horzLine: {
      color: '#6b7280',
      width: 1,
      style: LineStyle.Dashed,
      labelBackgroundColor: '#4b5563',
      visible: true,
      labelVisible: true,
    },
  },
  timeScale: {
    timeVisible: true,
    secondsVisible: true,
    borderColor: '#374151',
    rightOffset: 5,
    barSpacing: 6,
  },
  rightPriceScale: {
    borderColor: '#374151',
    scaleMargins: { top: 0.05, bottom: 0.05 },
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
    axisDoubleClickReset: true,
  },
}

// Light theme chart options
export const LIGHT_CHART_OPTIONS: DeepPartial<ChartOptions> = {
  layout: {
    background: { type: ColorType.Solid, color: '#ffffff' },
    textColor: '#374151', // gray-700
    fontSize: 11,
    fontFamily: "'Inter', system-ui, sans-serif",
  },
  grid: {
    vertLines: { color: '#e5e7eb', style: LineStyle.Solid, visible: true }, // gray-200
    horzLines: { color: '#e5e7eb', style: LineStyle.Solid, visible: true },
  },
  crosshair: {
    mode: CrosshairMode.Normal,
    vertLine: {
      color: '#9ca3af',
      width: 1,
      style: LineStyle.Dashed,
      labelBackgroundColor: '#f3f4f6',
      visible: true,
      labelVisible: true,
    },
    horzLine: {
      color: '#9ca3af',
      width: 1,
      style: LineStyle.Dashed,
      labelBackgroundColor: '#f3f4f6',
      visible: true,
      labelVisible: true,
    },
  },
  timeScale: {
    timeVisible: true,
    secondsVisible: true,
    borderColor: '#e5e7eb',
    rightOffset: 5,
    barSpacing: 6,
  },
  rightPriceScale: {
    borderColor: '#e5e7eb',
    scaleMargins: { top: 0.05, bottom: 0.05 },
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
    axisDoubleClickReset: true,
  },
}

// FHR series options (blue line)
export const FHR_SERIES_OPTIONS: DeepPartial<LineSeriesOptions> = {
  color: '#3b82f6', // blue-500
  lineWidth: 2,
  crosshairMarkerVisible: true,
  crosshairMarkerRadius: 4,
}

// UC series options (orange line)
export const UC_SERIES_OPTIONS: DeepPartial<LineSeriesOptions> = {
  color: '#f97316', // orange-500
  lineWidth: 2,
  crosshairMarkerVisible: true,
  crosshairMarkerRadius: 4,
}

// Clinical ranges for FHR
export const CLINICAL_RANGES = {
  fhr: { min: 60, max: 200 },      // bpm
  uc: { min: 0, max: 100 },        // arbitrary units
  normalBaseline: { min: 110, max: 160 },
  bradycardia: 110,
  tachycardia: 160,
} as const

// Time and buffer constants
export const CHART_CONSTANTS = {
  SAMPLING_RATE_HZ: 4,
  DEFAULT_WINDOW_MINUTES: 20,
  BUFFER_SIZE: 4800,            // 20 min * 60 sec * 4 Hz
  SPARKLINE_POINTS: 60,         // 15 seconds at 4Hz
  UPDATE_THROTTLE_MS: 100,
  SPARKLINE_UPDATE_MS: 500,
} as const

// Red zone colors by severity
export const RED_ZONE_COLORS = {
  info: 'rgba(59, 130, 246, 0.15)',      // Blue
  warning: 'rgba(249, 115, 22, 0.25)',   // Orange
  critical: 'rgba(239, 68, 68, 0.35)',   // Red
  bradycardia: 'rgba(59, 130, 246, 0.3)', // Blue for low FHR
  tachycardia: 'rgba(239, 68, 68, 0.3)',  // Red for high FHR
} as const

// Category colors for sparklines
export const CATEGORY_SPARKLINE_COLORS = {
  1: '#22c55e', // green-500 (Normal)
  2: '#f97316', // orange-500 (Suspicious)
  3: '#ef4444', // red-500 (Pathological)
  normal: '#22c55e',
  suspicious: '#f97316',
  pathological: '#ef4444',
} as const

// Reference lines configuration
export const REFERENCE_LINES = {
  fhrBaseline: {
    upper: { value: 160, color: '#fbbf24', style: 2 }, // yellow dashed
    lower: { value: 110, color: '#fbbf24', style: 2 },
  },
} as const
