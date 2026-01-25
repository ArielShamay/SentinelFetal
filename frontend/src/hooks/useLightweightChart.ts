/**
 * Hook for managing lightweight-charts instance lifecycle
 */

import { useRef, useEffect, useCallback } from 'react'
import { createChart, IChartApi, ISeriesApi, LineData, Time, PriceScaleMode } from 'lightweight-charts'
import { DEFAULT_CHART_OPTIONS, LIGHT_CHART_OPTIONS, FHR_SERIES_OPTIONS, UC_SERIES_OPTIONS } from '../utils/chartConfig'
import type { ChartDataPoint } from '../types/chart'

interface UseLightweightChartOptions {
  container: HTMLDivElement | null
  autoSize?: boolean
  darkMode?: boolean
}

interface ChartReturn {
  chart: IChartApi | null
  fhrSeries: ISeriesApi<'Line'> | null
  ucSeries: ISeriesApi<'Line'> | null
  updateFHRData: (data: ChartDataPoint[]) => void
  updateUCData: (data: ChartDataPoint[]) => void
  setTimeRange: (start: Time, end: Time) => void
  fitContent: () => void
  scrollToRealTime: () => void
  zoomIn: () => void
  zoomOut: () => void
  panLeft: () => void
  panRight: () => void
}

/**
 * Convert our chart data to lightweight-charts format
 */
function toLightweightData(data: ChartDataPoint[]): LineData<Time>[] {
  return data.map(point => ({
    time: point.time as Time,
    value: point.value
  }))
}

export function useLightweightChart(options: UseLightweightChartOptions): ChartReturn {
  const { container, autoSize = true, darkMode = true } = options

  const chartRef = useRef<IChartApi | null>(null)
  const fhrSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const ucSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)

  // Initialize chart
  useEffect(() => {
    if (!container) return

    // Select chart options based on theme
    const chartOptions = darkMode ? DEFAULT_CHART_OPTIONS : LIGHT_CHART_OPTIONS

    // Create chart instance
    const chart = createChart(container, {
      ...chartOptions,
      width: container.clientWidth,
      height: container.clientHeight,
    })

    chartRef.current = chart

    // Create FHR series (top pane)
    const fhrSeries = chart.addLineSeries({
      ...FHR_SERIES_OPTIONS,
      priceScaleId: 'right',
    })
    fhrSeriesRef.current = fhrSeries
    fhrSeries.applyOptions({
      autoscaleInfoProvider: () => ({
        priceRange: { minValue: 50, maxValue: 210 },
      }),
    })

    // Create UC series (bottom pane)
    const ucSeries = chart.addLineSeries({
      ...UC_SERIES_OPTIONS,
      priceScaleId: 'left',
    })
    ucSeriesRef.current = ucSeries
    ucSeries.applyOptions({
      autoscaleInfoProvider: () => ({
        priceRange: { minValue: 0, maxValue: 100 },
      }),
    })

    // Configure price scales with theme-appropriate colors
    const borderColor = darkMode ? '#3d3d3d' : '#e5e7eb'

    const fhrScale = chart.priceScale('right')
    fhrScale.applyOptions({
      scaleMargins: {
        top: 0.05,
        bottom: 0.55,
      },
      borderVisible: true,
      borderColor,
      mode: PriceScaleMode.Normal,
      visible: true,
    })
    const ucScale = chart.priceScale('left')
    ucScale.applyOptions({
      scaleMargins: {
        top: 0.55,
        bottom: 0.05,
      },
      borderVisible: true,
      borderColor,
      mode: PriceScaleMode.Normal,
      visible: true,
    })

    // Handle resize
    const resizeObserver = autoSize ? new ResizeObserver(entries => {
      const entry = entries[0]
      if (entry && chartRef.current) {
        const { width, height } = entry.contentRect
        chartRef.current.applyOptions({ width, height })
      }
    }) : null

    if (resizeObserver) {
      resizeObserver.observe(container)
    }

    // Cleanup
    return () => {
      if (resizeObserver) {
        resizeObserver.disconnect()
      }
      chart.remove()
      chartRef.current = null
      fhrSeriesRef.current = null
      ucSeriesRef.current = null
    }
  }, [container, autoSize, darkMode])

  /**
   * Update FHR series data
   */
  const updateFHRData = useCallback((data: ChartDataPoint[]) => {
    if (!fhrSeriesRef.current || data.length === 0) return
    fhrSeriesRef.current.setData(toLightweightData(data))
  }, [])

  /**
   * Update UC series data
   */
  const updateUCData = useCallback((data: ChartDataPoint[]) => {
    if (!ucSeriesRef.current || data.length === 0) return
    ucSeriesRef.current.setData(toLightweightData(data))
  }, [])

  /**
   * Set visible time range
   */
  const setTimeRange = useCallback((start: Time, end: Time) => {
    if (!chartRef.current) return
    chartRef.current.timeScale().setVisibleRange({ from: start, to: end })
  }, [])

  /**
   * Fit all content in view
   */
  const fitContent = useCallback(() => {
    if (!chartRef.current) return
    chartRef.current.timeScale().fitContent()
  }, [])

  /**
   * Scroll to real-time (latest data)
   */
  const scrollToRealTime = useCallback(() => {
    if (!chartRef.current) return
    chartRef.current.timeScale().scrollToRealTime()
  }, [])

  /**
   * Zoom in - reduce visible time range by 20%
   */
  const zoomIn = useCallback(() => {
    if (!chartRef.current) return
    const timeScale = chartRef.current.timeScale()
    const visibleRange = timeScale.getVisibleLogicalRange()
    if (!visibleRange) return

    const { from, to } = visibleRange
    const range = to - from
    const newRange = range * 0.8 // Zoom in by 20%
    const center = (from + to) / 2
    const newFrom = center - newRange / 2
    const newTo = center + newRange / 2
    timeScale.setVisibleLogicalRange({ from: newFrom, to: newTo })
  }, [])

  /**
   * Zoom out - increase visible time range by 25%
   */
  const zoomOut = useCallback(() => {
    if (!chartRef.current) return
    const timeScale = chartRef.current.timeScale()
    const visibleRange = timeScale.getVisibleLogicalRange()
    if (!visibleRange) return

    const { from, to } = visibleRange
    const range = to - from
    const newRange = range * 1.25 // Zoom out by 25%
    const center = (from + to) / 2
    const newFrom = center - newRange / 2
    const newTo = center + newRange / 2

    timeScale.setVisibleLogicalRange({ from: newFrom, to: newTo })
  }, [])

  /**
   * Pan left - scroll back in time
   */
  const panLeft = useCallback(() => {
    if (!chartRef.current) return
    const timeScale = chartRef.current.timeScale()
    const visibleRange = timeScale.getVisibleLogicalRange()
    if (!visibleRange) return

    const { from, to } = visibleRange
    const range = to - from
    const shift = range * 0.2 // Pan by 20% of visible range

    timeScale.setVisibleLogicalRange({ from: from - shift, to: to - shift })
  }, [])

  /**
   * Pan right - scroll forward in time
   */
  const panRight = useCallback(() => {
    if (!chartRef.current) return
    const timeScale = chartRef.current.timeScale()
    const visibleRange = timeScale.getVisibleLogicalRange()
    if (!visibleRange) return

    const { from, to } = visibleRange
    const range = to - from
    const shift = range * 0.2 // Pan by 20% of visible range

    timeScale.setVisibleLogicalRange({ from: from + shift, to: to + shift })
  }, [])

  return {
    chart: chartRef.current,
    fhrSeries: fhrSeriesRef.current,
    ucSeries: ucSeriesRef.current,
    updateFHRData,
    updateUCData,
    setTimeRange,
    fitContent,
    scrollToRealTime,
    zoomIn,
    zoomOut,
    panLeft,
    panRight,
  }
}
