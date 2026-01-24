/**
 * Hook for managing lightweight-charts instance lifecycle
 */

import { useRef, useEffect, useCallback } from 'react'
import { createChart, IChartApi, ISeriesApi, LineData, Time } from 'lightweight-charts'
import { DEFAULT_CHART_OPTIONS, FHR_SERIES_OPTIONS, UC_SERIES_OPTIONS } from '../utils/chartConfig'
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

    // Create chart instance
    const chart = createChart(container, {
      ...DEFAULT_CHART_OPTIONS,
      width: container.clientWidth,
      height: container.clientHeight,
    })

    chartRef.current = chart

    // Create FHR series (top pane)
    const fhrSeries = chart.addLineSeries({
      ...FHR_SERIES_OPTIONS,
      priceScaleId: 'fhr',
    })
    fhrSeriesRef.current = fhrSeries

    // Create UC series (bottom pane)
    const ucSeries = chart.addLineSeries({
      ...UC_SERIES_OPTIONS,
      priceScaleId: 'uc',
    })
    ucSeriesRef.current = ucSeries

    // Configure price scales
    chart.priceScale('fhr').applyOptions({
      scaleMargins: {
        top: 0.05,
        bottom: 0.55,
      },
      borderVisible: true,
      borderColor: '#3d3d3d',
    })

    chart.priceScale('uc').applyOptions({
      scaleMargins: {
        top: 0.55,
        bottom: 0.05,
      },
      borderVisible: true,
      borderColor: '#3d3d3d',
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

  return {
    chart: chartRef.current,
    fhrSeries: fhrSeriesRef.current,
    ucSeries: ucSeriesRef.current,
    updateFHRData,
    updateUCData,
    setTimeRange,
    fitContent,
    scrollToRealTime,
  }
}
