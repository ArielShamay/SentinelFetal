/**
 * Hook for managing chart data buffers
 */

import { useCallback, useRef, useMemo } from 'react'
import { RingBuffer } from '../utils/ringBuffer'
import { CHART_CONSTANTS } from '../utils/chartConfig'
import type { ChartDataPoint } from '../types/chart'

interface UseChartDataOptions {
  bufferSize?: number
  samplingRate?: number
}

interface ChartDataReturn {
  appendFHR: (samples: number[], timestamps?: number[]) => void
  appendUC: (samples: number[], timestamps?: number[]) => void
  setFHRData: (data: ChartDataPoint[]) => void
  setUCData: (data: ChartDataPoint[]) => void
  getFHRData: () => ChartDataPoint[]
  getUCData: () => ChartDataPoint[]
  getLatestFHR: (n: number) => number[]
  getLatestUC: (n: number) => number[]
  clear: () => void
  fhrLength: number
  ucLength: number
}

export function useChartData(options: UseChartDataOptions = {}): ChartDataReturn {
  const { 
    bufferSize = CHART_CONSTANTS.BUFFER_SIZE,
    samplingRate = CHART_CONSTANTS.SAMPLING_RATE_HZ
  } = options

  const fhrBuffer = useRef(new RingBuffer<ChartDataPoint>(bufferSize))
  const ucBuffer = useRef(new RingBuffer<ChartDataPoint>(bufferSize))

  /**
   * Append new FHR samples to the buffer
   */
  const appendFHR = useCallback((samples: number[], timestamps?: number[]) => {
    const currentTime = Date.now() / 1000
    const sampleInterval = 1 / samplingRate

    samples.forEach((value, i) => {
      const time = timestamps?.[i] ?? (currentTime - (samples.length - 1 - i) * sampleInterval)
      fhrBuffer.current.push({ time, value })
    })
  }, [samplingRate])

  /**
   * Append new UC samples to the buffer
   */
  const appendUC = useCallback((samples: number[], timestamps?: number[]) => {
    const currentTime = Date.now() / 1000
    const sampleInterval = 1 / samplingRate

    samples.forEach((value, i) => {
      const time = timestamps?.[i] ?? (currentTime - (samples.length - 1 - i) * sampleInterval)
      ucBuffer.current.push({ time, value })
    })
  }, [samplingRate])

  /**
   * Set FHR data directly (for initial load)
   */
  const setFHRData = useCallback((data: ChartDataPoint[]) => {
    fhrBuffer.current.clear()
    data.forEach(point => fhrBuffer.current.push(point))
  }, [])

  /**
   * Set UC data directly (for initial load)
   */
  const setUCData = useCallback((data: ChartDataPoint[]) => {
    ucBuffer.current.clear()
    data.forEach(point => ucBuffer.current.push(point))
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
   * Get latest N FHR samples for sparkline (values only)
   */
  const getLatestFHR = useCallback((n: number): number[] => {
    return fhrBuffer.current.getLatest(n).map(p => p.value)
  }, [])

  /**
   * Get latest N UC samples (values only)
   */
  const getLatestUC = useCallback((n: number): number[] => {
    return ucBuffer.current.getLatest(n).map(p => p.value)
  }, [])

  /**
   * Clear all buffers
   */
  const clear = useCallback(() => {
    fhrBuffer.current.clear()
    ucBuffer.current.clear()
  }, [])

  return useMemo(() => ({
    appendFHR,
    appendUC,
    setFHRData,
    setUCData,
    getFHRData,
    getUCData,
    getLatestFHR,
    getLatestUC,
    clear,
    get fhrLength() { return fhrBuffer.current.length },
    get ucLength() { return ucBuffer.current.length },
  }), [appendFHR, appendUC, setFHRData, setUCData, getFHRData, getUCData, getLatestFHR, getLatestUC, clear])
}
