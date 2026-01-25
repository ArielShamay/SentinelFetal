/**
 * FHR Sparkline Component - Mini CTG chart for patient cards
 * 
 * Lightweight-Charts based mini CTG with axes and fixed clinical ranges.
 */

import { memo, useEffect, useMemo, useRef, useState } from 'react'
import { useLightweightChart } from '../../hooks/useLightweightChart'
import { CHART_CONSTANTS } from '../../utils/chartConfig'
import type { SparklineProps } from '../../types/chart'

const FHRSparkline = memo(({
  fhrData,
  ucData = [],
  timestamps,
  width = 240,
  height = 120,
  className = '',
}: SparklineProps) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const [container, setContainer] = useState<HTMLDivElement | null>(null)

  useEffect(() => {
    setContainer(containerRef.current)
  }, [])

  const { updateFHRData, updateUCData } = useLightweightChart({
    container,
    autoSize: true,
    darkMode: false,
  })

  const { fhrPoints, ucPoints } = useMemo(() => {
    if (!fhrData || fhrData.length === 0) {
      return { fhrPoints: [], ucPoints: [] }
    }

    const sampleInterval = 1 / CHART_CONSTANTS.SAMPLING_RATE_HZ
    const resolvedTimestamps = timestamps && timestamps.length === fhrData.length
      ? timestamps
      : fhrData.map((_, idx) => (Date.now() / 1000) - (fhrData.length - 1 - idx) * sampleInterval)

    const fhrPoints = fhrData.map((value, idx) => ({
      time: resolvedTimestamps[idx],
      value,
    }))

    const ucPoints = ucData && ucData.length === fhrData.length
      ? ucData.map((value, idx) => ({
          time: resolvedTimestamps[idx],
          value,
        }))
      : []

    return { fhrPoints, ucPoints }
  }, [fhrData, ucData, timestamps])

  useEffect(() => {
    if (fhrPoints.length) {
      updateFHRData(fhrPoints)
    }
    if (ucPoints.length) {
      updateUCData(ucPoints)
    }
  }, [fhrPoints, ucPoints, updateFHRData, updateUCData])

  if (!fhrData || fhrData.length < 2) {
    return (
      <div
        className={`flex items-center justify-center text-gray-500 text-xs ${className}`}
        style={{ width, height }}
      >
        No data
      </div>
    )
  }

  return (
    <div
      className={`relative rounded border border-gray-200 bg-white ${className}`}
      style={{ width, height }}
    >
      <div ref={containerRef} className="w-full h-full" />
    </div>
  )
})

FHRSparkline.displayName = 'FHRSparkline'

export default FHRSparkline
