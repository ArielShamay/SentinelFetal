/**
 * CTG Chart Component - Main real-time CTG visualization
 *
 * Displays FHR (top) and UC (bottom) traces using lightweight-charts.
 */

import { useRef, useEffect, useState, useCallback, useMemo, memo } from 'react'
import type { Time } from 'lightweight-charts'
import { useLightweightChart } from '../../hooks/useLightweightChart'
import { useChartData } from '../../hooks/useChartData'
import { CLINICAL_RANGES, RED_ZONE_COLORS, CHART_CONSTANTS } from '../../utils/chartConfig'
import type { CTGChartProps, ChartHighlightRegion } from '../../types/chart'

const CTGChart = memo(({
  snapshot,
  liveUpdate,
  fhrData,
  ucData,
  timestamps,
  width = '100%',
  height = 400,
  showControls = true,
  isLive = true,
  onTimeRangeChange: _onTimeRangeChange,
}: CTGChartProps) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const [isFollowingRealTime, setIsFollowingRealTime] = useState(true)
  const [container, setContainer] = useState<HTMLDivElement | null>(null)
  const initializedRef = useRef(false)

  useEffect(() => {
    setContainer(containerRef.current)
  }, [])

  const {
    chart,
    updateFHRData,
    updateUCData,
    fitContent,
    scrollToRealTime,
    zoomIn,
    zoomOut,
    panLeft,
    panRight,
  } = useLightweightChart({
    container,
    autoSize: true,
    darkMode: false,
  })

  const chartData = useChartData({
    bufferSize: CHART_CONSTANTS.BUFFER_SIZE,
    samplingRate: CHART_CONSTANTS.SAMPLING_RATE_HZ,
  })

  const buildDataPoints = useCallback((samples: number[], sampleTimestamps?: number[]) => {
    if (!samples.length) {
      return []
    }

    const sampleInterval = 1 / CHART_CONSTANTS.SAMPLING_RATE_HZ

    if (sampleTimestamps && sampleTimestamps.length === samples.length) {
      return samples.map((value, idx) => ({
        time: sampleTimestamps[idx],
        value,
      }))
    }

    const endTime = Date.now() / 1000
    const startTime = endTime - (samples.length - 1) * sampleInterval

    return samples.map((value, idx) => ({
      time: startTime + idx * sampleInterval,
      value,
    }))
  }, [])

  const snapshotKey = useMemo(() => {
    if (!snapshot) return 'none'
    const historyLength = snapshot.fhr_history?.length ?? 0
    const ucLength = snapshot.uc_history?.length ?? 0
    return `${snapshot.patient_id}:${snapshot.last_update}:${historyLength}:${ucLength}`
  }, [snapshot])

  const { baseFhr, baseUc, baseTimestamps } = useMemo(() => {
    const resolvedFhr = snapshot?.fhr_history?.length
      ? snapshot.fhr_history
      : fhrData && fhrData.length
        ? fhrData
        : []

    const resolvedUc = snapshot?.uc_history?.length
      ? snapshot.uc_history
      : ucData && ucData.length
        ? ucData
        : []

    let resolvedTimestamps = snapshot?.timestamps && snapshot.timestamps.length === resolvedFhr.length
      ? snapshot.timestamps
      : timestamps && timestamps.length === resolvedFhr.length
        ? timestamps
        : undefined

    if (!resolvedTimestamps && resolvedFhr.length) {
      const sampleInterval = 1 / CHART_CONSTANTS.SAMPLING_RATE_HZ
      const endTime = Date.now() / 1000
      const startTime = endTime - (resolvedFhr.length - 1) * sampleInterval
      resolvedTimestamps = resolvedFhr.map((_value: number, idx: number) => startTime + idx * sampleInterval)
    }

    return {
      baseFhr: resolvedFhr,
      baseUc: resolvedUc,
      baseTimestamps: resolvedTimestamps,
    }
  }, [snapshot, fhrData, ucData, timestamps])

  useEffect(() => {
    let updated = false

    if (baseFhr.length) {
      if (chartData.fhrLength === 0 || baseFhr.length > chartData.fhrLength) {
        chartData.setFHRData(buildDataPoints(baseFhr, baseTimestamps))
        updateFHRData(chartData.getFHRData())
        updated = true
      }
    }

    if (baseUc.length) {
      if (chartData.ucLength === 0 || baseUc.length > chartData.ucLength) {
        chartData.setUCData(buildDataPoints(baseUc, baseTimestamps))
        updateUCData(chartData.getUCData())
        updated = true
      }
    }

    if (updated) {
      if (!initializedRef.current) {
        initializedRef.current = true
        setIsFollowingRealTime(false)
      }
      fitContent()
    }
  }, [snapshotKey, baseFhr, baseUc, baseTimestamps, chartData, buildDataPoints, updateFHRData, updateUCData, fitContent])

  useEffect(() => {
    if (!liveUpdate) {
      return
    }

    if (liveUpdate.fhr_latest && liveUpdate.fhr_latest.length) {
      chartData.appendFHR(liveUpdate.fhr_latest)
      updateFHRData(chartData.getFHRData())
    }

    if (liveUpdate.uc_latest && liveUpdate.uc_latest.length) {
      chartData.appendUC(liveUpdate.uc_latest)
      updateUCData(chartData.getUCData())
    }

    if (isLive && isFollowingRealTime) {
      scrollToRealTime()
    }
  }, [liveUpdate, chartData, updateFHRData, updateUCData, isLive, isFollowingRealTime, scrollToRealTime])

  const highlightRegions = useMemo<ChartHighlightRegion[]>(() => {
    const rawRegions = snapshot?.highlight_regions ?? []
    if (!rawRegions.length || !baseTimestamps || baseTimestamps.length === 0) {
      return []
    }

    const windowLength = Math.max(
      ...rawRegions.map(region => region.end_idx ?? 0),
      0
    )
    const offset = Math.max(0, baseTimestamps.length - windowLength)

    return rawRegions.reduce<ChartHighlightRegion[]>((acc, region) => {
        const startIndex = Math.min(baseTimestamps.length - 1, offset + region.start_idx)
        const endIndex = Math.min(baseTimestamps.length - 1, offset + region.end_idx)
        const startTime = baseTimestamps[startIndex]
        const endTime = baseTimestamps[endIndex]

        if (startTime == null || endTime == null) {
          return acc
        }

        acc.push({
          startTime,
          endTime,
          color: region.color ?? 'rgba(220, 53, 69, 0.35)',
          label: region.label,
          severity: region.severity,
        })

        return acc
      }, [])
  }, [snapshot, baseTimestamps])

  const [overlayRegions, setOverlayRegions] = useState<Array<{ left: number; width: number; color: string; label?: string }>>([])

  const computeOverlayRegions = useCallback(() => {
    if (!chart || !container || highlightRegions.length === 0) {
      setOverlayRegions([])
      return
    }

    const timeScale = chart.timeScale()
    const nextRegions = highlightRegions.reduce<Array<{ left: number; width: number; color: string; label?: string }>>(
      (acc, region) => {
        const startX = timeScale.timeToCoordinate(region.startTime as Time)
        const endX = timeScale.timeToCoordinate(region.endTime as Time)

        if (startX == null || endX == null) {
          return acc
        }

        const left = Math.min(startX, endX)
        const width = Math.max(1, Math.abs(endX - startX))

        acc.push({
          left,
          width,
          color: region.color,
          label: region.label,
        })

        return acc
      },
      []
    )

    setOverlayRegions(nextRegions)
  }, [chart, container, highlightRegions])

  useEffect(() => {
    computeOverlayRegions()
  }, [computeOverlayRegions])

  useEffect(() => {
    if (!chart) return
    const timeScale = chart.timeScale()
    const handleRangeChange = () => computeOverlayRegions()
    timeScale.subscribeVisibleTimeRangeChange(handleRangeChange)
    return () => timeScale.unsubscribeVisibleTimeRangeChange(handleRangeChange)
  }, [chart, computeOverlayRegions])

  const handleLiveToggle = useCallback(() => {
    if (!isFollowingRealTime) {
      scrollToRealTime()
    }
    setIsFollowingRealTime(prev => !prev)
  }, [isFollowingRealTime, scrollToRealTime])

  return (
    <div className="ctg-chart-container relative bg-white rounded-lg overflow-hidden border border-gray-200">
      {/* Chart Header */}
      <div className="chart-header flex justify-between items-center p-2 bg-gray-50 border-b border-gray-200">
        <div className="flex items-center gap-4">
          <span className="text-sm text-gray-600">
            FHR: <span className="text-blue-600 font-mono">{CLINICAL_RANGES.fhr.min}-{CLINICAL_RANGES.fhr.max} bpm</span>
          </span>
          <span className="text-sm text-gray-600">
            UC: <span className="text-orange-600 font-mono">{CLINICAL_RANGES.uc.min}-{CLINICAL_RANGES.uc.max} mmHg</span>
          </span>
        </div>

        {showControls && (
          <div className="flex items-center gap-2">
            <button
              onClick={panLeft}
              className="p-1.5 text-xs bg-gray-200 hover:bg-gray-300 rounded text-gray-700 transition-colors"
              title="Pan Left"
            >
              ◀
            </button>
            <button
              onClick={zoomOut}
              className="p-1.5 text-xs bg-gray-200 hover:bg-gray-300 rounded text-gray-700 transition-colors"
              title="Zoom Out"
            >
              −
            </button>
            <button
              onClick={zoomIn}
              className="p-1.5 text-xs bg-gray-200 hover:bg-gray-300 rounded text-gray-700 transition-colors"
              title="Zoom In"
            >
              +
            </button>
            <button
              onClick={panRight}
              className="p-1.5 text-xs bg-gray-200 hover:bg-gray-300 rounded text-gray-700 transition-colors"
              title="Pan Right"
            >
              ▶
            </button>
            <div className="w-px h-4 bg-gray-300 mx-1" />
            <button
              onClick={() => {
                setIsFollowingRealTime(false)
                fitContent()
              }}
              className="px-2 py-1 text-xs bg-gray-200 hover:bg-gray-300 rounded text-gray-700 transition-colors"
            >
              Fit All
            </button>
            <button
              onClick={handleLiveToggle}
              className={`px-2 py-1 text-xs rounded transition-colors ${
                isFollowingRealTime
                  ? 'bg-green-500 hover:bg-green-600 text-white'
                  : 'bg-gray-200 hover:bg-gray-300 text-gray-700'
              }`}
            >
              LIVE
            </button>
          </div>
        )}
      </div>

      {/* Clinical Ranges Indicator */}
      <div className="absolute left-0 top-12 bottom-0 w-1 z-10">
        <div
          className="absolute w-full"
          style={{
            top: '5%',
            height: '45%',
            background: `linear-gradient(to bottom, ${RED_ZONE_COLORS.bradycardia} 0%, transparent 15%, transparent 85%, ${RED_ZONE_COLORS.tachycardia} 100%)`,
          }}
        />
      </div>

      {/* Main Chart Area */}
      <div
        className="relative"
        style={{
          width: typeof width === 'number' ? `${width}px` : width,
          height: typeof height === 'number' ? `${height}px` : height,
        }}
      >
        <div
          ref={containerRef}
          className="chart-area"
          style={{
            width: '100%',
            height: '100%',
          }}
        />
        <div className="absolute inset-0 pointer-events-none">
          {overlayRegions.map((region, idx) => (
            <div key={`${region.left}-${region.width}-${idx}`} className="absolute inset-y-0">
              <div
                className="absolute inset-y-0"
                style={{
                  left: region.left,
                  width: region.width,
                  backgroundColor: region.color,
                }}
              />
              <div
                className="absolute inset-y-0"
                style={{
                  left: region.left,
                  width: 2,
                  backgroundColor: 'rgba(220, 53, 69, 0.9)',
                }}
              />
              <div
                className="absolute inset-y-0"
                style={{
                  left: region.left + region.width,
                  width: 2,
                  backgroundColor: 'rgba(220, 53, 69, 0.9)',
                }}
              />
              {region.label && (
                <div
                  className="absolute top-2 text-[10px] font-semibold text-red-600"
                  style={{ left: region.left + 4 }}
                >
                  {region.label}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Y-Axis Labels */}
      <div className="absolute right-2 top-14 text-xs text-gray-500 font-mono">
        <div>FHR (bpm)</div>
      </div>
      <div className="absolute left-2 bottom-14 text-xs text-gray-500 font-mono">
        <div>UC (mmHg)</div>
      </div>

      {/* Live Indicator */}
      {isLive && isFollowingRealTime && (
        <div className="absolute top-14 right-4 flex items-center gap-1">
          <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
          <span className="text-xs text-red-500 font-medium">LIVE</span>
        </div>
      )}
    </div>
  )
})

CTGChart.displayName = 'CTGChart'

export default CTGChart

