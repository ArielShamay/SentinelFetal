/**
 * CTG Chart Component - Main real-time CTG visualization
 * 
 * Displays FHR (top) and UC (bottom) traces using lightweight-charts
 */

import React, { useRef, useEffect, useCallback, useState, memo } from 'react'
import { useLightweightChart } from '../../hooks/useLightweightChart'
import { useChartData } from '../../hooks/useChartData'
import { CLINICAL_RANGES, RED_ZONE_COLORS, CHART_CONSTANTS } from '../../utils/chartConfig'
import type { CTGChartProps } from '../../types/chart'

/**
 * Main CTG Chart for real-time monitoring
 */
const CTGChart: React.FC<CTGChartProps> = memo(({
  width = '100%',
  height = 400,
  showControls = true,
  isLive = true,
  onTimeRangeChange: _onTimeRangeChange,
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const [isFollowingRealTime, setIsFollowingRealTime] = useState(true)
  const [container, setContainer] = useState<HTMLDivElement | null>(null)

  // Set container after mount
  useEffect(() => {
    setContainer(containerRef.current)
  }, [])

  // Initialize chart
  const {
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

  // Initialize data buffers
  const chartData = useChartData({
    bufferSize: CHART_CONSTANTS.BUFFER_SIZE,
    samplingRate: CHART_CONSTANTS.SAMPLING_RATE_HZ,
  })

  /**
   * Handle new data from WebSocket
   */
  const handleNewData = useCallback((fhrSamples: number[], ucSamples: number[]) => {
    if (fhrSamples.length > 0) {
      chartData.appendFHR(fhrSamples)
      updateFHRData(chartData.getFHRData())
    }
    
    if (ucSamples.length > 0) {
      chartData.appendUC(ucSamples)
      updateUCData(chartData.getUCData())
    }

    // Auto-scroll to latest if in live mode
    if (isFollowingRealTime && isLive) {
      scrollToRealTime()
    }
  }, [chartData, updateFHRData, updateUCData, scrollToRealTime, isFollowingRealTime, isLive])

  // Expose data handler to parent via ref or context
  useEffect(() => {
    // This effect can be used to connect to WebSocket data
    // Parent component should call handleNewData when receiving data
    (window as any).__ctgChartDataHandler = handleNewData
    return () => {
      delete (window as any).__ctgChartDataHandler
    }
  }, [handleNewData])

  /**
   * Handle live toggle
   */
  const handleLiveToggle = useCallback(() => {
    if (!isFollowingRealTime) {
      scrollToRealTime()
    }
    setIsFollowingRealTime(!isFollowingRealTime)
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
            {/* Pan Left */}
            <button
              onClick={panLeft}
              className="p-1.5 text-xs bg-gray-200 hover:bg-gray-300 rounded text-gray-700 transition-colors"
              title="Pan Left"
            >
              ◀
            </button>
            {/* Zoom Out */}
            <button
              onClick={zoomOut}
              className="p-1.5 text-xs bg-gray-200 hover:bg-gray-300 rounded text-gray-700 transition-colors"
              title="Zoom Out"
            >
              −
            </button>
            {/* Zoom In */}
            <button
              onClick={zoomIn}
              className="p-1.5 text-xs bg-gray-200 hover:bg-gray-300 rounded text-gray-700 transition-colors"
              title="Zoom In"
            >
              +
            </button>
            {/* Pan Right */}
            <button
              onClick={panRight}
              className="p-1.5 text-xs bg-gray-200 hover:bg-gray-300 rounded text-gray-700 transition-colors"
              title="Pan Right"
            >
              ▶
            </button>
            <div className="w-px h-4 bg-gray-300 mx-1" />
            <button
              onClick={fitContent}
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
        {/* Normal FHR zone */}
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
        ref={containerRef}
        className="chart-area"
        style={{
          width: typeof width === 'number' ? `${width}px` : width,
          height: typeof height === 'number' ? `${height}px` : height,
        }}
      />

      {/* Y-Axis Labels */}
      <div className="absolute right-2 top-14 text-xs text-gray-500 font-mono">
        <div>FHR (bpm)</div>
      </div>
      <div className="absolute right-2 bottom-14 text-xs text-gray-500 font-mono">
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
