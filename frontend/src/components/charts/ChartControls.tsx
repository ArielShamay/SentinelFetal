/**
 * Chart Controls Component - Zoom, pan, and time range controls
 */

import React, { memo, useState, useCallback } from 'react'
import type { ChartControlsProps } from '../../types/chart'

const TIME_RANGES = [
  { label: '1 min', minutes: 1 },
  { label: '5 min', minutes: 5 },
  { label: '10 min', minutes: 10 },
  { label: '30 min', minutes: 30 },
  { label: '1 hour', minutes: 60 },
  { label: 'All', minutes: null },
]

const ChartControls: React.FC<ChartControlsProps> = memo(({
  onZoomIn,
  onZoomOut,
  onReset,
  onTimeRangeChange,
  selectedRange = 10,
  className = '',
}) => {
  const [activeRange, setActiveRange] = useState<number | null>(selectedRange)

  const handleRangeChange = useCallback((minutes: number | null) => {
    setActiveRange(minutes)
    onTimeRangeChange?.(minutes)
  }, [onTimeRangeChange])

  return (
    <div className={`flex items-center gap-4 p-2 bg-gray-50 rounded-lg border border-gray-200 ${className}`}>
      {/* Zoom Controls */}
      <div className="flex items-center gap-1">
        <button
          onClick={onZoomIn}
          className="p-1.5 bg-white hover:bg-gray-100 border border-gray-200 rounded text-gray-600 transition-colors"
          title="Zoom In"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8" />
            <path d="M21 21l-4.35-4.35M11 8v6M8 11h6" />
          </svg>
        </button>

        <button
          onClick={onZoomOut}
          className="p-1.5 bg-white hover:bg-gray-100 border border-gray-200 rounded text-gray-600 transition-colors"
          title="Zoom Out"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8" />
            <path d="M21 21l-4.35-4.35M8 11h6" />
          </svg>
        </button>

        <button
          onClick={onReset}
          className="p-1.5 bg-white hover:bg-gray-100 border border-gray-200 rounded text-gray-600 transition-colors"
          title="Reset View"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
            <path d="M3 3v5h5" />
          </svg>
        </button>
      </div>

      {/* Divider */}
      <div className="w-px h-6 bg-gray-300" />

      {/* Time Range Selector */}
      <div className="flex items-center gap-1">
        <span className="text-xs text-gray-500 mr-2">Time Range:</span>
        {TIME_RANGES.map(({ label, minutes }) => (
          <button
            key={label}
            onClick={() => handleRangeChange(minutes)}
            className={`px-2 py-1 text-xs rounded transition-colors ${
              activeRange === minutes
                ? 'bg-blue-500 text-white'
                : 'bg-white hover:bg-gray-100 border border-gray-200 text-gray-600'
            }`}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  )
})

ChartControls.displayName = 'ChartControls'

export default ChartControls
