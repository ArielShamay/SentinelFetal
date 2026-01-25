/**
 * FHR Sparkline Component - Mini chart for patient cards
 * 
 * Lightweight SVG-based sparkline for quick FHR trend visualization
 */

import React, { memo, useMemo } from 'react'
import { generateSparklinePath, isNormalFHR } from '../../utils/chartHelpers'
import { CATEGORY_SPARKLINE_COLORS } from '../../utils/chartConfig'
import type { SparklineProps } from '../../types/chart'

const FHR_MIN = 50
const FHR_MAX = 210

/**
 * Mini sparkline for patient cards
 */
const FHRSparkline: React.FC<SparklineProps> = memo(({
  data,
  width = 120,
  height = 32,
  color,
  showTrend = true,
  className = '',
}) => {
  // Calculate path and stats
  const { path, minValue, maxValue, trend, latestValue } = useMemo(() => {
    if (!data || data.length < 2) {
      return { path: '', minValue: FHR_MIN, maxValue: FHR_MAX, trend: 'stable' as const, latestValue: FHR_MIN }
    }

    const clampedData = data.map(value => Math.min(FHR_MAX, Math.max(FHR_MIN, value)))

    // Generate SVG path
    const pathString = generateSparklinePath(clampedData, width, height, 2, { min: FHR_MIN, max: FHR_MAX })
    
    // Calculate trend
    const recentData = data.slice(-10)
    const firstHalf = recentData.slice(0, Math.floor(recentData.length / 2))
    const secondHalf = recentData.slice(Math.floor(recentData.length / 2))
    
    const avgFirst = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length
    const avgSecond = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length
    
    let trendValue: 'up' | 'down' | 'stable' = 'stable'
    const diff = avgSecond - avgFirst
    if (diff > 5) trendValue = 'up'
    else if (diff < -5) trendValue = 'down'
    
    return {
      path: pathString,
      minValue: FHR_MIN,
      maxValue: FHR_MAX,
      trend: trendValue,
      latestValue: clampedData[clampedData.length - 1],
    }
  }, [data, width, height])

  // Determine color based on data health
  const lineColor = useMemo(() => {
    if (color) return color

    // Check if data is in normal range
    const abnormalCount = data.filter(v => !isNormalFHR(v)).length
    const abnormalPercent = abnormalCount / data.length

    if (abnormalPercent > 0.3) return CATEGORY_SPARKLINE_COLORS.pathological
    if (abnormalPercent > 0.1) return CATEGORY_SPARKLINE_COLORS.suspicious
    return CATEGORY_SPARKLINE_COLORS.normal
  }, [color, data])

  // No data state
  if (!data || data.length < 2) {
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
    <div className={`relative ${className}`} style={{ width, height }}>
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        className="overflow-visible"
      >
        {/* Gradient fill under the line */}
        <defs>
          <linearGradient id={`sparkline-gradient-${lineColor.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={lineColor} stopOpacity="0.3" />
            <stop offset="100%" stopColor={lineColor} stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Fill area */}
        <path
          d={`${path} L ${width - 2},${height - 2} L 2,${height - 2} Z`}
          fill={`url(#sparkline-gradient-${lineColor.replace('#', '')})`}
        />

        {/* Main line */}
        <path
          d={path}
          fill="none"
          stroke={lineColor}
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* End point */}
        <circle
          cx={width - 2}
          cy={height - 2 - ((latestValue - minValue) / (maxValue - minValue || 1)) * (height - 4)}
          r="2"
          fill={lineColor}
        />
      </svg>

      {/* Trend indicator */}
      {showTrend && trend !== 'stable' && (
        <div className="absolute -right-3 top-1/2 -translate-y-1/2">
          {trend === 'up' ? (
            <svg className="w-2 h-2 text-yellow-400" viewBox="0 0 8 8">
              <path d="M4 0 L8 6 L0 6 Z" fill="currentColor" />
            </svg>
          ) : (
            <svg className="w-2 h-2 text-blue-400" viewBox="0 0 8 8">
              <path d="M4 6 L8 0 L0 0 Z" fill="currentColor" />
            </svg>
          )}
        </div>
      )}
    </div>
  )
})

FHRSparkline.displayName = 'FHRSparkline'

export default FHRSparkline
