/**
 * Trend Panel - Shows trend analysis over 60 minutes
 */

import React, { useMemo } from 'react'
import { useTranslation } from 'react-i18next'

interface TrendData {
  deteriorationScore: number  // 0-100
  variabilityTrend: 'increasing' | 'decreasing' | 'stable'
  decelsIn30min: number
  lateDecelsIn15min: number
  alerts: string[]
}

interface TrendPanelProps {
  data?: TrendData
  className?: string
}

export const TrendPanel: React.FC<TrendPanelProps> = ({
  data,
  className = '',
}) => {
  const { t } = useTranslation()

  // Default values if no data
  const trendData: TrendData = data ?? {
    deteriorationScore: 0,
    variabilityTrend: 'stable',
    decelsIn30min: 0,
    lateDecelsIn15min: 0,
    alerts: [],
  }

  // Score color
  const scoreColor = useMemo(() => {
    if (trendData.deteriorationScore >= 70) return 'bg-red-500'
    if (trendData.deteriorationScore >= 40) return 'bg-orange-500'
    return 'bg-green-500'
  }, [trendData.deteriorationScore])

  // Trend icon
  const trendIcon = {
    increasing: '📈',
    decreasing: '📉',
    stable: '➡️',
  }[trendData.variabilityTrend]

  return (
    <div className={`bg-gray-800 rounded-xl p-4 border border-gray-700 ${className}`}>
      <h3 className="text-md font-semibold text-white mb-4 flex items-center gap-2">
        <span>📊</span>
        {t('trend.title')}
      </h3>

      {/* Deterioration Score */}
      <div className="mb-4">
        <div className="flex justify-between text-sm mb-1">
          <span className="text-gray-400">{t('trend.deteriorationScore')}</span>
          <span className="text-white font-mono">{trendData.deteriorationScore}</span>
        </div>
        <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full ${scoreColor} transition-all duration-500`}
            style={{ width: `${trendData.deteriorationScore}%` }}
          />
        </div>
      </div>

      {/* Variability Trend */}
      <div className="flex items-center justify-between py-2 border-b border-gray-700">
        <span className="text-gray-400">{t('trend.variability')}</span>
        <span className="flex items-center gap-1 text-white">
          {trendIcon}
          <span className={`text-sm ${
            trendData.variabilityTrend === 'decreasing' ? 'text-yellow-400' :
            trendData.variabilityTrend === 'increasing' ? 'text-green-400' :
            'text-gray-300'
          }`}>
            {t(`trend.${trendData.variabilityTrend}`)}
          </span>
        </span>
      </div>

      {/* Deceleration Counts */}
      <div className="space-y-2 py-2 border-b border-gray-700">
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">{t('trend.decels30')}</span>
          <span className="text-white font-mono">{trendData.decelsIn30min}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">{t('trend.lateDecels15')}</span>
          <span className={`font-mono ${
            trendData.lateDecelsIn15min > 0 ? 'text-red-400' : 'text-white'
          }`}>
            {trendData.lateDecelsIn15min}
            {trendData.lateDecelsIn15min > 0 && ' ⚠️'}
          </span>
        </div>
      </div>

      {/* Trend Alerts */}
      {trendData.alerts.length > 0 && (
        <div className="mt-3 space-y-1">
          {trendData.alerts.map((alert, idx) => (
            <div
              key={idx}
              className="flex items-start gap-2 text-sm text-yellow-300 bg-yellow-900/20 p-2 rounded"
            >
              <span>⚠️</span>
              <span>{alert}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default TrendPanel
