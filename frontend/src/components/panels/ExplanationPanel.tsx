/**
 * Explanation Panel - Shows AI classification reasoning
 */

import React from 'react'
import { useTranslation } from 'react-i18next'

interface ExplanationData {
  category: 1 | 2 | 3
  primaryReason: string
  factors: string[]
  confidence: number  // 0-100
}

interface ExplanationPanelProps {
  data?: ExplanationData
  className?: string
}

export const ExplanationPanel: React.FC<ExplanationPanelProps> = ({
  data,
  className = '',
}) => {
  const { t } = useTranslation()

  if (!data) {
    return (
      <div className={`bg-gray-800 rounded-xl p-4 border border-gray-700 ${className}`}>
        <h3 className="text-md font-semibold text-white mb-3 flex items-center gap-2">
          <span>🔍</span>
          {t('explanation.title')}
        </h3>
        <p className="text-sm text-gray-500">{t('explanation.noExplanation')}</p>
      </div>
    )
  }

  // Category colors
  const categoryColors = {
    1: 'text-green-400 bg-green-900/20',
    2: 'text-orange-400 bg-orange-900/20',
    3: 'text-red-400 bg-red-900/20',
  }

  // Confidence color
  const confidenceColor = 
    data.confidence >= 80 ? 'text-green-400' :
    data.confidence >= 60 ? 'text-yellow-400' :
    'text-red-400'

  return (
    <div className={`bg-gray-800 rounded-xl p-4 border border-gray-700 ${className}`}>
      <h3 className="text-md font-semibold text-white mb-4 flex items-center gap-2">
        <span>🔍</span>
        {t('explanation.title')}
      </h3>

      {/* Why Category X? */}
      <div className={`p-3 rounded-lg mb-4 ${categoryColors[data.category]}`}>
        <div className="font-medium">
          {t('explanation.why')} {t(`category.${data.category}`)}?
        </div>
        <div className="text-sm mt-1 opacity-90">
          {data.primaryReason}
        </div>
      </div>

      {/* Contributing Factors */}
      {data.factors.length > 0 && (
        <div className="mb-4">
          <div className="text-sm text-gray-400 mb-2">
            {t('explanation.factors')}:
          </div>
          <ul className="space-y-1.5">
            {data.factors.map((factor, idx) => (
              <li 
                key={idx}
                className="flex items-start gap-2 text-sm text-gray-300"
              >
                <span className="text-gray-500 mt-0.5">•</span>
                <span>{factor}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Confidence */}
      <div className="flex items-center justify-between pt-3 border-t border-gray-700">
        <span className="text-sm text-gray-400">{t('explanation.confidence')}</span>
        <span className={`font-mono font-semibold ${confidenceColor}`}>
          {data.confidence}%
        </span>
      </div>
    </div>
  )
}

export default ExplanationPanel
