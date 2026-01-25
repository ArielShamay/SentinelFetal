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
      <div className={`bg-white rounded-xl p-4 border border-gray-200 shadow-sm ${className}`}>
        <h3 className="text-md font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <span>🔍</span>
          {t('explanation.title')}
        </h3>
        <p className="text-sm text-gray-400">{t('explanation.noExplanation')}</p>
      </div>
    )
  }

  // Category colors
  const categoryColors = {
    1: 'text-green-700 bg-green-50 border border-green-200',
    2: 'text-orange-700 bg-orange-50 border border-orange-200',
    3: 'text-red-700 bg-red-50 border border-red-200',
  }

  // Confidence color
  const confidenceColor =
    data.confidence >= 80 ? 'text-green-600' :
    data.confidence >= 60 ? 'text-yellow-600' :
    'text-red-600'

  return (
    <div className={`bg-white rounded-xl p-4 border border-gray-200 shadow-sm ${className}`}>
      <h3 className="text-md font-semibold text-gray-900 mb-4 flex items-center gap-2">
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
          <div className="text-sm text-gray-500 mb-2">
            {t('explanation.factors')}:
          </div>
          <ul className="space-y-1.5">
            {data.factors.map((factor, idx) => (
              <li
                key={idx}
                className="flex items-start gap-2 text-sm text-gray-600"
              >
                <span className="text-gray-400 mt-0.5">•</span>
                <span>{factor}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Confidence */}
      <div className="flex items-center justify-between pt-3 border-t border-gray-200">
        <span className="text-sm text-gray-500">{t('explanation.confidence')}</span>
        <span className={`font-mono font-semibold ${confidenceColor}`}>
          {data.confidence}%
        </span>
      </div>
    </div>
  )
}

export default ExplanationPanel
