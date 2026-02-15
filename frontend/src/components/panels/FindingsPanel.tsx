/**
 * FindingsPanel - Displays Clinical Findings from Rule Engine
 * ===========================================================
 */

import React from 'react'
import { Activity, TrendingDown, Heart, Waves } from 'lucide-react'
import type { ClinicalFindings } from '../../types'



interface FindingsPanelProps {
  findings: ClinicalFindings
  className?: string
}

export const FindingsPanel: React.FC<FindingsPanelProps> = ({
  findings,
  className = ''
}) => {
  const { decelerations, variability, baseline, contraction_frequency } = findings

  const getVariabilityColor = (category: string) => {
    switch (category) {
      case 'absent': return 'text-red-600 bg-red-50'
      case 'minimal': return 'text-orange-600 bg-orange-50'
      case 'moderate': return 'text-green-600 bg-green-50'
      case 'marked': return 'text-yellow-600 bg-yellow-50'
      default: return 'text-gray-600 bg-gray-50'
    }
  }

  const getBaselineColor = (status: string) => {
    switch (status) {
      case 'normal': return 'text-green-600 bg-green-50'
      case 'tachycardia': return 'text-orange-600 bg-orange-50'
      case 'bradycardia': return 'text-red-600 bg-red-50'
      default: return 'text-gray-600 bg-gray-50'
    }
  }

  return (
    <div className={`bg-white rounded-xl shadow-sm border border-gray-100 p-4 ${className}`}>
      <div className="flex items-center gap-2 mb-4 pb-3 border-b border-gray-100">
        <Activity className="w-5 h-5 text-blue-600" />
        <h3 className="font-semibold text-gray-900">Clinical Findings</h3>
      </div>

      <div className="grid grid-cols-2 gap-4">

        {/* Decelerations */}
        <div className="col-span-2 bg-gray-50 rounded-lg p-3">
          <div className="flex items-center gap-2 mb-2">
            <TrendingDown className="w-4 h-4 text-gray-600" />
            <span className="font-medium text-gray-700 text-sm">Decelerations</span>
            {decelerations.recurrent && (
              <span className="px-2 py-0.5 text-xs font-medium bg-red-100 text-red-700 rounded-full">
                RECURRENT
              </span>
            )}
          </div>

          <div className="grid grid-cols-4 gap-2 text-center">
            <div className={`p-2 rounded ${decelerations.late_count > 0 ? 'bg-red-100' : 'bg-white'}`}>
              <div className={`text-lg font-bold ${decelerations.late_count > 0 ? 'text-red-600' : 'text-gray-400'}`}>
                {decelerations.late_count}
              </div>
              <div className="text-xs text-gray-500">Late</div>
            </div>
            <div className={`p-2 rounded ${decelerations.variable_count > 0 ? 'bg-orange-100' : 'bg-white'}`}>
              <div className={`text-lg font-bold ${decelerations.variable_count > 0 ? 'text-orange-600' : 'text-gray-400'}`}>
                {decelerations.variable_count}
              </div>
              <div className="text-xs text-gray-500">Variable</div>
            </div>
            <div className={`p-2 rounded ${decelerations.early_count > 0 ? 'bg-green-100' : 'bg-white'}`}>
              <div className={`text-lg font-bold ${decelerations.early_count > 0 ? 'text-green-600' : 'text-gray-400'}`}>
                {decelerations.early_count}
              </div>
              <div className="text-xs text-gray-500">Early</div>
            </div>
            <div className={`p-2 rounded ${decelerations.prolonged_count > 0 ? 'bg-red-100' : 'bg-white'}`}>
              <div className={`text-lg font-bold ${decelerations.prolonged_count > 0 ? 'text-red-700' : 'text-gray-400'}`}>
                {decelerations.prolonged_count}
              </div>
              <div className="text-xs text-gray-500">Prolonged</div>
            </div>
          </div>
        </div>

        {/* Baseline */}
        <div className="bg-gray-50 rounded-lg p-3">
          <div className="flex items-center gap-2 mb-2">
            <Heart className="w-4 h-4 text-gray-600" />
            <span className="font-medium text-gray-700 text-sm">Baseline</span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-gray-900">
              {baseline.value_bpm.toFixed(0)}
            </span>
            <span className="text-sm text-gray-500">bpm</span>
          </div>
          <span className={`inline-block mt-1 px-2 py-0.5 text-xs font-medium rounded-full capitalize ${getBaselineColor(baseline.status)}`}>
            {baseline.status}
          </span>
        </div>

        {/* Variability */}
        <div className="bg-gray-50 rounded-lg p-3">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-gray-600" />
            <span className="font-medium text-gray-700 text-sm">Variability</span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-gray-900">
              {variability.value_bpm.toFixed(1)}
            </span>
            <span className="text-sm text-gray-500">bpm</span>
          </div>
          <span className={`inline-block mt-1 px-2 py-0.5 text-xs font-medium rounded-full capitalize ${getVariabilityColor(variability.category)}`}>
            {variability.category}
          </span>
        </div>

        {/* Contractions */}
        <div className="col-span-2 bg-gray-50 rounded-lg p-3">
          <div className="flex items-center gap-2 mb-2">
            <Waves className="w-4 h-4 text-gray-600" />
            <span className="font-medium text-gray-700 text-sm">Contractions</span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className={`text-2xl font-bold ${contraction_frequency > 5 ? 'text-red-600' : 'text-gray-900'}`}>
              {contraction_frequency.toFixed(1)}
            </span>
            <span className="text-sm text-gray-500">/10 min</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default FindingsPanel
