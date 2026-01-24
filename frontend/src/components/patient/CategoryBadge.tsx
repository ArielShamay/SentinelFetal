import React from 'react'
import type { Category } from '../../types'

interface CategoryBadgeProps {
  category: Category
  size?: 'sm' | 'md' | 'lg'
  showLabel?: boolean
  className?: string
}

const categoryConfig: Record<Category, { 
  label: string
  bgColor: string
  textColor: string
  borderColor: string
  description: string
}> = {
  1: {
    label: 'Normal',
    bgColor: 'bg-green-900/60',
    textColor: 'text-green-300',
    borderColor: 'border-green-500/50',
    description: 'Normal CTG - No intervention needed'
  },
  2: {
    label: 'Suspicious',
    bgColor: 'bg-yellow-900/60',
    textColor: 'text-yellow-300',
    borderColor: 'border-yellow-500/50',
    description: 'Suspicious CTG - Enhanced monitoring recommended'
  },
  3: {
    label: 'Pathological',
    bgColor: 'bg-red-900/60',
    textColor: 'text-red-300',
    borderColor: 'border-red-500/50',
    description: 'Pathological CTG - Immediate attention required'
  }
}

const sizeClasses = {
  sm: 'px-1.5 py-0.5 text-xs',
  md: 'px-2 py-1 text-sm',
  lg: 'px-3 py-1.5 text-base'
}

export const CategoryBadge: React.FC<CategoryBadgeProps> = ({ 
  category, 
  size = 'md',
  showLabel = true,
  className = '' 
}) => {
  const config = categoryConfig[category] ?? categoryConfig[1]
  
  return (
    <span 
      className={`
        inline-flex items-center gap-1 rounded font-medium border
        ${config.bgColor} ${config.textColor} ${config.borderColor}
        ${sizeClasses[size]}
        ${className}
      `}
      title={config.description}
    >
      {/* Category indicator dot */}
      <span className={`
        w-2 h-2 rounded-full 
        ${category === 1 ? 'bg-green-400' : ''}
        ${category === 2 ? 'bg-yellow-400' : ''}
        ${category === 3 ? 'bg-red-400 animate-pulse' : ''}
      `} />
      
      {showLabel && <span>{config.label}</span>}
    </span>
  )
}

// Export category utilities
export const getCategoryPriority = (category: Category): number => {
  return category // 1, 2, 3 already represent priority order (reversed for sorting)
}

export const sortByCategory = (a: Category, b: Category): number => {
  return b - a // Higher category = higher priority
}

export default CategoryBadge
