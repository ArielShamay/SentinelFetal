/**
 * Language Toggle Component - EN/HE switch
 */

import React, { useCallback } from 'react'
import { useTranslation } from 'react-i18next'

interface LanguageToggleProps {
  className?: string
}

export const LanguageToggle: React.FC<LanguageToggleProps> = ({
  className = '',
}) => {
  const { i18n } = useTranslation()
  const currentLang = i18n.language

  const toggleLanguage = useCallback(() => {
    const newLang = currentLang === 'en' ? 'he' : 'en'
    i18n.changeLanguage(newLang)
  }, [currentLang, i18n])

  return (
    <button
      onClick={toggleLanguage}
      className={`
        flex items-center gap-2 px-3 py-1.5 rounded-lg
        bg-gray-700 hover:bg-gray-600
        text-sm font-medium text-gray-200
        transition-colors duration-200
        ${className}
      `}
      title={currentLang === 'en' ? 'Switch to Hebrew' : 'החלף לאנגלית'}
    >
      <span className={`transition-opacity ${currentLang === 'en' ? 'opacity-100' : 'opacity-50'}`}>
        EN
      </span>
      <span className="text-gray-500">/</span>
      <span className={`transition-opacity ${currentLang === 'he' ? 'opacity-100' : 'opacity-50'}`}>
        עב
      </span>
    </button>
  )
}

export default LanguageToggle
