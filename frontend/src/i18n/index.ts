/**
 * i18n Configuration - Internationalization setup
 */

import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import en from './en.json'
import he from './he.json'

const resources = {
  en: { translation: en },
  he: { translation: he },
}

// Get saved language or detect from browser
const getSavedLanguage = (): string => {
  try {
    const saved = localStorage.getItem('language')
    if (saved && (saved === 'en' || saved === 'he')) {
      return saved
    }
  } catch {
    // localStorage not available
  }
  // Detect from browser
  const browserLang = navigator.language.toLowerCase()
  return browserLang.startsWith('he') ? 'he' : 'en'
}

const defaultLang = getSavedLanguage()

i18n
  .use(initReactI18next)
  .init({
    resources,
    lng: defaultLang,
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false,
    },
    react: {
      useSuspense: false,
    },
  })

// Update document direction when language changes
i18n.on('languageChanged', (lng) => {
  const dir = lng === 'he' ? 'rtl' : 'ltr'
  document.documentElement.dir = dir
  document.documentElement.lang = lng
  try {
    localStorage.setItem('language', lng)
  } catch {
    // localStorage not available
  }
})

// Set initial direction
document.documentElement.dir = defaultLang === 'he' ? 'rtl' : 'ltr'
document.documentElement.lang = defaultLang

export default i18n

// Helper to get current language
export const getCurrentLanguage = (): 'en' | 'he' => {
  return (i18n.language || defaultLang) as 'en' | 'he'
}

// Helper to toggle language
export const toggleLanguage = (): void => {
  const current = getCurrentLanguage()
  const next = current === 'en' ? 'he' : 'en'
  i18n.changeLanguage(next)
}

// Check if current language is RTL
export const isRTL = (): boolean => {
  return getCurrentLanguage() === 'he'
}
