# Phase 5: Feature Parity & Polish - Technical Specifications

**Phase:** 5 of 6
**Document Type:** Technical Specifications
**Target Audience:** Frontend Developers

---

## 1. Project Structure Additions

```
frontend/src/
├── components/
│   ├── godmode/
│   │   ├── GodModePanel.tsx       # Event injection UI
│   │   ├── EventTypeSelect.tsx    # Event dropdown
│   │   ├── SeveritySelector.tsx   # Severity radio buttons
│   │   └── DurationSlider.tsx     # Duration input
│   │
│   ├── panels/
│   │   ├── TrendPanel.tsx         # Trend analysis
│   │   ├── ExplanationPanel.tsx   # AI explanation
│   │   └── AlertPanel.tsx         # Current alert display
│   │
│   ├── controls/
│   │   ├── SimulationControls.tsx # Start/Pause/Reset
│   │   ├── PatientCountSlider.tsx # Patient count
│   │   └── LanguageToggle.tsx     # EN/HE switch
│   │
│   └── common/
│       ├── ProgressBar.tsx        # Deterioration bar
│       └── Toast.tsx              # Notification wrapper
│
├── i18n/
│   ├── index.ts                   # i18n configuration
│   ├── en.json                    # English strings
│   └── he.json                    # Hebrew strings
│
├── hooks/
│   ├── useSimulation.ts           # Simulation control
│   └── useGodMode.ts              # Event injection
│
└── styles/
    └── rtl.css                    # RTL-specific styles
```

---

## 2. Internationalization Setup

### 2.1 src/i18n/index.ts

```typescript
import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import en from './en.json'
import he from './he.json'

const resources = {
  en: { translation: en },
  he: { translation: he },
}

// Get saved language or detect from browser
const savedLang = localStorage.getItem('language')
const browserLang = navigator.language.startsWith('he') ? 'he' : 'en'
const defaultLang = savedLang || browserLang

i18n
  .use(initReactI18next)
  .init({
    resources,
    lng: defaultLang,
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false,
    },
  })

// Update document direction when language changes
i18n.on('languageChanged', (lng) => {
  const dir = lng === 'he' ? 'rtl' : 'ltr'
  document.documentElement.dir = dir
  document.documentElement.lang = lng
  localStorage.setItem('language', lng)
})

// Set initial direction
document.documentElement.dir = defaultLang === 'he' ? 'rtl' : 'ltr'
document.documentElement.lang = defaultLang

export default i18n
```

### 2.2 src/i18n/en.json

```json
{
  "app": {
    "title": "SentinelFetal",
    "version": "V3.0"
  },
  "nav": {
    "ward": "Ward View",
    "detail": "Patient Detail",
    "back": "Back"
  },
  "patient": {
    "baseline": "Baseline",
    "variability": "Variability",
    "trendScore": "Trend Score",
    "category": "Category",
    "metrics": "Metrics",
    "viewDetails": "View Details"
  },
  "category": {
    "1": "Normal",
    "2": "Intermediate",
    "3": "Pathological"
  },
  "alert": {
    "headline1": "Green Alert - Category 1 (Normal)",
    "headline2": "Orange Alert - Category 2 (Intermediate)",
    "headline3": "Red Alert - Category 3 (Pathological)",
    "mhrWarning": "MHR Contamination Suspected"
  },
  "recommendation": {
    "normal": "Routine monitoring",
    "intermediate": "Increased surveillance recommended",
    "pathological": "Immediate clinical evaluation required"
  },
  "trend": {
    "title": "Trend Analysis (60 min)",
    "deteriorationScore": "Deterioration Score",
    "variability": "Variability",
    "increasing": "Increasing",
    "decreasing": "Decreasing",
    "stable": "Stable",
    "decels30": "Decels (30 min)",
    "lateDecels15": "Late Decels (15 min)"
  },
  "explanation": {
    "title": "Classification Explanation",
    "why": "Why",
    "factors": "Contributing Factors",
    "confidence": "Confidence"
  },
  "godmode": {
    "title": "God Mode - Event Injection",
    "eventType": "Event Type",
    "severity": "Severity",
    "mild": "Mild",
    "moderate": "Moderate",
    "severe": "Severe",
    "duration": "Duration",
    "target": "Target Patient",
    "detection": "Expected Detection",
    "inject": "Inject Event",
    "injecting": "Injecting...",
    "success": "Event injected successfully",
    "error": "Failed to inject event"
  },
  "events": {
    "LATE_DECEL": "Late Deceleration",
    "VARIABLE_DECEL": "Variable Deceleration",
    "EARLY_DECEL": "Early Deceleration",
    "PROLONGED_DECEL": "Prolonged Deceleration",
    "TACHYCARDIA": "Tachycardia",
    "BRADYCARDIA": "Bradycardia",
    "REDUCED_VARIABILITY": "Reduced Variability",
    "SINUSOIDAL": "Sinusoidal Pattern"
  },
  "simulation": {
    "status": "Status",
    "running": "Running",
    "paused": "Paused",
    "stopped": "Stopped",
    "start": "Start",
    "pause": "Pause",
    "resume": "Resume",
    "reset": "Reset",
    "patients": "Patients"
  },
  "connection": {
    "connected": "Live",
    "disconnected": "Offline",
    "reconnecting": "Reconnecting..."
  },
  "common": {
    "loading": "Loading...",
    "error": "Error",
    "noData": "No data available",
    "seconds": "seconds",
    "minutes": "minutes",
    "bpm": "bpm"
  }
}
```

### 2.3 src/i18n/he.json

```json
{
  "app": {
    "title": "SentinelFetal",
    "version": "V3.0"
  },
  "nav": {
    "ward": "תצוגת מחלקה",
    "detail": "פרטי מטופל",
    "back": "חזרה"
  },
  "patient": {
    "baseline": "קו בסיס",
    "variability": "וריאביליות",
    "trendScore": "ציון מגמה",
    "category": "קטגוריה",
    "metrics": "מדדים",
    "viewDetails": "צפה בפרטים"
  },
  "category": {
    "1": "תקין",
    "2": "בינוני",
    "3": "פתולוגי"
  },
  "alert": {
    "headline1": "התראה ירוקה - קטגוריה 1 (תקין)",
    "headline2": "התראה כתומה - קטגוריה 2 (בינוני)",
    "headline3": "התראה אדומה - קטגוריה 3 (פתולוגי)",
    "mhrWarning": "חשד לזיהום MHR"
  },
  "recommendation": {
    "normal": "מעקב שגרתי",
    "intermediate": "הגברת מעקב מומלצת",
    "pathological": "נדרשת הערכה קלינית מיידית"
  },
  "trend": {
    "title": "ניתוח מגמות (60 דקות)",
    "deteriorationScore": "ציון התדרדרות",
    "variability": "וריאביליות",
    "increasing": "עולה",
    "decreasing": "יורד",
    "stable": "יציב",
    "decels30": "האטות (30 דק׳)",
    "lateDecels15": "האטות מאוחרות (15 דק׳)"
  },
  "explanation": {
    "title": "הסבר סיווג",
    "why": "למה",
    "factors": "גורמים תורמים",
    "confidence": "רמת ביטחון"
  },
  "godmode": {
    "title": "מצב אל - הזרקת אירועים",
    "eventType": "סוג אירוע",
    "severity": "חומרה",
    "mild": "קל",
    "moderate": "בינוני",
    "severe": "חמור",
    "duration": "משך",
    "target": "מטופל יעד",
    "detection": "זמן גילוי צפוי",
    "inject": "הזרק אירוע",
    "injecting": "מזריק...",
    "success": "האירוע הוזרק בהצלחה",
    "error": "הזרקת האירוע נכשלה"
  },
  "events": {
    "LATE_DECEL": "האטה מאוחרת",
    "VARIABLE_DECEL": "האטה משתנה",
    "EARLY_DECEL": "האטה מוקדמת",
    "PROLONGED_DECEL": "האטה ממושכת",
    "TACHYCARDIA": "טכיקרדיה",
    "BRADYCARDIA": "ברדיקרדיה",
    "REDUCED_VARIABILITY": "וריאביליות מופחתת",
    "SINUSOIDAL": "דפוס סינוסואידלי"
  },
  "simulation": {
    "status": "סטטוס",
    "running": "פועל",
    "paused": "מושהה",
    "stopped": "עצור",
    "start": "התחל",
    "pause": "השהה",
    "resume": "המשך",
    "reset": "איפוס",
    "patients": "מטופלים"
  },
  "connection": {
    "connected": "מחובר",
    "disconnected": "מנותק",
    "reconnecting": "מתחבר מחדש..."
  },
  "common": {
    "loading": "טוען...",
    "error": "שגיאה",
    "noData": "אין נתונים",
    "seconds": "שניות",
    "minutes": "דקות",
    "bpm": "פעימות/דקה"
  }
}
```

---

## 3. God Mode Components

### 3.1 src/components/godmode/GodModePanel.tsx

```tsx
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useMutation } from '@tanstack/react-query'
import toast from 'react-hot-toast'
import { api } from '../../services/api'
import { useAllPatients } from '../../stores/patientStore'
import { EventTypeSelect } from './EventTypeSelect'
import { SeveritySelector } from './SeveritySelector'
import { DurationSlider } from './DurationSlider'

const DETECTION_TIMES: Record<string, string> = {
  LATE_DECEL: '~15s',
  VARIABLE_DECEL: '~10s',
  EARLY_DECEL: '~12s',
  PROLONGED_DECEL: '~8s',
  TACHYCARDIA: '~20s',
  BRADYCARDIA: '~5s',
  REDUCED_VARIABILITY: '~30s',
  SINUSOIDAL: '~25s',
}

export function GodModePanel() {
  const { t } = useTranslation()
  const patients = useAllPatients()

  const [eventType, setEventType] = useState('LATE_DECEL')
  const [severity, setSeverity] = useState<'mild' | 'moderate' | 'severe'>('moderate')
  const [duration, setDuration] = useState(120)
  const [targetPatient, setTargetPatient] = useState(patients[0]?.patient_id ?? '')

  const injectMutation = useMutation({
    mutationFn: (params: { patientId: string; event: any }) =>
      api.injectEvent(params.patientId, params.event),
    onSuccess: () => {
      toast.success(t('godmode.success'))
    },
    onError: () => {
      toast.error(t('godmode.error'))
    },
  })

  const handleInject = () => {
    if (!targetPatient) return

    injectMutation.mutate({
      patientId: targetPatient,
      event: {
        event_type: eventType,
        severity,
        duration_seconds: duration,
      },
    })
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border p-4">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <span>⚡</span>
        {t('godmode.title')}
      </h3>

      <div className="space-y-4">
        {/* Event Type */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            {t('godmode.eventType')}
          </label>
          <EventTypeSelect value={eventType} onChange={setEventType} />
        </div>

        {/* Severity */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            {t('godmode.severity')}
          </label>
          <SeveritySelector value={severity} onChange={setSeverity} />
        </div>

        {/* Duration */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            {t('godmode.duration')}
          </label>
          <DurationSlider value={duration} onChange={setDuration} />
        </div>

        {/* Target Patient */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            {t('godmode.target')}
          </label>
          <select
            value={targetPatient}
            onChange={(e) => setTargetPatient(e.target.value)}
            className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
          >
            {patients.map((p) => (
              <option key={p.patient_id} value={p.patient_id}>
                {p.patient_id}
              </option>
            ))}
          </select>
        </div>

        {/* Detection Time */}
        <div className="text-sm text-gray-600">
          <span className="font-medium">{t('godmode.detection')}: </span>
          {DETECTION_TIMES[eventType]}
        </div>

        {/* Inject Button */}
        <button
          onClick={handleInject}
          disabled={injectMutation.isPending || !targetPatient}
          className="w-full py-2 px-4 bg-purple-600 text-white rounded-lg font-medium
                     hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed
                     transition-colors flex items-center justify-center gap-2"
        >
          <span>💉</span>
          {injectMutation.isPending ? t('godmode.injecting') : t('godmode.inject')}
        </button>
      </div>
    </div>
  )
}
```

### 3.2 src/components/godmode/EventTypeSelect.tsx

```tsx
import { useTranslation } from 'react-i18next'

const EVENT_TYPES = [
  'LATE_DECEL',
  'VARIABLE_DECEL',
  'EARLY_DECEL',
  'PROLONGED_DECEL',
  'TACHYCARDIA',
  'BRADYCARDIA',
  'REDUCED_VARIABILITY',
  'SINUSOIDAL',
] as const

interface EventTypeSelectProps {
  value: string
  onChange: (value: string) => void
}

export function EventTypeSelect({ value, onChange }: EventTypeSelectProps) {
  const { t } = useTranslation()

  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
    >
      {EVENT_TYPES.map((type) => (
        <option key={type} value={type}>
          {t(`events.${type}`)}
        </option>
      ))}
    </select>
  )
}
```

### 3.3 src/components/godmode/SeveritySelector.tsx

```tsx
import { useTranslation } from 'react-i18next'

type Severity = 'mild' | 'moderate' | 'severe'

interface SeveritySelectorProps {
  value: Severity
  onChange: (value: Severity) => void
}

export function SeveritySelector({ value, onChange }: SeveritySelectorProps) {
  const { t } = useTranslation()

  const options: Severity[] = ['mild', 'moderate', 'severe']

  return (
    <div className="flex gap-4">
      {options.map((severity) => (
        <label key={severity} className="flex items-center gap-2 cursor-pointer">
          <input
            type="radio"
            name="severity"
            value={severity}
            checked={value === severity}
            onChange={() => onChange(severity)}
            className="w-4 h-4 text-blue-600"
          />
          <span className="text-sm">{t(`godmode.${severity}`)}</span>
        </label>
      ))}
    </div>
  )
}
```

### 3.4 src/components/godmode/DurationSlider.tsx

```tsx
interface DurationSliderProps {
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
}

export function DurationSlider({
  value,
  onChange,
  min = 30,
  max = 600,
}: DurationSliderProps) {
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="flex items-center gap-4">
      <input
        type="range"
        min={min}
        max={max}
        step={30}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
      />
      <span className="text-sm font-mono w-16 text-right">
        {formatTime(value)}
      </span>
    </div>
  )
}
```

---

## 4. Panels

### 4.1 src/components/panels/TrendPanel.tsx

```tsx
import { useTranslation } from 'react-i18next'
import { ProgressBar } from '../common/ProgressBar'
import type { TrendData } from '../../types/patient'

interface TrendPanelProps {
  data: TrendData | null
}

export function TrendPanel({ data }: TrendPanelProps) {
  const { t } = useTranslation()

  if (!data) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-4">
        <h3 className="font-semibold mb-3">{t('trend.title')}</h3>
        <p className="text-sm text-gray-500">{t('common.noData')}</p>
      </div>
    )
  }

  const {
    deterioration_score,
    variability_slope,
    decel_count_30min,
    late_decel_count_15min,
    alerts,
  } = data

  // Determine trend direction
  let trendIcon: string
  let trendText: string
  if (variability_slope < -0.5) {
    trendIcon = '📉'
    trendText = t('trend.decreasing')
  } else if (variability_slope > 0.5) {
    trendIcon = '📈'
    trendText = t('trend.increasing')
  } else {
    trendIcon = '➡️'
    trendText = t('trend.stable')
  }

  // Determine score color
  const scoreColor =
    deterioration_score < 30
      ? 'green'
      : deterioration_score < 70
      ? 'orange'
      : 'red'

  return (
    <div className="bg-white rounded-lg shadow-sm border p-4">
      <h3 className="font-semibold mb-3 flex items-center gap-2">
        <span>📈</span>
        {t('trend.title')}
      </h3>

      <div className="space-y-3">
        {/* Deterioration Score */}
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-600">{t('trend.deteriorationScore')}</span>
            <span className="font-medium">{Math.round(deterioration_score)}</span>
          </div>
          <ProgressBar value={deterioration_score} color={scoreColor} />
        </div>

        {/* Variability Trend */}
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">{t('trend.variability')}</span>
          <span className="font-medium">
            {trendIcon} {trendText}
          </span>
        </div>

        {/* Deceleration Counts */}
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">{t('trend.decels30')}</span>
          <span className="font-medium">{decel_count_30min}</span>
        </div>

        <div className="flex justify-between text-sm">
          <span className="text-gray-600">{t('trend.lateDecels15')}</span>
          <span className={`font-medium ${late_decel_count_15min > 0 ? 'text-red-600' : ''}`}>
            {late_decel_count_15min}
            {late_decel_count_15min > 0 && ' ⚠️'}
          </span>
        </div>

        {/* Alerts */}
        {alerts.length > 0 && (
          <div className="pt-2 border-t space-y-1">
            {alerts.slice(0, 3).map((alert, idx) => (
              <div
                key={idx}
                className={`text-sm px-2 py-1 rounded ${
                  alert.severity === 'HIGH'
                    ? 'bg-red-50 text-red-700'
                    : 'bg-yellow-50 text-yellow-700'
                }`}
              >
                {alert.severity === 'HIGH' && '⚠️ '}
                {alert.message}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
```

### 4.2 src/components/panels/ExplanationPanel.tsx

```tsx
import { useTranslation } from 'react-i18next'
import type { ExplanationData, Category } from '../../types/patient'

interface ExplanationPanelProps {
  data: ExplanationData | null
  category: Category
}

export function ExplanationPanel({ data, category }: ExplanationPanelProps) {
  const { t } = useTranslation()

  if (!data) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-4">
        <h3 className="font-semibold mb-3">{t('explanation.title')}</h3>
        <p className="text-sm text-gray-500">{t('common.noData')}</p>
      </div>
    )
  }

  const { primary_reason, contributing_factors, confidence } = data
  const confidencePercent = Math.round(confidence * 100)

  return (
    <div className="bg-white rounded-lg shadow-sm border p-4">
      <h3 className="font-semibold mb-3 flex items-center gap-2">
        <span>🔍</span>
        {t('explanation.title')}
      </h3>

      <div className="space-y-3">
        {/* Primary Reason */}
        {primary_reason && (
          <div>
            <span className="text-sm text-gray-600">
              {t('explanation.why')} {t(`category.${category}`)}?
            </span>
            <p className="font-medium mt-1">{primary_reason}</p>
          </div>
        )}

        {/* Contributing Factors */}
        {contributing_factors.length > 0 && (
          <div>
            <span className="text-sm text-gray-600">{t('explanation.factors')}:</span>
            <ul className="mt-1 space-y-1">
              {contributing_factors.slice(0, 4).map((factor, idx) => (
                <li key={idx} className="text-sm flex items-start gap-2">
                  <span className="text-gray-400">•</span>
                  <span>{factor}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Confidence */}
        {confidence > 0 && (
          <div className="text-sm text-gray-600">
            {t('explanation.confidence')}: {confidencePercent}%
          </div>
        )}
      </div>
    </div>
  )
}
```

---

## 5. Simulation Controls

### 5.1 src/components/controls/SimulationControls.tsx

```tsx
import { useTranslation } from 'react-i18next'
import { useSimulation } from '../../hooks/useSimulation'
import { PatientCountSlider } from './PatientCountSlider'

export function SimulationControls() {
  const { t } = useTranslation()
  const {
    status,
    isLoading,
    start,
    pause,
    resume,
    reset,
    setPatientCount,
  } = useSimulation()

  const { running, paused, patient_count } = status ?? {
    running: false,
    paused: false,
    patient_count: 4,
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border p-4">
      <h3 className="font-semibold mb-4">{t('simulation.status')}</h3>

      {/* Status Indicator */}
      <div className="mb-4 flex items-center gap-2">
        <div
          className={`w-3 h-3 rounded-full ${
            running && !paused
              ? 'bg-green-500 animate-pulse'
              : paused
              ? 'bg-yellow-500'
              : 'bg-gray-400'
          }`}
        />
        <span className="text-sm font-medium">
          {running && !paused
            ? t('simulation.running')
            : paused
            ? t('simulation.paused')
            : t('simulation.stopped')}
        </span>
      </div>

      {/* Control Buttons */}
      <div className="flex gap-2 mb-4">
        {!running ? (
          <button
            onClick={start}
            disabled={isLoading}
            className="flex-1 py-2 px-4 bg-green-600 text-white rounded-lg
                       hover:bg-green-700 disabled:bg-gray-400 transition-colors"
          >
            ▶ {t('simulation.start')}
          </button>
        ) : paused ? (
          <button
            onClick={resume}
            disabled={isLoading}
            className="flex-1 py-2 px-4 bg-green-600 text-white rounded-lg
                       hover:bg-green-700 disabled:bg-gray-400 transition-colors"
          >
            ▶ {t('simulation.resume')}
          </button>
        ) : (
          <button
            onClick={pause}
            disabled={isLoading}
            className="flex-1 py-2 px-4 bg-yellow-600 text-white rounded-lg
                       hover:bg-yellow-700 disabled:bg-gray-400 transition-colors"
          >
            ⏸ {t('simulation.pause')}
          </button>
        )}

        <button
          onClick={reset}
          disabled={isLoading}
          className="py-2 px-4 bg-gray-200 text-gray-800 rounded-lg
                     hover:bg-gray-300 disabled:bg-gray-100 transition-colors"
        >
          ⏹ {t('simulation.reset')}
        </button>
      </div>

      {/* Patient Count */}
      <PatientCountSlider
        value={patient_count}
        onChange={setPatientCount}
      />
    </div>
  )
}
```

### 5.2 src/components/controls/PatientCountSlider.tsx

```tsx
import { useTranslation } from 'react-i18next'

interface PatientCountSliderProps {
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
}

export function PatientCountSlider({
  value,
  onChange,
  min = 1,
  max = 20,
}: PatientCountSliderProps) {
  const { t } = useTranslation()

  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-gray-600">{t('simulation.patients')}</span>
        <span className="font-medium">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
      />
    </div>
  )
}
```

### 5.3 src/components/controls/LanguageToggle.tsx

```tsx
import { useTranslation } from 'react-i18next'

export function LanguageToggle() {
  const { i18n } = useTranslation()

  const toggleLanguage = () => {
    const newLang = i18n.language === 'he' ? 'en' : 'he'
    i18n.changeLanguage(newLang)
  }

  return (
    <button
      onClick={toggleLanguage}
      className="px-3 py-1 text-sm bg-gray-100 rounded-lg hover:bg-gray-200
                 transition-colors font-medium"
    >
      {i18n.language === 'he' ? 'EN' : 'עב'}
    </button>
  )
}
```

---

## 6. Hooks

### 6.1 src/hooks/useSimulation.ts

```typescript
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api, queryKeys } from '../services/api'
import { usePatientStore } from '../stores/patientStore'

export function useSimulation() {
  const queryClient = useQueryClient()
  const setSimulationState = usePatientStore((s) => s.setSimulationState)

  const { data: status, isLoading: isStatusLoading } = useQuery({
    queryKey: queryKeys.simulationStatus,
    queryFn: api.getSimulationStatus,
    refetchInterval: 5000,
    onSuccess: (data) => {
      setSimulationState(data.running, data.paused)
    },
  })

  const startMutation = useMutation({
    mutationFn: api.startSimulation,
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.simulationStatus, data)
      setSimulationState(data.running, data.paused)
    },
  })

  const pauseMutation = useMutation({
    mutationFn: api.pauseSimulation,
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.simulationStatus, data)
      setSimulationState(data.running, data.paused)
    },
  })

  const resumeMutation = useMutation({
    mutationFn: api.resumeSimulation,
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.simulationStatus, data)
      setSimulationState(data.running, data.paused)
    },
  })

  const resetMutation = useMutation({
    mutationFn: api.resetSimulation,
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.simulationStatus, data)
      setSimulationState(data.running, data.paused)
      usePatientStore.getState().reset()
    },
  })

  const configMutation = useMutation({
    mutationFn: (patientCount: number) =>
      api.updateSimulationConfig({ patient_count: patientCount }),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.simulationStatus, data)
    },
  })

  const isLoading =
    isStatusLoading ||
    startMutation.isPending ||
    pauseMutation.isPending ||
    resumeMutation.isPending ||
    resetMutation.isPending

  return {
    status,
    isLoading,
    start: startMutation.mutate,
    pause: pauseMutation.mutate,
    resume: resumeMutation.mutate,
    reset: resetMutation.mutate,
    setPatientCount: configMutation.mutate,
  }
}
```

---

## 7. RTL Styles

### 7.1 src/styles/rtl.css

```css
/* RTL-specific overrides */
[dir="rtl"] {
  /* Flip icons that indicate direction */
  .icon-arrow-right {
    transform: scaleX(-1);
  }

  /* Adjust spacing for RTL */
  .gap-2 > :not(:first-child) {
    margin-right: 0.5rem;
    margin-left: 0;
  }

  /* Fix slider thumb direction */
  input[type="range"] {
    direction: ltr;
  }

  /* Flip back arrow */
  .back-arrow {
    transform: scaleX(-1);
  }
}

/* Ensure numbers display LTR even in RTL context */
[dir="rtl"] .numeric {
  direction: ltr;
  unicode-bidi: embed;
}
```

---

## 8. App Integration

### 8.1 Update src/main.tsx

```tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'react-hot-toast'
import App from './App'
import './i18n'  // Initialize i18n
import './styles/globals.css'
import './styles/rtl.css'

const queryClient = new QueryClient()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 3000,
          style: {
            direction: document.documentElement.dir,
          },
        }}
      />
    </QueryClientProvider>
  </React.StrictMode>,
)
```

---

## 9. Verification Checklist

```bash
# Test language switching
npm run dev
# Toggle EN/HE and verify all strings change

# Test RTL layout
# Switch to Hebrew and verify layout flips correctly

# Test God Mode
# Inject event and verify toast notification

# Test simulation controls
# Start/Pause/Resume/Reset flow

# Verify no console errors
# Check browser DevTools
```

---

*End of Phase 5 Technical Specifications*
