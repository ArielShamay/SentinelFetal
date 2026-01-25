# Phase 3: Frontend Scaffold (React + Vite) - Technical Specifications

**Phase:** 3 of 6
**Document Type:** Technical Specifications
**Target Audience:** Frontend Developers

---

## ✅ IMPLEMENTATION STATUS: COMPLETE

**Implemented:** January 2026
**Developer:** AI Assistant (Claude)

### Implementation Notes:
All specifications have been implemented as described below. The actual implementation follows the spec with these enhancements:
1. Enhanced WardView with search, filtering by category, and sorting options
2. PatientCard with compact mode for dense grids
3. CategoryBadge with utility functions for sorting
4. WebSocketManager with full lifecycle management
5. uiStore with persist middleware for user preferences

### Key Implementation Decisions:
- QueryClient moved from main.tsx to App.tsx for cleaner setup
- All components organized into subfolder structure as specified
- Map used for patient storage for O(1) lookups
- Custom `animate-pulse-subtle` animation added to Tailwind config

---

## 1. Project Structure

```
frontend/
├── public/
│   ├── index.html
│   └── favicon.svg
│
├── src/
│   ├── main.tsx                 # Entry point
│   ├── App.tsx                  # Root component + router
│   │
│   ├── pages/
│   │   ├── WardView.tsx         # Multi-patient grid
│   │   ├── DetailView.tsx       # Single patient focus
│   │   └── NotFound.tsx         # 404 page
│   │
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Header.tsx
│   │   │   ├── Footer.tsx
│   │   │   └── Layout.tsx
│   │   ├── patient/
│   │   │   ├── PatientCard.tsx
│   │   │   ├── PatientInfo.tsx
│   │   │   └── CategoryBadge.tsx
│   │   ├── status/
│   │   │   ├── ConnectionStatus.tsx
│   │   │   └── SimulationControls.tsx
│   │   └── charts/
│   │       └── CTGPlaceholder.tsx  # Placeholder for Phase 4
│   │
│   ├── hooks/
│   │   ├── usePatientStream.ts  # WebSocket connection
│   │   ├── usePatientStore.ts   # Store selector hooks
│   │   └── useSimulation.ts     # Simulation control
│   │
│   ├── stores/
│   │   ├── patientStore.ts      # Zustand patient state
│   │   └── uiStore.ts           # UI preferences
│   │
│   ├── services/
│   │   ├── api.ts               # REST API client
│   │   └── websocket.ts         # WebSocket manager
│   │
│   ├── types/
│   │   ├── index.ts             # All types
│   │   ├── patient.ts           # Patient types
│   │   └── api.ts               # API response types
│   │
│   ├── utils/
│   │   ├── constants.ts         # App constants
│   │   ├── formatters.ts        # Value formatters
│   │   └── msgpack.ts           # MessagePack helpers
│   │
│   └── styles/
│       └── globals.css          # Global styles + Tailwind
│
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
└── postcss.config.js
```

---

## 2. Core Files

### 2.1 src/main.tsx

```tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from './App'
import './styles/globals.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000,
      refetchOnWindowFocus: false,
    },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
)
```

### 2.2 src/App.tsx

```tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { useEffect } from 'react'
import { Layout } from './components/layout/Layout'
import { WardView } from './pages/WardView'
import { DetailView } from './pages/DetailView'
import { NotFound } from './pages/NotFound'
import { usePatientStream } from './hooks/usePatientStream'

function App() {
  // Initialize WebSocket connection
  const { connect } = usePatientStream()

  useEffect(() => {
    connect()
  }, [connect])

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<WardView />} />
          <Route path="patient/:patientId" element={<DetailView />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
```

---

## 3. Type Definitions

### 3.1 src/types/patient.ts

```typescript
/**
 * Patient data types matching API schemas
 */

export type Category = 1 | 2 | 3

export interface HighlightRegion {
  start_idx: number
  end_idx: number
  severity: 'info' | 'warning' | 'critical'
  label: string
  color?: string
}

export interface TrendData {
  deterioration_score: number
  variability_slope: number
  decel_count_30min: number
  late_decel_count_15min: number
  alerts: TrendAlert[]
}

export interface TrendAlert {
  severity: 'LOW' | 'MEDIUM' | 'HIGH'
  message: string
}

export interface ExplanationData {
  primary_reason: string
  contributing_factors: string[]
  confidence: number
}

export interface PatientSnapshot {
  patient_id: string
  category: Category
  baseline: number
  variability: number
  fhr_latest: number[]
  uc_latest: number[]
  mhr_alert: boolean
  highlight_regions: HighlightRegion[]
  trend_score: number
  active_event: string | null
}

export interface PatientDetail extends PatientSnapshot {
  fhr_buffer: number[]
  uc_buffer: number[]
  trend_data?: TrendData
  explanation?: ExplanationData
  alert: {
    headline: string
    recommendation: string
    category: Category
  }
}

export const CATEGORY_NAMES: Record<Category, string> = {
  1: 'Normal',
  2: 'Intermediate',
  3: 'Pathological',
}

export const CATEGORY_COLORS: Record<Category, string> = {
  1: '#28a745', // Green
  2: '#fd7e14', // Orange
  3: '#dc3545', // Red
}
```

### 3.2 src/types/api.ts

```typescript
/**
 * API request/response types
 */

export interface SimulationStatus {
  running: boolean
  paused: boolean
  patient_count: number
  tick_count: number
  uptime_seconds: number
}

export interface EventInjectionRequest {
  event_type: string
  severity: 'mild' | 'moderate' | 'severe'
  duration_seconds: number
}

export interface WebSocketMessage {
  type: 'connected' | 'patient_update' | 'ping' | 'error'
  timestamp: number
  client_id?: string
  patients?: import('./patient').PatientSnapshot[]
  error?: string
}
```

---

## 4. State Management

### 4.1 src/stores/patientStore.ts

```typescript
import { create } from 'zustand'
import { devtools, subscribeWithSelector } from 'zustand/middleware'
import type { PatientSnapshot, PatientDetail } from '../types/patient'

interface PatientState {
  // Connection state
  connected: boolean
  clientId: string | null
  lastUpdate: number

  // Patient data
  patients: Map<string, PatientSnapshot>
  patientDetails: Map<string, PatientDetail>
  selectedPatientId: string | null

  // Simulation state
  simulationRunning: boolean
  simulationPaused: boolean

  // Actions
  setConnected: (connected: boolean, clientId?: string) => void
  updatePatients: (patients: PatientSnapshot[]) => void
  setPatientDetail: (detail: PatientDetail) => void
  selectPatient: (patientId: string | null) => void
  setSimulationState: (running: boolean, paused: boolean) => void
  reset: () => void
}

export const usePatientStore = create<PatientState>()(
  devtools(
    subscribeWithSelector((set, get) => ({
      // Initial state
      connected: false,
      clientId: null,
      lastUpdate: 0,
      patients: new Map(),
      patientDetails: new Map(),
      selectedPatientId: null,
      simulationRunning: false,
      simulationPaused: false,

      // Actions
      setConnected: (connected, clientId) =>
        set({ connected, clientId: clientId ?? get().clientId }),

      updatePatients: (patients) =>
        set((state) => {
          const newMap = new Map(state.patients)
          for (const patient of patients) {
            newMap.set(patient.patient_id, patient)
          }
          return { patients: newMap, lastUpdate: Date.now() }
        }),

      setPatientDetail: (detail) =>
        set((state) => {
          const newMap = new Map(state.patientDetails)
          newMap.set(detail.patient_id, detail)
          return { patientDetails: newMap }
        }),

      selectPatient: (patientId) => set({ selectedPatientId: patientId }),

      setSimulationState: (running, paused) =>
        set({ simulationRunning: running, simulationPaused: paused }),

      reset: () =>
        set({
          patients: new Map(),
          patientDetails: new Map(),
          selectedPatientId: null,
        }),
    })),
    { name: 'patient-store' }
  )
)

// Selector hooks for optimized re-renders
export const usePatient = (patientId: string) =>
  usePatientStore((state) => state.patients.get(patientId))

export const useAllPatients = () =>
  usePatientStore((state) => Array.from(state.patients.values()))

export const useConnectionStatus = () =>
  usePatientStore((state) => ({
    connected: state.connected,
    lastUpdate: state.lastUpdate,
  }))

export const useSelectedPatient = () =>
  usePatientStore((state) => {
    const id = state.selectedPatientId
    return id ? state.patientDetails.get(id) ?? state.patients.get(id) : null
  })
```

### 4.2 src/stores/uiStore.ts

```typescript
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface UIState {
  // Preferences
  darkMode: boolean
  soundEnabled: boolean
  language: 'he' | 'en'

  // Layout
  sidebarOpen: boolean
  gridColumns: number

  // Actions
  toggleDarkMode: () => void
  toggleSound: () => void
  setLanguage: (lang: 'he' | 'en') => void
  setSidebarOpen: (open: boolean) => void
  setGridColumns: (cols: number) => void
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      darkMode: false,
      soundEnabled: true,
      language: 'he',
      sidebarOpen: true,
      gridColumns: 4,

      toggleDarkMode: () => set((s) => ({ darkMode: !s.darkMode })),
      toggleSound: () => set((s) => ({ soundEnabled: !s.soundEnabled })),
      setLanguage: (language) => set({ language }),
      setSidebarOpen: (sidebarOpen) => set({ sidebarOpen }),
      setGridColumns: (gridColumns) => set({ gridColumns }),
    }),
    { name: 'ui-preferences' }
  )
)
```

---

## 5. WebSocket Integration

### 5.1 src/services/websocket.ts

```typescript
import msgpack from '@ygoe/msgpack'
import type { WebSocketMessage } from '../types/api'

type MessageHandler = (message: WebSocketMessage) => void
type StatusHandler = (connected: boolean) => void

class WebSocketManager {
  private ws: WebSocket | null = null
  private url: string
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000
  private messageHandler: MessageHandler | null = null
  private statusHandler: StatusHandler | null = null
  private clientId: string | null = null

  constructor(url: string = 'ws://localhost:8000/ws/stream') {
    this.url = url
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return
    }

    const urlWithClientId = this.clientId
      ? `${this.url}?client_id=${this.clientId}`
      : this.url

    this.ws = new WebSocket(urlWithClientId)
    this.ws.binaryType = 'arraybuffer'

    this.ws.onopen = () => {
      console.log('WebSocket connected')
      this.reconnectAttempts = 0
      this.statusHandler?.(true)
    }

    this.ws.onmessage = (event) => {
      try {
        const data = msgpack.decode(new Uint8Array(event.data)) as WebSocketMessage

        // Handle connection message
        if (data.type === 'connected' && data.client_id) {
          this.clientId = data.client_id
        }

        // Handle ping
        if (data.type === 'ping') {
          this.sendPong()
          return
        }

        // Forward to handler
        this.messageHandler?.(data)
      } catch (error) {
        console.error('Failed to decode message:', error)
      }
    }

    this.ws.onclose = () => {
      console.log('WebSocket disconnected')
      this.statusHandler?.(false)
      this.attemptReconnect()
    }

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('Max reconnect attempts reached')
      return
    }

    this.reconnectAttempts++
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`)

    setTimeout(() => {
      this.connect()
    }, delay)
  }

  private sendPong(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      const pong = msgpack.encode({ type: 'pong' })
      this.ws.send(pong)
    }
  }

  subscribe(patientIds: string[]): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      const message = msgpack.encode({
        type: 'subscribe',
        patient_ids: patientIds,
      })
      this.ws.send(message)
    }
  }

  onMessage(handler: MessageHandler): void {
    this.messageHandler = handler
  }

  onStatusChange(handler: StatusHandler): void {
    this.statusHandler = handler
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }
}

export const wsManager = new WebSocketManager()
```

### 5.2 src/hooks/usePatientStream.ts

```typescript
import { useCallback, useEffect, useRef } from 'react'
import { wsManager } from '../services/websocket'
import { usePatientStore } from '../stores/patientStore'
import type { WebSocketMessage } from '../types/api'

export function usePatientStream() {
  const updatePatients = usePatientStore((s) => s.updatePatients)
  const setConnected = usePatientStore((s) => s.setConnected)
  const initialized = useRef(false)

  const handleMessage = useCallback(
    (message: WebSocketMessage) => {
      if (message.type === 'connected') {
        setConnected(true, message.client_id)
      } else if (message.type === 'patient_update' && message.patients) {
        updatePatients(message.patients)
      }
    },
    [updatePatients, setConnected]
  )

  const handleStatus = useCallback(
    (connected: boolean) => {
      setConnected(connected)
    },
    [setConnected]
  )

  const connect = useCallback(() => {
    if (initialized.current) return
    initialized.current = true

    wsManager.onMessage(handleMessage)
    wsManager.onStatusChange(handleStatus)
    wsManager.connect()
  }, [handleMessage, handleStatus])

  const disconnect = useCallback(() => {
    wsManager.disconnect()
    initialized.current = false
  }, [])

  const subscribe = useCallback((patientIds: string[]) => {
    wsManager.subscribe(patientIds)
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])

  return {
    connect,
    disconnect,
    subscribe,
    isConnected: wsManager.isConnected,
  }
}
```

---

## 6. Components

### 6.1 src/components/layout/Layout.tsx

```tsx
import { Outlet } from 'react-router-dom'
import { Header } from './Header'

export function Layout() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <main className="container mx-auto px-4 py-6">
        <Outlet />
      </main>
    </div>
  )
}
```

### 6.2 src/components/layout/Header.tsx

```tsx
import { Link } from 'react-router-dom'
import { ConnectionStatus } from '../status/ConnectionStatus'

export function Header() {
  return (
    <header className="bg-white shadow-sm border-b">
      <div className="container mx-auto px-4 py-3 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2">
          <span className="text-xl font-bold text-gray-800">
            SentinelFetal
          </span>
          <span className="text-sm text-gray-500">V3.0</span>
        </Link>

        <div className="flex items-center gap-4">
          <ConnectionStatus />
        </div>
      </div>
    </header>
  )
}
```

### 6.3 src/components/status/ConnectionStatus.tsx

```tsx
import { useConnectionStatus } from '../../stores/patientStore'

export function ConnectionStatus() {
  const { connected, lastUpdate } = useConnectionStatus()

  return (
    <div className="flex items-center gap-2">
      <div
        className={`w-2 h-2 rounded-full ${
          connected ? 'bg-green-500' : 'bg-red-500'
        }`}
        title={connected ? 'Connected' : 'Disconnected'}
      />
      <span className="text-sm text-gray-600">
        {connected ? 'Live' : 'Offline'}
      </span>
      {connected && lastUpdate > 0 && (
        <span className="text-xs text-gray-400">
          {new Date(lastUpdate).toLocaleTimeString()}
        </span>
      )}
    </div>
  )
}
```

### 6.4 src/components/patient/PatientCard.tsx

```tsx
import { Link } from 'react-router-dom'
import type { PatientSnapshot } from '../../types/patient'
import { CategoryBadge } from './CategoryBadge'

interface PatientCardProps {
  patient: PatientSnapshot
}

export function PatientCard({ patient }: PatientCardProps) {
  const {
    patient_id,
    category,
    baseline,
    variability,
    mhr_alert,
    active_event,
  } = patient

  return (
    <Link
      to={`/patient/${patient_id}`}
      className="block bg-white rounded-lg shadow-sm border hover:shadow-md transition-shadow p-4"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <span className="font-medium text-gray-800">{patient_id}</span>
        <CategoryBadge category={category} />
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div>
          <span className="text-gray-500">Baseline</span>
          <p className="font-medium">{baseline.toFixed(0)} bpm</p>
        </div>
        <div>
          <span className="text-gray-500">Variability</span>
          <p className="font-medium">{variability.toFixed(1)} bpm</p>
        </div>
      </div>

      {/* Alerts */}
      {(mhr_alert || active_event) && (
        <div className="mt-3 flex flex-wrap gap-1">
          {mhr_alert && (
            <span className="px-2 py-0.5 bg-yellow-100 text-yellow-800 text-xs rounded">
              MHR
            </span>
          )}
          {active_event && (
            <span className="px-2 py-0.5 bg-red-100 text-red-800 text-xs rounded">
              {active_event}
            </span>
          )}
        </div>
      )}

      {/* Mini sparkline placeholder */}
      <div className="mt-3 h-12 bg-gray-50 rounded flex items-center justify-center text-xs text-gray-400">
        Sparkline (Phase 4)
      </div>
    </Link>
  )
}
```

### 6.5 src/components/patient/CategoryBadge.tsx

```tsx
import type { Category } from '../../types/patient'
import { CATEGORY_COLORS, CATEGORY_NAMES } from '../../types/patient'

interface CategoryBadgeProps {
  category: Category
  size?: 'sm' | 'md' | 'lg'
}

export function CategoryBadge({ category, size = 'sm' }: CategoryBadgeProps) {
  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-3 py-1 text-sm',
    lg: 'px-4 py-1.5 text-base',
  }

  return (
    <span
      className={`rounded-full font-medium text-white ${sizeClasses[size]}`}
      style={{ backgroundColor: CATEGORY_COLORS[category] }}
    >
      {CATEGORY_NAMES[category]}
    </span>
  )
}
```

---

## 7. Pages

### 7.1 src/pages/WardView.tsx

```tsx
import { useAllPatients } from '../stores/patientStore'
import { PatientCard } from '../components/patient/PatientCard'

export function WardView() {
  const patients = useAllPatients()

  if (patients.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">
          No patients available. Start the simulation to begin.
        </p>
      </div>
    )
  }

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Ward Monitor</h1>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {patients.map((patient) => (
          <PatientCard key={patient.patient_id} patient={patient} />
        ))}
      </div>
    </div>
  )
}
```

### 7.2 src/pages/DetailView.tsx

```tsx
import { useParams, Link } from 'react-router-dom'
import { useEffect } from 'react'
import { usePatient, usePatientStore } from '../stores/patientStore'
import { CategoryBadge } from '../components/patient/CategoryBadge'
import { usePatientStream } from '../hooks/usePatientStream'

export function DetailView() {
  const { patientId } = useParams<{ patientId: string }>()
  const patient = usePatient(patientId ?? '')
  const selectPatient = usePatientStore((s) => s.selectPatient)
  const { subscribe } = usePatientStream()

  // Subscribe to this patient for higher-fidelity updates
  useEffect(() => {
    if (patientId) {
      selectPatient(patientId)
      subscribe([patientId])
    }
    return () => {
      selectPatient(null)
      subscribe([]) // Unsubscribe
    }
  }, [patientId, selectPatient, subscribe])

  if (!patient) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">Patient not found</p>
        <Link to="/" className="text-blue-600 hover:underline mt-4 inline-block">
          Back to Ward
        </Link>
      </div>
    )
  }

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <Link
            to="/"
            className="text-gray-500 hover:text-gray-700"
          >
            ← Back
          </Link>
          <h1 className="text-2xl font-bold">{patient.patient_id}</h1>
          <CategoryBadge category={patient.category} size="md" />
        </div>
      </div>

      {/* Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Chart Area - 3 columns */}
        <div className="lg:col-span-3">
          <div className="bg-white rounded-lg shadow-sm border p-4">
            <h2 className="text-lg font-semibold mb-4">CTG Monitor</h2>
            <div className="h-96 bg-gray-50 rounded flex items-center justify-center text-gray-400">
              CTG Chart (Phase 4)
            </div>
          </div>
        </div>

        {/* Info Panel - 1 column */}
        <div className="space-y-4">
          {/* Metrics */}
          <div className="bg-white rounded-lg shadow-sm border p-4">
            <h3 className="font-semibold mb-3">Metrics</h3>
            <dl className="space-y-2">
              <div className="flex justify-between">
                <dt className="text-gray-500">Baseline</dt>
                <dd className="font-medium">{patient.baseline.toFixed(0)} bpm</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">Variability</dt>
                <dd className="font-medium">{patient.variability.toFixed(1)} bpm</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">Trend Score</dt>
                <dd className="font-medium">{patient.trend_score}</dd>
              </div>
            </dl>
          </div>

          {/* Alerts */}
          {patient.mhr_alert && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <p className="text-yellow-800 text-sm font-medium">
                MHR Contamination Suspected
              </p>
            </div>
          )}

          {/* Active Event */}
          {patient.active_event && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-red-800 text-sm font-medium">
                Active: {patient.active_event}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
```

### 7.3 src/pages/NotFound.tsx

```tsx
import { Link } from 'react-router-dom'

export function NotFound() {
  return (
    <div className="text-center py-12">
      <h1 className="text-4xl font-bold text-gray-800 mb-4">404</h1>
      <p className="text-gray-500 mb-6">Page not found</p>
      <Link to="/" className="text-blue-600 hover:underline">
        Back to Ward
      </Link>
    </div>
  )
}
```

---

## 8. API Service

### 8.1 src/services/api.ts

```typescript
import { QueryClient } from '@tanstack/react-query'
import type { SimulationStatus, EventInjectionRequest } from '../types/api'
import type { PatientDetail } from '../types/patient'

const API_BASE = '/api'

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  })

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`)
  }

  return response.json()
}

export const api = {
  // Simulation
  getSimulationStatus: () =>
    fetchJSON<SimulationStatus>('/simulation/status'),

  startSimulation: () =>
    fetchJSON<SimulationStatus>('/simulation/start', { method: 'POST' }),

  pauseSimulation: () =>
    fetchJSON<SimulationStatus>('/simulation/pause', { method: 'POST' }),

  resumeSimulation: () =>
    fetchJSON<SimulationStatus>('/simulation/resume', { method: 'POST' }),

  resetSimulation: () =>
    fetchJSON<SimulationStatus>('/simulation/reset', { method: 'POST' }),

  // Patients
  getPatientDetail: (patientId: string) =>
    fetchJSON<PatientDetail>(`/patients/${patientId}`),

  injectEvent: (patientId: string, event: EventInjectionRequest) =>
    fetchJSON(`/patients/${patientId}/inject`, {
      method: 'POST',
      body: JSON.stringify(event),
    }),
}

// React Query keys
export const queryKeys = {
  simulationStatus: ['simulation', 'status'] as const,
  patientDetail: (id: string) => ['patient', id] as const,
}
```

---

## 9. Utilities

### 9.1 src/utils/constants.ts

```typescript
export const APP_NAME = 'SentinelFetal'
export const APP_VERSION = '3.0.0'

export const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/stream'
export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const SAMPLING_RATE = 4 // Hz
export const SPARKLINE_POINTS = 60 // 15 seconds
export const CHART_WINDOW_MINUTES = 20
```

### 9.2 src/utils/formatters.ts

```typescript
export function formatBPM(value: number): string {
  return `${Math.round(value)} bpm`
}

export function formatVariability(value: number): string {
  return `${value.toFixed(1)} bpm`
}

export function formatTimestamp(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString()
}

export function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins}:${secs.toString().padStart(2, '0')}`
}
```

---

## 10. Styles

### 10.1 src/styles/globals.css

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

/* Custom base styles */
@layer base {
  body {
    @apply antialiased text-gray-900;
  }
}

/* Custom component styles */
@layer components {
  .btn {
    @apply px-4 py-2 rounded-lg font-medium transition-colors;
  }

  .btn-primary {
    @apply bg-blue-600 text-white hover:bg-blue-700;
  }

  .btn-secondary {
    @apply bg-gray-200 text-gray-800 hover:bg-gray-300;
  }

  .card {
    @apply bg-white rounded-lg shadow-sm border p-4;
  }
}

/* RTL support for Hebrew */
[dir="rtl"] {
  text-align: right;
}
```

---

## 11. Verification

```bash
# Install dependencies
cd frontend && npm install

# Type check
npm run type-check

# Lint
npm run lint

# Dev server
npm run dev

# Build
npm run build
```

---

*End of Phase 3 Technical Specifications*
