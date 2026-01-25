import { create } from 'zustand'
import { devtools, subscribeWithSelector } from 'zustand/middleware'
import type { PatientSnapshot, PatientSummary, WSPatientUpdate } from '../types'

interface PatientState {
  // Connection state
  connected: boolean
  clientId: string | null
  lastUpdate: number

  // Patient data
  patients: Map<string, PatientSnapshot>
  patientSummaries: Map<string, PatientSummary>
  liveUpdates: Map<string, WSPatientUpdate>
  selectedPatientId: string | null

  // Simulation state
  simulationRunning: boolean
  simulationPaused: boolean
  patientCount: number
  tickCount: number

  // Actions
  setConnected: (connected: boolean, clientId?: string) => void
  updateFromWebSocket: (update: WSPatientUpdate) => void
  updatePatientSnapshot: (snapshot: PatientSnapshot) => void
  updatePatientSummaries: (summaries: PatientSummary[]) => void
  selectPatient: (patientId: string | null) => void
  setSimulationState: (running: boolean, paused: boolean, patientCount?: number) => void
  incrementTickCount: () => void
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
      patientSummaries: new Map(),
      liveUpdates: new Map(),
      selectedPatientId: null,
      simulationRunning: false,
      simulationPaused: false,
      patientCount: 4,
      tickCount: 0,

      // Actions
      setConnected: (connected, clientId) =>
        set({ connected, clientId: clientId ?? get().clientId }),

      updateFromWebSocket: (update) =>
        set((state) => {
          console.log('🔄 updateFromWebSocket called:', update.patient_id, 'FHR:', update.fhr_latest?.length, 'samples')
          // DEBUG: Update DOM for debugging
          const debugDiv = document.getElementById('debug-info')
          if (debugDiv) {
            debugDiv.innerHTML = `Patient: ${update.patient_id}, FHR samples: ${update.fhr_latest?.length}, Updates: ${state.liveUpdates.size + 1}`
          }
          
          const newUpdates = new Map(state.liveUpdates)
          newUpdates.set(update.patient_id, update)

          const newPatients = new Map(state.patients)
          const existingPatient = newPatients.get(update.patient_id)
          if (existingPatient) {
            const latestFhr = update.fhr_latest?.[update.fhr_latest.length - 1]
            const latestUc = update.uc_latest?.[update.uc_latest.length - 1]
            newPatients.set(update.patient_id, {
              ...existingPatient,
              category: update.category,
              category_name: update.category === 1 ? 'Normal' : update.category === 2 ? 'Suspicious' : 'Pathological',
              metrics: {
                ...existingPatient.metrics,
                baseline_fhr: update.baseline,
                current_fhr: latestFhr ?? existingPatient.metrics.current_fhr,
                variability: update.variability,
                current_uc: latestUc ?? existingPatient.metrics.current_uc,
              },
              explanation: update.explanation ?? existingPatient.explanation,
              highlight_regions: update.highlight_regions ?? existingPatient.highlight_regions,
              last_update: Date.now(),
            })
          }
          
          // Also update summary if exists
          const newSummaries = new Map(state.patientSummaries)
          const existingSummary = newSummaries.get(update.patient_id)
          if (existingSummary) {
            newSummaries.set(update.patient_id, {
              ...existingSummary,
              category: update.category,
              current_fhr: update.fhr_latest[update.fhr_latest.length - 1] ?? existingSummary.current_fhr,
              baseline_fhr: update.baseline,
              last_update: Date.now(),
            })
          }
          
          console.log('✅ Store updated, liveUpdates.size:', newUpdates.size)
          return { 
            liveUpdates: newUpdates, 
            patientSummaries: newSummaries,
            patients: newPatients,
            lastUpdate: Date.now() 
          }
        }),

      updatePatientSnapshot: (snapshot) =>
        set((state) => {
          const newPatients = new Map(state.patients)
          newPatients.set(snapshot.patient_id, snapshot)
          return { patients: newPatients }
        }),

      updatePatientSummaries: (summaries) =>
        set((state) => {
          const newMap = new Map(state.patientSummaries)
          for (const summary of summaries) {
            newMap.set(summary.patient_id, summary)
          }
          return { patientSummaries: newMap, patientCount: summaries.length }
        }),

      selectPatient: (patientId) => set({ selectedPatientId: patientId }),

      setSimulationState: (running, paused, patientCount) =>
        set((state) => ({
          simulationRunning: running,
          simulationPaused: paused,
          patientCount: patientCount ?? state.patientCount,
        })),

      incrementTickCount: () =>
        set((state) => ({ tickCount: state.tickCount + 1 })),

      reset: () =>
        set({
          patients: new Map(),
          patientSummaries: new Map(),
          liveUpdates: new Map(),
          selectedPatientId: null,
          tickCount: 0,
        }),
    })),
    { name: 'patient-store' }
  )
)

// Selector hooks for optimized re-renders
export const usePatient = (patientId: string) =>
  usePatientStore((state) => state.patients.get(patientId))

export const useLivePatientUpdate = (patientId: string) =>
  usePatientStore((state) => state.liveUpdates.get(patientId))

export const usePatientSummary = (patientId: string) =>
  usePatientStore((state) => state.patientSummaries.get(patientId))

export const useAllPatientSummaries = () =>
  usePatientStore((state) => Array.from(state.patientSummaries.values()))

export const useAllLiveUpdates = () =>
  usePatientStore((state) => Array.from(state.liveUpdates.values()))

export const useConnectionStatus = () =>
  usePatientStore((state) => ({
    connected: state.connected,
    clientId: state.clientId,
    lastUpdate: state.lastUpdate,
  }))

export const useSimulationStatus = () =>
  usePatientStore((state) => ({
    running: state.simulationRunning,
    paused: state.simulationPaused,
    patientCount: state.patientCount,
    tickCount: state.tickCount,
  }))

export const useSelectedPatient = () =>
  usePatientStore((state) => {
    const id = state.selectedPatientId
    return id ? state.patients.get(id) : null
  })

// Sort patients by category (Category 3 first)
export const useSortedPatients = () =>
  usePatientStore((state) => {
    const summaries = Array.from(state.patientSummaries.values())
    return summaries.sort((a, b) => {
      // Category 3 first, then 2, then 1
      if (a.category !== b.category) {
        return b.category - a.category
      }
      // Then by patient ID
      return a.patient_id.localeCompare(b.patient_id)
    })
  })
