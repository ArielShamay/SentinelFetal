/**
 * REST API Client
 * ================
 * Typed API client for SentinelFetal backend
 */

import type { 
  PatientSnapshot, 
  PatientSummary, 
  SimulationStatus, 
  SimulationResponse,
  EventInjection,
  EventInjectionResponse,
} from '../types'

// Get base URL from environment variable or use default
const API_BASE = (import.meta as unknown as { env: Record<string, string> }).env?.VITE_API_URL || '/api'

class APIClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(error.detail || `HTTP ${response.status}`)
    }

    return response.json()
  }

  // ==========================================================================
  // Health
  // ==========================================================================

  async healthCheck(): Promise<{ status: string; version: string; timestamp: number }> {
    return this.request('/health')
  }

  // ==========================================================================
  // Patients
  // ==========================================================================

  async getPatients(durationMinutes: number = 5): Promise<{ patients: PatientSnapshot[]; count: number }> {
    return this.request(`/patients?duration_minutes=${durationMinutes}`)
  }

  async getPatientsSummary(): Promise<{ patients: PatientSummary[]; count: number }> {
    return this.request('/patients/summary')
  }

  async getPatient(patientId: string, durationMinutes: number = 5): Promise<PatientSnapshot> {
    return this.request(`/patients/${patientId}?duration_minutes=${durationMinutes}`)
  }

  /** Alias for getPatient for DetailView compatibility */
  async getPatientSnapshot(patientId: string, durationMinutes: number = 5): Promise<PatientSnapshot> {
    return this.getPatient(patientId, durationMinutes)
  }

  async getPatientHistory(
    patientId: string, 
    durationMinutes: number = 30
  ): Promise<{ patient_id: string; fhr: number[]; uc: number[]; timestamps: number[] }> {
    return this.request(`/patients/${patientId}/history?duration_minutes=${durationMinutes}`)
  }

  async injectEvent(patientId: string, event: EventInjection): Promise<EventInjectionResponse> {
    return this.request(`/patients/${patientId}/event`, {
      method: 'POST',
      body: JSON.stringify(event),
    })
  }

  // ==========================================================================
  // Simulation
  // ==========================================================================

  async getSimulationStatus(): Promise<SimulationStatus> {
    return this.request('/simulation/status')
  }

  async startSimulation(): Promise<SimulationResponse> {
    return this.request('/simulation/start', { method: 'POST' })
  }

  async stopSimulation(): Promise<SimulationResponse> {
    return this.request('/simulation/stop', { method: 'POST' })
  }

  async pauseSimulation(): Promise<SimulationResponse> {
    return this.request('/simulation/pause', { method: 'POST' })
  }

  async resumeSimulation(): Promise<SimulationResponse> {
    return this.request('/simulation/resume', { method: 'POST' })
  }

  async resetSimulation(): Promise<SimulationResponse> {
    return this.request('/simulation/reset', { method: 'POST' })
  }

  async updateConfig(config: { patient_count?: number; speed_multiplier?: number }): Promise<SimulationResponse> {
    return this.request('/simulation/config', {
      method: 'PATCH',
      body: JSON.stringify(config),
    })
  }

  // ==========================================================================
  // WebSocket Stats
  // ==========================================================================

  async getWebSocketStats(): Promise<{ connected_clients: number; queue_size: number; running: boolean }> {
    // Note: This goes to /ws/stats not /api/ws/stats
    const response = await fetch(`${this.baseUrl.replace('/api', '')}/ws/stats`)
    if (!response.ok) throw new Error('Failed to get WebSocket stats')
    return response.json()
  }
}

// Export singleton instance
export const api = new APIClient()

// Export class for testing
export { APIClient }
