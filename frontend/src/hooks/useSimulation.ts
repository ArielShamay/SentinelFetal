import { useState, useCallback } from 'react'
import { api } from '../services/api'
import { usePatientStore } from '../stores/patientStore'
import type { SimulationStatus } from '../types'

export function useSimulation() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  const setSimulationState = usePatientStore((s) => s.setSimulationState)
  const resetStore = usePatientStore((s) => s.reset)

  const updateState = useCallback((status: SimulationStatus) => {
    setSimulationState(status.running, status.paused, status.patient_count)
  }, [setSimulationState])

  const fetchStatus = useCallback(async () => {
    try {
      const status = await api.getSimulationStatus()
      updateState(status)
      return status
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch status')
      return null
    }
  }, [updateState])

  const start = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await api.startSimulation()
      updateState(response.status)
      return response.success
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start')
      return false
    } finally {
      setLoading(false)
    }
  }, [updateState])

  const stop = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await api.stopSimulation()
      updateState(response.status)
      return response.success
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to stop')
      return false
    } finally {
      setLoading(false)
    }
  }, [updateState])

  const pause = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await api.pauseSimulation()
      updateState(response.status)
      return response.success
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to pause')
      return false
    } finally {
      setLoading(false)
    }
  }, [updateState])

  const resume = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await api.resumeSimulation()
      updateState(response.status)
      return response.success
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to resume')
      return false
    } finally {
      setLoading(false)
    }
  }, [updateState])

  const reset = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await api.resetSimulation()
      updateState(response.status)
      resetStore()
      return response.success
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reset')
      return false
    } finally {
      setLoading(false)
    }
  }, [updateState, resetStore])

  const setPatientCount = useCallback(async (count: number) => {
    setLoading(true)
    setError(null)
    try {
      const response = await api.updateConfig({ patient_count: count })
      updateState(response.status)
      return response.success
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update config')
      return false
    } finally {
      setLoading(false)
    }
  }, [updateState])

  return {
    loading,
    error,
    fetchStatus,
    start,
    stop,
    pause,
    resume,
    reset,
    setPatientCount,
  }
}
