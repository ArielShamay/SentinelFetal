import { useCallback, useEffect, useRef } from 'react'
import { wsManager } from '../services/websocket'
import { usePatientStore } from '../stores/patientStore'
import type { WSMessage, WSPatientUpdate, MHRAlert } from '../types'

export function usePatientStream() {
  const updateFromWebSocket = usePatientStore((s) => s.updateFromWebSocket)
  const setConnected = usePatientStore((s) => s.setConnected)
  const incrementTickCount = usePatientStore((s) => s.incrementTickCount)
  const setSimulationState = usePatientStore((s) => s.setSimulationState)
  const initialized = useRef(false)
  const simulationSynced = useRef(false)

  // Store callbacks in refs to avoid stale closure issues
  const updateFromWebSocketRef = useRef(updateFromWebSocket)
  const setConnectedRef = useRef(setConnected)
  const incrementTickCountRef = useRef(incrementTickCount)
  const setSimulationStateRef = useRef(setSimulationState)

  // Keep refs updated
  useEffect(() => {
    updateFromWebSocketRef.current = updateFromWebSocket
    setConnectedRef.current = setConnected
    incrementTickCountRef.current = incrementTickCount
    setSimulationStateRef.current = setSimulationState
  }, [updateFromWebSocket, setConnected, incrementTickCount, setSimulationState])

  const handleMessage = useCallback(
    (message: WSMessage) => {
      if (message.type === 'connected') {
        setConnectedRef.current(true, message.client_id)
      } else if (message.type === 'patient_update') {
        // Receiving patient data means the simulation is running
        if (!simulationSynced.current) {
          simulationSynced.current = true
          setSimulationStateRef.current(true, false)
        }

        // Handle batch update
        if (message.patients) {
          for (const patient of message.patients) {
            updateFromWebSocketRef.current(patient)
          }
        }
        // Handle single patient update - extract WSPatientUpdate fields from message
        else if (message.patient_id) {
          const patientUpdate: WSPatientUpdate = {
            patient_id: message.patient_id,
            category: message.category as 1 | 2 | 3,
            baseline: message.baseline as number,
            variability: message.variability as number,
            fhr_latest: message.fhr_latest as number[],
            uc_latest: message.uc_latest as number[],
            fsqi: message.fsqi as number,
            confidence: message.confidence as number,
            findings: message.findings as Record<string, unknown>,
            mhr_alert: message.mhr_alert as MHRAlert | null | undefined,
            trend_score: message.trend_score as number | undefined,
            trend_slope: message.trend_slope as number | undefined,
            explanation: message.explanation as WSPatientUpdate['explanation'],
            highlight_regions: message.highlight_regions as WSPatientUpdate['highlight_regions'],
          }
          updateFromWebSocketRef.current(patientUpdate)
        }
        incrementTickCountRef.current()
      }
    },
    [] // No dependencies - uses refs
  )

  const handleStatus = useCallback(
    (connected: boolean) => {
      setConnectedRef.current(connected)
    },
    [] // No dependencies - uses ref
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
    setConnectedRef.current(false)
  }, [])

  const subscribe = useCallback((patientIds: string[]) => {
    wsManager.subscribe(patientIds)
  }, [])

  const unsubscribe = useCallback(() => {
    wsManager.unsubscribe()
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
    unsubscribe,
    isConnected: wsManager.isConnected,
  }
}
