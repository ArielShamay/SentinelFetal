import { useCallback, useEffect, useRef } from 'react'
import { wsManager } from '../services/websocket'
import { usePatientStore } from '../stores/patientStore'
import type { WSMessage, WSPatientUpdate, MHRAlert } from '../types'

export function usePatientStream() {
  const updateFromWebSocket = usePatientStore((s) => s.updateFromWebSocket)
  const setConnected = usePatientStore((s) => s.setConnected)
  const incrementTickCount = usePatientStore((s) => s.incrementTickCount)
  const initialized = useRef(false)

  const handleMessage = useCallback(
    (message: WSMessage) => {
      console.log('🎯 handleMessage:', message.type, message)
      if (message.type === 'connected') {
        setConnected(true, message.client_id)
      } else if (message.type === 'patient_update') {
        console.log('📊 Patient update - patients:', message.patients?.length, 'single:', message.patient_id)
        // Handle batch update
        if (message.patients) {
          for (const patient of message.patients) {
            updateFromWebSocket(patient)
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
          }
          console.log('📌 Extracted patient update:', patientUpdate)
          updateFromWebSocket(patientUpdate)
        }
        incrementTickCount()
      }
    },
    [updateFromWebSocket, setConnected, incrementTickCount]
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
    setConnected(false)
  }, [setConnected])

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
