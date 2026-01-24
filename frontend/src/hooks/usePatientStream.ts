import { useCallback, useEffect, useRef } from 'react'
import { wsManager } from '../services/websocket'
import { usePatientStore } from '../stores/patientStore'
import type { WSMessage, WSPatientUpdate } from '../types'

export function usePatientStream() {
  const updateFromWebSocket = usePatientStore((s) => s.updateFromWebSocket)
  const setConnected = usePatientStore((s) => s.setConnected)
  const incrementTickCount = usePatientStore((s) => s.incrementTickCount)
  const initialized = useRef(false)

  const handleMessage = useCallback(
    (message: WSMessage) => {
      if (message.type === 'connected') {
        setConnected(true, message.client_id)
      } else if (message.type === 'patient_update') {
        // Handle batch update
        if (message.patients) {
          for (const patient of message.patients) {
            updateFromWebSocket(patient)
          }
        }
        // Handle single patient update
        else if (message.patient_id) {
          updateFromWebSocket(message as unknown as WSPatientUpdate)
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
