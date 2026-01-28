/**
 * WebSocket Manager
 * ==================
 * Manages WebSocket connection to the backend streaming endpoint
 */

import type { WSMessage } from '../types'

type MessageHandler = (message: WSMessage) => void
type StatusHandler = (connected: boolean) => void

class WebSocketManager {
  private ws: WebSocket | null = null
  private url: string
  private reconnectAttempts = 0
  private maxReconnectAttempts = 10
  private reconnectDelay = 1000
  private messageHandler: MessageHandler | null = null
  private statusHandler: StatusHandler | null = null
  private clientId: string | null = null
  private shouldReconnect = true

  constructor(url?: string) {
    // Use relative URL so Vite proxy handles it in dev, and works in production too
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    this.url = url || `${wsProtocol}//${window.location.host}/ws/stream`
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return
    }

    this.shouldReconnect = true

    const urlWithParams = this.clientId
      ? `${this.url}?client_id=${this.clientId}&format=json`
      : `${this.url}?format=json`

    console.log('[WS] Connecting to:', urlWithParams)

    this.ws = new WebSocket(urlWithParams)

    this.ws.onopen = () => {
      console.log('[WS] Connected')
      this.reconnectAttempts = 0
      this.statusHandler?.(true)
    }

    this.ws.onmessage = (event) => {
      try {
        let data: WSMessage

        // Handle binary data (Blob or ArrayBuffer)
        if (event.data instanceof Blob) {
          event.data.text().then(text => {
            try {
              data = JSON.parse(text) as WSMessage
              this.processMessage(data)
            } catch (error) {
              console.error('[WS] Failed to parse Blob data:', error)
            }
          })
          return
        } else if (event.data instanceof ArrayBuffer) {
          const decoder = new TextDecoder()
          const text = decoder.decode(event.data)
          data = JSON.parse(text) as WSMessage
        } else {
          data = JSON.parse(event.data) as WSMessage
        }

        this.processMessage(data)
      } catch (error) {
        console.error('[WS] Failed to parse message:', error)
      }
    }

    this.ws.onclose = (event) => {
      console.log('[WS] Disconnected:', event.code, event.reason)
      this.statusHandler?.(false)

      if (this.shouldReconnect) {
        this.attemptReconnect()
      }
    }

    this.ws.onerror = (error) => {
      console.error('[WS] Error:', error)
    }
  }

  private processMessage(data: WSMessage): void {
    // Handle connection message
    if (data.type === 'connected' && data.client_id) {
      console.log('[WS] Assigned client ID:', data.client_id)
      this.clientId = data.client_id
    }

    // Handle ping - respond with pong
    if (data.type === 'ping') {
      this.sendPong()
      return
    }

    // Forward to handler
    if (this.messageHandler) {
      this.messageHandler(data)
    }
  }

  disconnect(): void {
    this.shouldReconnect = false
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect')
      this.ws = null
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('[WS] Max reconnect attempts reached')
      return
    }

    this.reconnectAttempts++
    const delay = Math.min(
      this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1),
      30000 // Max 30 seconds
    )

    console.log(`[WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`)

    setTimeout(() => {
      if (this.shouldReconnect) {
        this.connect()
      }
    }, delay)
  }

  private sendPong(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'pong' }))
    }
  }

  subscribe(patientIds: string[]): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      const message = JSON.stringify({
        type: 'subscribe',
        patient_ids: patientIds,
      })
      this.ws.send(message)
    }
  }

  unsubscribe(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      const message = JSON.stringify({
        type: 'unsubscribe',
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

  get currentClientId(): string | null {
    return this.clientId
  }
}

// Export singleton instance
export const wsManager = new WebSocketManager()

// Export class for testing
export { WebSocketManager }
