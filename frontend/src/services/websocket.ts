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
    // Use relative WebSocket URL if not specified
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    this.url = url || `${wsProtocol}//${window.location.host}/ws/stream`
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected')
      return
    }

    this.shouldReconnect = true

    const urlWithParams = this.clientId
      ? `${this.url}?client_id=${this.clientId}&format=json`
      : `${this.url}?format=json`

    console.log('Connecting to WebSocket:', urlWithParams)
    
    this.ws = new WebSocket(urlWithParams)

    this.ws.onopen = () => {
      console.log('WebSocket connected')
      this.reconnectAttempts = 0
      this.statusHandler?.(true)
    }

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as WSMessage

        // Handle connection message
        if (data.type === 'connected' && data.client_id) {
          console.log('Assigned client ID:', data.client_id)
          this.clientId = data.client_id
        }

        // Handle ping - respond with pong
        if (data.type === 'ping') {
          this.sendPong()
          return
        }

        // Forward to handler
        this.messageHandler?.(data)
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error)
      }
    }

    this.ws.onclose = (event) => {
      console.log('WebSocket disconnected:', event.code, event.reason)
      this.statusHandler?.(false)
      
      if (this.shouldReconnect) {
        this.attemptReconnect()
      }
    }

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error)
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
      console.log('Max reconnect attempts reached')
      return
    }

    this.reconnectAttempts++
    const delay = Math.min(
      this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1),
      30000 // Max 30 seconds
    )

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`)

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
