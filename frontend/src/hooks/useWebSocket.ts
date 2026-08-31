import { useState, useEffect, useRef, useCallback } from 'react'

export interface FraudAlert {
  alert_id: string
  verification_id: string
  timestamp: number
  risk_level: 'HIGH' | 'CRITICAL'
  trust_score: number
  alert_type: string
  title: string
  message: string
  details: Record<string, any>
  user_id?: string
  file_name?: string
  file_type?: string
}

interface WSMessage {
  type: 'alert' | 'history' | 'stats_update' | 'heartbeat' | 'pong'
  alert?: FraudAlert
  alerts?: FraudAlert[]
  stats?: any
  connected_clients?: number
}

interface UseWebSocketReturn {
  connected: boolean
  alerts: FraudAlert[]
  latestAlert: FraudAlert | null
  connectedClients: number
  unreadCount: number
  clearAlerts: () => void
  dismissAlert: (id: string) => void
  clearUnread: () => void
}

export function useWebSocket(): UseWebSocketReturn {
  const [connected, setConnected] = useState(false)
  const [alerts, setAlerts] = useState<FraudAlert[]>([])
  const [latestAlert, setLatestAlert] = useState<FraudAlert | null>(null)
  const [connectedClients, setConnectedClients] = useState(0)
  const [unreadCount, setUnreadCount] = useState(0)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>(null)
  const heartbeatRef = useRef<ReturnType<typeof setInterval>>(null)

  const connect = useCallback(() => {
    // Determine WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const wsUrl = `${protocol}//${host}/ws/alerts`

    try {
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        console.log('[WS] Connected to alert stream')
        // Start heartbeat
        heartbeatRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }))
          }
        }, 25000)
      }

      ws.onmessage = (event) => {
        try {
          const msg: WSMessage = JSON.parse(event.data)

          switch (msg.type) {
            case 'alert':
              if (msg.alert) {
                setAlerts((prev) => [msg.alert!, ...prev].slice(0, 100))
                setLatestAlert(msg.alert)
                setUnreadCount((prev) => prev + 1)
                // Play notification sound for CRITICAL
                if (msg.alert.risk_level === 'CRITICAL') {
                  playAlertSound('critical')
                } else {
                  playAlertSound('high')
                }
              }
              break

            case 'history':
              if (msg.alerts) {
                setAlerts(msg.alerts.reverse())
              }
              if (msg.connected_clients !== undefined) {
                setConnectedClients(msg.connected_clients)
              }
              break

            case 'stats_update':
              // Stats are handled by the polling interval in App
              break

            case 'heartbeat':
              // Keep-alive response
              break

            case 'pong':
              break
          }
        } catch {
          // Ignore parse errors
        }
      }

      ws.onclose = () => {
        setConnected(false)
        if (heartbeatRef.current) {
          clearInterval(heartbeatRef.current)
        }
        console.log('[WS] Disconnected, reconnecting in 3s...')
        reconnectTimeoutRef.current = setTimeout(connect, 3000)
      }

      ws.onerror = () => {
        ws.close()
      }
    } catch {
      // Connection failed, retry
      reconnectTimeoutRef.current = setTimeout(connect, 3000)
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (heartbeatRef.current) {
        clearInterval(heartbeatRef.current)
      }
    }
  }, [connect])

  const clearAlerts = useCallback(() => {
    setAlerts([])
    setUnreadCount(0)
  }, [])

  const dismissAlert = useCallback((id: string) => {
    setAlerts((prev) => prev.filter((a) => a.alert_id !== id))
  }, [])

  const clearUnread = useCallback(() => {
    setUnreadCount(0)
  }, [])

  return {
    connected,
    alerts,
    latestAlert,
    connectedClients,
    unreadCount,
    clearAlerts,
    dismissAlert,
    clearUnread,
  }
}

function playAlertSound(severity: 'high' | 'critical') {
  try {
    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)()
    const osc = ctx.createOscillator()
    const gain = ctx.createGain()

    osc.connect(gain)
    gain.connect(ctx.destination)

    if (severity === 'critical') {
      osc.frequency.setValueAtTime(880, ctx.currentTime)
      osc.frequency.setValueAtTime(660, ctx.currentTime + 0.1)
      osc.frequency.setValueAtTime(880, ctx.currentTime + 0.2)
      gain.gain.setValueAtTime(0.15, ctx.currentTime)
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.4)
      osc.start(ctx.currentTime)
      osc.stop(ctx.currentTime + 0.4)
    } else {
      osc.frequency.setValueAtTime(660, ctx.currentTime)
      gain.gain.setValueAtTime(0.1, ctx.currentTime)
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.25)
      osc.start(ctx.currentTime)
      osc.stop(ctx.currentTime + 0.25)
    }
  } catch {
    // Audio not available
  }
}
