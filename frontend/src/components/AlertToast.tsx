import { useState, useEffect } from 'react'
import { AlertTriangle, XCircle, Shield, X, Bell, BellOff } from 'lucide-react'
import type { FraudAlert } from '../hooks/useWebSocket'

interface Props {
  alerts: FraudAlert[]
  latestAlert: FraudAlert | null
  connected: boolean
  unreadCount: number
  onDismiss: (id: string) => void
  onClearAll: () => void
  onClearUnread: () => void
}

export function AlertToast({
  alerts,
  latestAlert,
  connected,
  unreadCount,
  onDismiss,
  onClearAll,
  onClearUnread,
}: Props) {
  const [expanded, setExpanded] = useState(false)
  const [autoDismissId, setAutoDismissId] = useState<string | null>(null)

  // Auto-dismiss latest toast after 6 seconds
  useEffect(() => {
    if (latestAlert) {
      setAutoDismissId(latestAlert.alert_id)
      const timer = setTimeout(() => setAutoDismissId(null), 6000)
      return () => clearTimeout(timer)
    }
  }, [latestAlert])

  const formatTime = (ts: number) => {
    const d = new Date(ts * 1000)
    return d.toLocaleTimeString()
  }

  const getAlertIcon = (level: string) => {
    if (level === 'CRITICAL') return <XCircle className="w-5 h-5 text-red-400" />
    return <AlertTriangle className="w-5 h-5 text-amber-400" />
  }

  const getAlertBg = (level: string) => {
    if (level === 'CRITICAL') return 'bg-red-500/10 border-red-500/30'
    return 'bg-amber-500/10 border-amber-500/30'
  }

  return (
    <>
      {/* Floating notification badge (bottom-right) */}
      <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end gap-3">
        {/* Latest alert toast */}
        {latestAlert && autoDismissId === latestAlert.alert_id && (
          <div
            className={`max-w-md w-full ${getAlertBg(latestAlert.risk_level)} border rounded-2xl p-4 shadow-2xl animate-slide-up backdrop-blur-xl`}
            role="alert"
          >
            <div className="flex items-start gap-3">
              {getAlertIcon(latestAlert.risk_level)}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-sm font-bold text-white">{latestAlert.title}</span>
                  <span className={`text-xs px-1.5 py-0.5 rounded-full font-bold
                    ${latestAlert.risk_level === 'CRITICAL'
                      ? 'bg-red-500/30 text-red-300'
                      : 'bg-amber-500/30 text-amber-300'
                    }`}>
                    {latestAlert.risk_level}
                  </span>
                </div>
                <p className="text-xs text-gray-300 leading-relaxed">{latestAlert.message}</p>
                <div className="flex items-center gap-3 mt-2 text-xs text-gray-500">
                  {latestAlert.file_name && <span>{latestAlert.file_name}</span>}
                  <span>{formatTime(latestAlert.timestamp)}</span>
                  <span>Score: {latestAlert.trust_score.toFixed(0)}</span>
                </div>
              </div>
              <button
                onClick={() => setAutoDismissId(null)}
                className="text-gray-500 hover:text-gray-300 shrink-0"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {/* Alert list panel (click to expand) */}
        {alerts.length > 0 && (
          <div className="relative">
            <button
              onClick={() => { setExpanded(!expanded); onClearUnread() }}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gray-900/90 border border-gray-700 backdrop-blur-xl shadow-xl hover:bg-gray-800/90 transition-all"
            >
              <Bell className="w-4 h-4 text-amber-400" />
              <span className="text-sm font-medium text-gray-200">Alerts</span>
              {unreadCount > 0 && (
                <span className="flex items-center justify-center w-5 h-5 rounded-full bg-red-500 text-white text-xs font-bold animate-pulse">
                  {unreadCount > 99 ? '99+' : unreadCount}
                </span>
              )}
            </button>

            {/* Expanded alert list */}
            {expanded && (
              <div className="absolute bottom-14 right-0 w-96 max-h-[60vh] overflow-y-auto bg-gray-900/95 border border-gray-700 rounded-2xl shadow-2xl backdrop-blur-xl animate-slide-up">
                <div className="flex items-center justify-between p-4 border-b border-gray-800">
                  <h3 className="text-sm font-semibold text-white">
                    Fraud Alerts ({alerts.length})
                  </h3>
                  <button
                    onClick={onClearAll}
                    className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
                  >
                    Clear all
                  </button>
                </div>

                <div className="divide-y divide-gray-800/50">
                  {alerts.slice(0, 20).map((alert) => (
                    <div
                      key={alert.alert_id}
                      className={`p-3 hover:bg-gray-800/30 transition-colors ${getAlertBg(alert.risk_level)} border-l-0 border-t-0 border-b-0 border-r-0 rounded-none`}
                    >
                      <div className="flex items-start gap-2">
                        {getAlertIcon(alert.risk_level)}
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-semibold text-white">{alert.title}</p>
                          <p className="text-xs text-gray-400 mt-0.5 line-clamp-2">{alert.message}</p>
                          <div className="flex items-center gap-2 mt-1 text-xs text-gray-500">
                            {alert.file_name && <span className="truncate max-w-[120px]">{alert.file_name}</span>}
                            <span>{formatTime(alert.timestamp)}</span>
                          </div>
                        </div>
                        <button
                          onClick={() => onDismiss(alert.alert_id)}
                          className="text-gray-600 hover:text-gray-400 shrink-0"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Connection indicator (top-right corner) */}
      <div className="fixed top-20 right-6 z-40">
        <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs
          ${connected
            ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
            : 'bg-gray-800/80 text-gray-500 border border-gray-700/50'
          }`}>
          <div className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-emerald-400 animate-pulse' : 'bg-gray-600'}`} />
          {connected ? 'Live Alerts' : 'Reconnecting...'}
        </div>
      </div>
    </>
  )
}
