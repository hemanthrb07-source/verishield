import { useState, useEffect } from 'react'
import { api, SystemStats } from './services/api'
import { useWebSocket } from './hooks/useWebSocket'
import { UploadPanel } from './components/UploadPanel'
import { TrustScoreGauge } from './components/TrustScoreGauge'
import { ResultsPanel } from './components/ResultsPanel'
import { FraudGraph } from './components/FraudGraph'
import { StatsBar } from './components/StatsBar'
import { VerificationHistory } from './components/VerificationHistory'
import { AlertToast } from './components/AlertToast'
import { AdversarialTest } from './components/AdversarialTest'
import { LivenessTest } from './components/LivenessTest'
import { Shield, Activity, Zap, Clock, Target, Eye } from 'lucide-react'
import type { VerificationResult } from './services/api'

type Tab = 'upload' | 'history' | 'graph' | 'adversarial' | 'liveness'

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('upload')
  const [stats, setStats] = useState<SystemStats | null>(null)
  const [currentResult, setCurrentResult] = useState<VerificationResult | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Real-time WebSocket alerts
  const {
    connected: wsConnected,
    alerts: wsAlerts,
    latestAlert,
    connectedClients,
    unreadCount,
    clearAlerts,
    dismissAlert,
    clearUnread,
  } = useWebSocket()

  useEffect(() => {
    loadStats()
    const interval = setInterval(loadStats, 10000)
    return () => clearInterval(interval)
  }, [])

  const loadStats = async () => {
    try {
      const s = await api.getStats()
      setStats(s)
    } catch {
      // Stats endpoint might not be available yet
    }
  }

  const handleVerificationComplete = (result: VerificationResult) => {
    setCurrentResult(result)
    setIsProcessing(false)
    loadStats()
  }

  const handleStartProcessing = () => {
    setIsProcessing(true)
    setError(null)
    setCurrentResult(null)
  }

  const handleError = (msg: string) => {
    setError(msg)
    setIsProcessing(false)
  }

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-gray-800/50 bg-gray-950/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center">
                <Shield className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-white tracking-tight">VeriShield</h1>
                <p className="text-xs text-gray-500">AI Fraud & Deepfake Detection</p>
              </div>
            </div>

            <nav className="flex items-center gap-1">
              {([
                { key: 'upload' as Tab, label: 'Verify', icon: Zap },
                { key: 'history' as Tab, label: 'History', icon: Clock },
                { key: 'graph' as Tab, label: 'Graph', icon: Activity },
                { key: 'adversarial' as Tab, label: 'Adversarial', icon: Target },
                { key: 'liveness' as Tab, label: 'Liveness', icon: Eye },
              ]).map(({ key, label, icon: Icon }) => (
                <button
                  key={key}
                  onClick={() => setActiveTab(key)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all
                    ${activeTab === key
                      ? 'bg-brand-600/20 text-brand-400 border border-brand-500/30'
                      : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
                    }`}
                >
                  <Icon className="w-4 h-4" />
                  {label}
                </button>
              ))}
            </nav>
          </div>
        </div>
      </header>

      {/* Real-time Alert Toasts */}
      <AlertToast
        alerts={wsAlerts}
        latestAlert={latestAlert}
        connected={wsConnected}
        unreadCount={unreadCount}
        onDismiss={dismissAlert}
        onClearAll={clearAlerts}
        onClearUnread={clearUnread}
      />

      {/* Stats Bar */}
      {stats && <StatsBar stats={stats} />}

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'upload' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left: Upload Panel */}
            <div className="lg:col-span-2 space-y-6">
              <UploadPanel
                onStart={handleStartProcessing}
                onComplete={handleVerificationComplete}
                onError={handleError}
                isProcessing={isProcessing}
              />

              {error && (
                <div className="card border-red-500/30 bg-red-500/10 animate-slide-up">
                  <p className="text-red-400 text-sm">{error}</p>
                </div>
              )}

              {currentResult && (
                <ResultsPanel result={currentResult} />
              )}
            </div>

            {/* Right: Trust Score Gauge */}
            <div className="space-y-6">
              <TrustScoreGauge
                score={currentResult?.trust_score ?? null}
                riskLevel={currentResult?.risk_level ?? null}
                isProcessing={isProcessing}
              />

              {/* Quick Info */}
              <div className="card">
                <h3 className="text-sm font-semibold text-gray-300 mb-4">Pipeline Stages</h3>
                <div className="space-y-3">
                  {[
                    { label: 'Document Intelligence', desc: 'OCR, font & tampering analysis' },
                    { label: 'Deepfake Detection', desc: 'CNN classifier, GAN fingerprinting' },
                    { label: 'Face Matching', desc: 'ArcFace embedding comparison' },
                    { label: 'Fraud Graph', desc: 'Relationship network analysis' },
                    { label: 'Risk Scoring', desc: 'Weighted trust score computation' },
                  ].map((stage, i) => (
                    <div key={i} className="flex items-start gap-3">
                      <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold shrink-0
                        ${isProcessing ? 'bg-brand-500/20 text-brand-400 animate-pulse' : 'bg-gray-800 text-gray-500'}`}>
                        {i + 1}
                      </div>
                      <div>
                        <p className="text-sm font-medium text-gray-300">{stage.label}</p>
                        <p className="text-xs text-gray-500">{stage.desc}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'history' && (
          <VerificationHistory />
        )}

        {activeTab === 'graph' && (
          <FraudGraph />
        )}

        {activeTab === 'adversarial' && (
          <AdversarialTest />
        )}

        {activeTab === 'liveness' && (
          <LivenessTest />
        )}
      </main>
    </div>
  )
}

export default App
