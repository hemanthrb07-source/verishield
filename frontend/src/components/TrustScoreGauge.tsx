import { useEffect, useState } from 'react'
import { Shield, ShieldAlert, ShieldCheck, ShieldQuestion } from 'lucide-react'

interface Props {
  score: number | null
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' | null
  isProcessing: boolean
}

export function TrustScoreGauge({ score, riskLevel, isProcessing }: Props) {
  const [animatedScore, setAnimatedScore] = useState(0)

  useEffect(() => {
    if (score === null) {
      setAnimatedScore(0)
      return
    }
    // Animate score
    const duration = 1000
    const startTime = Date.now()
    const start = animatedScore
    const diff = score - start

    const animate = () => {
      const elapsed = Date.now() - startTime
      const progress = Math.min(elapsed / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3) // ease-out cubic
      setAnimatedScore(start + diff * eased)
      if (progress < 1) requestAnimationFrame(animate)
    }
    requestAnimationFrame(animate)
  }, [score])

  const getColor = () => {
    if (riskLevel === 'CRITICAL') return '#dc2626'
    if (riskLevel === 'HIGH') return '#ef4444'
    if (riskLevel === 'MEDIUM') return '#f59e0b'
    if (riskLevel === 'LOW') return '#10b981'
    return '#6b7280'
  }

  const getBgColor = () => {
    if (riskLevel === 'CRITICAL') return 'from-red-500/20 to-red-600/5'
    if (riskLevel === 'HIGH') return 'from-red-500/10 to-orange-600/5'
    if (riskLevel === 'MEDIUM') return 'from-amber-500/10 to-yellow-600/5'
    if (riskLevel === 'LOW') return 'from-emerald-500/10 to-green-600/5'
    return 'from-gray-800/50 to-gray-900/50'
  }

  const getIcon = () => {
    if (riskLevel === 'CRITICAL' || riskLevel === 'HIGH')
      return <ShieldAlert className="w-6 h-6" />
    if (riskLevel === 'MEDIUM')
      return <ShieldQuestion className="w-6 h-6" />
    if (riskLevel === 'LOW')
      return <ShieldCheck className="w-6 h-6" />
    return <Shield className="w-6 h-6" />
  }

  const color = getColor()
  const circumference = 2 * Math.PI * 40
  const strokeDashoffset = circumference - (animatedScore / 100) * circumference

  return (
    <div className={`card bg-gradient-to-b ${getBgColor()}`}>
      <h3 className="text-sm font-semibold text-gray-300 mb-6 text-center">Trust Score</h3>

      <div className="relative flex items-center justify-center">
        <svg width="160" height="160" className="transform -rotate-90">
          {/* Background circle */}
          <circle
            cx="80" cy="80" r="40"
            fill="none"
            stroke="#1f2937"
            strokeWidth="8"
          />
          {/* Score arc */}
          <circle
            cx="80" cy="80" r="40"
            fill="none"
            stroke={color}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            style={{ transition: 'stroke-dashoffset 0.5s ease-out, stroke 0.3s ease' }}
          />
        </svg>

        {/* Center content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          {isProcessing ? (
            <div className="w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
          ) : score !== null ? (
            <>
              <div style={{ color }} className="mb-1">
                {getIcon()}
              </div>
              <span className="text-3xl font-bold" style={{ color }}>
                {Math.round(animatedScore)}
              </span>
              <span className="text-xs text-gray-500 mt-0.5">/ 100</span>
            </>
          ) : (
            <>
              <Shield className="w-6 h-6 text-gray-600 mb-1" />
              <span className="text-lg text-gray-500">--</span>
            </>
          )}
        </div>
      </div>

      {/* Risk Level Badge */}
      {riskLevel && (
        <div className="flex justify-center mt-4">
          <span className={`badge text-sm px-4 py-1.5
            ${riskLevel === 'LOW' ? 'badge-green' :
              riskLevel === 'MEDIUM' ? 'badge-yellow' :
              'badge-red'}`}>
            {riskLevel === 'CRITICAL' ? '🔴 CRITICAL' :
             riskLevel === 'HIGH' ? '🟠 HIGH RISK' :
             riskLevel === 'MEDIUM' ? '🟡 MEDIUM RISK' :
             '🟢 LOW RISK'}
          </span>
        </div>
      )}

      {/* Confidence */}
      {score !== null && (
        <p className="text-center text-xs text-gray-500 mt-3">
          Classification confidence varies by analysis depth
        </p>
      )}
    </div>
  )
}
