import type { SystemStats } from '../services/api'
import { Shield, AlertTriangle, BarChart3, Activity, Link } from 'lucide-react'

interface Props {
  stats: SystemStats
}

export function StatsBar({ stats }: Props) {
  return (
    <div className="border-b border-gray-800/50 bg-gray-900/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-6">
            <StatItem
              icon={<BarChart3 className="w-4 h-4" />}
              label="Total"
              value={stats.total_verifications}
            />
            <StatItem
              icon={<Shield className="w-4 h-4" />}
              label="Completed"
              value={stats.completed}
              color="text-emerald-400"
            />
            <StatItem
              icon={<AlertTriangle className="w-4 h-4" />}
              label="High Risk"
              value={stats.high_risk_detected}
              color="text-red-400"
            />
            <StatItem
              icon={<Activity className="w-4 h-4" />}
              label="Avg Score"
              value={stats.avg_trust_score}
              color="text-brand-400"
            />
          </div>
          <div className="flex items-center gap-4 text-xs text-gray-500">
            <span className="flex items-center gap-1">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              Graph: {stats.graph_nodes} nodes
            </span>
            <span className="flex items-center gap-1">
              <Link className="w-3 h-3" />
              Chain: {stats.blockchain_blocks} blocks
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

function StatItem({ icon, label, value, color = 'text-gray-300' }: {
  icon: React.ReactNode
  label: string
  value: number
  color?: string
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-gray-500">{icon}</span>
      <span className="text-xs text-gray-500">{label}:</span>
      <span className={`text-sm font-semibold ${color}`}>
        {typeof value === 'number' && value % 1 !== 0 ? value.toFixed(1) : value}
      </span>
    </div>
  )
}
