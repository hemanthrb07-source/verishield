import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, PieChart, Pie, Legend
} from 'recharts'

interface Props {
  componentScores: Record<string, number>
  trustScore: number
  riskLevel: string
}

const LABELS: Record<string, string> = {
  document_authenticity: 'Document Authenticity',
  deepfake_detection: 'Deepfake Detection',
  face_match: 'Face Matching',
  graph_risk: 'Fraud Graph',
}

const COLORS: Record<string, string> = {
  document_authenticity: '#f59e0b',
  deepfake_detection: '#a855f7',
  face_match: '#3b82f6',
  graph_risk: '#10b981',
}

function getScoreColor(score: number): string {
  if (score >= 70) return '#10b981'
  if (score >= 40) return '#f59e0b'
  return '#ef4444'
}

export function TrustScoreBreakdown({ componentScores, trustScore, riskLevel }: Props) {
  // Prepare bar chart data
  const barData = Object.entries(componentScores).map(([key, value]) => ({
    name: LABELS[key] || key,
    score: Math.round(value),
    fill: getScoreColor(value),
    key,
  }))

  // Prepare radar chart data
  const radarData = Object.entries(componentScores).map(([key, value]) => ({
    subject: LABELS[key]?.replace(' ', '\n') || key,
    score: Math.round(value),
    fullMark: 100,
  }))

  // Prepare pie chart for weighted contribution
  const WEIGHTS: Record<string, number> = {
    document_authenticity: 0.25,
    deepfake_detection: 0.30,
    face_match: 0.25,
    graph_risk: 0.20,
  }
  const pieData = Object.entries(componentScores).map(([key, value]) => ({
    name: LABELS[key] || key,
    value: Math.round(value * (WEIGHTS[key] || 0.25)),
    color: COLORS[key] || '#6b7280',
  }))

  const hasData = Object.keys(componentScores).length > 0
  if (!hasData) return null

  return (
    <div className="card space-y-4">
      <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
        📊 Trust Score Breakdown
      </h3>

      {/* Overall score bar */}
      <div className="p-4 bg-gray-800/50 rounded-xl">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-gray-400">Overall Trust Score</span>
          <span className={`text-lg font-bold ${
            riskLevel === 'LOW' ? 'text-emerald-400' :
            riskLevel === 'MEDIUM' ? 'text-amber-400' : 'text-red-400'
          }`}>
            {trustScore.toFixed(0)} / 100
          </span>
        </div>
        <div className="w-full h-3 bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-1000"
            style={{
              width: `${trustScore}%`,
              backgroundColor: getScoreColor(trustScore),
            }}
          />
        </div>
        <div className="flex justify-between mt-1">
          <span className="text-xs text-red-400">0 — Dangerous</span>
          <span className="text-xs text-amber-400">50</span>
          <span className="text-xs text-emerald-400">100 — Safe</span>
        </div>
      </div>

      {/* Bar chart of component scores */}
      <div className="p-4 bg-gray-800/50 rounded-xl">
        <p className="text-xs text-gray-400 mb-3">Score by Detection Area</p>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={barData} layout="vertical" margin={{ left: 10, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis type="number" domain={[0, 100]} tick={{ fill: '#9ca3af', fontSize: 11 }} />
            <YAxis
              type="category"
              dataKey="name"
              width={130}
              tick={{ fill: '#d1d5db', fontSize: 11 }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1f2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#e5e7eb',
                fontSize: 12,
              }}
              formatter={(value: number) => [`${value}/100`, 'Score']}
            />
            <Bar dataKey="score" radius={[0, 6, 6, 0]} barSize={20}>
              {barData.map((entry, index) => (
                <Cell key={index} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Radar chart */}
      <div className="p-4 bg-gray-800/50 rounded-xl">
        <p className="text-xs text-gray-400 mb-3">Detection Coverage Radar</p>
        <ResponsiveContainer width="100%" height={250}>
          <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="75%">
            <PolarGrid stroke="#374151" />
            <PolarAngleAxis
              dataKey="subject"
              tick={{ fill: '#d1d5db', fontSize: 10 }}
            />
            <PolarRadiusAxis
              angle={30}
              domain={[0, 100]}
              tick={{ fill: '#9ca3af', fontSize: 10 }}
            />
            <Radar
              name="Score"
              dataKey="score"
              stroke="#6366f1"
              fill="#6366f1"
              fillOpacity={0.3}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* Weighted contribution pie */}
      <div className="p-4 bg-gray-800/50 rounded-xl">
        <p className="text-xs text-gray-400 mb-3">Weighted Contribution to Trust Score</p>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie
              data={pieData}
              cx="50%"
              cy="50%"
              innerRadius={50}
              outerRadius={80}
              paddingAngle={3}
              dataKey="value"
              label={({ name, value }) => `${value}`}
            >
              {pieData.map((entry, index) => (
                <Cell key={index} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                backgroundColor: '#1f2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#e5e7eb',
                fontSize: 12,
              }}
              formatter={(value: number, name: string) => [`${value} pts`, name]}
            />
            <Legend
              wrapperStyle={{ fontSize: 11, color: '#9ca3af' }}
              formatter={(value) => <span style={{ color: '#d1d5db' }}>{value}</span>}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Per-component detail cards */}
      <div className="grid grid-cols-2 gap-2">
        {barData.map((item) => (
          <div key={item.key} className="p-3 bg-gray-800/50 rounded-xl border border-gray-700/50">
            <div className="flex items-center gap-2 mb-1">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: item.fill }} />
              <span className="text-xs text-gray-400">{item.name}</span>
            </div>
            <p className={`text-lg font-bold ${
              item.score >= 70 ? 'text-emerald-400' :
              item.score >= 40 ? 'text-amber-400' : 'text-red-400'
            }`}>
              {item.score}
            </p>
            <p className="text-xs text-gray-500">
              Weight: {((WEIGHTS[item.key] || 0.25) * 100).toFixed(0)}%
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}
