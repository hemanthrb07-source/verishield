import { useState, useEffect } from 'react'
import { api } from '../services/api'
import { Clock, Shield, AlertTriangle, ChevronLeft, ChevronRight } from 'lucide-react'

export function VerificationHistory() {
  const [items, setItems] = useState<any[]>([])
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<string>('')
  const limit = 20

  const loadData = async () => {
    setLoading(true)
    try {
      const result = await api.getVerifications({
        limit,
        offset,
        risk_level: filter || undefined,
      })
      setItems(result.items)
      setTotal(result.total)
    } catch {
      // Not available yet
    }
    setLoading(false)
  }

  useEffect(() => {
    loadData()
  }, [offset, filter])

  const getRiskColor = (level: string) => {
    if (level === 'LOW') return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20'
    if (level === 'MEDIUM') return 'text-amber-400 bg-amber-500/10 border-amber-500/20'
    if (level === 'HIGH' || level === 'CRITICAL') return 'text-red-400 bg-red-500/10 border-red-500/20'
    return 'text-gray-400 bg-gray-500/10 border-gray-500/20'
  }

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-lg font-semibold text-white">Verification History</h2>
            <p className="text-xs text-gray-500 mt-1">{total} total records</p>
          </div>
          <div className="flex gap-2">
            {['', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].map(level => (
              <button
                key={level}
                onClick={() => { setFilter(level); setOffset(0) }}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all
                  ${filter === level
                    ? 'bg-brand-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
              >
                {level || 'All'}
              </button>
            ))}
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="w-6 h-6 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : items.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <Shield className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p className="text-sm">No verification records found</p>
            <p className="text-xs mt-1">Upload files to start verifying</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="text-left py-3 px-4 text-xs font-medium text-gray-500">Time</th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-gray-500">File</th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-gray-500">Type</th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-gray-500">Trust Score</th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-gray-500">Risk</th>
                  <th className="text-left py-3 px-4 text-xs font-medium text-gray-500">Status</th>
                </tr>
              </thead>
              <tbody>
                {items.map((item: any, i: number) => (
                  <tr key={item.id || i} className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors">
                    <td className="py-3 px-4 text-gray-400 text-xs">
                      {item.created_at ? new Date(item.created_at).toLocaleString() : '--'}
                    </td>
                    <td className="py-3 px-4 text-gray-200 text-xs font-medium truncate max-w-[200px]">
                      {item.file_name || '--'}
                    </td>
                    <td className="py-3 px-4">
                      <span className="badge bg-gray-700 text-gray-300">{item.file_type}</span>
                    </td>
                    <td className="py-3 px-4">
                      <span className={`font-bold ${
                        (item.trust_score ?? 0) >= 70 ? 'text-emerald-400' :
                        (item.trust_score ?? 0) >= 40 ? 'text-amber-400' : 'text-red-400'
                      }`}>
                        {item.trust_score != null ? item.trust_score.toFixed(0) : '--'}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      {item.risk_level ? (
                        <span className={`badge border ${getRiskColor(item.risk_level)}`}>
                          {item.risk_level}
                        </span>
                      ) : '--'}
                    </td>
                    <td className="py-3 px-4">
                      <span className={`flex items-center gap-1.5 text-xs
                        ${item.status === 'COMPLETED' ? 'text-emerald-400' :
                          item.status === 'FAILED' ? 'text-red-400' :
                          item.status === 'PROCESSING' ? 'text-brand-400' : 'text-gray-500'
                        }`}>
                        {item.status === 'PROCESSING' && (
                          <div className="w-3 h-3 border border-brand-400 border-t-transparent rounded-full animate-spin" />
                        )}
                        {item.status === 'COMPLETED' && <Shield className="w-3 h-3" />}
                        {item.status === 'FAILED' && <AlertTriangle className="w-3 h-3" />}
                        {item.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination */}
        {total > limit && (
          <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-800">
            <p className="text-xs text-gray-500">
              Showing {offset + 1}–{Math.min(offset + limit, total)} of {total}
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => setOffset(Math.max(0, offset - limit))}
                disabled={offset === 0}
                className="btn-secondary text-xs py-1.5 px-3 disabled:opacity-50"
              >
                <ChevronLeft className="w-3 h-3" />
              </button>
              <button
                onClick={() => setOffset(offset + limit)}
                disabled={offset + limit >= total}
                className="btn-secondary text-xs py-1.5 px-3 disabled:opacity-50"
              >
                <ChevronRight className="w-3 h-3" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
