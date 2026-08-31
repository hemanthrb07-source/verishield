import { useState, useEffect, useRef, useCallback } from 'react'
import { api, GraphData } from '../services/api'
import { RefreshCw, AlertTriangle } from 'lucide-react'

export function FraudGraph() {
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [loading, setLoading] = useState(true)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animFrameRef = useRef<number>(0)

  const loadData = async () => {
    setLoading(true)
    try {
      const data = await api.getGraphData()
      setGraphData(data)
    } catch {
      // Graph might not be available
    }
    setLoading(false)
  }

  useEffect(() => {
    loadData()
  }, [])

  useEffect(() => {
    if (!graphData || !canvasRef.current) return
    drawGraph()
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
    }
  }, [graphData])

  const drawGraph = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !graphData) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const rect = canvas.parentElement?.getBoundingClientRect()
    if (!rect) return
    canvas.width = rect.width
    canvas.height = 400

    const width = canvas.width
    const height = canvas.height
    const cx = width / 2
    const cy = height / 2

    ctx.clearRect(0, 0, width, height)

    if (graphData.nodes.length === 0) {
      ctx.fillStyle = '#6b7280'
      ctx.font = '14px -apple-system, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('No graph data yet. Run verifications to populate the fraud graph.', cx, cy)
      return
    }

    // Layout nodes in a force-directed style (simplified)
    const nodes = graphData.nodes.map((node, i) => {
      const angle = (2 * Math.PI * i) / graphData.nodes.length
      const radius = Math.min(width, height) * 0.3
      return {
        ...node,
        x: cx + Math.cos(angle) * radius + (Math.random() - 0.5) * 20,
        y: cy + Math.sin(angle) * radius + (Math.random() - 0.5) * 20,
        vx: 0,
        vy: 0,
      }
    })

    // Simple force simulation
    for (let iter = 0; iter < 50; iter++) {
      // Repulsion between nodes
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          let dx = nodes[j].x - nodes[i].x
          let dy = nodes[j].y - nodes[i].y
          let dist = Math.sqrt(dx * dx + dy * dy) || 1
          let force = 5000 / (dist * dist)
          nodes[i].vx -= (dx / dist) * force
          nodes[i].vy -= (dy / dist) * force
          nodes[j].vx += (dx / dist) * force
          nodes[j].vy += (dy / dist) * force
        }
      }

      // Attraction along edges
      const nodeMap = new Map(nodes.map(n => [n.id, n]))
      for (const edge of graphData.edges) {
        const s = nodeMap.get(edge.source)
        const t = nodeMap.get(edge.target)
        if (s && t) {
          let dx = t.x - s.x
          let dy = t.y - s.y
          let dist = Math.sqrt(dx * dx + dy * dy) || 1
          let force = (dist - 80) * 0.01
          s.vx += (dx / dist) * force
          s.vy += (dy / dist) * force
          t.vx -= (dx / dist) * force
          t.vy -= (dy / dist) * force
        }
      }

      // Center gravity
      for (const node of nodes) {
        node.vx += (cx - node.x) * 0.005
        node.vy += (cy - node.y) * 0.005
        // Apply velocity with damping
        node.x += node.vx * 0.3
        node.y += node.vy * 0.3
        node.vx *= 0.8
        node.vy *= 0.8
        // Keep in bounds
        node.x = Math.max(30, Math.min(width - 30, node.x))
        node.y = Math.max(30, Math.min(height - 30, node.y))
      }
    }

    // Draw edges
    const nodeMap = new Map(nodes.map(n => [n.id, n]))
    for (const edge of graphData.edges) {
      const s = nodeMap.get(edge.source)
      const t = nodeMap.get(edge.target)
      if (s && t) {
        ctx.beginPath()
        ctx.moveTo(s.x, s.y)
        ctx.lineTo(t.x, t.y)
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.2)'
        ctx.lineWidth = 1
        ctx.stroke()
      }
    }

    // Draw nodes
    for (const node of nodes) {
      const radius = 12
      const color =
        node.risk_score > 0.7 ? '#ef4444' :
        node.risk_score > 0.4 ? '#f59e0b' :
        node.type === 'user' ? '#6366f1' :
        node.type === 'ip' ? '#8b5cf6' :
        node.type === 'device' ? '#06b6d4' :
        '#10b981'

      // Glow for high risk
      if (node.risk_score > 0.5) {
        ctx.beginPath()
        ctx.arc(node.x, node.y, radius + 4, 0, Math.PI * 2)
        ctx.fillStyle = `${color}33`
        ctx.fill()
      }

      ctx.beginPath()
      ctx.arc(node.x, node.y, radius, 0, Math.PI * 2)
      ctx.fillStyle = color
      ctx.fill()

      // Label
      ctx.fillStyle = '#9ca3af'
      ctx.font = '10px -apple-system, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(node.label, node.x, node.y + radius + 14)
      ctx.fillText(`(${node.type})`, node.x, node.y + radius + 26)
    }
  }, [graphData])

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-lg font-semibold text-white">Fraud Relationship Graph</h2>
            <p className="text-xs text-gray-500 mt-1">
              Visualizes connections between users, IPs, devices, and face embeddings
            </p>
          </div>
          <button onClick={loadData} className="btn-secondary text-sm">
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>

        <div className="relative bg-gray-800/30 rounded-xl overflow-hidden" style={{ minHeight: 400 }}>
          <canvas ref={canvasRef} className="w-full" style={{ height: 400 }} />
        </div>

        {/* Legend */}
        <div className="flex flex-wrap gap-4 mt-4 text-xs text-gray-400">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-indigo-500" /> User
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-purple-500" /> IP Address
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-cyan-500" /> Device
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-emerald-500" /> Face
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-red-500" /> High Risk
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-amber-500" /> Medium Risk
          </span>
        </div>
      </div>

      {/* Suspicious Clusters */}
      {graphData && graphData.suspicious_clusters.length > 0 && (
        <div className="card border-amber-500/30">
          <h3 className="flex items-center gap-2 text-sm font-semibold text-amber-400 mb-3">
            <AlertTriangle className="w-4 h-4" />
            Suspicious Clusters Detected
          </h3>
          <div className="space-y-2">
            {graphData.suspicious_clusters.map((cluster: any, i: number) => (
              <div key={i} className="p-3 bg-amber-500/10 rounded-xl border border-amber-500/20">
                <p className="text-sm text-gray-200">{cluster.description}</p>
                <p className="text-xs text-gray-400 mt-1">
                  Risk: {(cluster.risk_score * 100).toFixed(0)}%
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Graph Stats */}
      {graphData && (
        <div className="grid grid-cols-3 gap-4">
          <div className="card text-center">
            <p className="text-3xl font-bold text-brand-400">{graphData.stats.total_nodes}</p>
            <p className="text-xs text-gray-500 mt-1">Total Nodes</p>
          </div>
          <div className="card text-center">
            <p className="text-3xl font-bold text-purple-400">{graphData.stats.total_edges}</p>
            <p className="text-xs text-gray-500 mt-1">Total Edges</p>
          </div>
          <div className="card text-center">
            <p className="text-3xl font-bold text-amber-400">{graphData.suspicious_clusters.length}</p>
            <p className="text-xs text-gray-500 mt-1">Suspicious Clusters</p>
          </div>
        </div>
      )}
    </div>
  )
}
