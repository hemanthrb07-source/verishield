import { useState } from 'react'
import { Layers, Eye, EyeOff, ChevronDown, ChevronUp } from 'lucide-react'

interface HeatmapRegion {
  type: string
  x: number
  y: number
  w: number
  h: number
  confidence: number
  label: string
  color: string
}

interface HeatmapData {
  image: string
  regions: HeatmapRegion[]
  summary: {
    total_regions: number
    tampered_count: number
    text_regions_count: number
    frequency_anomalies_count: number
    reference_diffs_count: number
    overall_risk?: string
    gan_artifacts?: number
    color_anomalies?: number
    is_deepfake?: boolean
    probability?: number
  }
}

interface Props {
  heatmap: HeatmapData | null
  title?: string
}

const COLOR_MAP: Record<string, string> = {
  red: 'bg-red-500',
  orange: 'bg-orange-500',
  yellow: 'bg-amber-500',
  green: 'bg-emerald-500',
  purple: 'bg-purple-500',
  cyan: 'bg-cyan-500',
  blue: 'bg-blue-500',
}

const TYPE_LABELS: Record<string, string> = {
  tampering: 'Tampered Region',
  font_inconsistency: 'Font Inconsistency',
  spacing_anomaly: 'Spacing Anomaly',
  text_region: 'Detected Text',
  frequency_anomaly: 'Frequency Anomaly',
  reference_diff: 'Differs from Reference',
  gan_artifact: 'GAN Artifact',
  color_anomaly: 'Color Anomaly',
}

export function HeatmapDisplay({ heatmap, title = 'AI Analysis Heatmap' }: Props) {
  const [showOverlay, setShowOverlay] = useState(true)
  const [selectedRegion, setSelectedRegion] = useState<number | null>(null)
  const [expanded, setExpanded] = useState(false)

  if (!heatmap || !heatmap.image) {
    return (
      <div className="card">
        <h3 className="text-sm font-semibold text-gray-300 mb-2">{title}</h3>
        <p className="text-xs text-gray-500">No heatmap data available</p>
      </div>
    )
  }

  const regionsByType = heatmap.regions.reduce((acc, r) => {
    const type = r.type
    if (!acc[type]) acc[type] = []
    acc[type].push(r)
    return acc
  }, {} as Record<string, HeatmapRegion[]>)

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-brand-400" />
          <h3 className="text-sm font-semibold text-gray-300">{title}</h3>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowOverlay(!showOverlay)}
            className={`flex items-center gap-1 px-2 py-1 rounded text-xs transition-all ${
              showOverlay ? 'bg-brand-600/20 text-brand-400' : 'bg-gray-800 text-gray-500'
            }`}
          >
            {showOverlay ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
            {showOverlay ? 'Overlay On' : 'Overlay Off'}
          </button>
        </div>
      </div>

      {/* Heatmap Image */}
      <div className="relative rounded-xl overflow-hidden bg-gray-800/50">
        <img
          src={`data:image/png;base64,${heatmap.image}`}
          alt="Analysis Heatmap"
          className="w-full h-auto"
          style={{ imageRendering: 'auto' }}
        />

        {/* Region count badge */}
        <div className="absolute top-2 right-2 px-2 py-1 rounded-lg bg-black/60 backdrop-blur-sm text-xs text-white">
          {heatmap.regions.length} regions detected
        </div>
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-2">
        {Object.entries(regionsByType).map(([type, regions]) => (
          <button
            key={type}
            onClick={() => setSelectedRegion(selectedRegion === parseInt(type) ? null : regions[0] ? heatmap.regions.indexOf(regions[0]) : null)}
            className={`flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs transition-all ${
              selectedRegion !== null && regions.some((_, i) => heatmap.regions.indexOf(regions[i]) === selectedRegion)
                ? 'bg-white/10 border border-white/20'
                : 'bg-gray-800/50 border border-gray-700/50 hover:bg-gray-700/50'
            }`}
          >
            <div className={`w-2 h-2 rounded-full ${COLOR_MAP[regions[0]?.color] || 'bg-gray-500'}`} />
            <span className="text-gray-300">{TYPE_LABELS[type] || type}</span>
            <span className="text-gray-500">({regions.length})</span>
          </button>
        ))}
      </div>

      {/* Region Details */}
      {selectedRegion !== null && heatmap.regions[selectedRegion] && (
        <div className="mt-3 p-3 bg-gray-800/50 rounded-xl animate-slide-up">
          <div className="flex items-center gap-2 mb-2">
            <div className={`w-3 h-3 rounded-full ${COLOR_MAP[heatmap.regions[selectedRegion].color] || 'bg-gray-500'}`} />
            <span className="text-sm font-medium text-white">
              {TYPE_LABELS[heatmap.regions[selectedRegion].type] || heatmap.regions[selectedRegion].type}
            </span>
          </div>
          <p className="text-xs text-gray-400">{heatmap.regions[selectedRegion].label}</p>
          <div className="flex gap-4 mt-2 text-xs text-gray-500">
            <span>Position: ({heatmap.regions[selectedRegion].x}, {heatmap.regions[selectedRegion].y})</span>
            <span>Size: {heatmap.regions[selectedRegion].w} x {heatmap.regions[selectedRegion].h}</span>
            <span>Confidence: {(heatmap.regions[selectedRegion].confidence * 100).toFixed(0)}%</span>
          </div>
        </div>
      )}

      {/* Summary */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full mt-3 flex items-center justify-between text-xs text-gray-500 hover:text-gray-300"
      >
        <span>Analysis Summary</span>
        {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
      </button>

      {expanded && (
        <div className="mt-2 grid grid-cols-3 gap-2">
          {heatmap.summary.tampered_count > 0 && (
            <div className="p-2 bg-red-500/10 rounded-lg text-center">
              <p className="text-lg font-bold text-red-400">{heatmap.summary.tampered_count}</p>
              <p className="text-xs text-gray-500">Tampered</p>
            </div>
          )}
          {heatmap.summary.gan_artifacts !== undefined && heatmap.summary.gan_artifacts > 0 && (
            <div className="p-2 bg-purple-500/10 rounded-lg text-center">
              <p className="text-lg font-bold text-purple-400">{heatmap.summary.gan_artifacts}</p>
              <p className="text-xs text-gray-500">GAN Artifacts</p>
            </div>
          )}
          {heatmap.summary.text_regions_count > 0 && (
            <div className="p-2 bg-emerald-500/10 rounded-lg text-center">
              <p className="text-lg font-bold text-emerald-400">{heatmap.summary.text_regions_count}</p>
              <p className="text-xs text-gray-500">Text Regions</p>
            </div>
          )}
          {heatmap.summary.frequency_anomalies_count > 0 && (
            <div className="p-2 bg-purple-500/10 rounded-lg text-center">
              <p className="text-lg font-bold text-purple-400">{heatmap.summary.frequency_anomalies_count}</p>
              <p className="text-xs text-gray-500">Freq Anomalies</p>
            </div>
          )}
          {heatmap.summary.reference_diffs_count > 0 && (
            <div className="p-2 bg-cyan-500/10 rounded-lg text-center">
              <p className="text-lg font-bold text-cyan-400">{heatmap.summary.reference_diffs_count}</p>
              <p className="text-xs text-gray-500">Ref Diffs</p>
            </div>
          )}
          {heatmap.summary.overall_risk && (
            <div className={`p-2 rounded-lg text-center ${
              heatmap.summary.overall_risk === 'HIGH' ? 'bg-red-500/10' :
              heatmap.summary.overall_risk === 'MEDIUM' ? 'bg-amber-500/10' : 'bg-emerald-500/10'
            }`}>
              <p className={`text-lg font-bold ${
                heatmap.summary.overall_risk === 'HIGH' ? 'text-red-400' :
                heatmap.summary.overall_risk === 'MEDIUM' ? 'text-amber-400' : 'text-emerald-400'
              }`}>{heatmap.summary.overall_risk}</p>
              <p className="text-xs text-gray-500">Overall Risk</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
