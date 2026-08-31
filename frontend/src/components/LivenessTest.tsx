import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { api } from '../services/api'
import {
  Eye, EyeOff, Upload, Loader2, Zap, AlertTriangle,
  CheckCircle, ChevronDown, ChevronUp, Camera, Box,
  Layers, Scan
} from 'lucide-react'

interface LivenessResult {
  is_live: boolean
  liveness_score: number
  confidence: number
  head_pose: {
    yaw?: number
    pitch?: number
    roll?: number
    within_valid_range?: boolean
    natural_motion?: boolean
    avg_yaw?: number
    avg_pitch?: number
    avg_roll?: number
  }
  depth_analysis: {
    mean_depth: number
    depth_variance: number
    depth_range: number
    is_flat: boolean
    has_3d_structure: boolean
    confidence: number
    temporal_depth_variance?: number
  }
  texture_analysis: {
    moire_pattern: number
    screen_reflection: number
    screen_edge: number
    print_artifact: number
  }
  spoof_type: string | null
  frame_count: number
  temporal_consistency?: {
    consistent: boolean
    inconsistency_type: string
    issues: string[]
    frames_analyzed: number
  }
}

interface FullResult {
  verification_id: string
  status: string
  trust_score: number
  risk_level: string
  confidence: number
  reasons: string[]
  detailed_results: {
    liveness_analysis: LivenessResult
    risk_assessment: any
  }
  processing_time_ms: number
}

export function LivenessTest() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [testing, setTesting] = useState(false)
  const [result, setResult] = useState<FullResult | null>(null)
  const [expanded, setExpanded] = useState<string | null>(null)

  const onDrop = useCallback((accepted: File[]) => {
    if (accepted.length > 0) setSelectedFile(accepted[0])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg'],
      'video/*': ['.mp4', '.avi', '.mov'],
    },
    maxFiles: 1,
  })

  const runTest = async () => {
    if (!selectedFile) return
    setTesting(true)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('user_id', 'liveness_test')
      const res = await api.verifyLiveness(selectedFile)
      setResult(res)
    } catch (err: any) {
      console.error('Liveness test failed:', err)
    }
    setTesting(false)
  }

  const toggle = (section: string) => setExpanded(expanded === section ? null : section)

  const liveness = result?.detailed_results?.liveness_analysis

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
            <Scan className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">Liveness Detection</h2>
            <p className="text-xs text-gray-500">Anti-spoofing via head-pose, depth, and texture analysis</p>
          </div>
        </div>

        {/* Attack types info */}
        <div className="grid grid-cols-4 gap-2 mb-6">
          {[
            { icon: <Camera className="w-4 h-4" />, label: 'Photo Replay', color: 'text-amber-400' },
            { icon: <Layers className="w-4 h-4" />, label: 'Screen Replay', color: 'text-red-400' },
            { icon: <Box className="w-4 h-4" />, label: 'Printed Photo', color: 'text-purple-400' },
            { icon: <Eye className="w-4 h-4" />, label: 'Video Replay', color: 'text-blue-400' },
          ].map(({ icon, label, color }, i) => (
            <div key={i} className="flex items-center gap-2 p-2 bg-gray-800/50 rounded-lg">
              <span className={color}>{icon}</span>
              <span className="text-xs text-gray-400">{label}</span>
            </div>
          ))}
        </div>

        {/* Upload */}
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all
            ${isDragActive ? 'border-cyan-500 bg-cyan-500/10' :
              selectedFile ? 'border-emerald-500/50 bg-emerald-500/5' :
              'border-gray-700 hover:border-gray-600 bg-gray-800/30'}`}
        >
          <input {...getInputProps()} />
          {selectedFile ? (
            <div className="flex items-center justify-center gap-3">
              {selectedFile.type.startsWith('video/') ?
                <Layers className="w-6 h-6 text-purple-400" /> :
                <Eye className="w-6 h-6 text-cyan-400" />}
              <div>
                <p className="text-sm font-medium text-white">{selectedFile.name}</p>
                <p className="text-xs text-gray-500">
                  {(selectedFile.size / 1024).toFixed(0)} KB
                  {selectedFile.type.startsWith('video/') && ' (video — frame analysis)'}
                </p>
              </div>
            </div>
          ) : (
            <>
              <Eye className="w-12 h-12 text-gray-500 mx-auto mb-3" />
              <p className="text-sm text-gray-300 mb-1">Upload image or video for liveness check</p>
              <p className="text-xs text-gray-500">Supports PNG, JPEG, MP4, AVI</p>
            </>
          )}
        </div>

        <button
          onClick={runTest}
          disabled={!selectedFile || testing}
          className="btn-primary w-full mt-4 disabled:opacity-50"
        >
          {testing ? (
            <><Loader2 className="w-5 h-5 mr-2 animate-spin" /> Analyzing liveness...</>
          ) : (
            <><Zap className="w-5 h-5 mr-2" /> Run Liveness Detection</>
          )}
        </button>
      </div>

      {/* Main Result */}
      {liveness && (
        <div className={`card bg-gradient-to-b ${
          liveness.is_live ? 'from-emerald-500/20' : 'from-red-500/20'
        }`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className={`w-16 h-16 rounded-2xl flex items-center justify-center ${
                liveness.is_live ? 'bg-emerald-500/20' : 'bg-red-500/20'
              }`}>
                {liveness.is_live ?
                  <Eye className="w-8 h-8 text-emerald-400" /> :
                  <EyeOff className="w-8 h-8 text-red-400" />}
              </div>
              <div>
                <h3 className={`text-2xl font-bold ${liveness.is_live ? 'text-emerald-400' : 'text-red-400'}`}>
                  {liveness.is_live ? 'LIVE' : 'SPOOF DETECTED'}
                </h3>
                <p className="text-sm text-gray-400">
                  {liveness.is_live ? 'Subject appears to be a live person' : `Attack type: ${liveness.spoof_type || 'unknown'}`}
                </p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-3xl font-bold text-white">
                {(liveness.liveness_score * 100).toFixed(0)}
              </p>
              <p className="text-xs text-gray-500">/100 liveness</p>
              <p className="text-xs text-gray-500 mt-1">
                {(liveness.confidence * 100).toFixed(0)}% confidence
              </p>
            </div>
          </div>

          {/* Liveness score bar */}
          <div className="mt-4 h-3 bg-gray-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-1000 ${
                liveness.liveness_score >= 0.7 ? 'bg-emerald-500' :
                liveness.liveness_score >= 0.4 ? 'bg-amber-500' : 'bg-red-500'
              }`}
              style={{ width: `${liveness.liveness_score * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Head Pose Analysis */}
      {liveness && liveness.head_pose && (
        <div className="card cursor-pointer" onClick={() => toggle('pose')}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Camera className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-semibold text-gray-300">Head Pose Estimation</h3>
              {liveness.head_pose.within_valid_range !== undefined && (
                <span className={`badge text-xs ${liveness.head_pose.within_valid_range ? 'badge-green' : 'badge-red'}`}>
                  {liveness.head_pose.within_valid_range ? 'Valid' : 'Abnormal'}
                </span>
              )}
            </div>
            {expanded === 'pose' ? <ChevronUp className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
          </div>

          {expanded === 'pose' && (
            <div className="mt-4 space-y-4">
              {/* 3D Pose Visualization */}
              <div className="grid grid-cols-3 gap-4">
                {[
                  { label: 'Yaw (Left-Right)', value: liveness.head_pose.yaw ?? liveness.head_pose.avg_yaw, range: [-45, 45], color: 'text-cyan-400' },
                  { label: 'Pitch (Up-Down)', value: liveness.head_pose.pitch ?? liveness.head_pose.avg_pitch, range: [-30, 30], color: 'text-blue-400' },
                  { label: 'Roll (Tilt)', value: liveness.head_pose.roll ?? liveness.head_pose.avg_roll, range: [-20, 20], color: 'text-purple-400' },
                ].map(({ label, value, range, color }, i) => {
                  const normalized = value != null ? ((value - range[0]) / (range[1] - range[0])) * 100 : 50
                  const inRange = value != null && value >= range[0] && value <= range[1]
                  return (
                    <div key={i} className="p-3 bg-gray-800/50 rounded-xl">
                      <p className="text-xs text-gray-500 mb-2">{label}</p>
                      <div className="relative h-2 bg-gray-700 rounded-full mb-2">
                        {/* Valid range indicator */}
                        <div className="absolute inset-y-0 left-[10%] right-[10%] bg-gray-600/30 rounded-full" />
                        {/* Current position */}
                        <div
                          className={`absolute top-0 w-3 h-2 rounded-full ${inRange ? 'bg-emerald-400' : 'bg-red-400'}`}
                          style={{ left: `${Math.max(0, Math.min(95, normalized - 2))}%` }}
                        />
                      </div>
                      <p className={`text-lg font-bold text-center ${color}`}>
                        {value != null ? `${value.toFixed(1)}°` : '--'}
                      </p>
                    </div>
                  )
                })}
              </div>

              {liveness.head_pose.natural_motion !== undefined && (
                <p className={`text-xs ${liveness.head_pose.natural_motion ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {liveness.head_pose.natural_motion ? 'Natural head motion detected' : 'Head motion is unnaturally static'}
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {/* Depth Analysis */}
      {liveness && liveness.depth_analysis && (
        <div className="card cursor-pointer" onClick={() => toggle('depth')}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Box className="w-4 h-4 text-blue-400" />
              <h3 className="text-sm font-semibold text-gray-300">Depth Analysis</h3>
              <span className={`badge text-xs ${liveness.depth_analysis.has_3d_structure ? 'badge-green' : liveness.depth_analysis.is_flat ? 'badge-red' : 'badge-yellow'}`}>
                {liveness.depth_analysis.has_3d_structure ? '3D' : liveness.depth_analysis.is_flat ? 'Flat' : 'Uncertain'}
              </span>
            </div>
            {expanded === 'depth' ? <ChevronUp className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
          </div>

          {expanded === 'depth' && (
            <div className="mt-4 space-y-3">
              <div className="grid grid-cols-3 gap-3">
                <div className="p-3 bg-gray-800/50 rounded-xl text-center">
                  <p className="text-xs text-gray-500">Mean Depth</p>
                  <p className="text-lg font-bold text-blue-400">{liveness.depth_analysis.mean_depth.toFixed(3)}</p>
                </div>
                <div className="p-3 bg-gray-800/50 rounded-xl text-center">
                  <p className="text-xs text-gray-500">Variance</p>
                  <p className={`text-lg font-bold ${liveness.depth_analysis.depth_variance > 0.05 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {liveness.depth_analysis.depth_variance.toFixed(4)}
                  </p>
                </div>
                <div className="p-3 bg-gray-800/50 rounded-xl text-center">
                  <p className="text-xs text-gray-500">Depth Range</p>
                  <p className="text-lg font-bold text-purple-400">{liveness.depth_analysis.depth_range.toFixed(3)}</p>
                </div>
              </div>

              {/* Depth visualization bar */}
              <div className="p-3 bg-gray-800/50 rounded-xl">
                <p className="text-xs text-gray-500 mb-2">3D Structure Assessment</p>
                <div className="h-4 bg-gray-700 rounded-full overflow-hidden relative">
                  <div
                    className={`h-full rounded-full ${
                      liveness.depth_analysis.has_3d_structure ? 'bg-emerald-500' :
                      liveness.depth_analysis.is_flat ? 'bg-red-500' : 'bg-amber-500'
                    }`}
                    style={{ width: `${Math.min(liveness.depth_analysis.depth_variance * 1000, 100)}%` }}
                  />
                  {/* Threshold line */}
                  <div className="absolute top-0 bottom-0 left-[50%] w-px bg-white/20" />
                </div>
                <div className="flex justify-between mt-1 text-xs text-gray-500">
                  <span>Flat (2D)</span>
                  <span>3D Structure</span>
                </div>
              </div>

              {liveness.depth_analysis.temporal_depth_variance !== undefined && (
                <p className="text-xs text-gray-400">
                  Temporal depth variance: {liveness.depth_analysis.temporal_depth_variance.toFixed(6)}
                </p>
              )}

              <p className="text-xs text-gray-400">
                {liveness.depth_analysis.is_flat
                  ? 'Depth profile is too flat — consistent with a 2D photo or screen'
                  : liveness.depth_analysis.has_3d_structure
                    ? 'Depth profile shows natural 3D facial structure'
                    : 'Depth profile is inconclusive'}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Texture Analysis */}
      {liveness && liveness.texture_analysis && (
        <div className="card cursor-pointer" onClick={() => toggle('texture')}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-amber-400" />
              <h3 className="text-sm font-semibold text-gray-300">Texture Analysis</h3>
            </div>
            {expanded === 'texture' ? <ChevronUp className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
          </div>

          {expanded === 'texture' && (
            <div className="mt-4 space-y-3">
              {[
                { label: 'Moiré Pattern', key: 'moire_pattern', desc: 'Screen pixel interference patterns', icon: '🔍' },
                { label: 'Screen Reflection', key: 'screen_reflection', desc: 'Glossy screen surface reflections', icon: '💡' },
                { label: 'Screen Edge', key: 'screen_edge', desc: 'Device bezel/edge detection', icon: '📱' },
                { label: 'Print Artifact', key: 'print_artifact', desc: 'Printed paper texture artifacts', icon: '🖨️' },
              ].map(({ label, key, desc, icon }, i) => {
                const score = liveness.texture_analysis[key as keyof typeof liveness.texture_analysis] || 0
                return (
                  <div key={i} className="p-3 bg-gray-800/50 rounded-xl">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span>{icon}</span>
                        <span className="text-sm text-gray-300">{label}</span>
                      </div>
                      <span className={`text-sm font-bold ${score > 0.6 ? 'text-red-400' : score > 0.3 ? 'text-amber-400' : 'text-emerald-400'}`}>
                        {(score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          score > 0.6 ? 'bg-red-500' : score > 0.3 ? 'bg-amber-500' : 'bg-emerald-500'
                        }`}
                        style={{ width: `${score * 100}%` }}
                      />
                    </div>
                    <p className="text-xs text-gray-500 mt-1">{desc}</p>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}

      {/* Temporal Consistency (video only) */}
      {liveness?.temporal_consistency && (
        <div className="card cursor-pointer" onClick={() => toggle('temporal')}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Eye className="w-4 h-4 text-purple-400" />
              <h3 className="text-sm font-semibold text-gray-300">Temporal Consistency</h3>
              <span className={`badge text-xs ${liveness.temporal_consistency.consistent ? 'badge-green' : 'badge-red'}`}>
                {liveness.temporal_consistency.consistent ? 'Consistent' : 'Inconsistent'}
              </span>
            </div>
            {expanded === 'temporal' ? <ChevronUp className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
          </div>

          {expanded === 'temporal' && (
            <div className="mt-4 space-y-3">
              <p className="text-xs text-gray-400">
                Analyzed {liveness.temporal_consistency.frames_analyzed} frames
              </p>
              {liveness.temporal_consistency.issues.length > 0 ? (
                <div className="space-y-2">
                  {liveness.temporal_consistency.issues.map((issue, i) => (
                    <div key={i} className="p-2 bg-amber-500/10 rounded-lg border border-amber-500/20">
                      <p className="text-xs text-amber-400">{issue.replace(/_/g, ' ')}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-emerald-400">All temporal checks passed</p>
              )}
            </div>
          )}
        </div>
      )}

      {/* Frame Analysis (video) */}
      {liveness?.frame_analysis && liveness.frame_analysis.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Frame-by-Frame Analysis</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-2 px-3 text-gray-500">Frame</th>
                  <th className="text-left py-2 px-3 text-gray-500">Live</th>
                  <th className="text-left py-2 px-3 text-gray-500">Score</th>
                  <th className="text-left py-2 px-3 text-gray-500">Yaw</th>
                  <th className="text-left py-2 px-3 text-gray-500">Pitch</th>
                  <th className="text-left py-2 px-3 text-gray-500">Roll</th>
                </tr>
              </thead>
              <tbody>
                {liveness.frame_analysis.map((f, i) => (
                  <tr key={i} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                    <td className="py-2 px-3 text-gray-400">#{f.frame_index}</td>
                    <td className="py-2 px-3">
                      {f.is_live ?
                        <CheckCircle className="w-3 h-3 text-emerald-400 inline" /> :
                        <AlertTriangle className="w-3 h-3 text-red-400 inline" />}
                    </td>
                    <td className="py-2 px-3">
                      <span className={`font-bold ${f.liveness_score > 0.7 ? 'text-emerald-400' : f.liveness_score > 0.4 ? 'text-amber-400' : 'text-red-400'}`}>
                        {(f.liveness_score * 100).toFixed(0)}
                      </span>
                    </td>
                    <td className="py-2 px-3 font-mono text-gray-400">{f.yaw?.toFixed(1)}°</td>
                    <td className="py-2 px-3 font-mono text-gray-400">{f.pitch?.toFixed(1)}°</td>
                    <td className="py-2 px-3 font-mono text-gray-400">{f.roll?.toFixed(1)}°</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Reasons */}
      {result && result.reasons.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Detection Reasons</h3>
          <ul className="space-y-2">
            {result.reasons.map((reason, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-300">
                {liveness?.is_live ?
                  <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" /> :
                  <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />}
                {reason}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Processing info */}
      {result && (
        <div className="card bg-gray-900/50 border-gray-800">
          <div className="flex items-center justify-between text-xs text-gray-500">
            <span>Processing time: {result.processing_time_ms}ms</span>
            <span className="flex items-center gap-1">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              Blockchain logged
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
