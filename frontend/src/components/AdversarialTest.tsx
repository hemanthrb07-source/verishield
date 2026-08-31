import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { api } from '../services/api'
import {
  Shield, ShieldAlert, ShieldCheck, Upload,
  Loader2, Zap, AlertTriangle, CheckCircle,
  ChevronDown, ChevronUp, Target, Layers
} from 'lucide-react'

interface AttackResult {
  method: string
  epsilon?: number
  success_rate?: number
  prediction_changed?: boolean
  original_confidence?: number
  adversarial_confidence?: number
  l2_perturbation?: number
  linf_perturbation?: number
  stability_rate?: number
  std?: number
}

interface RobustnessReport {
  overall_robustness_score: number
  fgsm_results: AttackResult[]
  pgd_results: AttackResult[]
  noise_results: AttackResult[]
  spatial_results: AttackResult & { num_transforms?: number }
  brightness_results: AttackResult & { num_variations?: number }
  compression_results: AttackResult & { num_qualities?: number }
  vulnerabilities: { type: string; severity: string; description: string; attack: string; epsilon?: number }[]
  recommendations: string[]
  perturbation_visualizations: {
    epsilon: number
    perturbation_magnitude: number
    original_confidence: number
    adversarial_confidence: number
    prediction_changed: boolean
    l2_norm: number
    linf_norm: number
  }[]
}

interface ModelComparison {
  model: string
  robustness_score: number | null
  vulnerability_count?: number
  fgsm_success_rate?: number
  pgd_success_rate?: number
  error?: string
  note?: string
}

export function AdversarialTest() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [testing, setTesting] = useState(false)
  const [report, setReport] = useState<RobustnessReport | null>(null)
  const [comparison, setComparison] = useState<ModelComparison[] | null>(null)
  const [mode, setMode] = useState<'robustness' | 'compare'>('robustness')
  const [expanded, setExpanded] = useState<string | null>(null)

  const onDrop = useCallback((accepted: File[]) => {
    if (accepted.length > 0) setSelectedFile(accepted[0])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg'] },
    maxFiles: 1,
  })

  const runTest = async () => {
    if (!selectedFile) return
    setTesting(true)
    setReport(null)
    setComparison(null)

    try {
      if (mode === 'robustness') {
        const formData = new FormData()
        formData.append('file', selectedFile)
        formData.append('test_all', 'true')
        const result = await api.request('/test/adversarial', { method: 'POST', body: formData })
        setReport(result)
      } else {
        const formData = new FormData()
        formData.append('file', selectedFile)
        formData.append('epsilon', '0.03')
        const result = await api.request('/test/adversarial/compare', { method: 'POST', body: formData })
        setComparison(result.comparisons)
      }
    } catch (err: any) {
      console.error('Test failed:', err)
    }
    setTesting(false)
  }

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-emerald-400'
    if (score >= 60) return 'text-amber-400'
    if (score >= 40) return 'text-orange-400'
    return 'text-red-400'
  }

  const getScoreBg = (score: number) => {
    if (score >= 80) return 'from-emerald-500/20'
    if (score >= 60) return 'from-amber-500/20'
    if (score >= 40) return 'from-orange-500/20'
    return 'from-red-500/20'
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-red-500 to-orange-600 flex items-center justify-center">
              <Target className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">Adversarial Robustness Testing</h2>
              <p className="text-xs text-gray-500">Evaluate model resilience against perturbation attacks</p>
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setMode('robustness')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${mode === 'robustness'
                ? 'bg-brand-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}
            >
              <Target className="w-4 h-4 mr-1 inline" />
              Robustness
            </button>
            <button
              onClick={() => setMode('compare')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${mode === 'compare'
                ? 'bg-brand-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}
            >
              <Layers className="w-4 h-4 mr-1 inline" />
              Compare Models
            </button>
          </div>
        </div>

        {/* Upload */}
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all
            ${isDragActive ? 'border-brand-500 bg-brand-500/10' :
              selectedFile ? 'border-emerald-500/50 bg-emerald-500/5' :
              'border-gray-700 hover:border-gray-600 bg-gray-800/30'}`}
        >
          <input {...getInputProps()} />
          {selectedFile ? (
            <div className="flex items-center justify-center gap-3">
              <Shield className="w-5 h-5 text-emerald-400" />
              <span className="text-sm text-white">{selectedFile.name}</span>
              <span className="text-xs text-gray-500">({(selectedFile.size / 1024).toFixed(0)} KB)</span>
            </div>
          ) : (
            <>
              <Upload className="w-8 h-8 text-gray-500 mx-auto mb-2" />
              <p className="text-sm text-gray-400">Drop an image to test model robustness</p>
            </>
          )}
        </div>

        <button
          onClick={runTest}
          disabled={!selectedFile || testing}
          className="btn-primary w-full mt-4 disabled:opacity-50"
        >
          {testing ? (
            <><Loader2 className="w-5 h-5 mr-2 animate-spin" /> Running attack simulations...</>
          ) : (
            <><Zap className="w-5 h-5 mr-2" /> Run {mode === 'robustness' ? 'Robustness Tests' : 'Model Comparison'}</>
          )}
        </button>
      </div>

      {/* Overall Robustness Score */}
      {report && (
        <div className={`card bg-gradient-to-b ${getScoreBg(report.overall_robustness_score)}`}>
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-gray-300 mb-1">Overall Robustness Score</h3>
              <p className="text-xs text-gray-500">Across all tested attack vectors</p>
            </div>
            <div className="text-right">
              <span className={`text-5xl font-bold ${getScoreColor(report.overall_robustness_score)}`}>
                {report.overall_robustness_score.toFixed(0)}
              </span>
              <span className="text-lg text-gray-500">/100</span>
            </div>
          </div>

          {/* Score bar */}
          <div className="mt-4 h-3 bg-gray-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-1000
                ${report.overall_robustness_score >= 80 ? 'bg-emerald-500' :
                  report.overall_robustness_score >= 60 ? 'bg-amber-500' :
                  report.overall_robustness_score >= 40 ? 'bg-orange-500' : 'bg-red-500'}`}
              style={{ width: `${report.overall_robustness_score}%` }}
            />
          </div>

          <p className="text-xs text-gray-500 mt-2">
            {report.overall_robustness_score >= 80 ? 'Excellent resilience — model is well-defended' :
             report.overall_robustness_score >= 60 ? 'Moderate resilience — some attack vectors succeed' :
             report.overall_robustness_score >= 40 ? 'Weak resilience — significant vulnerabilities found' :
             'Critical vulnerabilities — model is highly susceptible to attacks'}
          </p>
        </div>
      )}

      {/* FGSM Results */}
      {report && report.fgsm_results.length > 0 && (
        <AttackSection
          title="FGSM (Fast Gradient Sign Method)"
          icon={<Zap className="w-4 h-4 text-amber-400" />}
          isExpanded={expanded === 'fgsm'}
          onToggle={() => setExpanded(expanded === 'fgsm' ? null : 'fgsm')}
        >
          <p className="text-xs text-gray-400 mb-3">
            Single-step gradient attack. Measures how small pixel perturbations affect predictions.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-2 px-3 text-gray-500">Epsilon</th>
                  <th className="text-left py-2 px-3 text-gray-500">Success Rate</th>
                  <th className="text-left py-2 px-3 text-gray-500">Pred Changed</th>
                  <th className="text-left py-2 px-3 text-gray-500">L2 Norm</th>
                  <th className="text-left py-2 px-3 text-gray-500">Linf Norm</th>
                </tr>
              </thead>
              <tbody>
                {report.fgsm_results.map((r, i) => (
                  <tr key={i} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                    <td className="py-2 px-3 font-mono text-gray-300">{r.epsilon?.toFixed(3)}</td>
                    <td className="py-2 px-3">
                      <span className={`font-bold ${r.success_rate && r.success_rate > 0.5 ? 'text-red-400' :
                        r.success_rate && r.success_rate > 0.2 ? 'text-amber-400' : 'text-emerald-400'}`}>
                        {r.success_rate != null ? `${(r.success_rate * 100).toFixed(0)}%` : '--'}
                      </span>
                    </td>
                    <td className="py-2 px-3">
                      {r.prediction_changed ?
                        <span className="text-red-400 flex items-center gap-1"><AlertTriangle className="w-3 h-3" /> Yes</span> :
                        <span className="text-emerald-400 flex items-center gap-1"><CheckCircle className="w-3 h-3" /> No</span>
                      }
                    </td>
                    <td className="py-2 px-3 font-mono text-gray-400">{r.l2_perturbation?.toFixed(4)}</td>
                    <td className="py-2 px-3 font-mono text-gray-400">{r.linf_perturbation?.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </AttackSection>
      )}

      {/* PGD Results */}
      {report && report.pgd_results.length > 0 && (
        <AttackSection
          title="PGD (Projected Gradient Descent)"
          icon={<ShieldAlert className="w-4 h-4 text-red-400" />}
          isExpanded={expanded === 'pgd'}
          onToggle={() => setExpanded(expanded === 'pgd' ? null : 'pgd')}
        >
          <p className="text-xs text-gray-400 mb-3">
            Iterative multi-step attack — the strongest first-order adversarial attack.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-2 px-3 text-gray-500">Epsilon</th>
                  <th className="text-left py-2 px-3 text-gray-500">Steps</th>
                  <th className="text-left py-2 px-3 text-gray-500">Success Rate</th>
                  <th className="text-left py-2 px-3 text-gray-500">Confidence Drop</th>
                </tr>
              </thead>
              <tbody>
                {report.pgd_results.map((r, i) => {
                  const confDrop = (r.original_confidence ?? 0) - (r.adversarial_confidence ?? 0)
                  return (
                    <tr key={i} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                      <td className="py-2 px-3 font-mono text-gray-300">{r.epsilon?.toFixed(3)}</td>
                      <td className="py-2 px-3 text-gray-400">10</td>
                      <td className="py-2 px-3">
                        <span className={`font-bold ${r.success_rate && r.success_rate > 0.5 ? 'text-red-400' :
                          r.success_rate && r.success_rate > 0.2 ? 'text-amber-400' : 'text-emerald-400'}`}>
                          {r.success_rate != null ? `${(r.success_rate * 100).toFixed(0)}%` : '--'}
                        </span>
                      </td>
                      <td className="py-2 px-3 font-mono text-gray-400">
                        {confDrop > 0 ? `-${(confDrop * 100).toFixed(1)}%` : '--'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </AttackSection>
      )}

      {/* Noise, Spatial, Brightness, Compression */}
      {report && (
        <div className="grid grid-cols-2 gap-4">
          {report.noise_results.length > 0 && (
            <div className="card">
              <h4 className="text-xs font-semibold text-gray-300 mb-3">Gaussian Noise</h4>
              <div className="space-y-2">
                {report.noise_results.map((r, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <span className="text-xs text-gray-500 w-16">std={r.std}</span>
                    <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${r.stability_rate && r.stability_rate >= 0.8 ? 'bg-emerald-500' : r.stability_rate && r.stability_rate >= 0.6 ? 'bg-amber-500' : 'bg-red-500'}`}
                        style={{ width: `${(r.stability_rate ?? 0) * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-400 w-10 text-right">
                      {r.stability_rate != null ? `${(r.stability_rate * 100).toFixed(0)}%` : '--'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {report.spatial_results && (
            <div className="card">
              <h4 className="text-xs font-semibold text-gray-300 mb-3">Spatial Transforms</h4>
              <div className="text-center py-2">
                <span className={`text-3xl font-bold ${
                  report.spatial_results.stability_rate >= 0.8 ? 'text-emerald-400' :
                  report.spatial_results.stability_rate >= 0.6 ? 'text-amber-400' : 'text-red-400'
                }`}>
                  {report.spatial_results.stability_rate != null ? `${(report.spatial_results.stability_rate * 100).toFixed(0)}%` : '--'}
                </span>
                <p className="text-xs text-gray-500 mt-1">stability ({report.spatial_results.num_transforms} transforms)</p>
              </div>
            </div>
          )}

          {report.brightness_results && (
            <div className="card">
              <h4 className="text-xs font-semibold text-gray-300 mb-3">Brightness/Contrast</h4>
              <div className="text-center py-2">
                <span className={`text-3xl font-bold ${
                  report.brightness_results.stability_rate >= 0.8 ? 'text-emerald-400' :
                  report.brightness_results.stability_rate >= 0.6 ? 'text-amber-400' : 'text-red-400'
                }`}>
                  {report.brightness_results.stability_rate != null ? `${(report.brightness_results.stability_rate * 100).toFixed(0)}%` : '--'}
                </span>
                <p className="text-xs text-gray-500 mt-1">stability ({report.brightness_results.num_variations} variations)</p>
              </div>
            </div>
          )}

          {report.compression_results && (
            <div className="card">
              <h4 className="text-xs font-semibold text-gray-300 mb-3">JPEG Compression</h4>
              <div className="text-center py-2">
                <span className={`text-3xl font-bold ${
                  report.compression_results.stability_rate >= 0.8 ? 'text-emerald-400' :
                  report.compression_results.stability_rate >= 0.6 ? 'text-amber-400' : 'text-red-400'
                }`}>
                  {report.compression_results.stability_rate != null ? `${(report.compression_results.stability_rate * 100).toFixed(0)}%` : '--'}
                </span>
                <p className="text-xs text-gray-500 mt-1">stability ({report.compression_results.num_qualities} qualities)</p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Vulnerabilities */}
      {report && report.vulnerabilities.length > 0 && (
        <div className="card border-red-500/30">
          <h3 className="flex items-center gap-2 text-sm font-semibold text-red-400 mb-3">
            <AlertTriangle className="w-4 h-4" />
            Vulnerabilities Detected ({report.vulnerabilities.length})
          </h3>
          <div className="space-y-2">
            {report.vulnerabilities.map((v, i) => (
              <div key={i} className={`p-3 rounded-xl border ${
                v.severity === 'critical' ? 'bg-red-500/10 border-red-500/20' :
                v.severity === 'high' ? 'bg-orange-500/10 border-orange-500/20' :
                'bg-amber-500/10 border-amber-500/20'
              }`}>
                <div className="flex items-center justify-between">
                  <p className="text-sm text-gray-200">{v.description}</p>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                    v.severity === 'critical' ? 'bg-red-500/30 text-red-300' :
                    v.severity === 'high' ? 'bg-orange-500/30 text-orange-300' :
                    'bg-amber-500/30 text-amber-300'
                  }`}>{v.severity}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      {report && report.recommendations.length > 0 && (
        <div className="card">
          <h3 className="flex items-center gap-2 text-sm font-semibold text-gray-300 mb-3">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            Recommendations
          </h3>
          <ul className="space-y-2">
            {report.recommendations.map((rec, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-300">
                <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                {rec}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Perturbation Visualizations */}
      {report && report.perturbation_visualizations.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Perturbation Analysis</h3>
          <div className="grid grid-cols-3 gap-4">
            {report.perturbation_visualizations.map((pv, i) => (
              <div key={i} className="p-4 bg-gray-800/50 rounded-xl text-center">
                <p className="text-xs text-gray-500 mb-2">Epsilon: {pv.epsilon}</p>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-500">Perturbation L2</span>
                    <span className="text-gray-300 font-mono">{pv.l2_norm.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-500">Perturbation Linf</span>
                    <span className="text-gray-300 font-mono">{pv.linf_norm.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-500">Conf Change</span>
                    <span className={`font-mono ${pv.prediction_changed ? 'text-red-400' : 'text-emerald-400'}`}>
                      {pv.original_confidence != null ? `${(pv.original_confidence * 100).toFixed(0)}%` : '--'}
                      {' → '}
                      {pv.adversarial_confidence != null ? `${(pv.adversarial_confidence * 100).toFixed(0)}%` : '--'}
                    </span>
                  </div>
                  <div className="mt-2">
                    {pv.prediction_changed ?
                      <span className="text-xs text-red-400 flex items-center justify-center gap-1">
                        <AlertTriangle className="w-3 h-3" /> FOOLED
                      </span> :
                      <span className="text-xs text-emerald-400 flex items-center justify-center gap-1">
                        <CheckCircle className="w-3 h-3" /> ROBUST
                      </span>
                    }
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Model Comparison */}
      {comparison && (
        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-4">Cross-Model Comparison</h3>
          <div className="grid grid-cols-3 gap-4">
            {comparison.map((c, i) => (
              <div key={i} className="p-4 bg-gray-800/50 rounded-xl text-center">
                <p className="text-sm font-semibold text-white mb-2">{c.model}</p>
                {c.error ? (
                  <p className="text-xs text-red-400">{c.error}</p>
                ) : c.robustness_score != null ? (
                  <>
                    <span className={`text-3xl font-bold ${getScoreColor(c.robustness_score)}`}>
                      {c.robustness_score.toFixed(0)}
                    </span>
                    <p className="text-xs text-gray-500">/100</p>
                    <div className="mt-3 space-y-1 text-xs text-gray-400">
                      <p>FGSM success: {c.fgsm_success_rate != null ? `${(c.fgsm_success_rate * 100).toFixed(0)}%` : '--'}</p>
                      <p>PGD success: {c.pgd_success_rate != null ? `${(c.pgd_success_rate * 100).toFixed(0)}%` : '--'}</p>
                      <p>Vulnerabilities: {c.vulnerability_count}</p>
                    </div>
                  </>
                ) : (
                  <p className="text-xs text-gray-500">{c.note}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function AttackSection({ title, icon, isExpanded, onToggle, children }: {
  title: string; icon: React.ReactNode; isExpanded: boolean; onToggle: () => void; children: React.ReactNode
}) {
  return (
    <div className="card cursor-pointer" onClick={onToggle}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">{icon}<h3 className="text-sm font-semibold text-gray-300">{title}</h3></div>
        {isExpanded ? <ChevronUp className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
      </div>
      {isExpanded && <div className="mt-4">{children}</div>}
    </div>
  )
}
