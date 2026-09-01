import { useState } from 'react'
import type { VerificationResult } from '../services/api'
import { HeatmapDisplay } from './HeatmapDisplay'
import { TrustScoreBreakdown } from './TrustScoreBreakdown'
import {
  CheckCircle, XCircle, AlertTriangle, Clock,
  FileText, Eye, Fingerprint, Network, Layers,
  ChevronDown, ChevronUp, ExternalLink
} from 'lucide-react'

interface Props {
  result: VerificationResult
}

export function ResultsPanel({ result }: Props) {
  const [expanded, setExpanded] = useState<string | null>(null)

  const toggle = (section: string) => {
    setExpanded(expanded === section ? null : section)
  }

  const details = result.detailed_results || {}
  const riskAssessment = details.risk_assessment || {}
  const contributingFactors = riskAssessment.contributing_factors || []
  const recommendations = riskAssessment.recommendations || []

  return (
    <div className="space-y-4 animate-slide-up">
      {/* Header */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-lg font-semibold text-white">Verification Results</h2>
            <p className="text-xs text-gray-500 mt-1">
              ID: {result.verification_id.slice(0, 8)}...
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="badge bg-gray-700 text-gray-300">
              {result.file_type}
            </span>
            <span className="flex items-center gap-1 text-xs text-gray-500">
              <Clock className="w-3 h-3" />
              {result.processing_time_ms}ms
            </span>
          </div>
        </div>

        {/* Score Summary */}
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center p-3 bg-gray-800/50 rounded-xl">
            <p className="text-xs text-gray-500 mb-1">Trust Score</p>
            <p className={`text-2xl font-bold ${
              (result.trust_score ?? 0) >= 70 ? 'text-emerald-400' :
              (result.trust_score ?? 0) >= 40 ? 'text-amber-400' : 'text-red-400'
            }`}>
              {result.trust_score?.toFixed(0) ?? '--'}
            </p>
          </div>
          <div className="text-center p-3 bg-gray-800/50 rounded-xl">
            <p className="text-xs text-gray-500 mb-1">Confidence</p>
            <p className="text-2xl font-bold text-brand-400">
              {((result.confidence ?? 0) * 100).toFixed(0)}%
            </p>
          </div>
          <div className="text-center p-3 bg-gray-800/50 rounded-xl">
            <p className="text-xs text-gray-500 mb-1">Risk Level</p>
            <p className={`text-lg font-bold ${
              result.risk_level === 'LOW' ? 'text-emerald-400' :
              result.risk_level === 'MEDIUM' ? 'text-amber-400' : 'text-red-400'
            }`}>
              {result.risk_level}
            </p>
          </div>
        </div>
      </div>

      {/* Trust Score Breakdown Chart */}
      {riskAssessment.component_scores && (
        <TrustScoreBreakdown
          componentScores={riskAssessment.component_scores}
          trustScore={result.trust_score ?? 0}
          riskLevel={result.risk_level ?? 'UNKNOWN'}
        />
      )}

      {/* Contributing Factors */}
      {contributingFactors.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Contributing Factors</h3>
          <div className="space-y-2">
            {contributingFactors.map((factor: any, i: number) => (
              <div key={i} className={`flex items-start gap-3 p-3 rounded-xl
                ${factor.severity === 'critical' || factor.severity === 'high'
                  ? 'bg-red-500/10 border border-red-500/20'
                  : factor.severity === 'medium'
                    ? 'bg-amber-500/10 border border-amber-500/20'
                    : 'bg-gray-800/50 border border-gray-700/50'
                }`}>
                {factor.severity === 'critical' || factor.severity === 'high' ? (
                  <XCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
                ) : factor.severity === 'medium' ? (
                  <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
                ) : (
                  <CheckCircle className="w-5 h-5 text-gray-500 shrink-0 mt-0.5" />
                )}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-200">{factor.description}</p>
                  {factor.details && (
                    <p className="text-xs text-gray-400 mt-1">{factor.details}</p>
                  )}
                </div>
                <span className={`text-xs px-2 py-0.5 rounded-full
                  ${factor.severity === 'critical' ? 'bg-red-500/20 text-red-400' :
                    factor.severity === 'high' ? 'bg-red-500/10 text-red-300' :
                    factor.severity === 'medium' ? 'bg-amber-500/10 text-amber-400' :
                    'bg-gray-700 text-gray-400'}`}>
                  {factor.severity}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Plain-English Explanation */}
      {result.reasons && result.reasons.length > 0 && (
        <div className="card border-brand-500/30">
          <h3 className="text-sm font-semibold text-brand-400 mb-3 flex items-center gap-2">
            📋 What We Found — Simple Explanation
          </h3>
          <div className="space-y-3">
            {result.reasons.map((reason, i) => {
              const isPositive = reason.startsWith('✅')
              const isWarning = reason.startsWith('⚠️')
              const isNegative = reason.startsWith('❌')
              return (
                <div key={i} className={`p-3 rounded-xl text-sm leading-relaxed
                  ${isNegative ? 'bg-red-500/10 border border-red-500/20 text-red-200'
                  : isWarning ? 'bg-amber-500/10 border border-amber-500/20 text-amber-200'
                  : isPositive ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-200'
                  : 'bg-gray-800/50 border border-gray-700/50 text-gray-300'}`}
                >
                  {reason}
                </div>
              )
            })}
          </div>
          <p className="text-xs text-gray-500 mt-3 italic">
            These explanations describe what our AI detected in the uploaded file, explained in everyday language.
          </p>
        </div>
      )}

      {/* Detailed Analysis Sections */}
      <div className="space-y-2">
        {/* Document Analysis */}
        {details.document_analysis && (
          <AnalysisSection
            title="Document Intelligence"
            icon={<FileText className="w-4 h-4 text-amber-400" />}
            isExpanded={expanded === 'doc'}
            onToggle={() => toggle('doc')}
          >
            <DocumentAnalysis data={details.document_analysis} />
          </AnalysisSection>
        )}

        {/* Deepfake Analysis */}
        {details.deepfake_analysis && (
          <AnalysisSection
            title="Deepfake Detection"
            icon={<Eye className="w-4 h-4 text-purple-400" />}
            isExpanded={expanded === 'deepfake'}
            onToggle={() => toggle('deepfake')}
          >
            <DeepfakeAnalysis data={details.deepfake_analysis} />
          </AnalysisSection>
        )}

        {/* Face Analysis */}
        {details.face_analysis && (
          <AnalysisSection
            title="Face Matching"
            icon={<Fingerprint className="w-4 h-4 text-blue-400" />}
            isExpanded={expanded === 'face'}
            onToggle={() => toggle('face')}
          >
            <FaceAnalysis data={details.face_analysis} />
          </AnalysisSection>
        )}

        {/* Recommendations */}
        {recommendations.length > 0 && (
          <AnalysisSection
            title="Recommendations"
            icon={<Network className="w-4 h-4 text-emerald-400" />}
            isExpanded={expanded === 'recs'}
            onToggle={() => toggle('recs')}
          >
            <ul className="space-y-2">
              {recommendations.map((rec: string, i: number) => (
                <li key={i} className="flex items-start gap-2 text-sm text-gray-300">
                  <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  {rec}
                </li>
              ))}
            </ul>
          </AnalysisSection>
        )}
      </div>

      {/* Heatmap */}
      {details.heatmap && (
        <HeatmapDisplay heatmap={details.heatmap} title="AI Analysis Heatmap" />
      )}

      {/* Reference Comparison */}
      {details.reference_comparison && (
        <div className="card border-cyan-500/30">
          <h3 className="flex items-center gap-2 text-sm font-semibold text-cyan-400 mb-2">
            <Layers className="w-4 h-4" />
            Reference Comparison
          </h3>
          <div className="grid grid-cols-3 gap-3">
            <div className="p-2 bg-gray-800/50 rounded-lg text-center">
              <p className="text-xs text-gray-500">Reference Score</p>
              <p className="text-lg font-bold text-cyan-400">
                {(details.reference_comparison.reference_authenticity * 100).toFixed(0)}%
              </p>
            </div>
            <div className="p-2 bg-gray-800/50 rounded-lg text-center">
              <p className="text-xs text-gray-500">Current Score</p>
              <p className="text-lg font-bold text-white">
                {(details.reference_comparison.current_authenticity * 100).toFixed(0)}%
              </p>
            </div>
            <div className="p-2 bg-gray-800/50 rounded-lg text-center">
              <p className="text-xs text-gray-500">Difference</p>
              <p className={`text-lg font-bold ${
                details.reference_comparison.difference > 0.2 ? 'text-red-400' :
                details.reference_comparison.difference > 0.1 ? 'text-amber-400' : 'text-emerald-400'
              }`}>
                {(details.reference_comparison.difference * 100).toFixed(0)}%
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Blockchain */}
      {result.blockchain_tx_hash && (
        <div className="card bg-gray-900/50 border-gray-800">
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            Logged to blockchain
            <code className="ml-2 text-gray-400 font-mono text-xs">
              {result.blockchain_tx_hash.slice(0, 16)}...
            </code>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────

function AnalysisSection({ title, icon, isExpanded, onToggle, children }: {
  title: string
  icon: React.ReactNode
  isExpanded: boolean
  onToggle: () => void
  children: React.ReactNode
}) {
  return (
    <div className="card cursor-pointer" onClick={onToggle}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {icon}
          <h3 className="text-sm font-semibold text-gray-300">{title}</h3>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-4 h-4 text-gray-500" />
        ) : (
          <ChevronDown className="w-4 h-4 text-gray-500" />
        )}
      </div>
      {isExpanded && <div className="mt-4">{children}</div>}
    </div>
  )
}

function DocumentAnalysis({ data }: { data: any }) {
  const score = data.authenticity_score * 100
  let verdict = ''
  let verdictColor = ''
  if (score >= 80) { verdict = 'This document looks authentic — no signs of editing were found.'; verdictColor = 'text-emerald-400' }
  else if (score >= 60) { verdict = 'This document has some minor irregularities, but nothing clearly suspicious.'; verdictColor = 'text-amber-400' }
  else if (score >= 40) { verdict = 'This document has several concerns — parts of it may have been edited or altered.'; verdictColor = 'text-amber-400' }
  else { verdict = 'This document looks suspicious — multiple signs of tampering or manipulation were found.'; verdictColor = 'text-red-400' }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <StatBox label="Authenticity Score" value={`${score.toFixed(0)} / 100`}
          good={data.authenticity_score > 0.7} />
        <StatBox label="Tampering" value={data.tampering_detected ? 'Detected' : 'None'}
          good={!data.tampering_detected} />
      </div>
      <div className={`p-3 rounded-xl bg-gray-800/50 border border-gray-700/50 text-sm ${verdictColor}`}>
        {verdict}
      </div>
      <div className="grid grid-cols-3 gap-2 text-center">
        <div className="p-2 bg-gray-800/50 rounded-lg">
          <p className="text-xs text-gray-500">Text Style Issues</p>
          <p className="text-lg font-bold text-gray-200">{data.font_inconsistencies}</p>
        </div>
        <div className="p-2 bg-gray-800/50 rounded-lg">
          <p className="text-xs text-gray-500">Edited Regions</p>
          <p className="text-lg font-bold text-gray-200">{data.tampered_regions}</p>
        </div>
        <div className="p-2 bg-gray-800/50 rounded-lg">
          <p className="text-xs text-gray-500">Spacing Problems</p>
          <p className="text-lg font-bold text-gray-200">{data.spacing_anomalies}</p>
        </div>
      </div>
    </div>
  )
}

function DeepfakeAnalysis({ data }: { data: any }) {
  const prob = data.probability * 100
  let verdict = ''
  let verdictColor = ''
  if (prob < 20) { verdict = 'This image/video looks real — very low chance of being AI-generated.'; verdictColor = 'text-emerald-400' }
  else if (prob < 50) { verdict = 'Some minor signs were detected, but this is likely a real image.'; verdictColor = 'text-emerald-400' }
  else if (prob < 70) { verdict = 'There is a moderate chance this is AI-generated or manipulated.'; verdictColor = 'text-amber-400' }
  else if (prob < 90) { verdict = 'This is likely a deepfake — strong signs of AI generation were found.'; verdictColor = 'text-red-400' }
  else { verdict = '⚠️ Almost certainly AI-generated — this is very likely a deepfake.'; verdictColor = 'text-red-400' }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <StatBox label="Deepfake Detected" value={data.is_deepfake ? 'YES' : 'NO'}
          good={!data.is_deepfake} />
        <StatBox label="Fake Chance" value={`${prob.toFixed(1)}%`}
          good={data.probability < 0.5} />
      </div>
      <div className={`p-3 rounded-xl bg-gray-800/50 border border-gray-700/50 text-sm ${verdictColor}`}>
        {verdict}
      </div>
      <div className="grid grid-cols-3 gap-2 text-center">
        <div className="p-2 bg-gray-800/50 rounded-lg">
          <p className="text-xs text-gray-500">Face Issues</p>
          <p className={`text-sm font-bold ${data.face_artifacts ? 'text-red-400' : 'text-emerald-400'}`}>
            {data.face_artifacts ? 'Found' : 'None'}
          </p>
        </div>
        <div className="p-2 bg-gray-800/50 rounded-lg">
          <p className="text-xs text-gray-500">Eye Behavior</p>
          <p className={`text-sm font-bold ${data.blinking_anomaly ? 'text-red-400' : 'text-emerald-400'}`}>
            {data.blinking_anomaly ? 'Unnatural' : 'Normal'}
          </p>
        </div>
        <div className="p-2 bg-gray-800/50 rounded-lg">
          <p className="text-xs text-gray-500">AI Signature</p>
          <p className={`text-sm font-bold ${data.gan_fingerprint ? 'text-red-400' : 'text-emerald-400'}`}>
            {data.gan_fingerprint ? 'Detected' : 'None'}
          </p>
        </div>
      </div>
      {data.frame_count > 0 && (
        <p className="text-xs text-gray-500">Analyzed {data.frame_count} video frames</p>
      )}
    </div>
  )
}

function FaceAnalysis({ data }: { data: any }) {
  let verdict = ''
  let verdictColor = ''
  if (data.match_score === null) {
    verdict = 'No reference photo was provided for comparison.'
    verdictColor = 'text-gray-400'
  } else if (data.is_match) {
    verdict = 'The face in this image matches the reference photo — this appears to be the same person.'
    verdictColor = 'text-emerald-400'
  } else if (data.match_score < 0.3) {
    verdict = 'The face does NOT match the reference — this could be a completely different person.'
    verdictColor = 'text-red-400'
  } else {
    verdict = 'The face does not closely match the reference — some features differ from the original.'
    verdictColor = 'text-amber-400'
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <StatBox label="Faces Found" value={String(data.faces_detected)}
          good={data.faces_detected === 1} />
        <StatBox label="Similarity" value={data.match_score !== null ? `${(data.match_score * 100).toFixed(0)}%` : 'N/A'}
          good={data.is_match} />
      </div>
      <div className={`p-3 rounded-xl bg-gray-800/50 border border-gray-700/50 text-sm ${verdictColor}`}>
        {verdict}
      </div>
      <div className="grid grid-cols-2 gap-3">
        <StatBox label="Match" value={data.is_match ? 'SAME PERSON' : (data.match_score !== null ? 'MISMATCH' : 'N/A')}
          good={data.is_match || data.match_score === null} />
        <StatBox label="Image Quality" value={`${(data.quality_score * 100).toFixed(0)}%`}
          good={data.quality_score > 0.5} />
      </div>
    </div>
  )
}

function StatBox({ label, value, good }: { label: string; value: string; good: boolean }) {
  return (
    <div className="p-3 bg-gray-800/50 rounded-xl text-center">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className={`text-sm font-bold ${good ? 'text-emerald-400' : 'text-red-400'}`}>
        {value}
      </p>
    </div>
  )
}
