"""
Risk Scoring Engine.
Combines outputs from all detection services into a final Trust Score.
"""
from typing import Optional
import numpy as np


class RiskScoringEngine:
    """Compute final Trust Score (0-100) from component analyses."""

    # Weight configuration for each component
    WEIGHTS = {
        'document_authenticity': 0.25,
        'deepfake_detection': 0.30,
        'face_match': 0.25,
        'graph_risk': 0.20,
    }

    # Risk level thresholds (trust score based)
    HIGH_RISK_THRESHOLD = 40
    MEDIUM_RISK_THRESHOLD = 70

    def compute_risk_score(
        self,
        document_results: Optional[dict] = None,
        deepfake_results: Optional[dict] = None,
        face_match_results: Optional[dict] = None,
        graph_risk_score: Optional[float] = None,
        verification_metadata: Optional[dict] = None,
    ) -> dict:
        """
        Compute the final Trust Score and risk assessment.

        Returns dict with:
            - trust_score: 0-100 (higher = more trustworthy)
            - risk_level: LOW / MEDIUM / HIGH / CRITICAL
            - confidence: 0-1
            - component_scores: individual scores
            - contributing_factors: top factors affecting score
            - recommendations: suggested actions
        """
        component_scores = {}
        contributing_factors = []
        weights_used = {}

        # ── Document Authenticity Score (0-100) ──
        if document_results is not None:
            doc_score = document_results.get('authenticity_score', 0.5) * 100
            component_scores['document_authenticity'] = doc_score
            weights_used['document_authenticity'] = self.WEIGHTS['document_authenticity']

            if document_results.get('tampering_detected'):
                factors = []
                if document_results.get('font_inconsistencies'):
                    factors.append(f"{len(document_results['font_inconsistencies'])} font inconsistency(ies)")
                if document_results.get('tampered_regions'):
                    factors.append(f"{len(document_results['tampered_regions'])} tampered region(s)")
                if document_results.get('spacing_anomalies'):
                    factors.append(f"{len(document_results['spacing_anomalies'])} spacing anomaly(ies)")

                contributing_factors.append({
                    'component': 'document',
                    'impact': 'negative',
                    'severity': 'high' if doc_score < 30 else 'medium',
                    'description': f"Document authenticity: {doc_score:.0f}/100",
                    'details': '; '.join(factors) if factors else 'Various tampering indicators',
                })

        # ── Deepfake Detection Score (0-100, inverted: low probability = high trust) ──
        if deepfake_results is not None:
            deepfake_prob = deepfake_results.get('deepfake_probability', 0.0)
            deepfake_score = (1.0 - deepfake_prob) * 100
            component_scores['deepfake_detection'] = deepfake_score
            weights_used['deepfake_detection'] = self.WEIGHTS['deepfake_detection']

            if deepfake_results.get('is_deepfake'):
                details = []
                if deepfake_results.get('face_artifacts'):
                    details.append('face artifacts')
                if deepfake_results.get('blinking_anomaly'):
                    details.append('abnormal blinking')
                if deepfake_results.get('gan_fingerprint'):
                    details.append('GAN fingerprint')

                contributing_factors.append({
                    'component': 'deepfake',
                    'impact': 'negative',
                    'severity': 'critical' if deepfake_prob > 0.8 else 'high',
                    'description': f"Deepfake probability: {deepfake_prob:.1%}",
                    'details': ', '.join(details) if details else 'General deepfake indicators',
                })

        # ── Face Match Score (0-100) ──
        if face_match_results is not None:
            if face_match_results.get('match_score') is not None:
                face_score = face_match_results['match_score'] * 100
            elif face_match_results.get('faces_detected', 0) == 0:
                face_score = 30  # No face detected is moderately suspicious
            else:
                face_score = 50  # Default when no reference available

            component_scores['face_match'] = face_score
            weights_used['face_match'] = self.WEIGHTS['face_match']

            if not face_match_results.get('is_match', True):
                contributing_factors.append({
                    'component': 'face_match',
                    'impact': 'negative',
                    'severity': 'high',
                    'description': f"Face match score: {face_match_results.get('match_score', 0):.3f}",
                    'details': 'Face does not match reference document',
                })

            if face_match_results.get('faces_detected', 0) == 0:
                contributing_factors.append({
                    'component': 'face_match',
                    'impact': 'negative',
                    'severity': 'medium',
                    'description': 'No face detected in image',
                    'details': 'Unable to verify identity',
                })

        # ── Graph Risk Score (0-100, inverted: high risk = low trust) ──
        if graph_risk_score is not None:
            graph_trust = (1.0 - graph_risk_score) * 100
            component_scores['graph_risk'] = graph_trust
            weights_used['graph_risk'] = self.WEIGHTS['graph_risk']

            if graph_risk_score > 0.5:
                contributing_factors.append({
                    'component': 'fraud_graph',
                    'impact': 'negative',
                    'severity': 'high' if graph_risk_score > 0.7 else 'medium',
                    'description': f"Fraud graph risk: {graph_risk_score:.2f}",
                    'details': 'Suspicious relationships detected in fraud network',
                })

        # ── Compute Weighted Trust Score ──
        if not component_scores:
            trust_score = 50.0  # No data = neutral
            confidence = 0.1
        else:
            total_weight = sum(weights_used.values())
            if total_weight > 0:
                weighted_sum = sum(
                    component_scores[k] * weights_used[k]
                    for k in component_scores
                )
                trust_score = weighted_sum / total_weight
            else:
                trust_score = np.mean(list(component_scores.values()))

            # Confidence based on how many components were available
            coverage = len(component_scores) / 4
            confidence = min(0.3 + coverage * 0.7, 1.0)

        trust_score = max(0.0, min(100.0, trust_score))

        # ── Determine Risk Level ──
        if trust_score < self.HIGH_RISK_THRESHOLD:
            risk_level = 'HIGH'
            if trust_score < 20:
                risk_level = 'CRITICAL'
        elif trust_score < self.MEDIUM_RISK_THRESHOLD:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'

        # ── Generate Recommendations ──
        recommendations = self._generate_recommendations(
            trust_score, risk_level, component_scores, contributing_factors
        )

        # Sort contributing factors by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'info': 4}
        contributing_factors.sort(key=lambda x: severity_order.get(x.get('severity', 'low'), 5))

        return {
            'trust_score': round(trust_score, 1),
            'risk_level': risk_level,
            'confidence': round(confidence, 3),
            'component_scores': component_scores,
            'contributing_factors': contributing_factors,
            'recommendations': recommendations,
        }

    def _generate_recommendations(
        self,
        trust_score: float,
        risk_level: str,
        component_scores: dict,
        contributing_factors: list,
    ) -> list[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        if risk_level in ('CRITICAL', 'HIGH'):
            recommendations.append('Recommend immediate manual review by fraud analyst')
            recommendations.append('Hold any automated approval processes')

        # Document-specific
        doc_score = component_scores.get('document_authenticity')
        if doc_score is not None and doc_score < 50:
            recommendations.append('Request original document or certified copy')
            recommendations.append('Cross-reference with issuing authority database')

        # Deepfake-specific
        deepfake_score = component_scores.get('deepfake_detection')
        if deepfake_score is not None and deepfake_score < 50:
            recommendations.append('Request live video verification with liveness detection')
            recommendations.append('Consider in-person identity verification')

        # Face match specific
        face_score = component_scores.get('face_match')
        if face_score is not None and face_score < 50:
            recommendations.append('Request additional government-issued ID for verification')
            recommendations.append('Conduct biometric cross-check')

        # Graph-specific
        graph_score = component_scores.get('graph_risk')
        if graph_score is not None and graph_score < 40:
            recommendations.append('Review user history and connected accounts')
            recommendations.append('Flag for enhanced due diligence')

        if trust_score >= 80:
            recommendations.append('Verification passed - proceed with standard workflow')

        if not recommendations:
            recommendations.append('No specific action required - continue monitoring')

        return recommendations
