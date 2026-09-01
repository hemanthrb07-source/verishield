"""
Risk Scoring Engine.
Combines outputs from all detection services into a final Trust Score.
All descriptions are written in plain English for non-technical users.
"""
from typing import Optional
import numpy as np


class RiskScoringEngine:
    """Compute final Trust Score (0-100) from component analyses."""

    WEIGHTS = {
        'document_authenticity': 0.25,
        'deepfake_detection': 0.30,
        'face_match': 0.25,
        'graph_risk': 0.20,
    }

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
        All descriptions are written in plain, everyday language.
        """
        component_scores = {}
        contributing_factors = []
        weights_used = {}

        # ── Document Authenticity Score ──
        if document_results is not None:
            doc_score = document_results.get('authenticity_score', 0.5) * 100
            component_scores['document_authenticity'] = doc_score
            weights_used['document_authenticity'] = self.WEIGHTS['document_authenticity']

            if document_results.get('tampering_detected'):
                # Collect what was found
                font_count = len(document_results.get('font_inconsistencies', []))
                tamper_count = len(document_results.get('tampered_regions', []))
                spacing_count = len(document_results.get('spacing_anomalies', []))

                issues = []
                if font_count:
                    issues.append(f"{font_count} text style{'s' if font_count > 1 else ''} look inconsistent (different fonts or sizes)")
                if tamper_count:
                    issues.append(f"{tamper_count} area{'s' if tamper_count > 1 else ''} appear to have been edited or altered")
                if spacing_count:
                    issues.append(f"{spacing_count} spacing issue{'s' if spacing_count > 1 else ''} detected (text or layout looks irregular)")

                detail_text = '; '.join(issues) if issues else "The document shows signs of being modified"

                # Severity depends on how low the score is
                if doc_score < 30:
                    severity = 'high'
                    desc = f"This document looks suspicious (score: {doc_score:.0f}/100) — multiple signs of tampering were found"
                else:
                    severity = 'medium'
                    desc = f"This document has some concerns (score: {doc_score:.0f}/100) — a few irregularities were detected"

                contributing_factors.append({
                    'component': 'document',
                    'impact': 'negative',
                    'severity': severity,
                    'description': desc,
                    'details': detail_text,
                })

        # ── Deepfake Detection Score ──
        if deepfake_results is not None:
            deepfake_prob = deepfake_results.get('deepfake_probability', 0.0)
            deepfake_score = (1.0 - deepfake_prob) * 100
            component_scores['deepfake_detection'] = deepfake_score
            weights_used['deepfake_detection'] = self.WEIGHTS['deepfake_detection']

            if deepfake_results.get('is_deepfake'):
                details = []
                if deepfake_results.get('face_artifacts'):
                    details.append("unusual patterns on the face region")
                if deepfake_results.get('blinking_anomaly'):
                    details.append("unnatural blinking or eye movement")
                if deepfake_results.get('gan_fingerprint'):
                    details.append("digital fingerprints typical of AI-generated images")
                if deepfake_results.get('frame_analysis'):
                    frame_count = len(deepfake_results['frame_analysis'])
                    details.append(f"analyzed {frame_count} frames and found inconsistencies between them")

                detail_text = ', '.join(details) if details else "The image or video shows signs of being AI-generated or manipulated"

                if deepfake_prob > 0.8:
                    severity = 'critical'
                    desc = (
                        f"⚠️ HIGH ALERT: This media is very likely fake or AI-generated "
                        f"(confidence: {deepfake_prob:.0%}). "
                        f"It was probably created using deepfake technology."
                    )
                elif deepfake_prob > 0.6:
                    severity = 'high'
                    desc = (
                        f"⚠️ WARNING: This media is likely fake or tampered with "
                        f"(confidence: {deepfake_prob:.0%}). "
                        f"Signs of AI manipulation were detected."
                    )
                else:
                    severity = 'high'
                    desc = (
                        f"There is a moderate chance this media is fake "
                        f"(confidence: {deepfake_prob:.0%}). "
                        f"Some indicators of manipulation were found."
                    )

                contributing_factors.append({
                    'component': 'deepfake',
                    'impact': 'negative',
                    'severity': severity,
                    'description': desc,
                    'details': detail_text,
                })
            else:
                # Even when not deepfake, provide informative feedback
                if deepfake_prob > 0.3:
                    contributing_factors.append({
                        'component': 'deepfake',
                        'impact': 'positive',
                        'severity': 'info',
                        'description': (
                            f"This media appears to be authentic "
                            f"(fake probability: {deepfake_prob:.0%}). "
                            f"No significant signs of AI manipulation were found."
                        ),
                        'details': 'The analysis did not detect typical deepfake patterns',
                    })

        # ── Face Match Score ──
        if face_match_results is not None:
            if face_match_results.get('match_score') is not None:
                face_score = face_match_results['match_score'] * 100
            elif face_match_results.get('faces_detected', 0) == 0:
                face_score = 30
            else:
                face_score = 50

            component_scores['face_match'] = face_score
            weights_used['face_match'] = self.WEIGHTS['face_match']

            if not face_match_results.get('is_match', True):
                match_score = face_match_results.get('match_score', 0)
                if match_score < 0.3:
                    desc = (
                        "❌ The person in this image does NOT match the reference photo. "
                        "The facial features are very different — this could be a different person."
                    )
                else:
                    desc = (
                        "⚠️ The person in this image does not closely match the reference photo. "
                        "Some facial features differ from the original."
                    )
                contributing_factors.append({
                    'component': 'face_match',
                    'impact': 'negative',
                    'severity': 'high',
                    'description': desc,
                    'details': 'The face comparison shows a mismatch between the submitted image and the reference document',
                })

            if face_match_results.get('faces_detected', 0) == 0:
                contributing_factors.append({
                    'component': 'face_match',
                    'impact': 'negative',
                    'severity': 'medium',
                    'description': (
                        "No face was found in this image. "
                        "The system could not verify the person's identity because no face was visible."
                    ),
                    'details': 'Make sure the face is clearly visible and not covered or too dark',
                })

        # ── Graph Risk Score ──
        if graph_risk_score is not None:
            graph_trust = (1.0 - graph_risk_score) * 100
            component_scores['graph_risk'] = graph_trust
            weights_used['graph_risk'] = self.WEIGHTS['graph_risk']

            if graph_risk_score > 0.5:
                if graph_risk_score > 0.7:
                    desc = (
                        "🔴 This person or device is linked to previous suspicious activity. "
                        "Similar identity documents or devices have been flagged before."
                    )
                    severity = 'high'
                else:
                    desc = (
                        "⚠️ Some connections to previously flagged activity were found. "
                        "This person or device has been seen in a suspicious pattern before."
                    )
                    severity = 'medium'

                contributing_factors.append({
                    'component': 'fraud_graph',
                    'impact': 'negative',
                    'severity': severity,
                    'description': desc,
                    'details': 'The fraud network analysis found links to previously flagged accounts or devices',
                })

        # ── Compute Weighted Trust Score ──
        if not component_scores:
            trust_score = 50.0
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
        """Generate clear, actionable recommendations in plain English."""
        recommendations = []

        if risk_level in ('CRITICAL', 'HIGH'):
            recommendations.append(
                "🚨 STOP — Do not proceed automatically. "
                "Have a human reviewer look at this case immediately."
            )
            recommendations.append(
                "Pause any automatic approvals or account creation until this is resolved."
            )

        # Document-specific
        doc_score = component_scores.get('document_authenticity')
        if doc_score is not None and doc_score < 50:
            recommendations.append(
                "📄 Ask the person to upload the ORIGINAL document (not a photo or screenshot). "
                "A clear scan or photo of the physical document works best."
            )
            recommendations.append(
                "📞 Contact the organization that issued this document to verify it is real."
            )

        # Deepfake-specific
        deepfake_score = component_scores.get('deepfake_detection')
        if deepfake_score is not None and deepfake_score < 50:
            recommendations.append(
                "🎥 Ask the person to take a LIVE photo or video right now, "
                "using their phone's front camera, so we can verify it's a real person."
            )
            recommendations.append(
                "👤 Consider meeting the person in person or via a live video call "
                "to confirm their identity."
            )

        # Face match specific
        face_score = component_scores.get('face_match')
        if face_score is not None and face_score < 50:
            recommendations.append(
                "🪪 Ask the person to upload a different government-issued photo ID "
                "(like a passport or driver's license) for additional verification."
            )
            recommendations.append(
                "🔍 Run a biometric check against the original ID photo to confirm the match."
            )

        # Graph-specific
        graph_score = component_scores.get('graph_risk')
        if graph_score is not None and graph_score < 40:
            recommendations.append(
                "📋 Review this person's account history and any connected devices or accounts."
            )
            recommendations.append(
                "🔒 Flag this case for extra security checks (enhanced due diligence)."
            )

        if trust_score >= 80:
            recommendations.append(
                "✅ This verification passed! Everything looks trustworthy. "
                "You can continue with your normal workflow."
            )

        if not recommendations:
            recommendations.append(
                "👍 No issues found. No special action is needed — keep monitoring as usual."
            )

        return recommendations
