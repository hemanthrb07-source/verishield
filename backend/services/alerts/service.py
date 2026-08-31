"""
Real-time Alert Service.
WebSocket manager that broadcasts fraud notifications
when high-risk verifications are detected.
"""
import asyncio
import json
import time
from typing import Optional
from fastapi import WebSocket
from dataclasses import dataclass, field, asdict


@dataclass
class FraudAlert:
    """A single fraud alert."""
    alert_id: str
    verification_id: str
    timestamp: float
    risk_level: str  # HIGH, CRITICAL
    trust_score: float
    alert_type: str  # deepfake, document_tampering, face_mismatch, fraud_graph, etc.
    title: str
    message: str
    details: dict = field(default_factory=dict)
    user_id: Optional[str] = None
    file_name: Optional[str] = None
    file_type: Optional[str] = None

    def to_dict(self):
        return asdict(self)


# Alert type constants
ALERT_DEEPFAKE = "deepfake_detected"
ALERT_DOC_TAMPER = "document_tampering"
ALERT_FACE_MISMATCH = "face_mismatch"
ALERT_FRAUD_GRAPH = "fraud_graph_risk"
ALERT_CRITICAL_SCORE = "critical_trust_score"
ALERT_BATCH_HIGH_RISK = "batch_high_risk"


class AlertManager:
    """
    Manages WebSocket connections and broadcasts fraud alerts.
    
    Architecture:
    - Clients connect via /ws/alerts
    - Each connection is tracked with optional channel filtering
    - When a verification completes with HIGH/CRITICAL risk,
      the manager broadcasts to all connected clients
    - Supports per-channel subscriptions (e.g. 'all', 'critical_only')
    """

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.alert_history: list[dict] = []
        self.max_history = 200
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str):
        """Register a new WebSocket client."""
        await websocket.accept()
        async with self._lock:
            self.active_connections[client_id] = websocket
        # Send recent alert history on connect
        await websocket.send_json({
            "type": "history",
            "alerts": self.alert_history[-50:],
            "connected_clients": len(self.active_connections),
        })

    async def disconnect(self, client_id: str):
        """Remove a WebSocket client."""
        async with self._lock:
            self.active_connections.pop(client_id, None)

    async def broadcast(self, alert: FraudAlert):
        """Broadcast an alert to all connected clients."""
        message = {
            "type": "alert",
            "alert": alert.to_dict(),
            "connected_clients": len(self.active_connections),
        }

        # Store in history
        self.alert_history.append(alert.to_dict())
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]

        # Broadcast to all connected clients
        disconnected = []
        for client_id, ws in self.active_connections.items():
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(client_id)

        # Clean up disconnected clients
        for cid in disconnected:
            self.active_connections.pop(cid, None)

    async def send_stats_update(self, stats: dict):
        """Broadcast a stats update to all clients."""
        message = {
            "type": "stats_update",
            "stats": stats,
        }
        disconnected = []
        for client_id, ws in self.active_connections.items():
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(client_id)
        for cid in disconnected:
            self.active_connections.pop(cid, None)

    def generate_alerts_for_result(self, verification_id: str, result: dict) -> list[FraudAlert]:
        """
        Analyze a verification result and generate alerts for any
        high-risk findings.
        """
        alerts = []
        risk_level = result.get("risk_level", "LOW")
        trust_score = result.get("trust_score", 100)
        details = result.get("detailed_results", {})
        file_name = result.get("file_name", "unknown")
        file_type = result.get("file_type", "UNKNOWN")
        user_id = result.get("user_id")

        # Only generate alerts for HIGH and CRITICAL
        if risk_level not in ("HIGH", "CRITICAL"):
            return alerts

        ts = time.time()

        # Critical trust score alert
        if trust_score < 20:
            alerts.append(FraudAlert(
                alert_id=f"alert_{verification_id[:8]}_critical",
                verification_id=verification_id,
                timestamp=ts,
                risk_level=risk_level,
                trust_score=trust_score,
                alert_type=ALERT_CRITICAL_SCORE,
                title="CRITICAL: Extremely Low Trust Score",
                message=f"Verification scored {trust_score:.0f}/100. "
                        f"Immediate manual review required.",
                file_name=file_name,
                file_type=file_type,
                user_id=user_id,
                details={"trust_score": trust_score},
            ))

        # Deepfake alert
        deepfake = details.get("deepfake_analysis") or details.get("deepfake")
        if deepfake and deepfake.get("is_deepfake"):
            prob = deepfake.get("probability", deepfake.get("deepfake_probability", 0))
            alerts.append(FraudAlert(
                alert_id=f"alert_{verification_id[:8]}_deepfake",
                verification_id=verification_id,
                timestamp=ts,
                risk_level=risk_level,
                trust_score=trust_score,
                alert_type=ALERT_DEEPFAKE,
                title="Deepfake Detected",
                message=f"Deepfake probability: {prob:.1%}. "
                        f"Face artifacts: {'Yes' if deepfake.get('face_artifacts') else 'No'}. "
                        f"GAN fingerprint: {'Yes' if deepfake.get('gan_fingerprint') else 'No'}.",
                file_name=file_name,
                file_type=file_type,
                user_id=user_id,
                details={
                    "probability": prob,
                    "face_artifacts": deepfake.get("face_artifacts", False),
                    "blinking_anomaly": deepfake.get("blinking_anomaly", False),
                    "gan_fingerprint": deepfake.get("gan_fingerprint", False),
                },
            ))

        # Document tampering alert
        doc = details.get("document_analysis") or details.get("document")
        if doc and doc.get("tampering_detected"):
            tampered_count = len(doc.get("tampered_regions", []))
            font_count = len(doc.get("font_inconsistencies", []))
            alerts.append(FraudAlert(
                alert_id=f"alert_{verification_id[:8]}_tamper",
                verification_id=verification_id,
                timestamp=ts,
                risk_level=risk_level,
                trust_score=trust_score,
                alert_type=ALERT_DOC_TAMPER,
                title="Document Tampering Detected",
                message=f"Authenticity: {doc.get('authenticity_score', 0):.0%}. "
                        f"{tampered_count} tampered region(s), "
                        f"{font_count} font inconsistency(ies).",
                file_name=file_name,
                file_type=file_type,
                user_id=user_id,
                details={
                    "authenticity_score": doc.get("authenticity_score", 0),
                    "tampered_regions": tampered_count,
                    "font_inconsistencies": font_count,
                },
            ))

        # Face mismatch alert
        face = details.get("face_analysis")
        if face and face.get("match_score") is not None and not face.get("is_match"):
            alerts.append(FraudAlert(
                alert_id=f"alert_{verification_id[:8]}_face",
                verification_id=verification_id,
                timestamp=ts,
                risk_level=risk_level,
                trust_score=trust_score,
                alert_type=ALERT_FACE_MISMATCH,
                title="Face Mismatch",
                message=f"Face match score: {face.get('match_score', 0):.3f}. "
                        f"Faces detected: {face.get('faces_detected', 0)}. "
                        f"Does not match reference.",
                file_name=file_name,
                file_type=file_type,
                user_id=user_id,
                details={
                    "match_score": face.get("match_score"),
                    "faces_detected": face.get("faces_detected", 0),
                    "quality_score": face.get("quality_score", 0),
                },
            ))

        # Fraud graph risk alert
        risk_assessment = details.get("risk_assessment") or details.get("risk")
        graph_score = None
        if risk_assessment:
            graph_score = risk_assessment.get("component_scores", {}).get("graph_risk")
            if graph_score is not None and graph_score < 30:
                alerts.append(FraudAlert(
                    alert_id=f"alert_{verification_id[:8]}_graph",
                    verification_id=verification_id,
                    timestamp=ts,
                    risk_level=risk_level,
                    trust_score=trust_score,
                    alert_type=ALERT_FRAUD_GRAPH,
                    title="Suspicious Fraud Graph Activity",
                    message=f"Graph risk score: {graph_score:.0f}/100. "
                            f"Suspicious relationships detected in network.",
                    file_name=file_name,
                    file_type=file_type,
                    user_id=user_id,
                    details={"graph_trust_score": graph_score},
                ))

        # Contributing factors summary (if we didn't generate specific alerts)
        if not alerts and risk_level in ("HIGH", "CRITICAL"):
            contributing = risk_assessment.get("contributing_factors", []) if risk_assessment else []
            top_factor = contributing[0] if contributing else {}
            alerts.append(FraudAlert(
                alert_id=f"alert_{verification_id[:8]}_general",
                verification_id=verification_id,
                timestamp=ts,
                risk_level=risk_level,
                trust_score=trust_score,
                alert_type="general_risk",
                title=f"High-Risk Verification ({risk_level})",
                message=top_factor.get("description", f"Trust score: {trust_score:.0f}/100"),
                file_name=file_name,
                file_type=file_type,
                user_id=user_id,
                details={"top_factor": top_factor},
            ))

        return alerts


# Global singleton
alert_manager = AlertManager()
