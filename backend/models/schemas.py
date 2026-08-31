"""
Schemas for API request/response models using Pydantic.
"""
from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FileType(str, Enum):
    DOCUMENT = "DOCUMENT"
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"


class VerificationStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ── Request Models ──────────────────────────────────────────────────────

class VerificationRequest(BaseModel):
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    device_info: Optional[dict] = None


class FaceVerificationRequest(BaseModel):
    user_id: Optional[str] = None
    reference_image_required: bool = True


class RiskScoreRequest(BaseModel):
    verification_id: str
    include_graph: bool = True


class BatchVerificationRequest(BaseModel):
    user_id: Optional[str] = None
    items: list[dict] = Field(default_factory=list, description="List of {file_path, file_type} objects")


# ── Response Models ─────────────────────────────────────────────────────

class DocumentAnalysisResult(BaseModel):
    authenticity_score: float
    tampering_detected: bool
    font_inconsistencies: list[dict] = []
    spacing_anomalies: list[dict] = []
    tampered_regions: list[dict] = []
    ocr_confidence: Optional[float] = None
    metadata_analysis: dict = {}
    reasons: list[str] = []


class DeepfakeAnalysisResult(BaseModel):
    is_deepfake: bool
    deepfake_probability: float
    face_artifacts: bool = False
    blinking_anomaly: bool = False
    gan_fingerprint: bool = False
    confidence: float = 0.0
    frame_analysis: Optional[list[dict]] = None
    reasons: list[str] = []


class FaceMatchResult(BaseModel):
    faces_detected: int
    match_score: Optional[float] = None
    is_match: bool = False
    quality_score: float = 0.0
    face_locations: list[dict] = []
    reasons: list[str] = []


class RiskScoreResult(BaseModel):
    trust_score: float
    risk_level: RiskLevel
    confidence: float
    component_scores: dict
    contributing_factors: list[dict]
    recommendations: list[str]


class VerificationResponse(BaseModel):
    verification_id: str
    status: VerificationStatus
    file_type: FileType
    trust_score: Optional[float] = None
    risk_level: Optional[RiskLevel] = None
    confidence: Optional[float] = None
    reasons: list[str] = []
    detailed_results: Optional[dict] = None
    processing_time_ms: Optional[int] = None
    blockchain_tx_hash: Optional[str] = None
    created_at: Optional[datetime] = None


class BatchVerificationResponse(BaseModel):
    batch_id: str
    total_items: int
    results: list[VerificationResponse]
    summary: dict


class VerificationListItem(BaseModel):
    id: str
    file_type: FileType
    file_name: str
    status: VerificationStatus
    trust_score: Optional[float] = None
    risk_level: Optional[RiskLevel] = None
    created_at: datetime


class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict[str, str]
    uptime_seconds: float


class GraphNode(BaseModel):
    id: str
    node_type: str
    label: str
    risk_score: float = 0.0


class GraphEdge(BaseModel):
    source: str
    target: str
    relationship: str
    weight: float = 1.0


class FraudGraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    suspicious_clusters: list[dict]
    risk_summary: dict
