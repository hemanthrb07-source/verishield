"""
SQLAlchemy database models for the verification system.
"""
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, Boolean, JSON, Enum,
    create_engine, ForeignKey, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import enum
import uuid

Base = declarative_base()


class RiskLevel(enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class VerificationStatus(enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class FileType(enum.Enum):
    DOCUMENT = "DOCUMENT"
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"


class Verification(Base):
    """Main verification record."""
    __tablename__ = "verifications"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=True, index=True)
    file_type = Column(Enum(FileType), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)
    file_path = Column(String(500), nullable=False)
    status = Column(Enum(VerificationStatus), default=VerificationStatus.PENDING)
    
    # Results
    trust_score = Column(Float, nullable=True)
    risk_level = Column(Enum(RiskLevel), nullable=True)
    confidence = Column(Float, nullable=True)
    reasons = Column(JSON, nullable=True)
    detailed_results = Column(JSON, nullable=True)
    
    # Metadata
    ip_address = Column(String(45), nullable=True)
    device_info = Column(JSON, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    blockchain_tx_hash = Column(String(66), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    document_results = relationship("DocumentResult", back_populates="verification", cascade="all, delete-orphan")
    deepfake_results = relationship("DeepfakeResult", back_populates="verification", cascade="all, delete-orphan")
    face_results = relationship("FaceMatchResult", back_populates="verification", cascade="all, delete-orphan")
    risk_assessments = relationship("RiskAssessment", back_populates="verification", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_verification_status", "status"),
        Index("idx_verification_created", "created_at"),
    )


class DocumentResult(Base):
    """Document intelligence analysis results."""
    __tablename__ = "document_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    verification_id = Column(String(36), ForeignKey("verifications.id"), nullable=False)
    
    ocr_text = Column(Text, nullable=True)
    ocr_confidence = Column(Float, nullable=True)
    font_inconsistencies = Column(JSON, nullable=True)
    spacing_anomalies = Column(JSON, nullable=True)
    tampered_regions = Column(JSON, nullable=True)
    metadata = Column(JSON, nullable=True)
    authenticity_score = Column(Float, nullable=False)
    tampering_detected = Column(Boolean, default=False)
    tampering_details = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    verification = relationship("Verification", back_populates="document_results")


class DeepfakeResult(Base):
    """Deepfake detection results."""
    __tablename__ = "deepfake_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    verification_id = Column(String(36), ForeignKey("verifications.id"), nullable=False)
    
    is_deepfake = Column(Boolean, nullable=False)
    deepfake_probability = Column(Float, nullable=False)
    face_artifacts_detected = Column(Boolean, default=False)
    blinking_anomaly = Column(Boolean, default=False)
    gan_fingerprint_detected = Column(Boolean, default=False)
    artifact_details = Column(JSON, nullable=True)
    frame_analysis = Column(JSON, nullable=True)  # For video
    confidence = Column(Float, nullable=False)
    model_version = Column(String(50), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    verification = relationship("Verification", back_populates="deepfake_results")


class FaceMatchResult(Base):
    """Face matching results."""
    __tablename__ = "face_match_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    verification_id = Column(String(36), ForeignKey("verifications.id"), nullable=False)
    
    faces_detected = Column(Integer, default=0)
    match_score = Column(Float, nullable=True)
    reference_face_embedding = Column(Text, nullable=True)  # JSON-serialized
    probe_face_embedding = Column(Text, nullable=True)
    face_locations = Column(JSON, nullable=True)
    landmarks = Column(JSON, nullable=True)
    is_match = Column(Boolean, nullable=True)
    quality_score = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    verification = relationship("Verification", back_populates="face_results")


class RiskAssessment(Base):
    """Risk scoring assessment."""
    __tablename__ = "risk_assessments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    verification_id = Column(String(36), ForeignKey("verifications.id"), nullable=False)
    
    final_score = Column(Float, nullable=False)
    risk_level = Column(Enum(RiskLevel), nullable=False)
    component_scores = Column(JSON, nullable=False)
    contributing_factors = Column(JSON, nullable=False)
    graph_risk_score = Column(Float, nullable=True)
    recommendations = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    verification = relationship("Verification", back_populates="risk_assessments")


class FraudNode(Base):
    """Graph node for fraud relationships (stored in PG as well for fast lookup)."""
    __tablename__ = "fraud_nodes"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    node_type = Column(String(50), nullable=False)  # user, device, ip, face_embedding
    node_value = Column(String(500), nullable=False, index=True)
    metadata = Column(JSON, nullable=True)
    risk_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_fraud_node_type_value", "node_type", "node_value", unique=True),
    )


class FraudEdge(Base):
    """Graph edges for fraud relationships."""
    __tablename__ = "fraud_edges"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_node_id = Column(String(36), ForeignKey("fraud_nodes.id"), nullable=False)
    target_node_id = Column(String(36), ForeignKey("fraud_nodes.id"), nullable=False)
    relationship = Column(String(50), nullable=False)  # uses, shares, belongs_to
    weight = Column(Float, default=1.0)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_fraud_edge_source", "source_node_id"),
        Index("idx_fraud_edge_target", "target_node_id"),
    )


class BlockchainLog(Base):
    """Blockchain hash log for tamper-proof audit trail."""
    __tablename__ = "blockchain_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    verification_id = Column(String(36), ForeignKey("verifications.id"), nullable=False)
    content_hash = Column(String(64), nullable=False)
    result_hash = Column(String(64), nullable=False)
    block_number = Column(Integer, nullable=True)
    tx_hash = Column(String(66), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    chain_data = Column(JSON, nullable=True)
    
    __table_args__ = (
        Index("idx_blockchain_verification", "verification_id"),
    )


# Database setup
DATABASE_URL = None  # Set at runtime


def get_engine(database_url: str = None):
    global DATABASE_URL
    if database_url:
        DATABASE_URL = database_url
    url = DATABASE_URL or "sqlite:///./verishield.db"
    return create_engine(url, echo=False, pool_pre_ping=True)


def get_session_factory(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(engine):
    Base.metadata.create_all(bind=engine)
