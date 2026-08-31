"""
Central configuration for the verification system.
Uses pydantic-settings for validation and .env support.
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "VeriShield - AI Fraud & Deepfake Detection"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    SECRET_KEY: str = "verishield-dev-secret-key-change-in-production"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Database
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "verishield"
    POSTGRES_USER: str = "verishield"
    POSTGRES_PASSWORD: str = "verishield_secret"
    
    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "verishield_neo4j"
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # ML Models
    DEEPFAKE_MODEL_PATH: str = "ml/models/deepfake/deepfake_detector.pth"
    FACE_MODEL_PATH: str = "ml/models/face/face_embedder.pth"
    DOCUMENT_MODEL_PATH: str = "ml/models/document/doc_analyzer.pth"
    
    # Thresholds
    DEEPFAKE_THRESHOLD: float = 0.5
    FACE_MATCH_THRESHOLD: float = 0.6
    DOC_AUTHENTICITY_THRESHOLD: float = 0.5
    HIGH_RISK_THRESHOLD: int = 40
    MEDIUM_RISK_THRESHOLD: int = 70
    
    # Processing
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_IMAGE_TYPES: list[str] = ["image/jpeg", "image/png", "image/tiff", "image/bmp"]
    ALLOWED_DOC_TYPES: list[str] = ["application/pdf", "image/jpeg", "image/png"]
    ALLOWED_VIDEO_TYPES: list[str] = ["video/mp4", "video/avi", "video/mov"]
    
    # Blockchain
    BLOCKCHAIN_NETWORK: str = "local"
    BLOCKCHAIN_NODE_URL: str = "http://localhost:8545"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()


def get_settings() -> Settings:
    return settings
