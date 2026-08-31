"""
VeriShield - AI-Based Fraud / Deepfake / Document Verification System
Main FastAPI Application
"""
import os
import sys
import time
import hashlib
import uuid
import json
import asyncio
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure backend is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config.settings import settings
from backend.core.preprocessor import Preprocessor
from backend.services.document_intelligence.service import DocumentIntelligenceService
from backend.services.deepfake_detection.service import DeepfakeDetectionService
from backend.services.face_matching.service import FaceMatchingService
from backend.services.risk_scoring.service import RiskScoringEngine
from backend.services.fraud_graph.service import FraudGraphService
from backend.services.blockchain_logger.service import BlockchainLogger
from backend.services.alerts.service import alert_manager
from backend.services.adversarial.service import AdversarialAttacks, RobustnessEvaluator
from backend.services.liveness.service import LivenessDetectionService

# ── Service Instances ───────────────────────────────────────────────────
preprocessor = Preprocessor()
doc_service = DocumentIntelligenceService()
deepfake_service = DeepfakeDetectionService()
face_service = FaceMatchingService()
risk_engine = RiskScoringEngine()
fraud_graph = FraudGraphService()
blockchain = BlockchainLogger()
liveness_service = LivenessDetectionService()

# In-memory verification store (replace with PostgreSQL in production)
verification_store = {}

START_TIME = time.time()


async def _broadcast_alerts(verification_id: str, result: dict):
    """Generate and broadcast fraud alerts for a completed verification."""
    alerts = alert_manager.generate_alerts_for_result(verification_id, result)
    for alert in alerts:
        await alert_manager.broadcast(alert)
    if alerts:
        # Also broadcast updated stats
        total = len(verification_store)
        completed = sum(1 for v in verification_store.values() if v.get('status') == 'COMPLETED')
        high_risk = sum(1 for v in verification_store.values() if v.get('risk_level') in ('HIGH', 'CRITICAL'))
        await alert_manager.send_stats_update({
            'total_verifications': total,
            'completed': completed,
            'high_risk_detected': high_risk,
            'avg_trust_score': _avg_trust_score(),
            'graph_nodes': len(fraud_graph.nodes),
            'blockchain_blocks': len(blockchain.chain),
        })


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    print(f"[VeriShield] Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"[VeriShield] Services initialized: Doc Intel, Deepfake, Face Match, Risk Scoring, Fraud Graph, Blockchain")
    yield
    print("[VeriShield] Shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-Based Fraud / Deepfake / Document Verification System",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health & Status ─────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "services": {
            "document_intelligence": "active",
            "deepfake_detection": "active",
            "face_matching": "active",
            "risk_scoring": "active",
            "fraud_graph": "active",
            "blockchain_logger": "active",
            "alert_websocket": "active",
        },
        "connected_clients": len(alert_manager.active_connections),
        "uptime_seconds": round(time.time() - START_TIME, 1),
    }


@app.get("/api/stats")
async def get_stats():
    """Return system statistics."""
    total = len(verification_store)
    completed = sum(1 for v in verification_store.values() if v.get('status') == 'COMPLETED')
    high_risk = sum(1 for v in verification_store.values() if v.get('risk_level') in ('HIGH', 'CRITICAL'))

    return {
        "total_verifications": total,
        "completed": completed,
        "high_risk_detected": high_risk,
        "avg_trust_score": _avg_trust_score(),
        "graph_nodes": len(fraud_graph.nodes),
        "blockchain_blocks": len(blockchain.chain),
    }


def _avg_trust_score() -> float:
    scores = [v['trust_score'] for v in verification_store.values() if v.get('trust_score') is not None]
    return round(sum(scores) / len(scores), 1) if scores else 0.0


# ── Verification Endpoints ──────────────────────────────────────────────

@app.post("/verify/document")
async def verify_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    ip_address: Optional[str] = Form(None),
):
    """Verify a document for authenticity, tampering, and fraud indicators."""
    start = time.time()
    verification_id = str(uuid.uuid4())

    # Validate file
    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, "File too large")

    file_hash = preprocessor.compute_file_hash(content)

    # Store verification record
    verification_store[verification_id] = {
        'id': verification_id,
        'status': 'PROCESSING',
        'file_type': 'DOCUMENT',
        'file_name': file.filename,
        'file_hash': file_hash,
        'created_at': datetime.utcnow().isoformat(),
    }

    try:
        # Preprocess
        preprocessed = preprocessor.preprocess_document(content, file.filename)

        # Document Intelligence Analysis
        doc_results = await doc_service.analyze(preprocessed)

        # Risk scoring (no deepfake/face for documents)
        graph_risk = await fraud_graph.lookup_risk(user_id=user_id, ip_address=ip_address)
        risk = risk_engine.compute_risk_score(
            document_results=doc_results,
            graph_risk_score=graph_risk,
        )

        # Add to fraud graph
        graph_result = await fraud_graph.add_verification_to_graph(
            verification_id=verification_id,
            user_id=user_id,
            ip_address=ip_address,
        )

        # Blockchain log
        result_hash = hashlib.sha256(json.dumps({
            'verification_id': verification_id,
            'trust_score': risk['trust_score'],
            'doc_score': doc_results['authenticity_score'],
        }, default=str).encode()).hexdigest()

        bc_result = await blockchain.log_verification(
            verification_id=verification_id,
            content_hash=file_hash,
            result_hash=result_hash,
        )

        processing_time = int((time.time() - start) * 1000)

        # Update store
        verification_store[verification_id].update({
            'status': 'COMPLETED',
            'trust_score': risk['trust_score'],
            'risk_level': risk['risk_level'],
            'confidence': risk['confidence'],
            'reasons': [f['description'] for f in risk['contributing_factors']],
            'detailed_results': {
                'document': doc_results,
                'risk_assessment': risk,
                'graph': graph_result,
            },
            'processing_time_ms': processing_time,
            'blockchain_tx_hash': bc_result.get('tx_hash'),
        })

        await _broadcast_alerts(verification_id, verification_store[verification_id])

        return {
            "verification_id": verification_id,
            "status": "COMPLETED",
            "file_type": "DOCUMENT",
            "trust_score": risk['trust_score'],
            "risk_level": risk['risk_level'],
            "confidence": risk['confidence'],
            "reasons": [f['description'] for f in risk['contributing_factors']],
            "detailed_results": {
                "document_analysis": {
                    "authenticity_score": doc_results['authenticity_score'],
                    "tampering_detected": doc_results['tampering_detected'],
                    "font_inconsistencies": len(doc_results['font_inconsistencies']),
                    "tampered_regions": len(doc_results['tampered_regions']),
                    "spacing_anomalies": len(doc_results['spacing_anomalies']),
                    "metadata_flags": len(doc_results['metadata_analysis'].get('suspicious_indicators', [])),
                },
                "risk_assessment": risk,
            },
            "processing_time_ms": processing_time,
            "blockchain_tx_hash": bc_result.get('tx_hash'),
            "created_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        verification_store[verification_id]['status'] = 'FAILED'
        raise HTTPException(500, f"Verification failed: {str(e)}")


@app.post("/verify/deepfake")
async def verify_deepfake(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    ip_address: Optional[str] = Form(None),
):
    """Detect deepfakes in images or videos."""
    start = time.time()
    verification_id = str(uuid.uuid4())

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, "File too large")

    file_hash = preprocessor.compute_file_hash(content)
    file_type = preprocessor.detect_file_type(file.filename, file.content_type or '')

    verification_store[verification_id] = {
        'id': verification_id,
        'status': 'PROCESSING',
        'file_type': file_type,
        'file_name': file.filename,
        'file_hash': file_hash,
        'created_at': datetime.utcnow().isoformat(),
    }

    try:
        # Preprocess
        if file_type == 'VIDEO':
            preprocessed = preprocessor.preprocess_video(content)
            preprocessed['document_type'] = 'video'
        else:
            preprocessed = preprocessor.preprocess_image(content)
            preprocessed['document_type'] = 'single_frame'

        # Deepfake detection
        deepfake_results = await deepfake_service.analyze(preprocessed)

        # Risk scoring
        graph_risk = await fraud_graph.lookup_risk(user_id=user_id, ip_address=ip_address)
        risk = risk_engine.compute_risk_score(
            deepfake_results=deepfake_results,
            graph_risk_score=graph_risk,
        )

        # Graph and blockchain
        graph_result = await fraud_graph.add_verification_to_graph(
            verification_id=verification_id,
            user_id=user_id,
            ip_address=ip_address,
        )

        result_hash = hashlib.sha256(json.dumps({
            'verification_id': verification_id,
            'trust_score': risk['trust_score'],
            'deepfake_prob': deepfake_results['deepfake_probability'],
        }, default=str).encode()).hexdigest()

        bc_result = await blockchain.log_verification(
            verification_id=verification_id,
            content_hash=file_hash,
            result_hash=result_hash,
        )

        processing_time = int((time.time() - start) * 1000)

        verification_store[verification_id].update({
            'status': 'COMPLETED',
            'trust_score': risk['trust_score'],
            'risk_level': risk['risk_level'],
            'confidence': risk['confidence'],
            'reasons': [f['description'] for f in risk['contributing_factors']],
            'processing_time_ms': processing_time,
        })

        await _broadcast_alerts(verification_id, verification_store[verification_id])

        return {
            "verification_id": verification_id,
            "status": "COMPLETED",
            "file_type": file_type,
            "trust_score": risk['trust_score'],
            "risk_level": risk['risk_level'],
            "confidence": risk['confidence'],
            "reasons": [f['description'] for f in risk['contributing_factors']],
            "detailed_results": {
                "deepfake_analysis": {
                    "is_deepfake": deepfake_results['is_deepfake'],
                    "probability": deepfake_results['deepfake_probability'],
                    "face_artifacts": deepfake_results['face_artifacts'],
                    "blinking_anomaly": deepfake_results['blinking_anomaly'],
                    "gan_fingerprint": deepfake_results['gan_fingerprint'],
                    "confidence": deepfake_results['confidence'],
                    "frame_count": len(deepfake_results.get('frame_analysis') or []),
                },
                "risk_assessment": risk,
            },
            "processing_time_ms": processing_time,
            "blockchain_tx_hash": bc_result.get('tx_hash'),
            "created_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        verification_store[verification_id]['status'] = 'FAILED'
        raise HTTPException(500, f"Deepfake verification failed: {str(e)}")


@app.post("/verify/face")
async def verify_face(
    file: UploadFile = File(...),
    reference_file: Optional[UploadFile] = File(None),
    user_id: Optional[str] = Form(None),
    ip_address: Optional[str] = Form(None),
):
    """Face matching: compare probe image against reference."""
    start = time.time()
    verification_id = str(uuid.uuid4())

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, "File too large")

    file_hash = preprocessor.compute_file_hash(content)

    verification_store[verification_id] = {
        'id': verification_id,
        'status': 'PROCESSING',
        'file_type': 'IMAGE',
        'file_name': file.filename,
        'file_hash': file_hash,
        'created_at': datetime.utcnow().isoformat(),
    }

    try:
        # Preprocess probe
        probe_data = preprocessor.preprocess_image(content)

        # Preprocess reference if provided
        ref_data = None
        if reference_file:
            ref_content = await reference_file.read()
            ref_data = preprocessor.preprocess_image(ref_content)

        # Face matching
        face_results = await face_service.analyze(probe_data, ref_data)

        # Risk scoring
        graph_risk = await fraud_graph.lookup_risk(user_id=user_id, ip_address=ip_address)
        risk = risk_engine.compute_risk_score(
            face_match_results=face_results,
            graph_risk_score=graph_risk,
        )

        # Graph and blockchain
        graph_result = await fraud_graph.add_verification_to_graph(
            verification_id=verification_id,
            user_id=user_id,
            ip_address=ip_address,
        )

        result_hash = hashlib.sha256(json.dumps({
            'verification_id': verification_id,
            'trust_score': risk['trust_score'],
            'match_score': face_results.get('match_score'),
        }, default=str).encode()).hexdigest()

        bc_result = await blockchain.log_verification(
            verification_id=verification_id,
            content_hash=file_hash,
            result_hash=result_hash,
        )

        processing_time = int((time.time() - start) * 1000)

        verification_store[verification_id].update({
            'status': 'COMPLETED',
            'trust_score': risk['trust_score'],
            'risk_level': risk['risk_level'],
            'confidence': risk['confidence'],
            'processing_time_ms': processing_time,
        })

        await _broadcast_alerts(verification_id, verification_store[verification_id])

        return {
            "verification_id": verification_id,
            "status": "COMPLETED",
            "file_type": "IMAGE",
            "trust_score": risk['trust_score'],
            "risk_level": risk['risk_level'],
            "confidence": risk['confidence'],
            "reasons": [f['description'] for f in risk['contributing_factors']],
            "detailed_results": {
                "face_analysis": {
                    "faces_detected": face_results['faces_detected'],
                    "match_score": face_results['match_score'],
                    "is_match": face_results['is_match'],
                    "quality_score": face_results['quality_score'],
                },
                "risk_assessment": risk,
            },
            "processing_time_ms": processing_time,
            "blockchain_tx_hash": bc_result.get('tx_hash'),
            "created_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        verification_store[verification_id]['status'] = 'FAILED'
        raise HTTPException(500, f"Face verification failed: {str(e)}")


@app.post("/verify/full")
async def full_verification(
    file: UploadFile = File(...),
    reference_file: Optional[UploadFile] = File(None),
    user_id: Optional[str] = Form(None),
    ip_address: Optional[str] = Form(None),
):
    """Full pipeline: document analysis + deepfake detection + face matching + risk scoring."""
    start = time.time()
    verification_id = str(uuid.uuid4())

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, "File too large")

    file_hash = preprocessor.compute_file_hash(content)
    file_type = preprocessor.detect_file_type(file.filename, file.content_type or '')

    verification_store[verification_id] = {
        'id': verification_id,
        'status': 'PROCESSING',
        'file_type': file_type,
        'file_name': file.filename,
        'file_hash': file_hash,
        'created_at': datetime.utcnow().isoformat(),
    }

    try:
        # Preprocess
        if file_type == 'VIDEO':
            preprocessed = preprocessor.preprocess_video(content)
            preprocessed['document_type'] = 'video'
        elif file_type == 'DOCUMENT':
            preprocessed = preprocessor.preprocess_document(content, file.filename)
        else:
            preprocessed = preprocessor.preprocess_image(content)
            preprocessed['document_type'] = 'single_frame'

        # Run all analyses
        doc_results = await doc_service.analyze(preprocessed)
        deepfake_results = await deepfake_service.analyze(preprocessed)

        ref_data = None
        if reference_file:
            ref_content = await reference_file.read()
            ref_data = preprocessor.preprocess_image(ref_content)

        probe_data = preprocessed if preprocessed.get('image_array') is not None else None
        face_results = await face_service.analyze(probe_data, ref_data) if probe_data else None

        # Risk scoring
        graph_risk = await fraud_graph.lookup_risk(user_id=user_id, ip_address=ip_address)
        risk = risk_engine.compute_risk_score(
            document_results=doc_results,
            deepfake_results=deepfake_results,
            face_match_results=face_results,
            graph_risk_score=graph_risk,
        )

        # Graph and blockchain
        graph_result = await fraud_graph.add_verification_to_graph(
            verification_id=verification_id,
            user_id=user_id,
            ip_address=ip_address,
        )

        result_hash = hashlib.sha256(json.dumps({
            'verification_id': verification_id,
            'trust_score': risk['trust_score'],
        }, default=str).encode()).hexdigest()

        bc_result = await blockchain.log_verification(
            verification_id=verification_id,
            content_hash=file_hash,
            result_hash=result_hash,
        )

        processing_time = int((time.time() - start) * 1000)

        verification_store[verification_id].update({
            'status': 'COMPLETED',
            'trust_score': risk['trust_score'],
            'risk_level': risk['risk_level'],
            'confidence': risk['confidence'],
            'reasons': [f['description'] for f in risk['contributing_factors']],
            'detailed_results': {
                'document': doc_results,
                'deepfake': deepfake_results,
                'face': face_results,
                'risk': risk,
            },
            'processing_time_ms': processing_time,
            'blockchain_tx_hash': bc_result.get('tx_hash'),
        })

        await _broadcast_alerts(verification_id, verification_store[verification_id])

        return {
            "verification_id": verification_id,
            "status": "COMPLETED",
            "file_type": file_type,
            "trust_score": risk['trust_score'],
            "risk_level": risk['risk_level'],
            "confidence": risk['confidence'],
            "reasons": [f['description'] for f in risk['contributing_factors']],
            "detailed_results": {
                "document_analysis": doc_results if doc_results else None,
                "deepfake_analysis": deepfake_results if deepfake_results else None,
                "face_analysis": face_results.__dict__ if face_results else None,
                "risk_assessment": risk,
            },
            "processing_time_ms": processing_time,
            "blockchain_tx_hash": bc_result.get('tx_hash'),
            "created_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        verification_store[verification_id]['status'] = 'FAILED'
        raise HTTPException(500, f"Full verification failed: {str(e)}")


# ── Batch Verification ─────────────────────────────────────────────────

@app.post("/verify/batch")
async def batch_verification(
    files: list[UploadFile] = File(...),
    user_id: Optional[str] = Form(None),
):
    """Batch verify multiple files."""
    batch_id = str(uuid.uuid4())
    results = []

    for file in files:
        try:
            # Quick inline verification
            content = await file.read()
            file_hash = preprocessor.compute_file_hash(content)
            file_type = preprocessor.detect_file_type(file.filename, file.content_type or '')

            vid = str(uuid.uuid4())

            if file_type == 'VIDEO':
                preprocessed = preprocessor.preprocess_video(content)
                preprocessed['document_type'] = 'video'
            else:
                preprocessed = preprocessor.preprocess_image(content)
                preprocessed['document_type'] = 'single_frame'

            deepfake_results = await deepfake_service.analyze(preprocessed)
            doc_results = await doc_service.analyze(preprocessed)
            graph_risk = await fraud_graph.lookup_risk(user_id=user_id)
            risk = risk_engine.compute_risk_score(
                document_results=doc_results,
                deepfake_results=deepfake_results,
                graph_risk_score=graph_risk,
            )

            await fraud_graph.add_verification_to_graph(
                verification_id=vid, user_id=user_id,
            )

            result_hash = hashlib.sha256(json.dumps({
                'vid': vid, 'trust_score': risk['trust_score'],
            }, default=str).encode()).hexdigest()
            bc_result = await blockchain.log_verification(
                vid, file_hash, result_hash,
            )

            results.append({
                "verification_id": vid,
                "file_name": file.filename,
                "file_type": file_type,
                "trust_score": risk['trust_score'],
                "risk_level": risk['risk_level'],
                "confidence": risk['confidence'],
                "blockchain_tx_hash": bc_result.get('tx_hash'),
            })
        except Exception as e:
            results.append({
                "file_name": file.filename,
                "error": str(e),
            })

    high_risk = sum(1 for r in results if r.get('risk_level') in ('HIGH', 'CRITICAL'))

    return {
        "batch_id": batch_id,
        "total_items": len(results),
        "results": results,
        "summary": {
            "high_risk_count": high_risk,
            "avg_trust_score": round(
                sum(r.get('trust_score', 0) for r in results if 'trust_score' in r) /
                max(len([r for r in results if 'trust_score' in r]), 1), 1
            ),
        },
    }


# ── Liveness Detection ────────────────────────────────────────────────

@app.post("/verify/liveness")
async def verify_liveness(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
):
    """
    Run liveness detection on an image or video.
    Detects photo/screen replay attacks via head-pose, depth, and texture analysis.
    """
    start = time.time()
    verification_id = str(uuid.uuid4())

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, "File too large")

    file_hash = preprocessor.compute_file_hash(content)
    file_type = preprocessor.detect_file_type(file.filename, file.content_type or '')

    verification_store[verification_id] = {
        'id': verification_id,
        'status': 'PROCESSING',
        'file_type': file_type,
        'file_name': file.filename,
        'file_hash': file_hash,
        'created_at': datetime.utcnow().isoformat(),
    }

    try:
        # Preprocess
        if file_type == 'VIDEO':
            preprocessed = preprocessor.preprocess_video(content)
            preprocessed['document_type'] = 'video'
        else:
            preprocessed = preprocessor.preprocess_image(content)
            preprocessed['document_type'] = 'single_frame'

        # Liveness analysis
        liveness_results = await liveness_service.analyze(preprocessed)

        # Risk scoring
        graph_risk = await fraud_graph.lookup_risk(user_id=user_id)
        
        # Compute trust score based on liveness
        liveness_trust = liveness_results['liveness_score'] * 100
        risk = risk_engine.compute_risk_score(
            document_results={
                'authenticity_score': liveness_results['liveness_score'],
                'tampering_detected': not liveness_results.get('is_live', True),
                'font_inconsistencies': [],
                'spacing_anomalies': [],
                'tampered_regions': [],
                'metadata_analysis': {},
            } if not liveness_results.get('is_live', True) else None,
            deepfake_results={
                'is_deepfake': not liveness_results.get('is_live', True),
                'deepfake_probability': 1.0 - liveness_results['liveness_score'],
                'face_artifacts': False,
                'blinking_anomaly': False,
                'gan_fingerprint': False,
                'confidence': liveness_results['confidence'],
            },
            graph_risk_score=graph_risk,
        )

        # Add to fraud graph
        graph_result = await fraud_graph.add_verification_to_graph(
            verification_id=verification_id,
            user_id=user_id,
        )

        # Blockchain log
        result_hash = hashlib.sha256(json.dumps({
            'verification_id': verification_id,
            'trust_score': risk['trust_score'],
            'liveness_score': liveness_results['liveness_score'],
        }, default=str).encode()).hexdigest()

        bc_result = await blockchain.log_verification(
            verification_id=verification_id,
            content_hash=file_hash,
            result_hash=result_hash,
        )

        processing_time = int((time.time() - start) * 1000)

        verification_store[verification_id].update({
            'status': 'COMPLETED',
            'trust_score': risk['trust_score'],
            'risk_level': risk['risk_level'],
            'confidence': risk['confidence'],
            'reasons': [f['description'] for f in risk['contributing_factors']],
            'processing_time_ms': processing_time,
        })

        await _broadcast_alerts(verification_id, verification_store[verification_id])

        return {
            'verification_id': verification_id,
            'status': 'COMPLETED',
            'file_type': file_type,
            'trust_score': risk['trust_score'],
            'risk_level': risk['risk_level'],
            'confidence': risk['confidence'],
            'reasons': [f['description'] for f in risk['contributing_factors']],
            'detailed_results': {
                'liveness_analysis': {
                    'is_live': liveness_results['is_live'],
                    'liveness_score': liveness_results['liveness_score'],
                    'confidence': liveness_results['confidence'],
                    'head_pose': liveness_results['head_pose'],
                    'depth_analysis': liveness_results['depth_analysis'],
                    'texture_analysis': liveness_results['texture_analysis'],
                    'spoof_type': liveness_results.get('spoof_type'),
                    'frame_count': len(liveness_results.get('frame_analysis') or []),
                    'temporal_consistency': liveness_results.get('temporal_consistency'),
                },
                'risk_assessment': risk,
            },
            'processing_time_ms': processing_time,
            'blockchain_tx_hash': bc_result.get('tx_hash'),
            'created_at': datetime.utcnow().isoformat(),
        })

    except Exception as e:
        verification_store[verification_id]['status'] = 'FAILED'
        raise HTTPException(500, f"Liveness verification failed: {str(e)}")


# ── Adversarial Robustness Testing ────────────────────────────────────

@app.post("/test/adversarial")
async def test_adversarial(
    file: UploadFile = File(...),
    epsilon: float = Form(0.03),
    test_all: bool = Form(True),
):
    """
    Run adversarial robustness tests on a model with the uploaded image.
    Tests FGSM, PGD, noise, spatial, brightness/contrast, and compression attacks.
    """
    import torch as _torch
    import numpy as _np

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, "File too large")

    # Preprocess
    preprocessed = preprocessor.preprocess_image(content)
    image_tensor = _torch.FloatTensor(preprocessed['image_array']).unsqueeze(0)

    # Use deepfake model for adversarial testing (most relevant)
    model = deepfake_service.model
    evaluator = RobustnessEvaluator(model)

    # Create a dummy label (use model's prediction as the 'true' label)
    with _torch.no_grad():
        output = model(image_tensor)
        label = output['logits'].argmax(dim=1)

    # Run full evaluation
    epsilons = [0.01, 0.03, 0.05, 0.1, 0.2]
    if not test_all:
        epsilons = [epsilon]

    report = evaluator.evaluate(image_tensor, label, epsilons=epsilons)

    # Convert tensors to serializable format for perturbation visualization
    perturbation_data = _generate_perturbation_visualization(
        model, image_tensor, label, epsilons[:3],
    )

    return {
        'overall_robustness_score': report['overall_robustness_score'],
        'fgsm_results': report['fgsm_results'],
        'pgd_results': report['pgd_results'],
        'noise_results': report['noise_results'],
        'spatial_results': report['spatial_results'],
        'brightness_results': report['brightness_results'],
        'compression_results': report['compression_results'],
        'vulnerabilities': report['vulnerabilities'],
        'recommendations': report['recommendations'],
        'perturbation_visualizations': perturbation_data,
        'original_prediction': label.item(),
        'model_type': 'deepfake_cnn',
    }


def _generate_perturbation_visualization(
    model, image_tensor, label, epsilons, max_show: int = 3,
) -> list[dict]:
    """Generate perturbation visualization data for frontend display."""
    import torch as _torch
    import numpy as _np

    vis_data = []
    original_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    with _torch.no_grad():
        orig_out = model(image_tensor)
        orig_pred = orig_out['logits'].argmax().item()
        orig_conf = orig_out['probability'].item()

    for eps in epsilons[:max_show]:
        adv_image, meta = AdversarialAttacks.fgsm(model, image_tensor, label, epsilon=eps)
        adv_np = adv_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        perturbation = _np.abs(adv_np - original_np)

        # Amplify perturbation for visibility (normalize to 0-1)
        if perturbation.max() > 0:
            perturbation_vis = perturbation / perturbation.max()
        else:
            perturbation_vis = perturbation

        vis_data.append({
            'epsilon': eps,
            'perturbation_magnitude': float(perturbation.mean()),
            'original_confidence': orig_conf,
            'adversarial_confidence': meta['adversarial_confidence'],
            'prediction_changed': meta['prediction_changed'],
            'l2_norm': meta['l2_perturbation'],
            'linf_norm': meta['linf_perturbation'],
        })

    return vis_data


@app.post("/test/adversarial/compare")
async def compare_models(
    file: UploadFile = File(...),
    epsilon: float = Form(0.03),
):
    """
    Compare adversarial robustness across all three models
    (deepfake, document, face) on the same input.
    """
    import torch as _torch

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, "File too large")

    preprocessed = preprocessor.preprocess_image(content)
    image_tensor = _torch.FloatTensor(preprocessed['image_array']).unsqueeze(0)

    comparisons = []

    # Test against deepfake model
    try:
        with _torch.no_grad():
            out = deepfake_service.model(image_tensor)
            label = out['logits'].argmax(dim=1)
        evaluator = RobustnessEvaluator(deepfake_service.model)
        report = evaluator.evaluate(image_tensor, label, epsilons=[epsilon])
        comparisons.append({
            'model': 'Deepfake CNN',
            'robustness_score': report['overall_robustness_score'],
            'vulnerability_count': len(report['vulnerabilities']),
            'fgsm_success_rate': report['fgsm_results'][0]['success_rate'] if report['fgsm_results'] else 0,
            'pgd_success_rate': report['pgd_results'][0]['success_rate'] if report['pgd_results'] else 0,
        })
    except Exception as e:
        comparisons.append({'model': 'Deepfake CNN', 'error': str(e)})

    # Test against document model
    try:
        with _torch.no_grad():
            out = doc_service.model(image_tensor) if hasattr(doc_service, 'model') and doc_service.model else None
            if out is not None:
                label = out['logits'].argmax(dim=1)
                evaluator = RobustnessEvaluator(doc_service.model)
                report = evaluator.evaluate(image_tensor, label, epsilons=[epsilon])
                comparisons.append({
                    'model': 'Document CNN',
                    'robustness_score': report['overall_robustness_score'],
                    'vulnerability_count': len(report['vulnerabilities']),
                    'fgsm_success_rate': report['fgsm_results'][0]['success_rate'] if report['fgsm_results'] else 0,
                    'pgd_success_rate': report['pgd_results'][0]['success_rate'] if report['pgd_results'] else 0,
                })
    except Exception as e:
        comparisons.append({'model': 'Document CNN', 'error': str(e)})

    # Test against face model
    try:
        with _torch.no_grad():
            out = face_service.model(image_tensor)
            # Face embedder returns embedding, not logits - skip classification test
            comparisons.append({
                'model': 'Face Embedder',
                'robustness_score': None,
                'note': 'Embedding model - tested via perturbation magnitude',
            })
    except Exception as e:
        comparisons.append({'model': 'Face Embedder', 'error': str(e)})

    return {
        'epsilon': epsilon,
        'comparisons': comparisons,
    }


# ── Graph Endpoints ─────────────────────────────────────────────────────

@app.get("/graph/data")
async def get_fraud_graph():
    """Get fraud graph data for visualization."""
    data = await fraud_graph.get_graph_data()
    return data


@app.get("/graph/suspicious")
async def get_suspicious_clusters():
    """Get suspicious clusters from fraud graph."""
    clusters = await fraud_graph.find_suspicious_clusters()
    return {"clusters": clusters}


# ── Blockchain Endpoints ────────────────────────────────────────────────

@app.get("/blockchain/verify")
async def verify_blockchain():
    """Verify blockchain integrity."""
    result = await blockchain.verify_integrity()
    return result


@app.get("/blockchain/record/{verification_id}")
async def get_blockchain_record(verification_id: str):
    """Get blockchain record for a verification."""
    record = await blockchain.get_verification_record(verification_id)
    if record is None:
        raise HTTPException(404, "Record not found")
    return record


# ── WebSocket Alerts ──────────────────────────────────────────────────

import uuid as _uuid

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time fraud alerts.
    
    Connects clients to a live stream of HIGH/CRITICAL risk alerts.
    On connect, receives recent alert history.
    During session, receives push notifications for new alerts.
    """
    client_id = str(_uuid.uuid4())[:8]
    try:
        await alert_manager.connect(websocket, client_id)
        print(f"[Alerts] Client {client_id} connected ({len(alert_manager.active_connections)} total)")
        # Keep connection alive, handle client messages (e.g. subscribe/filter)
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                msg = json.loads(data)
                if msg.get('type') == 'ping':
                    await websocket.send_json({'type': 'pong'})
            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await websocket.send_json({'type': 'heartbeat'})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[Alerts] Client {client_id} error: {e}")
    finally:
        await alert_manager.disconnect(client_id)
        print(f"[Alerts] Client {client_id} disconnected ({len(alert_manager.active_connections)} total)")


@app.get("/alerts/history")
async def get_alert_history(limit: int = 50):
    """Get recent fraud alert history."""
    return {
        "alerts": alert_manager.alert_history[-limit:],
        "total": len(alert_manager.alert_history),
        "connected_clients": len(alert_manager.active_connections),
    }


# ── History ─────────────────────────────────────────────────────────────

@app.get("/verifications")
async def list_verifications(
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    risk_level: Optional[str] = None,
):
    """List verification history."""
    items = list(verification_store.values())

    if status:
        items = [v for v in items if v.get('status') == status]
    if risk_level:
        items = [v for v in items if v.get('risk_level') == risk_level]

    items.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    total = len(items)
    items = items[offset:offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": items,
    }


@app.get("/verifications/{verification_id}")
async def get_verification(verification_id: str):
    """Get a specific verification result."""
    if verification_id not in verification_store:
        raise HTTPException(404, "Verification not found")
    return verification_store[verification_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
