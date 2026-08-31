# VeriShield - AI-Based Fraud / Deepfake / Document Verification System

A scalable, modular, real-time AI verification pipeline that detects fraud, deepfakes, and document tampering with explainable outputs.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ Upload   │ │ Results  │ │ Trust    │ │ Fraud Graph      │   │
│  │ Panel    │ │ Panel    │ │ Gauge    │ │ Visualization    │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────────┬─────────┘   │
│       └─────────────┴────────────┴────────────────┘             │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/REST
┌────────────────────────────┴────────────────────────────────────┐
│                    API GATEWAY (FastAPI)                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  POST /verify/document  | POST /verify/deepfake         │    │
│  │  POST /verify/face      | POST /verify/full             │    │
│  │  POST /verify/batch     | GET  /graph/data              │    │
│  │  GET  /blockchain/verify| GET  /verifications           │    │
│  └────────────────────────────┬────────────────────────────┘    │
│                               │                                  │
│  ┌────────────────────────────┴────────────────────────────┐    │
│  │              PREPROCESSING ENGINE                        │    │
│  │  • File type detection    • EXIF extraction              │    │
│  │  • Image resize/normalize • Frame sampling (video)       │    │
│  └────────────────────────────┬────────────────────────────┘    │
└───────────────────────────────┼──────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────┴────────┐  ┌──────────┴──────────┐  ┌────────┴────────┐
│  DOCUMENT      │  │  DEEPFAKE           │  │  FACE           │
│  INTELLIGENCE  │  │  DETECTION          │  │  MATCHING       │
│  SERVICE       │  │  SERVICE            │  │  SERVICE        │
│                │  │                     │  │                  │
│  • OCR         │  │  • CNN Classifier   │  │  • ArcFace       │
│  • Font Check  │  │  • Artifact Detect  │  │  • Embeddings    │
│  • Tamper Det  │  │  • Blink Analysis   │  │  • Cosine Sim    │
│  • Metadata    │  │  • GAN Fingerprint  │  │  • Quality Chk   │
└───────┬────────┘  └──────────┬──────────┘  └────────┬────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │       RISK SCORING ENGINE      │
                │  Weighted Trust Score (0-100)  │
                │  Component Scores + XAI        │
                └───────────────┬───────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
  ┌─────────┴────────┐ ┌───────┴──────┐ ┌─────────┴────────┐
  │  FRAUD GRAPH     │ │ BLOCKCHAIN   │ │  POSTGRESQL      │
  │  (In-Memory /    │ │ LOGGER       │ │  (Structured     │
  │   Neo4j)         │ │ (Audit Trail)│ │   Data)          │
  └──────────────────┘ └──────────────┘ └──────────────────┘
```

---

## Folder Structure

```
verishield/
├── backend/
│   ├── main.py                          # FastAPI application
│   ├── requirements.txt                 # Python dependencies
│   ├── Dockerfile
│   ├── config/
│   │   └── settings.py                  # Application settings
│   ├── core/
│   │   ├── preprocessor.py              # Input preprocessing
│   │   └── ml_models.py                 # PyTorch model architectures
│   ├── models/
│   │   ├── database.py                  # SQLAlchemy models
│   │   └── schemas.py                   # Pydantic schemas
│   ├── services/
│   │   ├── document_intelligence/       # OCR & tampering detection
│   │   │   └── service.py
│   │   ├── deepfake_detection/          # CNN deepfake classifier
│   │   │   └── service.py
│   │   ├── face_matching/               # ArcFace embeddings
│   │   │   └── service.py
│   │   ├── risk_scoring/                # Trust score computation
│   │   │   └── service.py
│   │   ├── fraud_graph/                 # Relationship graph
│   │   │   └── service.py
│   │   └── blockchain_logger/           # Tamper-proof audit log
│   │       └── service.py
│   └── db/
│       └── migrations/
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── Dockerfile
│   ├── nginx.conf
│   └── src/
│       ├── main.tsx
│       ├── App.tsx                      # Main application
│       ├── styles/globals.css
│       ├── services/api.ts              # API client
│       └── components/
│           ├── UploadPanel.tsx           # File upload UI
│           ├── TrustScoreGauge.tsx       # Score gauge visualization
│           ├── ResultsPanel.tsx          # Detailed results display
│           ├── FraudGraph.tsx            # Graph visualization
│           ├── StatsBar.tsx              # System statistics bar
│           └── VerificationHistory.tsx   # History table
├── ml/
│   └── models/                          # Pre-trained model weights
├── tests/
│   └── test_api.py                      # API test suite
├── scripts/
│   └── start_dev.sh                     # Dev startup script
├── docker/
│   └── nginx/
├── docker-compose.yml
└── README.md
```

---

## Quick Start

### Option 1: Local Development

```bash
# 1. Install Python dependencies
pip install -r backend/requirements.txt

# 2. Install frontend dependencies
cd frontend && npm install && cd ..

# 3. Start the backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Start the frontend (new terminal)
cd frontend && npm run dev
```

### Option 2: Docker Compose

```bash
docker-compose up --build
```

### Access Points

- **Frontend Dashboard:** http://localhost:5173 (dev) or http://localhost:3000 (docker)
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs
- **Neo4j Browser:** http://localhost:7474
- **PostgreSQL:** localhost:5432

---

## API Reference

### Verification Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/verify/document` | POST | Document authenticity analysis |
| `/verify/deepfake` | POST | Deepfake detection |
| `/verify/face` | POST | Face matching |
| `/verify/full` | POST | Full pipeline (all services) |
| `/verify/batch` | POST | Batch verification |

### System Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/stats` | GET | System statistics |
| `/graph/data` | GET | Fraud graph data |
| `/graph/suspicious` | GET | Suspicious clusters |
| `/blockchain/verify` | GET | Verify blockchain integrity |
| `/verifications` | GET | Verification history |

### Response Format

```json
{
  "verification_id": "uuid",
  "status": "COMPLETED",
  "file_type": "DOCUMENT",
  "trust_score": 82,
  "risk_level": "LOW",
  "confidence": 0.91,
  "reasons": ["Document authenticity verified"],
  "detailed_results": {
    "document_analysis": { ... },
    "deepfake_analysis": { ... },
    "face_analysis": { ... },
    "risk_assessment": {
      "trust_score": 82,
      "component_scores": { ... },
      "contributing_factors": [ ... ],
      "recommendations": [ ... ]
    }
  },
  "processing_time_ms": 1520,
  "blockchain_tx_hash": "0x..."
}
```

---

## Running Tests

```bash
# Start the backend first, then:
python tests/test_api.py
```

---

## Scalability Design

### Current Architecture
- **Microservices** pattern with independent service modules
- **Async processing** via FastAPI's async/await
- **In-memory caching** for graph lookups
- **SQLite** for development (swap to PostgreSQL)

### Scaling to Millions of Verifications

1. **Horizontal Scaling:**
   - Deploy multiple backend instances behind a load balancer
   - Use Docker Swarm or Kubernetes for orchestration
   - Each service is stateless and independently scalable

2. **Message Queue (Kafka):**
   - Decouple upload from processing
   - Enable asynchronous, non-blocking verification
   - Buffer burst traffic

3. **GPU Acceleration:**
   - Deploy ML models on GPU nodes
   - Use TensorRT for inference optimization
   - Model batching for throughput

4. **Caching (Redis):**
   - Cache face embeddings (avoid re-computation)
   - Cache document hashes (skip re-analysis)
   - Session and rate-limit tracking

5. **Database Sharding:**
   - Shard PostgreSQL by user_id or region
   - Use read replicas for query distribution
   - Archive old verifications to cold storage

6. **CDN + Edge:**
   - Serve frontend via CDN
   - Edge compute for pre-processing
   - Geographic distribution

### Target Metrics
- **Latency:** < 2s per verification (p95)
- **Throughput:** 1000+ concurrent verifications
- **Availability:** 99.9% uptime
- **Storage:** TB-scale with archival policy

---

## Demo Walkthrough

1. **Open the dashboard** at http://localhost:5173
2. **Upload a document** - drag & drop an image or PDF
3. **Select verification mode** - Full Pipeline, Document, Deepfake, or Face Match
4. **Click "Run Verification"** - watch the pipeline execute
5. **Review results** - Trust Score, risk classification, detailed analysis
6. **Check the History tab** - see all past verifications
7. **Explore the Graph tab** - visualize fraud relationships
8. **Verify blockchain** - confirm tamper-proof audit trail

---

## Tech Stack

- **Backend:** Python, FastAPI, PyTorch, SQLAlchemy
- **Frontend:** React 18, TypeScript, Tailwind CSS, Vite
- **ML:** Custom CNN classifiers, ArcFace embeddings, frequency analysis
- **Database:** PostgreSQL, SQLite (dev), Neo4j (graph)
- **Infra:** Docker, Nginx, Redis
- **Audit:** Local blockchain with proof-of-work

---

## License

MIT
