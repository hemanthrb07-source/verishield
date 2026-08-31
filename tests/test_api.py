"""
Test script for VeriShield API endpoints.
Run with: python tests/test_api.py
"""
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("=" * 60)
    print("TEST: Health Check")
    print("=" * 60)
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=10)
        print(f"Status: {r.status_code}")
        data = r.json()
        print(f"App: {data.get('version')}")
        print(f"Services: {json.dumps(data.get('services', {}), indent=2)}")
        print(f"Uptime: {data.get('uptime_seconds')}s")
        assert r.status_code == 200
        print("✅ PASS\n")
    except Exception as e:
        print(f"❌ FAIL: {e}\n")


def test_stats():
    """Test stats endpoint."""
    print("=" * 60)
    print("TEST: System Stats")
    print("=" * 60)
    try:
        r = httpx.get(f"{BASE_URL}/api/stats", timeout=10)
        print(f"Status: {r.status_code}")
        data = r.json()
        print(f"Total verifications: {data.get('total_verifications')}")
        print(f"Completed: {data.get('completed')}")
        print(f"High risk detected: {data.get('high_risk_detected')}")
        assert r.status_code == 200
        print("✅ PASS\n")
    except Exception as e:
        print(f"❌ FAIL: {e}\n")


def test_document_verification():
    """Test document verification with a sample image."""
    print("=" * 60)
    print("TEST: Document Verification")
    print("=" * 60)
    
    # Create a simple test image
    from PIL import Image, ImageDraw, ImageFont
    import io
    
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 380, 280], outline='black', width=2)
    draw.text((100, 50), "SAMPLE DOCUMENT", fill='black')
    draw.text((100, 100), "ID: 12345678", fill='black')
    draw.text((100, 140), "Name: Test User", fill='black')
    draw.text((100, 180), "Date: 2024-01-01", fill='black')
    
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    
    try:
        r = httpx.post(
            f"{BASE_URL}/verify/document",
            files={"file": ("test_doc.png", buf, "image/png")},
            data={"user_id": "test_user_001"},
            timeout=30,
        )
        print(f"Status: {r.status_code}")
        data = r.json()
        print(f"Verification ID: {data.get('verification_id', '')[:8]}...")
        print(f"Trust Score: {data.get('trust_score')}")
        print(f"Risk Level: {data.get('risk_level')}")
        print(f"Confidence: {data.get('confidence')}")
        print(f"Processing Time: {data.get('processing_time_ms')}ms")
        print(f"Blockchain TX: {data.get('blockchain_tx_hash', '')[:16]}...")
        print(f"Reasons: {data.get('reasons', [])}")
        
        assert r.status_code == 200
        assert data.get('trust_score') is not None
        assert data.get('risk_level') is not None
        print("✅ PASS\n")
        return data
    except Exception as e:
        print(f"❌ FAIL: {e}\n")
        return None


def test_deepfake_detection():
    """Test deepfake detection with a sample image."""
    print("=" * 60)
    print("TEST: Deepfake Detection")
    print("=" * 60)
    
    from PIL import Image
    import io
    
    # Create a face-like image (simple approximation)
    img = Image.new('RGB', (224, 224), color=(180, 140, 120))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    # Simple face shape
    draw.ellipse([40, 20, 184, 200], fill=(200, 160, 130))
    draw.ellipse([70, 70, 95, 90], fill='white')  # Left eye
    draw.ellipse([129, 70, 154, 90], fill='white')  # Right eye
    draw.ellipse([75, 75, 90, 85], fill='black')  # Left pupil
    draw.ellipse([134, 75, 149, 85], fill='black')  # Right pupil
    draw.arc([80, 110, 144, 160], 0, 180, fill='black', width=2)  # Mouth
    
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    
    try:
        r = httpx.post(
            f"{BASE_URL}/verify/deepfake",
            files={"file": ("test_face.png", buf, "image/png")},
            data={"user_id": "test_user_002"},
            timeout=30,
        )
        print(f"Status: {r.status_code}")
        data = r.json()
        print(f"Trust Score: {data.get('trust_score')}")
        print(f"Risk Level: {data.get('risk_level')}")
        
        deepfake = data.get('detailed_results', {}).get('deepfake_analysis', {})
        print(f"Is Deepfake: {deepfake.get('is_deepfake')}")
        print(f"Probability: {deepfake.get('probability', 0):.3f}")
        print(f"Face Artifacts: {deepfake.get('face_artifacts')}")
        print(f"GAN Fingerprint: {deepfake.get('gan_fingerprint')}")
        
        assert r.status_code == 200
        print("✅ PASS\n")
        return data
    except Exception as e:
        print(f"❌ FAIL: {e}\n")
        return None


def test_full_verification():
    """Test the full verification pipeline."""
    print("=" * 60)
    print("TEST: Full Pipeline Verification")
    print("=" * 60)
    
    from PIL import Image, ImageDraw
    import io
    
    img = Image.new('RGB', (400, 400), color=(200, 170, 140))
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 50, 300, 350], fill=(220, 180, 150))
    draw.ellipse([140, 130, 175, 165], fill='white')
    draw.ellipse([225, 130, 260, 165], fill='white')
    draw.ellipse([150, 140, 165, 155], fill='black')
    draw.ellipse([235, 140, 250, 155], fill='black')
    draw.text((130, 200), "Sample", fill='black')
    draw.text((110, 230), "Document", fill='black')
    
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    
    try:
        r = httpx.post(
            f"{BASE_URL}/verify/full",
            files={"file": ("full_test.png", buf, "image/png")},
            data={"user_id": "test_user_003"},
            timeout=30,
        )
        print(f"Status: {r.status_code}")
        data = r.json()
        print(f"Verification ID: {data.get('verification_id', '')[:8]}...")
        print(f"Trust Score: {data.get('trust_score')}")
        print(f"Risk Level: {data.get('risk_level')}")
        print(f"Confidence: {data.get('confidence')}")
        print(f"Reasons: {json.dumps(data.get('reasons', []), indent=2)}")
        print(f"Processing Time: {data.get('processing_time_ms')}ms")
        
        assert r.status_code == 200
        print("✅ PASS\n")
        return data
    except Exception as e:
        print(f"❌ FAIL: {e}\n")
        return None


def test_batch_verification():
    """Test batch verification."""
    print("=" * 60)
    print("TEST: Batch Verification")
    print("=" * 60)
    
    from PIL import Image, ImageDraw
    import io
    
    files = []
    for i in range(3):
        img = Image.new('RGB', (200, 200), color=(150 + i*20, 130, 110))
        draw = ImageDraw.Draw(img)
        draw.text((50, 80), f"Batch {i+1}", fill='black')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        files.append(("files", (f"batch_{i+1}.png", buf, "image/png")))
    
    try:
        r = httpx.post(
            f"{BASE_URL}/verify/batch",
            files=files,
            data={"user_id": "test_user_batch"},
            timeout=60,
        )
        print(f"Status: {r.status_code}")
        data = r.json()
        print(f"Batch ID: {data.get('batch_id', '')[:8]}...")
        print(f"Total Items: {data.get('total_items')}")
        print(f"Summary: {json.dumps(data.get('summary', {}), indent=2)}")
        
        for result in data.get('results', []):
            print(f"  - {result.get('file_name')}: "
                  f"Score={result.get('trust_score')}, "
                  f"Risk={result.get('risk_level')}")
        
        assert r.status_code == 200
        assert data.get('total_items') == 3
        print("✅ PASS\n")
        return data
    except Exception as e:
        print(f"❌ FAIL: {e}\n")
        return None


def test_graph():
    """Test fraud graph endpoints."""
    print("=" * 60)
    print("TEST: Fraud Graph")
    print("=" * 60)
    try:
        r = httpx.get(f"{BASE_URL}/graph/data", timeout=10)
        print(f"Status: {r.status_code}")
        data = r.json()
        print(f"Nodes: {len(data.get('nodes', []))}")
        print(f"Edges: {len(data.get('edges', []))}")
        print(f"Suspicious clusters: {len(data.get('suspicious_clusters', []))}")
        assert r.status_code == 200
        print("✅ PASS\n")
    except Exception as e:
        print(f"❌ FAIL: {e}\n")


def test_blockchain():
    """Test blockchain integrity."""
    print("=" * 60)
    print("TEST: Blockchain Integrity")
    print("=" * 60)
    try:
        r = httpx.get(f"{BASE_URL}/blockchain/verify", timeout=10)
        print(f"Status: {r.status_code}")
        data = r.json()
        print(f"Chain valid: {data.get('valid')}")
        print(f"Chain length: {data.get('chain_length')}")
        assert r.status_code == 200
        assert data.get('valid') is True
        print("✅ PASS\n")
    except Exception as e:
        print(f"❌ FAIL: {e}\n")


def test_history():
    """Test verification history."""
    print("=" * 60)
    print("TEST: Verification History")
    print("=" * 60)
    try:
        r = httpx.get(f"{BASE_URL}/verifications?limit=5", timeout=10)
        print(f"Status: {r.status_code}")
        data = r.json()
        print(f"Total records: {data.get('total')}")
        print(f"Items returned: {len(data.get('items', []))}")
        assert r.status_code == 200
        print("✅ PASS\n")
    except Exception as e:
        print(f"❌ FAIL: {e}\n")


def test_alert_history():
    """Test alert history endpoint."""
    print("=" * 60)
    print("TEST: Alert History")
    print("=" * 60)
    try:
        r = httpx.get(f"{BASE_URL}/alerts/history", timeout=10)
        print(f"Status: {r.status_code}")
        data = r.json()
        print(f"Total alerts: {data.get('total')}")
        print(f"Connected clients: {data.get('connected_clients')}")
        print(f"Recent alerts: {len(data.get('alerts', []))}")
        assert r.status_code == 200
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")


def test_adversarial_robustness():
    """Test adversarial robustness testing endpoint."""
    print("=" * 60)
    print("TEST: Adversarial Robustness Testing")
    print("=" * 60)
    
    from PIL import Image
    import io
    
    img = Image.new('RGB', (224, 224), color=(180, 140, 120))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse([40, 20, 184, 200], fill=(200, 160, 130))
    draw.ellipse([70, 70, 95, 90], fill='white')
    draw.ellipse([129, 70, 154, 90], fill='white')
    draw.ellipse([75, 75, 90, 85], fill='black')
    draw.ellipse([134, 75, 149, 85], fill='black')
    draw.arc([80, 110, 144, 160], 0, 180, fill='black', width=2)
    
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    
    try:
        r = httpx.post(
            f"{BASE_URL}/test/adversarial",
            files={"file": ("adv_test.png", buf, "image/png")},
            data={"test_all": "true"},
            timeout=60,
        )
        print(f"Status: {r.status_code}")
        data = r.json()
        print(f"Robustness Score: {data.get('overall_robustness_score')}")
        print(f"FGSM results: {len(data.get('fgsm_results', []))}")
        print(f"PGD results: {len(data.get('pgd_results', []))}")
        print(f"Noise results: {len(data.get('noise_results', []))}")
        print(f"Vulnerabilities: {len(data.get('vulnerabilities', []))}")
        print(f"Recommendations: {len(data.get('recommendations', []))}")
        
        assert r.status_code == 200
        assert 'overall_robustness_score' in data
        assert 'vulnerabilities' in data
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")


def test_liveness_detection():
    """Test liveness detection endpoint."""
    print("=" * 60)
    print("TEST: Liveness Detection")
    print("=" * 60)
    
    from PIL import Image
    import io
    
    img = Image.new('RGB', (224, 224), color=(180, 140, 120))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse([40, 20, 184, 200], fill=(200, 160, 130))
    draw.ellipse([70, 70, 95, 90], fill='white')
    draw.ellipse([129, 70, 154, 90], fill='white')
    draw.ellipse([75, 75, 90, 85], fill='black')
    draw.ellipse([134, 75, 149, 85], fill='black')
    draw.arc([80, 110, 144, 160], 0, 180, fill='black', width=2)
    
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    
    try:
        r = httpx.post(
            f"{BASE_URL}/verify/liveness",
            files={"file": ("liveness_test.png", buf, "image/png")},
            data={"user_id": "test_liveness"},
            timeout=30,
        )
        print(f"Status: {r.status_code}")
        data = r.json()
        print(f"Trust Score: {data.get('trust_score')}")
        print(f"Risk Level: {data.get('risk_level')}")
        
        liveness = data.get('detailed_results', {}).get('liveness_analysis', {})
        print(f"Is Live: {liveness.get('is_live')}")
        print(f"Liveness Score: {liveness.get('liveness_score')}")
        print(f"Head Pose: {liveness.get('head_pose', {})}")
        print(f"Depth is_flat: {liveness.get('depth_analysis', {}).get('is_flat')}")
        print(f"Spoof Type: {liveness.get('spoof_type')}")
        
        assert r.status_code == 200
        assert 'liveness_analysis' in data.get('detailed_results', {})
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")


def test_websocket_connection():
    """Test WebSocket connection."""
    print("=" * 60)
    print("TEST: WebSocket Alerts")
    print("=" * 60)
    try:
        import websocket
        ws = websocket.create_connection(f"ws://localhost:8000/ws/alerts", timeout=5)
        # Should receive history message
        result = ws.recv()
        data = json.loads(result)
        print(f"Message type: {data.get('type')}")
        print(f"Connected clients: {data.get('connected_clients')}")
        assert data.get('type') == 'history'
        ws.close()
        print("PASS")
    except ImportError:
        print("SKIP (websocket-client not installed)")
    except Exception as e:
        print(f"FAIL: {e}")


if __name__ == "__main__":
    print("\n" + "🛡️ " * 20)
    print("  VeriShield API Test Suite")
    print("🛡️ " * 20 + "\n")
    
    start = time.time()
    
    test_health()
    test_stats()
    test_document_verification()
    test_deepfake_detection()
    test_full_verification()
    test_batch_verification()
    test_graph()
    test_blockchain()
    test_history()
    test_alert_history()
    test_liveness_detection()
    test_adversarial_robustness()
    test_websocket_connection()
    
    elapsed = time.time() - start
    print("=" * 60)
    print(f"All tests completed in {elapsed:.1f}s")
    print("=" * 60)
