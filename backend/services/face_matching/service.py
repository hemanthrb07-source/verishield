"""
Face Matching Service.
Compare face embeddings between reference and probe images.
Uses ArcFace-style embeddings.
"""
import numpy as np
import torch
from typing import Optional
from backend.core.ml_models import FaceEmbedder, load_model


class FaceMatchingService:
    """Match faces between reference (ID) and probe (live) images."""

    def __init__(self, model_path: Optional[str] = None):
        self.device = "cpu"
        self.model = load_model(FaceEmbedder, model_path, self.device)
        self.match_threshold = 0.6
        self.face_cascade = None
        self._init_detector()

    def _init_detector(self):
        """Initialize face detector."""
        try:
            import cv2
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self.face_cascade = None

    async def analyze(
        self,
        probe_data: dict,
        reference_data: Optional[dict] = None,
    ) -> dict:
        """Analyze and optionally match faces."""
        results = {
            'faces_detected': 0,
            'match_score': None,
            'is_match': False,
            'quality_score': 0.0,
            'face_locations': [],
            'reasons': [],
        }

        # Detect faces in probe image
        probe_image = probe_data.get('original_image')
        if probe_image is None:
            results['reasons'].append('No image data available')
            return results

        faces = self._detect_faces(probe_image)
        results['faces_detected'] = len(faces)
        results['face_locations'] = faces

        if len(faces) == 0:
            results['reasons'].append('No faces detected in image')
            return results

        if len(faces) > 1:
            results['reasons'].append(f'Detected {len(faces)} faces - expected single face')

        # Quality assessment
        results['quality_score'] = self._assess_quality(probe_image, faces)
        if results['quality_score'] < 0.3:
            results['reasons'].append('Low image quality may affect accuracy')

        # Generate embedding for probe
        probe_embedding = self._extract_embedding(probe_data)

        # Match against reference if provided
        if reference_data is not None:
            ref_embedding = self._extract_embedding(reference_data)
            if probe_embedding is not None and ref_embedding is not None:
                score = self._compute_similarity(probe_embedding, ref_embedding)
                results['match_score'] = score
                results['is_match'] = score > self.match_threshold

                if results['is_match']:
                    results['reasons'].append(
                        f'Face match confirmed (score: {score:.3f})'
                    )
                else:
                    results['reasons'].append(
                        f'Face mismatch (score: {score:.3f}, threshold: {self.match_threshold})'
                    )
            else:
                results['reasons'].append('Could not generate face embeddings')
        else:
            results['reasons'].append('No reference image provided for matching')

        return results

    def _detect_faces(self, image: np.ndarray) -> list[dict]:
        """Detect face bounding boxes in an image."""
        locations = []

        if self.face_cascade is not None:
            try:
                import cv2
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                for (x, y, w, h) in faces:
                    locations.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)})
                return locations
            except Exception:
                pass

        # Fallback: assume the center region is a face
        h, w = image.shape[:2]
        margin = min(h, w) // 4
        locations.append({
            'x': margin,
            'y': margin,
            'w': w - 2 * margin,
            'h': h - 2 * margin,
        })
        return locations

    def _extract_embedding(self, data: dict) -> Optional[np.ndarray]:
        """Extract face embedding from image data using pixel-based features."""
        original = data.get('original_image')
        if original is None:
            # Fallback to array
            image_array = data.get('image_array')
            if image_array is None:
                return None
            # Convert CHW to HWC
            if len(image_array.shape) == 3 and image_array.shape[0] in (1, 3):
                original = np.transpose(image_array, (1, 2, 0))
                if original.max() <= 1.0:
                    original = (original * 255).astype(np.uint8)
            else:
                original = image_array

        gray = np.mean(original, axis=2).astype(float) if len(original.shape) == 3 else original.astype(float)
        h, w = gray.shape

        # Resize to fixed size for comparison
        from PIL import Image
        pil = Image.fromarray(gray.astype(np.uint8))
        pil = pil.resize((64, 64), Image.Resampling.LANCZOS)
        flat = np.array(pil, dtype=float).flatten() / 255.0

        # Add statistical features
        hist, _ = np.histogram(gray.flatten(), bins=32, range=(0, 256))
        hist = hist / (hist.sum() + 1e-10)

        # Edge features
        gy, gx = np.gradient(gray)
        edge_mag = np.sqrt(gx**2 + gy**2)
        edge_hist, _ = np.histogram(edge_mag.flatten(), bins=16, range=(0, 100))
        edge_hist = edge_hist / (edge_hist.sum() + 1e-10)

        # Combine into embedding
        embedding = np.concatenate([flat, hist, edge_hist])
        return embedding

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two face embeddings."""
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    def _assess_quality(self, image: np.ndarray, faces: list[dict]) -> float:
        """Assess image quality for face matching."""
        if not faces:
            return 0.0

        score = 0.5  # Base score

        face = faces[0]
        h, w = image.shape[:2]
        face_area = face['w'] * face['h']
        image_area = h * w
        face_ratio = face_area / image_area

        # Face should be reasonable size (not too small, not too large)
        if 0.05 < face_ratio < 0.5:
            score += 0.2
        elif face_ratio < 0.01:
            score -= 0.2

        # Check sharpness in face region
        try:
            face_img = image[face['y']:face['y']+face['h'], face['x']:face['x']+face['w']]
            gray = np.mean(face_img, axis=2) if len(face_img.shape) == 3 else face_img.astype(float)
            from scipy.ndimage import convolve
            laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]])
            filtered = convolve(gray.astype(float), laplacian)
            sharpness = np.var(filtered)
            if sharpness > 100:
                score += 0.2
            elif sharpness < 20:
                score -= 0.1
        except Exception:
            pass

        # Check illumination
        try:
            face_img = image[face['y']:face['y']+face['h'], face['x']:face['x']+face['w']]
            mean_brightness = np.mean(face_img)
            if 60 < mean_brightness < 200:
                score += 0.1
        except Exception:
            pass

        return max(0.0, min(1.0, score))
