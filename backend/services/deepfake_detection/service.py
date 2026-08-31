"""
Deepfake Detection Service.
CNN-based detection of face artifacts, blinking anomalies, GAN fingerprints.
"""
import numpy as np
import torch
from typing import Optional
from backend.core.ml_models import DeepfakeCNN, load_model


class DeepfakeDetectionService:
    """Detect deepfakes, face artifacts, and GAN-generated content."""

    def __init__(self, model_path: Optional[str] = None):
        self.device = "cpu"
        self.model = load_model(DeepfakeCNN, model_path, self.device)
        self.threshold = 0.5

    async def analyze(self, preprocessed_data: dict) -> dict:
        """Run deepfake analysis on preprocessed input."""
        results = {
            'is_deepfake': False,
            'deepfake_probability': 0.0,
            'face_artifacts': False,
            'blinking_anomaly': False,
            'gan_fingerprint': False,
            'confidence': 0.0,
            'frame_analysis': None,
            'reasons': [],
        }

        input_type = preprocessed_data.get('document_type', 'image_based')

        if 'frames' in preprocessed_data:
            results = await self._analyze_video(preprocessed_data, results)
        else:
            results = await self._analyze_single_frame(preprocessed_data, results)

        return results

    async def _analyze_single_frame(self, data: dict, results: dict) -> dict:
        """Analyze a single image for deepfake indicators."""
        image_array = data.get('image_array')
        if image_array is None:
            return results

        # Convert to tensor
        tensor = torch.FloatTensor(image_array).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)

        probability = output['probability'].item()
        results['deepfake_probability'] = probability
        results['is_deepfake'] = probability > self.threshold
        results['confidence'] = abs(probability - 0.5) * 2  # Distance from decision boundary

        # Analyze sub-detections
        results['face_artifacts'] = self._detect_artifacts(output['artifact_embedding'])
        results['blinking_anomaly'] = self._detect_blinking_anomaly(output['blink_embedding'])
        results['gan_fingerprint'] = self._detect_gan_fingerprint(output['gan_embedding'])

        # Build reasons
        if results['is_deepfake']:
            results['reasons'].append(f"Deepfake detected with {probability:.1%} probability")
        if results['face_artifacts']:
            results['reasons'].append("Face artifacts detected (unnatural textures)")
            results['confidence'] = min(results['confidence'] + 0.1, 1.0)
        if results['blinking_anomaly']:
            results['reasons'].append("Abnormal blinking pattern detected")
            results['confidence'] = min(results['confidence'] + 0.1, 1.0)
        if results['gan_fingerprint']:
            results['reasons'].append("GAN fingerprint detected in image")
            results['confidence'] = min(results['confidence'] + 0.15, 1.0)

        if not results['reasons']:
            results['reasons'].append("No deepfake indicators detected")

        return results

    async def _analyze_video(self, data: dict, results: dict) -> dict:
        """Analyze video frames for deepfake indicators."""
        frames = data.get('frames', [])
        if not frames:
            return results

        frame_results = []
        probabilities = []
        artifact_count = 0
        blink_count = 0
        gan_count = 0

        for frame_data in frames:
            frame_array = frame_data.get('frame_array')
            if frame_array is None:
                continue

            tensor = torch.FloatTensor(frame_array).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(tensor)

            prob = output['probability'].item()
            probabilities.append(prob)

            has_artifacts = self._detect_artifacts(output['artifact_embedding'])
            has_blink = self._detect_blinking_anomaly(output['blink_embedding'])
            has_gan = self._detect_gan_fingerprint(output['gan_embedding'])

            if has_artifacts:
                artifact_count += 1
            if has_blink:
                blink_count += 1
            if has_gan:
                gan_count += 1

            frame_results.append({
                'frame_index': frame_data.get('frame_index', 0),
                'probability': prob,
                'is_deepfake': prob > self.threshold,
                'artifacts': has_artifacts,
                'blinking': has_blink,
                'gan_fingerprint': has_gan,
            })

        if probabilities:
            avg_prob = np.mean(probabilities)
            max_prob = np.max(probabilities)
            std_prob = np.std(probabilities)

            # Temporal inconsistency can indicate deepfake
            temporal_inconsistency = std_prob > 0.15

            results['deepfake_probability'] = avg_prob
            results['is_deepfake'] = avg_prob > self.threshold or max_prob > 0.85
            results['confidence'] = abs(avg_prob - 0.5) * 2
            results['frame_analysis'] = frame_results

            total_frames = len(frame_results)
            results['face_artifacts'] = artifact_count / total_frames > 0.3
            results['blinking_anomaly'] = blink_count / total_frames > 0.3
            results['gan_fingerprint'] = gan_count / total_frames > 0.2

            # Build reasons
            if results['is_deepfake']:
                results['reasons'].append(
                    f"Deepfake detected: avg probability={avg_prob:.1%}, max={max_prob:.1%}"
                )
            if temporal_inconsistency:
                results['reasons'].append(
                    f"Temporal inconsistency detected (std={std_prob:.3f})"
                )
            if results['face_artifacts']:
                results['reasons'].append(
                    f"Face artifacts in {artifact_count}/{total_frames} frames"
                )
            if results['blinking_anomaly']:
                results['reasons'].append(
                    f"Abnormal blinking in {blink_count}/{total_frames} frames"
                )
            if results['gan_fingerprint']:
                results['reasons'].append(
                    f"GAN fingerprint in {gan_count}/{total_frames} frames"
                )

        if not results['reasons']:
            results['reasons'].append("No deepfake indicators detected in video")

        return results

    def _detect_artifacts(self, embedding: torch.Tensor) -> bool:
        """Detect face artifacts from embedding analysis."""
        emb_np = embedding.cpu().numpy().flatten()
        # Heuristic: check for high-magnitude components suggesting unnatural textures
        l2_norm = np.linalg.norm(emb_np)
        max_component = np.max(np.abs(emb_np))
        return l2_norm > 15.0 or max_component > 5.0

    def _detect_blinking_anomaly(self, embedding: torch.Tensor) -> bool:
        """Detect abnormal blinking patterns."""
        emb_np = embedding.cpu().numpy().flatten()
        # Check for patterns suggesting unnatural eye regions
        variance = np.var(emb_np)
        return variance > 8.0

    def _detect_gan_fingerprint(self, embedding: torch.Tensor) -> bool:
        """Detect GAN-generated content fingerprints."""
        emb_np = embedding.cpu().numpy().flatten()
        # GAN artifacts often show in specific frequency patterns of embeddings
        spectrum = np.abs(np.fft.rfft(emb_np))
        peak_ratio = np.max(spectrum) / (np.mean(spectrum) + 1e-6)
        return peak_ratio > 15.0
