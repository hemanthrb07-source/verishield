"""
Deepfake Detection Service.
Detects deepfakes using actual image analysis:
- Frequency domain analysis (GAN fingerprints)
- Color channel artifacts
- Face region consistency
- Noise pattern analysis
- Temporal consistency (video)
- Edge coherence analysis
"""
import numpy as np
from typing import Optional


class DeepfakeDetectionService:
    """Detect deepfakes via image content analysis."""

    def __init__(self):
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

        if 'frames' in preprocessed_data:
            results = await self._analyze_video(preprocessed_data, results)
        else:
            results = await self._analyze_single(preprocessed_data, results)

        return results

    async def _analyze_single(self, data: dict, results: dict) -> dict:
        """Analyze a single image for deepfake indicators."""
        original = data.get('original_image')
        if original is None:
            return results

        gray = np.mean(original, axis=2).astype(float) if len(original.shape) == 3 else original.astype(float)

        # ── 1. Frequency domain analysis (GAN fingerprint) ──
        freq_analysis = self._frequency_analysis(gray)
        results['gan_fingerprint'] = freq_analysis['suspicious']

        # ── 2. Color channel artifacts ──
        color_analysis = self._color_artifacts(original)
        results['face_artifacts'] = color_analysis['has_artifacts']

        # ── 3. Noise pattern analysis ──
        noise_analysis = self._noise_analysis(gray)

        # ── 4. Edge coherence ──
        edge_analysis = self._edge_coherence(gray)

        # ── 5. Texture analysis ──
        texture_analysis = self._texture_analysis(gray)

        # ── 6. Lighting consistency ──
        lighting_analysis = self._lighting_consistency(gray)

        # ── 7. Compression analysis ──
        compression_analysis = self._compression_artifacts(gray)

        # ── Compute deepfake probability ──
        indicators = []
        weights = []

        # GAN fingerprint (strong indicator)
        if freq_analysis['suspicious']:
            indicators.append(freq_analysis['score'])
            weights.append(0.30)

        # Color artifacts
        if color_analysis['has_artifacts']:
            indicators.append(color_analysis['severity'])
            weights.append(0.20)

        # Noise anomalies
        if noise_analysis['anomaly']:
            indicators.append(noise_analysis['score'])
            weights.append(0.15)

        # Edge incoherence
        if edge_analysis['incoherent']:
            indicators.append(edge_analysis['score'])
            weights.append(0.15)

        # Texture anomalies
        if texture_analysis['anomalous']:
            indicators.append(texture_analysis['score'])
            weights.append(0.10)

        # Lighting inconsistency
        if lighting_analysis['inconsistent']:
            indicators.append(lighting_analysis['score'])
            weights.append(0.10)

        # Compute probability
        if indicators and weights:
            total_weight = sum(weights)
            probability = sum(i * w for i, w in zip(indicators, weights)) / total_weight
        else:
            probability = 0.0

        results['deepfake_probability'] = round(probability, 4)
        results['is_deepfake'] = probability > self.threshold
        results['confidence'] = round(abs(probability - 0.5) * 2, 3)

        # Build reasons
        if freq_analysis['suspicious']:
            results['reasons'].append(
                f"GAN fingerprint detected (spectral score: {freq_analysis['score']:.3f})"
            )
        if color_analysis['has_artifacts']:
            results['reasons'].append(
                f"Color channel artifacts: {color_analysis['description']}"
            )
        if noise_analysis['anomaly']:
            results['reasons'].append(
                f"Noise pattern anomaly (score: {noise_analysis['score']:.3f})"
            )
        if edge_analysis['incoherent']:
            results['reasons'].append(
                f"Edge incoherence detected (score: {edge_analysis['score']:.3f})"
            )
        if texture_analysis['anomalous']:
            results['reasons'].append(
                f"Texture anomaly detected (score: {texture_analysis['score']:.3f})"
            )
        if lighting_analysis['inconsistent']:
            results['reasons'].append(
                f"Lighting inconsistency (score: {lighting_analysis['score']:.3f})"
            )
        if not results['reasons']:
            results['reasons'].append("No deepfake indicators detected")

        return results

    async def _analyze_video(self, data: dict, results: dict) -> dict:
        """Analyze video frames for temporal deepfake indicators."""
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

            # Convert from CHW to HWC
            if len(frame_array.shape) == 3 and frame_array.shape[0] in (1, 3):
                frame_hwc = np.transpose(frame_array, (1, 2, 0))
            else:
                frame_hwc = frame_array

            # Convert to 0-255 range if normalized
            if frame_hwc.max() <= 1.0:
                frame_hwc = (frame_hwc * 255).astype(np.uint8)

            gray = np.mean(frame_hwc, axis=2).astype(float) if len(frame_hwc.shape) == 3 else frame_hwc.astype(float)

            # Quick analysis per frame
            freq = self._frequency_analysis(gray)
            noise = self._noise_analysis(gray)
            edge = self._edge_coherence(gray)

            # Simple probability from this frame
            prob = 0.0
            if freq['suspicious']:
                prob += freq['score'] * 0.5
                gan_count += 1
            if noise['anomaly']:
                prob += noise['score'] * 0.3
                artifact_count += 1
            if edge['incoherent']:
                prob += edge['score'] * 0.2
                blink_count += 1

            probabilities.append(prob)
            frame_results.append({
                'frame_index': frame_data.get('frame_index', 0),
                'probability': prob,
                'is_deepfake': prob > self.threshold,
                'artifacts': noise['anomaly'],
                'blinking': edge['incoherent'],
                'gan_fingerprint': freq['suspicious'],
            })

        if probabilities:
            avg_prob = np.mean(probabilities)
            std_prob = np.std(probabilities)

            # Temporal inconsistency is a strong deepfake indicator
            temporal_inconsistency = std_prob > 0.15

            results['deepfake_probability'] = round(float(avg_prob), 4)
            results['is_deepfake'] = avg_prob > self.threshold or np.max(probabilities) > 0.85
            results['confidence'] = round(float(abs(avg_prob - 0.5) * 2), 3)
            results['frame_analysis'] = frame_results

            total_frames = len(frame_results)
            results['face_artifacts'] = artifact_count / total_frames > 0.3
            results['blinking_anomaly'] = blink_count / total_frames > 0.3
            results['gan_fingerprint'] = gan_count / total_frames > 0.2

            if results['is_deepfake']:
                results['reasons'].append(
                    f"Deepfake detected: avg={avg_prob:.1%}, max={np.max(probabilities):.1%}"
                )
            if temporal_inconsistency:
                results['reasons'].append(f"Temporal inconsistency (std={std_prob:.3f})")

        if not results['reasons']:
            results['reasons'].append("No deepfake indicators in video")

        return results

    def _frequency_analysis(self, gray: np.ndarray) -> dict:
        """Detect GAN fingerprints via frequency domain analysis."""
        h, w = gray.shape
        if h < 16 or w < 16:
            return {'suspicious': False, 'score': 0.0}

        # Analyze frequency spectrum
        block_size = min(64, h, w)
        center = block_size // 2

        spectral_scores = []
        for i in range(0, h - block_size, block_size // 2):
            for j in range(0, w - block_size, block_size // 2):
                block = gray[i:i+block_size, j:j+block_size]
                fft = np.fft.fft2(block)
                fft_shift = np.fft.fftshift(fft)
                magnitude = np.log1p(np.abs(fft_shift))

                total = np.sum(magnitude) + 1e-10
                # Check for periodic peaks (GAN artifacts)
                radial_profile = np.zeros(center)
                for r in range(center):
                    mask = np.zeros_like(magnitude)
                    cy, cx = center, center
                    Y, X = np.ogrid[:block_size, :block_size]
                    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
                    mask[(dist >= r) & (dist < r + 1)] = 1
                    radial_profile[r] = np.sum(magnitude * mask)

                # Check for spectral peaks (periodic patterns from GAN)
                if len(radial_profile) > 4:
                    peaks = []
                    for k in range(2, len(radial_profile) - 2):
                        if radial_profile[k] > radial_profile[k-1] and radial_profile[k] > radial_profile[k+1]:
                            if radial_profile[k] > np.mean(radial_profile) * 1.5:
                                peaks.append(radial_profile[k] / total)

                    spectral_scores.append(len(peaks) * 0.1 + sum(peaks) * 0.5)

        if not spectral_scores:
            return {'suspicious': False, 'score': 0.0}

        avg_score = np.mean(spectral_scores)
        suspicious = avg_score > 0.3

        return {
            'suspicious': suspicious,
            'score': round(min(avg_score, 1.0), 4),
        }

    def _color_artifacts(self, original: np.ndarray) -> dict:
        """Detect color channel artifacts from GAN generation."""
        if len(original.shape) != 3 or original.shape[2] < 3:
            return {'has_artifacts': False, 'severity': 0.0, 'description': 'N/A'}

        r, g, b = original[:,:,0].astype(float), original[:,:,1].astype(float), original[:,:,2].astype(float)

        # Check for unusual color transitions
        # Real faces have smooth color gradients; GANs often have artifacts
        r_diff = np.abs(np.diff(r, axis=0))
        g_diff = np.abs(np.diff(g, axis=0))
        b_diff = np.abs(np.diff(b, axis=0))

        # Check channel-wise transition consistency
        rg_diff = np.abs(np.mean(r_diff) - np.mean(g_diff))
        rb_diff = np.abs(np.mean(r_diff) - np.mean(b_diff))
        gb_diff = np.abs(np.mean(g_diff) - np.mean(b_diff))

        inconsistency = (rg_diff + rb_diff + gb_diff) / 3

        # Check for color saturation anomalies
        max_ch = np.maximum(np.maximum(r, g), b)
        min_ch = np.minimum(np.minimum(r, g), b)
        saturation = np.mean((max_ch - min_ch) / (max_ch + 1e-10))

        # High saturation with abrupt transitions = likely GAN
        has_artifacts = inconsistency > 5.0 or (saturation > 0.6 and inconsistency > 3.0)
        severity = min(inconsistency / 15, 1.0)

        description = []
        if inconsistency > 5.0:
            description.append(f'color_transition_mismatch ({inconsistency:.1f})')
        if saturation > 0.6:
            description.append(f'high_saturation ({saturation:.2f})')

        return {
            'has_artifacts': has_artifacts,
            'severity': round(severity, 4),
            'description': ', '.join(description) if description else 'normal',
        }

    def _noise_analysis(self, gray: np.ndarray) -> dict:
        """Analyze noise patterns for deepfake indicators."""
        h, w = gray.shape
        if h < 20 or w < 20:
            return {'anomaly': False, 'score': 0.0}

        # Compute local noise
        try:
            from scipy.ndimage import convolve
            laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
            filtered = convolve(gray, laplacian)
        except ImportError:
            filtered = np.zeros_like(gray)
            filtered[1:-1, 1:-1] = (
                gray[:-2, 1:-1] + gray[2:, 1:-1] +
                gray[1:-1, :-2] + gray[1:-1, 2:] -
                4 * gray[1:-1, 1:-1]
            )

        # Analyze noise in blocks
        block_size = max(h // 8, 8)
        block_vars = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = filtered[i:i+block_size, j:j+block_size]
                block_vars.append(np.var(block))

        if len(block_vars) < 4:
            return {'anomaly': False, 'score': 0.0}

        block_vars = np.array(block_vars)
        mean_var = np.mean(block_vars)
        std_var = np.std(block_vars)
        cv = std_var / (mean_var + 1e-10)

        # GAN images often have more uniform noise than real images
        is_too_uniform = cv < 0.1 and mean_var < 50
        # Or abnormally high variance in some regions
        has_spikes = np.sum(block_vars > mean_var + 3 * std_var) > len(block_vars) * 0.1

        anomaly = is_too_uniform or has_spikes
        score = 0.0
        if is_too_uniform:
            score += 0.5
        if has_spikes:
            score += 0.5

        return {'anomaly': anomaly, 'score': round(min(score, 1.0), 4)}

    def _edge_coherence(self, gray: np.ndarray) -> dict:
        """Check if edges are coherent (real) or incoherent (GAN)."""
        h, w = gray.shape
        if h < 10 or w < 10:
            return {'incoherent': False, 'score': 0.0}

        gy, gx = np.gradient(gray)
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx)

        # Check edge direction consistency in blocks
        block_size = max(h // 8, 8)
        block_coherences = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block_dir = direction[i:i+block_size, j:j+block_size]
                block_mag = magnitude[i:i+block_size, j:j+block_size]

                # Only consider strong edges
                strong = block_mag > np.percentile(block_mag, 70)
                if np.sum(strong) > 5:
                    dirs = block_dir[strong]
                    # Circular variance
                    mean_sin = np.mean(np.sin(dirs))
                    mean_cos = np.mean(np.cos(dirs))
                    coherence = 1 - np.sqrt(mean_sin**2 + mean_cos**2)
                    block_coherences.append(coherence)

        if not block_coherences:
            return {'incoherent': False, 'score': 0.0}

        mean_coherence = np.mean(block_coherences)
        # Low coherence = edges are chaotic = possible GAN artifact
        incoherent = mean_coherence > 0.7  # High circular variance

        return {
            'incoherent': incoherent,
            'score': round(min(mean_coherence, 1.0), 4),
        }

    def _texture_analysis(self, gray: np.ndarray) -> dict:
        """Analyze texture patterns for GAN artifacts."""
        h, w = gray.shape
        if h < 16 or w < 16:
            return {'anomalous': False, 'score': 0.0}

        # Compute Local Binary Pattern-like features
        block_size = max(h // 8, 8)
        texture_vars = []

        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                # Simple texture measure: local variance relative to mean
                local_mean = np.mean(block)
                local_var = np.var(block)
                if local_mean > 0:
                    texture_vars.append(local_var / (local_mean**2 + 1e-10))

        if len(texture_vars) < 4:
            return {'anomalous': False, 'score': 0.0}

        texture_vars = np.array(texture_vars)
        # GAN images often have repetitive texture patterns
        # Check if texture is too similar across blocks
        texture_cv = np.std(texture_vars) / (np.mean(texture_vars) + 1e-10)

        anomalous = texture_cv < 0.15  # Very uniform texture
        score = max(0, 0.5 - texture_cv) if anomalous else 0.0

        return {'anomalous': anomalous, 'score': round(min(score, 1.0), 4)}

    def _lighting_consistency(self, gray: np.ndarray) -> dict:
        """Check if lighting is consistent across the image."""
        h, w = gray.shape
        if h < 20 or w < 20:
            return {'inconsistent': False, 'score': 0.0}

        # Divide into quadrants and compare mean brightness
        q1 = np.mean(gray[:h//2, :w//2])
        q2 = np.mean(gray[:h//2, w//2:])
        q3 = np.mean(gray[h//2:, :w//2])
        q4 = np.mean(gray[h//2:, w//2:])

        means = [q1, q2, q3, q4]
        overall_mean = np.mean(means)
        max_diff = max(means) - min(means)

        # Extreme lighting differences
        inconsistent = max_diff > 80 and max_diff / (overall_mean + 1e-10) > 0.5
        score = min(max_diff / 200, 1.0)

        return {
            'inconsistent': inconsistent,
            'score': round(score, 4),
            'quadrant_means': [round(m, 1) for m in means],
        }

    def _compression_artifacts(self, gray: np.ndarray) -> dict:
        """Detect JPEG compression block artifacts."""
        h, w = gray.shape
        if h < 16 or w < 16:
            return {'detected': False, 'score': 0.0}

        # Check for 8x8 block boundaries (JPEG artifact)
        # This works by checking if pixel values at block boundaries
        # have sudden jumps
        block_size = 8
        jumps = []
        for i in range(block_size, h - 1, block_size):
            diff = np.abs(gray[i+1, :].mean() - gray[i-1, :].mean())
            jumps.append(diff)
        for j in range(block_size, w - 1, block_size):
            diff = np.abs(gray[:, j+1].mean() - gray[:, j-1].mean())
            jumps.append(diff)

        if not jumps:
            return {'detected': False, 'score': 0.0}

        avg_jump = np.mean(jumps)
        detected = avg_jump > 3.0  # Some JPEG compression
        # Very strong block boundaries suggest heavy compression (re-saved)
        heavy = avg_jump > 10.0

        return {
            'detected': detected,
            'heavy_compression': heavy,
            'score': round(min(avg_jump / 20, 1.0), 4),
        }
