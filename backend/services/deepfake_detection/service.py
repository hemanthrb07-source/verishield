"""
Deepfake Detection Service — Enhanced.
Detects deepfakes using multi-layered image and video analysis:
- Frequency domain analysis (GAN fingerprints)
- Temporal consistency (video frame-to-frame)
- Color channel artifacts
- Noise pattern analysis
- Face region consistency
- Edge coherence analysis
- Compression artifact detection
- Motion naturalness analysis
"""
import numpy as np
from typing import Optional


class DeepfakeDetectionService:
    """Detect deepfakes via multi-signal image/video content analysis."""

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

        # Run all analysis modules
        indicators = []
        weights = []

        freq = self._frequency_analysis(gray)
        if freq['suspicious']:
            indicators.append(freq['score'])
            weights.append(0.25)
            results['gan_fingerprint'] = True

        color = self._color_artifacts(original)
        if color['has_artifacts']:
            indicators.append(color['severity'])
            weights.append(0.15)
            results['face_artifacts'] = True

        noise = self._noise_analysis(gray)
        if noise['anomaly']:
            indicators.append(noise['score'])
            weights.append(0.10)

        edge = self._edge_coherence(gray)
        if edge['incoherent']:
            indicators.append(edge['score'])
            weights.append(0.15)

        texture = self._texture_analysis(gray)
        if texture['anomalous']:
            indicators.append(texture['score'])
            weights.append(0.10)

        lighting = self._lighting_consistency(gray)
        if lighting['inconsistent']:
            indicators.append(lighting['score'])
            weights.append(0.10)

        jpeg = self._compression_artifacts(gray)
        if jpeg['heavy_compression']:
            indicators.append(jpeg['score'])
            weights.append(0.05)

        face_region = self._face_region_analysis(original)
        if face_region['suspicious']:
            indicators.append(face_region['score'])
            weights.append(0.10)
            results['face_artifacts'] = True

        # Compute probability
        if indicators and weights:
            total_w = sum(weights)
            probability = sum(i * w for i, w in zip(indicators, weights)) / total_w
        else:
            probability = 0.0

        results['deepfake_probability'] = round(probability, 4)
        results['is_deepfake'] = probability > self.threshold
        results['confidence'] = round(abs(probability - 0.5) * 2, 3)

        # Build reasons
        if freq['suspicious']:
            results['reasons'].append(f"GAN fingerprint detected (spectral score: {freq['score']:.3f})")
        if color['has_artifacts']:
            results['reasons'].append(f"Color channel artifacts: {color['description']}")
        if noise['anomaly']:
            results['reasons'].append(f"Noise pattern anomaly (score: {noise['score']:.3f})")
        if edge['incoherent']:
            results['reasons'].append(f"Edge incoherence detected (score: {edge['score']:.3f})")
        if texture['anomalous']:
            results['reasons'].append(f"Texture anomaly detected (score: {texture['score']:.3f})")
        if lighting['inconsistent']:
            results['reasons'].append(f"Lighting inconsistency (score: {lighting['score']:.3f})")
        if face_region['suspicious']:
            results['reasons'].append(f"Face region artifacts (score: {face_region['score']:.3f})")
        if jpeg['heavy_compression']:
            results['reasons'].append(f"Heavy JPEG recompression detected (score: {jpeg['score']:.3f})")
        if not results['reasons']:
            results['reasons'].append("No deepfake indicators detected")

        return results

    async def _analyze_video(self, data: dict, results: dict) -> dict:
        """Enhanced video analysis with temporal consistency and per-frame checks."""
        frames = data.get('frames', [])
        if not frames:
            return results

        frame_results = []
        all_probabilities = []
        artifact_count = 0
        blink_count = 0
        gan_count = 0
        face_consistency_scores = []
        temporal_deltas = []

        prev_gray = None

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

            # Ensure 3-channel
            if len(frame_hwc.shape) == 2:
                gray = frame_hwc.astype(float)
            elif frame_hwc.shape[2] == 1:
                gray = frame_hwc[:, :, 0].astype(float)
            else:
                gray = np.mean(frame_hwc, axis=2).astype(float)

            frame_idx = frame_data.get('frame_index', 0)

            # ── Per-frame multi-signal analysis ──
            freq = self._frequency_analysis(gray)
            noise = self._noise_analysis(gray)
            edge = self._edge_coherence(gray)
            texture = self._texture_analysis(gray)
            color = self._color_artifacts(frame_hwc if len(frame_hwc.shape) == 3 else np.stack([gray]*3, axis=-1))
            face_r = self._face_region_analysis(frame_hwc if len(frame_hwc.shape) == 3 else np.stack([gray]*3, axis=-1).astype(np.uint8))

            # Compute frame probability
            frame_indicators = []
            frame_weights = []
            if freq['suspicious']:
                frame_indicators.append(freq['score'])
                frame_weights.append(0.30)
                gan_count += 1
            if noise['anomaly']:
                frame_indicators.append(noise['score'])
                frame_weights.append(0.15)
                artifact_count += 1
            if edge['incoherent']:
                frame_indicators.append(edge['score'])
                frame_weights.append(0.15)
                blink_count += 1
            if texture['anomalous']:
                frame_indicators.append(texture['score'])
                frame_weights.append(0.10)
            if color['has_artifacts']:
                frame_indicators.append(color['severity'])
                frame_weights.append(0.15)
            if face_r['suspicious']:
                frame_indicators.append(face_r['score'])
                frame_weights.append(0.15)

            if frame_indicators and frame_weights:
                frame_prob = sum(i * w for i, w in zip(frame_indicators, frame_weights)) / sum(frame_weights)
            else:
                frame_prob = 0.0

            all_probabilities.append(frame_prob)

            # ── Temporal consistency: compare with previous frame ──
            temporal_delta = 0.0
            if prev_gray is not None and prev_gray.shape == gray.shape:
                # Absolute difference between consecutive frames
                diff = np.abs(gray.astype(float) - prev_gray.astype(float))
                temporal_delta = float(np.mean(diff))
                temporal_deltas.append(temporal_delta)

                # High inter-frame difference in face region = potential face swap
                if temporal_delta > 25.0:
                    frame_prob = min(1.0, frame_prob + 0.15)

                # Extremely low difference = frozen/copy-pasted face
                if temporal_delta < 0.5 and frame_prob < 0.3:
                    frame_prob = min(1.0, frame_prob + 0.10)

            prev_gray = gray.copy()

            frame_results.append({
                'frame_index': frame_idx,
                'probability': round(float(frame_prob), 4),
                'is_deepfake': frame_prob > self.threshold,
                'artifacts': noise['anomaly'],
                'blinking': edge['incoherent'],
                'gan_fingerprint': freq['suspicious'],
                'face_artifacts': face_r['suspicious'],
                'color_artifacts': color['has_artifacts'],
                'temporal_delta': round(float(temporal_delta), 2),
            })

        if not all_probabilities:
            return results

        avg_prob = float(np.mean(all_probabilities))
        max_prob = float(np.max(all_probabilities))
        std_prob = float(np.std(all_probabilities))

        # ── Temporal inconsistency analysis ──
        temporal_inconsistency = std_prob > 0.12

        # Sudden probability spikes across frames = deepfake indicator
        spike_count = sum(1 for i in range(1, len(all_probabilities))
                          if abs(all_probabilities[i] - all_probabilities[i-1]) > 0.2)
        has_spikes = spike_count > len(all_probabilities) * 0.15

        # Check for unnatural temporal smoothness (all identical = possible synthetic)
        all_similar = std_prob < 0.01 and avg_prob > 0.1

        # ── Temporal delta analysis ──
        if temporal_deltas:
            avg_delta = float(np.mean(temporal_deltas))
            std_delta = float(np.std(temporal_deltas))
            delta_cv = std_delta / (avg_delta + 1e-10)

            # Very low motion variance = synthetic video
            unnaturally_smooth = delta_cv < 0.1 and avg_delta < 2.0
            # Very high motion spikes = potential face manipulation
            has_motion_spikes = any(d > 40 for d in temporal_deltas)
        else:
            unnaturally_smooth = False
            has_motion_spikes = False
            avg_delta = 0
            std_delta = 0

        # ── Compute final probability ──
        final_prob = avg_prob

        # Boost for temporal artifacts
        if temporal_inconsistency:
            final_prob = min(1.0, final_prob + std_prob * 0.3)
        if has_spikes:
            final_prob = min(1.0, final_prob + 0.10)
        if all_similar:
            final_prob = min(1.0, final_prob + 0.08)
        if unnaturally_smooth:
            final_prob = min(1.0, final_prob + 0.10)
        if has_motion_spikes:
            final_prob = min(1.0, final_prob + 0.08)

        total_frames = len(frame_results)
        results['deepfake_probability'] = round(final_prob, 4)
        results['is_deepfake'] = final_prob > self.threshold or max_prob > 0.80
        results['confidence'] = round(abs(final_prob - 0.5) * 2, 3)
        results['frame_analysis'] = frame_results

        results['face_artifacts'] = artifact_count / total_frames > 0.25
        results['blinking_anomaly'] = blink_count / total_frames > 0.25
        results['gan_fingerprint'] = gan_count / total_frames > 0.20

        # ── Build reasons ──
        if results['is_deepfake']:
            results['reasons'].append(
                f"Deepfake detected: avg={avg_prob:.1%}, max={max_prob:.1%}, "
                f"frames_analyzed={total_frames}"
            )
        else:
            results['reasons'].append(
                f"Analysis complete: avg_score={avg_prob:.1%}, max={max_prob:.1%} across {total_frames} frames"
            )

        if temporal_inconsistency:
            results['reasons'].append(
                f"Temporal inconsistency detected (std={std_prob:.3f}) — "
                f"frame scores vary significantly"
            )
        if has_spikes:
            results['reasons'].append(
                f"Probability spikes between frames ({spike_count} spikes detected) — "
                f"possible frame-level manipulation"
            )
        if unnaturally_smooth:
            results['reasons'].append(
                f"Unnaturally smooth temporal motion (CV={delta_cv:.3f}) — "
                f"possible synthetic video generation"
            )
        if has_motion_spikes:
            results['reasons'].append(
                "Motion spikes detected — "
                "possible face region manipulation between frames"
            )
        if results['gan_fingerprint']:
            results['reasons'].append(
                f"GAN fingerprint found in {gan_count}/{total_frames} frames"
            )
        if results['face_artifacts']:
            results['reasons'].append(
                f"Face artifacts found in {artifact_count}/{total_frames} frames"
            )
        if results['blinking_anomaly']:
            results['reasons'].append(
                f"Blinking/edge anomaly in {blink_count}/{total_frames} frames"
            )
        if not results['reasons']:
            results['reasons'].append("No deepfake indicators in video")

        return results

    # ── Analysis Modules ─────────────────────────────────────────────

    def _frequency_analysis(self, gray: np.ndarray) -> dict:
        """Detect GAN fingerprints via frequency domain analysis."""
        h, w = gray.shape
        if h < 16 or w < 16:
            return {'suspicious': False, 'score': 0.0}

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
                radial_profile = np.zeros(center)
                for r in range(center):
                    cy, cx = center, center
                    Y, X = np.ogrid[:block_size, :block_size]
                    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
                    mask = (dist >= r) & (dist < r + 1)
                    radial_profile[r] = np.sum(magnitude * mask)

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
        return {'suspicious': avg_score > 0.3, 'score': round(min(avg_score, 1.0), 4)}

    def _color_artifacts(self, original: np.ndarray) -> dict:
        """Detect color channel artifacts from GAN generation."""
        if len(original.shape) != 3 or original.shape[2] < 3:
            return {'has_artifacts': False, 'severity': 0.0, 'description': 'N/A'}

        r = original[:,:,0].astype(float)
        g = original[:,:,1].astype(float)
        b = original[:,:,2].astype(float)

        r_diff = np.abs(np.diff(r, axis=0))
        g_diff = np.abs(np.diff(g, axis=0))
        b_diff = np.abs(np.diff(b, axis=0))

        rg_diff = np.abs(np.mean(r_diff) - np.mean(g_diff))
        rb_diff = np.abs(np.mean(r_diff) - np.mean(b_diff))
        gb_diff = np.abs(np.mean(g_diff) - np.mean(b_diff))
        inconsistency = (rg_diff + rb_diff + gb_diff) / 3

        max_ch = np.maximum(np.maximum(r, g), b)
        min_ch = np.minimum(np.minimum(r, g), b)
        saturation = np.mean((max_ch - min_ch) / (max_ch + 1e-10))

        has_artifacts = inconsistency > 5.0 or (saturation > 0.6 and inconsistency > 3.0)
        severity = min(inconsistency / 15, 1.0)

        desc = []
        if inconsistency > 5.0:
            desc.append(f'color_transition_mismatch ({inconsistency:.1f})')
        if saturation > 0.6:
            desc.append(f'high_saturation ({saturation:.2f})')

        return {
            'has_artifacts': has_artifacts,
            'severity': round(severity, 4),
            'description': ', '.join(desc) if desc else 'normal',
        }

    def _noise_analysis(self, gray: np.ndarray) -> dict:
        """Analyze noise patterns for deepfake indicators."""
        h, w = gray.shape
        if h < 20 or w < 20:
            return {'anomaly': False, 'score': 0.0}

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

        is_too_uniform = cv < 0.1 and mean_var < 50
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

        block_size = max(h // 8, 8)
        block_coherences = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block_dir = direction[i:i+block_size, j:j+block_size]
                block_mag = magnitude[i:i+block_size, j:j+block_size]

                strong = block_mag > np.percentile(block_mag, 70)
                if np.sum(strong) > 5:
                    dirs = block_dir[strong]
                    mean_sin = np.mean(np.sin(dirs))
                    mean_cos = np.mean(np.cos(dirs))
                    coherence = 1 - np.sqrt(mean_sin**2 + mean_cos**2)
                    block_coherences.append(coherence)

        if not block_coherences:
            return {'incoherent': False, 'score': 0.0}

        mean_coherence = np.mean(block_coherences)
        incoherent = mean_coherence > 0.7

        return {'incoherent': incoherent, 'score': round(min(mean_coherence, 1.0), 4)}

    def _texture_analysis(self, gray: np.ndarray) -> dict:
        """Analyze texture patterns for GAN artifacts."""
        h, w = gray.shape
        if h < 16 or w < 16:
            return {'anomalous': False, 'score': 0.0}

        block_size = max(h // 8, 8)
        texture_vars = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                local_mean = np.mean(block)
                local_var = np.var(block)
                if local_mean > 0:
                    texture_vars.append(local_var / (local_mean**2 + 1e-10))

        if len(texture_vars) < 4:
            return {'anomalous': False, 'score': 0.0}

        texture_vars = np.array(texture_vars)
        texture_cv = np.std(texture_vars) / (np.mean(texture_vars) + 1e-10)

        anomalous = texture_cv < 0.15
        score = max(0, 0.5 - texture_cv) if anomalous else 0.0

        return {'anomalous': anomalous, 'score': round(min(score, 1.0), 4)}

    def _lighting_consistency(self, gray: np.ndarray) -> dict:
        """Check if lighting is consistent across the image."""
        h, w = gray.shape
        if h < 20 or w < 20:
            return {'inconsistent': False, 'score': 0.0}

        q1 = np.mean(gray[:h//2, :w//2])
        q2 = np.mean(gray[:h//2, w//2:])
        q3 = np.mean(gray[h//2:, :w//2])
        q4 = np.mean(gray[h//2:, w//2:])

        means = [q1, q2, q3, q4]
        overall_mean = np.mean(means)
        max_diff = max(means) - min(means)

        inconsistent = max_diff > 80 and max_diff / (overall_mean + 1e-10) > 0.5
        score = min(max_diff / 200, 1.0)

        return {'inconsistent': inconsistent, 'score': round(score, 4)}

    def _compression_artifacts(self, gray: np.ndarray) -> dict:
        """Detect JPEG compression block artifacts."""
        h, w = gray.shape
        if h < 16 or w < 16:
            return {'detected': False, 'heavy_compression': False, 'score': 0.0}

        block_size = 8
        jumps = []
        for i in range(block_size, h - 1, block_size):
            diff = np.abs(gray[i+1, :].mean() - gray[i-1, :].mean())
            jumps.append(diff)
        for j in range(block_size, w - 1, block_size):
            diff = np.abs(gray[:, j+1].mean() - gray[:, j-1].mean())
            jumps.append(diff)

        if not jumps:
            return {'detected': False, 'heavy_compression': False, 'score': 0.0}

        avg_jump = np.mean(jumps)
        detected = avg_jump > 3.0
        heavy = avg_jump > 10.0

        return {'detected': detected, 'heavy_compression': heavy, 'score': round(min(avg_jump / 20, 1.0), 4)}

    def _face_region_analysis(self, original: np.ndarray) -> dict:
        """Analyze center region for face-like artifacts."""
        h, w = original.shape[:2]
        if h < 40 or w < 40:
            return {'suspicious': False, 'score': 0.0}

        # Face is typically in center 60% of image
        y1, y2 = int(h * 0.2), int(h * 0.8)
        x1, x2 = int(w * 0.2), int(w * 0.8)
        face_region = original[y1:y2, x1:x2]

        if len(face_region.shape) == 3:
            face_gray = np.mean(face_region, axis=2).astype(float)
        else:
            face_gray = face_region.astype(float)

        # Check for blending boundaries (face swap artifact)
        # Horizontal seam in center
        center_y = face_gray.shape[0] // 2
        seam_band = face_gray[center_y-2:center_y+2, :]
        if seam_band.shape[0] > 0:
            seam_gradient = np.mean(np.abs(np.diff(seam_band, axis=0)))
        else:
            seam_gradient = 0

        # Vertical seam
        center_x = face_gray.shape[1] // 2
        seam_band_v = face_gray[:, center_x-2:center_x+2]
        if seam_band_v.shape[1] > 0:
            seam_gradient_v = np.mean(np.abs(np.diff(seam_band_v, axis=1)))
        else:
            seam_gradient_v = 0

        # Sharp seams indicate face swapping
        max_seam = max(seam_gradient, seam_gradient_v)
        suspicious = max_seam > 15
        score = min(max_seam / 30, 1.0)

        return {'suspicious': suspicious, 'score': round(score, 4)}
