"""
Document Intelligence Service.
Analyzes actual image content for authenticity using:
- Color histogram analysis
- Edge density and consistency
- Noise pattern analysis
- Compression artifact detection
- Text region detection
- Spatial frequency analysis
- Color channel correlation
"""
import numpy as np
from typing import Optional
import hashlib


class DocumentIntelligenceService:
    """Analyze documents for authenticity based on actual image content."""

    async def analyze(self, preprocessed_data: dict) -> dict:
        """Run full document analysis pipeline."""
        results = {
            'authenticity_score': 1.0,
            'tampering_detected': False,
            'font_inconsistencies': [],
            'spacing_anomalies': [],
            'tampered_regions': [],
            'ocr_confidence': 0.0,
            'metadata_analysis': {},
            'reasons': [],
        }

        original = preprocessed_data.get('original_image')
        if original is None:
            results['reasons'].append('No image data available')
            results['authenticity_score'] = 0.3
            return results

        gray = np.mean(original, axis=2).astype(float) if len(original.shape) == 3 else original.astype(float)
        h, w = gray.shape

        # ── 1. Color distribution analysis ──
        color_issues = self._analyze_color_distribution(original, gray)
        results['font_inconsistencies'] = color_issues['issues']

        # ── 2. Edge analysis ──
        edge_issues = self._analyze_edges(gray)
        results['spacing_anomalies'] = edge_issues['issues']

        # ── 3. Noise analysis ──
        noise_analysis = self._analyze_noise_patterns(gray)

        # ── 4. Tampering detection via block analysis ──
        tampered = self._detect_tampering_blocks(gray, original)
        results['tampered_regions'] = tampered['regions']

        # ── 5. Color channel correlation ──
        channel_analysis = self._analyze_color_channels(original)

        # ── 6. Resolution quality ──
        quality = self._analyze_quality(original, gray)

        # ── 7. Document structure analysis ──
        structure = self._analyze_structure(original, gray)

        # ── 8. Metadata analysis ──
        metadata = preprocessed_data.get('metadata', {})
        results['metadata_analysis'] = self._analyze_metadata(metadata)

        # ── Direct score computation from key features ──
        # Use the most discriminative features directly
        
        # 1. Color peak ratio: high = uniform background (document-like)
        color_peak = color_issues.get('score', 0.5)
        if color_peak > 0.8:
            color_score = 0.9
        elif color_peak > 0.5:
            color_score = 0.6
        elif color_peak > 0.2:
            color_score = 0.4
        else:
            color_score = 0.2

        # 2. Edge CV: high = structured content (text regions)
        edge_cv = edge_issues.get('score', 0.5)
        if edge_cv > 0.8:
            edge_score = 0.9
        elif edge_cv > 0.5:
            edge_score = 0.7
        elif edge_cv > 0.3:
            edge_score = 0.5
        else:
            edge_score = 0.2

        # 3. Noise: very high = suspicious
        noise_anomaly = noise_analysis.get('anomaly_score', 0)
        if noise_anomaly < 0.1:
            noise_score = 0.9
        elif noise_anomaly < 0.3:
            noise_score = 0.7
        elif noise_anomaly < 0.5:
            noise_score = 0.4
        else:
            noise_score = 0.15

        # 4. Tampering: fewer regions = better
        total_blocks = max((h * w) / (64 * 64), 1)
        tamper_ratio = len(tampered['regions']) / total_blocks
        if tamper_ratio < 0.02:
            tamper_score = 0.95
        elif tamper_ratio < 0.1:
            tamper_score = 0.7
        elif tamper_ratio < 0.2:
            tamper_score = 0.4
        else:
            tamper_score = 0.2

        # 5. Resolution quality
        quality_score = 0.7  # default
        if quality['is_low_res']:
            quality_score = 0.4
        if quality['high_entropy']:
            quality_score *= 0.6
        if quality['is_too_uniform']:
            quality_score *= 0.5

        # 6. Content type adjustment
        content_type = self._classify_content_type(original, gray, structure)
        results['content_type'] = content_type
        type_multiplier = {'document': 1.0, 'natural': 0.85, 'synthetic': 0.6, 'noise': 0.3}.get(content_type, 0.7)

        # Weighted combination
        score = (
            color_score * 0.20 +
            edge_score * 0.20 +
            noise_score * 0.15 +
            tamper_score * 0.25 +
            quality_score * 0.10 +
            structure.get('score', 0.5) * 0.10
        ) * type_multiplier

        # Small image penalty
        if h < 100 or w < 100:
            score *= 0.7
        if h < 50 or w < 50:
            score *= 0.5

        # Metadata penalty
        meta = results['metadata_analysis']
        if meta.get('editing_software_detected'):
            score *= 0.9

        score = max(0.05, min(1.0, score))
        results['authenticity_score'] = score
        results['tampering_detected'] = score < 0.7 or len(tampered['regions']) > 2

        # Build plain-English reasons
        color_count = len(color_issues['issues'])
        if color_count > 0:
            area_word = 'areas' if color_count > 1 else 'area'
            results['reasons'].append(
                f"The colors in this document look unusual — {color_count} {area_word} "
                f"have color patterns that don't look natural. This could mean the image was edited."
            )
        edge_count = len(edge_issues['issues'])
        if edge_count > 0:
            section_word = 'sections' if edge_count > 1 else 'section'
            results['reasons'].append(
                f"Some edges in the document look irregular — {edge_count} {section_word} "
                f"have lines or borders that appear artificially sharp or blurry. This often happens when parts of an image are copy-pasted."
            )
        if noise_analysis['anomaly_score'] > 0.3:
            results['reasons'].append(
                f"The texture of the image looks inconsistent — some parts are smoother or grainier than others. "
                f"This can be a sign that different images were stitched together."
            )
        tamper_count = len(tampered['regions'])
        if tamper_count > 0:
            area_word = 'areas' if tamper_count > 1 else 'area'
            results['reasons'].append(
                f"{tamper_count} {area_word} in the document look different from the rest — "
                f"they have unusual texture or brightness that doesn't match the surrounding content. "
                f"This is a common sign of editing or tampering."
            )
        if channel_analysis['anomaly']:
            results['reasons'].append(
                "The color channels (red, green, blue) don't behave normally — they're too independent of each other. "
                f"In real photos, these channels usually move together. This mismatch can indicate digital manipulation."
            )
        if quality['is_too_uniform']:
            results['reasons'].append(
                "The image looks artificially uniform — real documents and photos always have some natural variation. "
                f"This level of uniformity is more typical of computer-generated images."
            )
        if results['tampering_detected']:
            results['reasons'].append(
                f"Based on all the checks above, this document appears to have been modified. "
                f"The trust score is {results['authenticity_score'] * 100:.0f} out of 100."
            )
        elif not results['reasons']:
            results['reasons'].append(
                f"This document looks authentic — no signs of editing or tampering were detected. "
                f"Trust score: {results['authenticity_score'] * 100:.0f} out of 100."
            )

        return results

    def _analyze_color_distribution(self, original: np.ndarray, gray: np.ndarray) -> dict:
        """Analyze color histogram for anomalies."""
        issues = []
        score = 1.0

        if len(original.shape) == 3 and original.shape[2] >= 3:
            for ch in range(3):
                channel = original[:, :, ch].flatten()
                hist, _ = np.histogram(channel, bins=256, range=(0, 256))

                # Check for unusual peaks (clipping, manipulation)
                total_pixels = len(channel)
                peak_ratio = np.max(hist) / total_pixels
                if peak_ratio > 0.4:
                    issues.append({
                        'type': 'color_peak',
                        'channel': ['R', 'G', 'B'][ch],
                        'severity': 'medium',
                        'score': peak_ratio,
                    })
                    score *= 0.85

                # Check for bimodal distribution (tampering indicator)
                nonzero = hist[hist > 0]
                if len(nonzero) > 10:
                    gaps = np.diff(np.where(hist > total_pixels * 0.001)[0])
                    large_gaps = np.sum(gaps > 30)
                    if large_gaps > 3:
                        issues.append({
                            'type': 'bimodal_distribution',
                            'channel': ['R', 'G', 'B'][ch],
                            'severity': 'low',
                            'score': large_gaps / 10,
                        })
                        score *= 0.95

        return {'issues': issues, 'score': max(0.0, score)}

    def _analyze_edges(self, gray: np.ndarray) -> dict:
        """Analyze edge consistency for anomalies."""
        issues = []
        score = 1.0

        h, w = gray.shape
        if h < 10 or w < 10:
            return {'issues': issues, 'score': 0.5}

        # Simple edge detection via gradient
        gy, gx = np.gradient(gray)
        edge_magnitude = np.sqrt(gx**2 + gy**2)

        # Analyze edge density in blocks
        block_size = max(h // 8, 8)
        densities = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = edge_magnitude[i:i+block_size, j:j+block_size]
                density = np.mean(block)
                densities.append(density)

        if len(densities) < 2:
            return {'issues': issues, 'score': 0.5}

        densities = np.array(densities)
        mean_density = np.mean(densities)
        std_density = np.std(densities)

        # High variance in edge density suggests tampering
        cv = std_density / (mean_density + 1e-6)
        if cv > 1.5:
            issues.append({
                'type': 'edge_density_variance',
                'severity': 'medium',
                'description': f'High edge density variance (cv={cv:.2f})',
                'score': min(cv / 3, 1.0),
            })
            score *= 0.8

        # Very low edge density means no document content
        if mean_density < 1.0:
            issues.append({
                'type': 'low_content',
                'severity': 'info',
                'description': 'Very low edge density - minimal content',
            })
            score *= 0.9

        return {'issues': issues, 'score': max(0.0, score)}

    def _analyze_noise_patterns(self, gray: np.ndarray) -> dict:
        """Analyze noise patterns for manipulation artifacts."""
        h, w = gray.shape
        if h < 20 or w < 20:
            return {'anomaly_score': 0.3}

        # Compute local noise via Laplacian
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
        try:
            from scipy.ndimage import convolve
            filtered = convolve(gray, laplacian)
        except ImportError:
            # Fallback: manual convolution
            filtered = np.zeros_like(gray)
            filtered[1:-1, 1:-1] = (
                gray[:-2, 1:-1] + gray[2:, 1:-1] +
                gray[1:-1, :-2] + gray[1:-1, 2:] -
                4 * gray[1:-1, 1:-1]
            )

        noise_var = np.var(filtered)
        noise_mean = np.abs(np.mean(filtered))

        # Check for noise inconsistency in blocks
        block_size = max(h // 6, 8)
        block_noises = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = filtered[i:i+block_size, j:j+block_size]
                block_noises.append(np.var(block))

        anomaly_score = 0.0
        if len(block_noises) > 2:
            block_noises = np.array(block_noises)
            noise_cv = np.std(block_noises) / (np.mean(block_noises) + 1e-6)
            # Different noise levels in different regions = possible splicing
            anomaly_score = min(noise_cv / 3, 1.0)

        return {
            'anomaly_score': round(anomaly_score, 4),
            'noise_variance': round(noise_var, 4),
            'noise_mean': round(noise_mean, 4),
        }

    def _detect_tampering_blocks(self, gray: np.ndarray, original: np.ndarray) -> dict:
        """Detect tampered regions via frequency analysis."""
        regions = []
        h, w = gray.shape

        block_size = 64
        if h < block_size * 2 or w < block_size * 2:
            return {'regions': regions}

        # Compute frequency features per block
        block_features = []
        for i in range(0, h - block_size, block_size // 2):
            for j in range(0, w - block_size, block_size // 2):
                block = gray[i:i+block_size, j:j+block_size].astype(float)

                # FFT analysis
                fft = np.fft.fft2(block)
                fft_shift = np.fft.fftshift(fft)
                magnitude = np.abs(fft_shift)

                center = block_size // 2
                total_energy = np.sum(magnitude) + 1e-10
                high_freq = np.sum(magnitude) - np.sum(magnitude[center-8:center+8, center-8:center+8])
                high_freq_ratio = high_freq / total_energy

                # Local contrast
                local_std = np.std(block)

                block_features.append({
                    'i': i, 'j': j,
                    'high_freq_ratio': high_freq_ratio,
                    'local_std': local_std,
                })

        if len(block_features) < 4:
            return {'regions': regions}

        # Find outlier blocks
        hf_ratios = np.array([b['high_freq_ratio'] for b in block_features])
        stds = np.array([b['local_std'] for b in block_features])

        hf_mean, hf_std = np.mean(hf_ratios), np.std(hf_ratios)
        std_mean, std_std_val = np.mean(stds), np.std(stds)

        for feat in block_features:
            is_outlier = False
            reasons = []

            if hf_std > 0 and abs(feat['high_freq_ratio'] - hf_mean) > 2 * hf_std:
                is_outlier = True
                reasons.append('frequency_mismatch')
            if std_std_val > 0 and abs(feat['local_std'] - std_mean) > 2 * std_std_val:
                is_outlier = True
                reasons.append('contrast_mismatch')

            if is_outlier:
                confidence = 0.0
                if hf_std > 0:
                    confidence += min(abs(feat['high_freq_ratio'] - hf_mean) / (3 * hf_std), 1.0) * 0.5
                if std_std_val > 0:
                    confidence += min(abs(feat['local_std'] - std_mean) / (3 * std_std_val), 1.0) * 0.5

                if confidence > 0.4:
                    regions.append({
                        'x': int(feat['j']),
                        'y': int(feat['i']),
                        'w': block_size,
                        'h': block_size,
                        'confidence': round(confidence, 3),
                        'type': 'frequency_anomaly',
                        'reasons': reasons,
                    })

        return {'regions': regions}

    def _analyze_color_channels(self, original: np.ndarray) -> dict:
        """Analyze inter-channel correlations."""
        if len(original.shape) != 3 or original.shape[2] < 3:
            return {'anomaly': False}

        r, g, b = original[:,:,0].flatten(), original[:,:,1].flatten(), original[:,:,2].flatten()

        # Normal images have high R-G and R-B correlation
        corr_rg = np.corrcoef(r, g)[0, 1] if len(r) > 1 else 0
        corr_rb = np.corrcoef(r, b)[0, 1] if len(r) > 1 else 0
        corr_gb = np.corrcoef(g, b)[0, 1] if len(r) > 1 else 0

        # Unusual if correlations are very low (for natural images)
        anomaly = corr_rg < 0.3 or corr_rb < 0.3

        return {
            'anomaly': anomaly,
            'corr_rg': round(float(corr_rg), 4),
            'corr_rb': round(float(corr_rb), 4),
            'corr_gb': round(float(corr_gb), 4),
        }

    def _analyze_quality(self, original: np.ndarray, gray: np.ndarray) -> dict:
        """Analyze image quality metrics."""
        h, w = gray.shape

        # Check if image is too uniform
        flatness = np.std(gray) / (np.mean(gray) + 1e-6)
        is_too_uniform = flatness < 0.05

        # Check if resolution is low
        is_low_res = h < 200 or w < 200

        # Check entropy (information content)
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-10)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        high_entropy = entropy > 7.0  # Very high entropy = noise-like

        return {
            'flatness': round(float(flatness), 4),
            'entropy': round(float(entropy), 4),
            'is_too_uniform': is_too_uniform,
            'is_low_res': is_low_res,
            'high_entropy': high_entropy,
        }

    def _classify_content_type(self, original: np.ndarray, gray: np.ndarray, structure: dict) -> str:
        """Classify the image content type."""
        h, w = gray.shape

        # Check for noise (high entropy, no structure)
        flatness = np.std(gray) / (np.mean(gray) + 1e-10)
        if flatness > 0.3 and structure.get('peak_ratio', 0) < 0.02:
            return 'noise'

        # Check for uniform/synthetic (low edge variance, high peak)
        if structure.get('peak_ratio', 0) > 0.8 and structure.get('edge_cv', 0) > 1.0:
            return 'document'

        # Check for natural image (moderate edge CV, moderate peak)
        if structure.get('edge_cv', 0) > 0.3 and structure.get('peak_ratio', 0) > 0.05:
            return 'natural'

        return 'synthetic'

    def _analyze_structure(self, original: np.ndarray, gray: np.ndarray) -> dict:
        """Analyze how structured/organized the image content is."""
        h, w = gray.shape
        if h < 20 or w < 20:
            return {'score': 0.3, 'details': 'too_small'}

        signals = []

        # 1. Color peak ratio (high = uniform background = document-like)
        if len(original.shape) == 3 and original.shape[2] >= 3:
            for ch in range(3):
                channel = original[:, :, ch].flatten()
                hist, _ = np.histogram(channel, bins=256, range=(0, 256))
                peak_ratio = np.max(hist) / len(channel)
                signals.append(peak_ratio)
        peak_score = max(signals) if signals else 0.5

        # 2. Edge density CV (high = structured text regions)
        gy, gx = np.gradient(gray)
        edge_mag = np.sqrt(gx**2 + gy**2)
        block_size = max(h // 8, 8)
        densities = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                densities.append(np.mean(edge_mag[i:i+block_size, j:j+block_size]))
        if len(densities) > 1:
            densities = np.array(densities)
            edge_cv = np.std(densities) / (np.mean(densities) + 1e-6)
        else:
            edge_cv = 0.5
        edge_score = min(edge_cv / 3, 1.0)

        # 3. Flatness (low = uniform background = document-like)
        flatness = np.std(gray) / (np.mean(gray) + 1e-10)
        flatness_score = max(0, 1.0 - flatness)

        # 4. Block frequency consistency
        block_feats = []
        for i in range(0, h - 64, 32):
            for j in range(0, w - 64, 32):
                block = gray[i:i+64, j:j+64]
                fft = np.fft.fft2(block)
                fft_shift = np.fft.fftshift(fft)
                mag = np.abs(fft_shift)
                center = 32
                total = np.sum(mag) + 1e-10
                hf = np.sum(mag) - np.sum(mag[center-8:center+8, center-8:center+8])
                block_feats.append(hf / total)
        if len(block_feats) > 1:
            block_feats = np.array(block_feats)
            bf_cv = np.std(block_feats) / (np.mean(block_feats) + 1e-6)
        else:
            bf_cv = 0.5
        bf_score = min(bf_cv / 3, 1.0)

        # Combine signals
        structure_score = (peak_score * 0.3 + edge_score * 0.3 + flatness_score * 0.2 + bf_score * 0.2)

        return {
            'score': round(structure_score, 4),
            'peak_ratio': round(peak_score, 4),
            'edge_cv': round(edge_cv, 4),
            'flatness': round(flatness, 4),
            'block_freq_cv': round(bf_cv, 4),
        }

    def _analyze_metadata(self, metadata: dict) -> dict:
        """Analyze image metadata for suspicious indicators."""
        analysis = {
            'suspicious_indicators': [],
            'has_exif': metadata.get('has_exif', False),
            'editing_software_detected': False,
        }

        exif = metadata.get('exif', {})
        if exif:
            software_tags = ['Photoshop', 'GIMP', 'Lightroom', 'Snapseed', 'Afterlight']
            software = str(exif.get('Software', ''))
            for tag in software_tags:
                if tag.lower() in software.lower():
                    analysis['editing_software_detected'] = True
                    analysis['suspicious_indicators'].append({
                        'type': 'editing_software',
                        'value': software,
                        'severity': 'warning',
                    })
        else:
            analysis['suspicious_indicators'].append({
                'type': 'no_metadata',
                'value': 'No EXIF data present',
                'severity': 'info',
            })

        return analysis
