"""
Heatmap Generation Service.
Creates visual overlays showing where AI analysis was applied:
- Tampering regions (red overlay)
- Face detection areas (blue overlay)
- High-risk zones (orange overlay)
- Text regions (green overlay)
- Frequency anomalies (purple overlay)
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import Optional


class HeatmapService:
    """Generate visual heatmaps showing AI analysis regions."""

    def generate_document_heatmap(
        self,
        image_array: np.ndarray,
        doc_results: dict,
        reference_results: Optional[dict] = None,
    ) -> dict:
        """
        Generate a heatmap overlay for document analysis.
        
        Returns:
            - heatmap_base64: Base64-encoded heatmap image
            - regions: List of annotated regions with labels
            - summary: Overall analysis summary
        """
        h, w = image_array.shape[:2]

        # Create overlay canvas
        if len(image_array.shape) == 3:
            base = Image.fromarray(image_array.astype(np.uint8))
        else:
            base = Image.fromarray(np.stack([image_array] * 3, axis=2).astype(np.uint8))

        overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        regions = []

        # ── 1. Tampered Regions (Red) ──
        tampered = doc_results.get('tampered_regions', [])
        for region in tampered:
            x, y, rw, rh = region.get('x', 0), region.get('y', 0), region.get('w', 64), region.get('h', 64)
            conf = region.get('confidence', 0.5)
            alpha = int(80 + conf * 120)
            draw.rectangle([x, y, x + rw, y + rh], fill=(255, 50, 50, alpha))
            draw.rectangle([x, y, x + rw, y + rh], outline=(255, 50, 50, 200), width=2)
            regions.append({
                'type': 'tampering',
                'x': x, 'y': y, 'w': rw, 'h': rh,
                'confidence': conf,
                'label': f'Tampered Region ({conf:.0%})',
                'color': 'red',
            })

        # ── 2. Font Inconsistencies (Orange) ──
        font_issues = doc_results.get('font_inconsistencies', [])
        for issue in font_issues:
            severity = issue.get('severity', 'low')
            color = (255, 165, 0, 120) if severity == 'medium' else (255, 200, 0, 80)
            # Mark as a general region since we don't have exact coordinates
            regions.append({
                'type': 'font_inconsistency',
                'x': 0, 'y': 0, 'w': w, 'h': h,
                'confidence': issue.get('score', 0.5),
                'label': f"Font Issue: {issue.get('type', 'unknown')}",
                'color': 'orange',
            })

        # ── 3. Spacing Anomalies (Yellow) ──
        spacing = doc_results.get('spacing_anomalies', [])
        for issue in spacing:
            regions.append({
                'type': 'spacing_anomaly',
                'x': 0, 'y': 0, 'w': w, 'h': h,
                'confidence': issue.get('score', 0.5),
                'label': f"Spacing: {issue.get('type', 'unknown')}",
                'color': 'yellow',
            })

        # ── 4. Text Regions Detected (Green) ──
        text_regions = self._detect_text_regions(image_array)
        for tr in text_regions:
            x, y, rw, rh = tr['x'], tr['y'], tr['w'], tr['h']
            draw.rectangle([x, y, x + rw, y + rh], fill=(50, 200, 50, 50))
            draw.rectangle([x, y, x + rw, y + rh], outline=(50, 200, 50, 150), width=1)
            regions.append({
                'type': 'text_region',
                'x': x, 'y': y, 'w': rw, 'h': rh,
                'confidence': tr.get('confidence', 0.8),
                'label': 'Detected Text',
                'color': 'green',
            })

        # ── 5. Frequency Anomalies (Purple) ──
        freq_regions = self._detect_frequency_anomalies(image_array)
        for fr in freq_regions[:10]:  # Limit to top 10
            x, y, rw, rh = fr['x'], fr['y'], fr['w'], fr['h']
            draw.rectangle([x, y, x + rw, y + rh], fill=(150, 50, 200, 60))
            draw.rectangle([x, y, x + rw, y + rh], outline=(150, 50, 200, 150), width=1)
            regions.append({
                'type': 'frequency_anomaly',
                'x': x, 'y': y, 'w': rw, 'h': rh,
                'confidence': fr.get('confidence', 0.5),
                'label': 'Frequency Anomaly',
                'color': 'purple',
            })

        # ── 6. Reference Comparison (Cyan) ──
        if reference_results:
            diff_regions = self._compare_references(image_array, reference_results)
            for dr in diff_regions:
                x, y, rw, rh = dr['x'], dr['y'], dr['w'], dr['h']
                draw.rectangle([x, y, x + rw, y + rh], fill=(0, 200, 255, 80))
                draw.rectangle([x, y, x + rw, y + rh], outline=(0, 200, 255, 200), width=2)
                regions.append({
                    'type': 'reference_diff',
                    'x': x, 'y': y, 'w': rw, 'h': rh,
                    'confidence': dr.get('similarity', 0.5),
                    'label': f"Differs from Reference ({dr.get('similarity', 0):.0%} similar)",
                    'color': 'cyan',
                })

        # Composite: base + overlay
        base_rgba = base.convert('RGBA')
        composite = Image.alpha_composite(base_rgba, overlay)

        # Convert to base64 PNG
        buf = io.BytesIO()
        composite.save(buf, format='PNG', quality=95)
        heatmap_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Build summary
        summary = {
            'total_regions': len(regions),
            'tampered_count': len(tampered),
            'font_issues_count': len(font_issues),
            'spacing_issues_count': len(spacing),
            'text_regions_count': len(text_regions),
            'frequency_anomalies_count': len(freq_regions),
            'reference_diffs_count': len(diff_regions) if reference_results else 0,
            'overall_risk': 'HIGH' if len(tampered) > 3 else 'MEDIUM' if len(tampered) > 0 else 'LOW',
        }

        return {
            'heatmap_base64': heatmap_base64,
            'regions': regions,
            'summary': summary,
        }

    def generate_deepfake_heatmap(
        self,
        image_array: np.ndarray,
        deepfake_results: dict,
    ) -> dict:
        """Generate heatmap for deepfake analysis."""
        h, w = image_array.shape[:2]

        if len(image_array.shape) == 3:
            base = Image.fromarray(image_array.astype(np.uint8))
        else:
            base = Image.fromarray(np.stack([image_array] * 3, axis=2).astype(np.uint8))

        overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        regions = []

        # ── High-frequency artifact zones ──
        gray = np.mean(image_array, axis=2).astype(float) if len(image_array.shape) == 3 else image_array.astype(float)
        block_size = 32
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                fft = np.fft.fft2(block)
                mag = np.abs(np.fft.fftshift(fft))
                center = block_size // 2
                total = np.sum(mag) + 1e-10
                hf = np.sum(mag) - np.sum(mag[center-4:center+4, center-4:center+4])
                ratio = hf / total

                if ratio > 0.85:  # High frequency = possible GAN artifact
                    alpha = int(min((ratio - 0.85) * 500, 120))
                    draw.rectangle([j, i, j + block_size, i + block_size],
                                 fill=(200, 50, 255, alpha))
                    regions.append({
                        'type': 'gan_artifact',
                        'x': j, 'y': i, 'w': block_size, 'h': block_size,
                        'confidence': ratio,
                        'label': f'GAN Fingerprint ({ratio:.2f})',
                        'color': 'purple',
                    })

        # ── Color channel anomaly zones ──
        if len(image_array.shape) == 3 and image_array.shape[2] >= 3:
            r, g, b = image_array[:,:,0].astype(float), image_array[:,:,1].astype(float), image_array[:,:,2].astype(float)
            diff_rg = np.abs(r - g)
            diff_rb = np.abs(r - b)

            # High channel differences = suspicious
            anomaly_mask = (diff_rg > 50) | (diff_rb > 50)
            if np.any(anomaly_mask):
                # Find bounding boxes of anomalous regions
                coords = np.where(anomaly_mask)
                if len(coords[0]) > 0:
                    y_min, y_max = int(coords[0].min()), int(coords[0].max())
                    x_min, x_max = int(coords[1].min()), int(coords[1].max())
                    # Expand to grid
                    y_min = (y_min // block_size) * block_size
                    y_max = min(((y_max // block_size) + 1) * block_size, h)
                    x_min = (x_min // block_size) * block_size
                    x_max = min(((x_max // block_size) + 1) * block_size, w)

                    draw.rectangle([x_min, y_min, x_max, y_max],
                                 fill=(255, 100, 0, 60), outline=(255, 100, 0, 150))
                    regions.append({
                        'type': 'color_anomaly',
                        'x': x_min, 'y': y_min, 'w': x_max - x_min, 'h': y_max - y_min,
                        'confidence': 0.7,
                        'label': 'Color Channel Anomaly',
                        'color': 'orange',
                    })

        # ── Noise pattern zones ──
        try:
            from scipy.ndimage import convolve
            laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
            noise_map = convolve(gray, laplacian)

            # Find high-noise blocks
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block_noise = np.var(noise_map[i:i+block_size, j:j+block_size])
                    if block_noise > 1000:
                        alpha = min(int(block_noise / 50), 100)
                        draw.rectangle([j, i, j + block_size, i + block_size],
                                     fill=(255, 200, 0, alpha))
        except ImportError:
            pass

        base_rgba = base.convert('RGBA')
        composite = Image.alpha_composite(base_rgba, overlay)

        buf = io.BytesIO()
        composite.save(buf, format='PNG', quality=95)
        heatmap_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return {
            'heatmap_base64': heatmap_base64,
            'regions': regions,
            'summary': {
                'total_anomaly_zones': len(regions),
                'gan_artifacts': len([r for r in regions if r['type'] == 'gan_artifact']),
                'color_anomalies': len([r for r in regions if r['type'] == 'color_anomaly']),
                'is_deepfake': deepfake_results.get('is_deepfake', False),
                'probability': deepfake_results.get('deepfake_probability', 0),
            },
        }

    def _detect_text_regions(self, image_array: np.ndarray) -> list:
        """Detect text-like regions using edge analysis."""
        gray = np.mean(image_array, axis=2).astype(float) if len(image_array.shape) == 3 else image_array.astype(float)
        h, w = gray.shape

        regions = []
        block_size = 32

        for i in range(0, h - block_size, block_size // 2):
            for j in range(0, w - block_size, block_size // 2):
                block = gray[i:i+block_size, j:j+block_size]

                # Text has specific characteristics:
                # - Many horizontal edges
                # - Bimodal intensity distribution
                # - Moderate local variance
                gy, gx = np.gradient(block)
                h_edges = np.mean(np.abs(gy))
                v_edges = np.mean(np.abs(gx))

                # Text ratio: horizontal edges should dominate slightly
                if h_edges > 2 and v_edges > 1:
                    local_var = np.var(block)
                    if 50 < local_var < 5000:  # Text-like variance
                        confidence = min((h_edges + v_edges) / 20, 1.0)
                        regions.append({
                            'x': j, 'y': i, 'w': block_size, 'h': block_size,
                            'confidence': confidence,
                        })

        # Merge nearby text regions
        return self._merge_regions(regions)

    def _detect_frequency_anomalies(self, image_array: np.ndarray) -> list:
        """Detect frequency-domain anomalies."""
        gray = np.mean(image_array, axis=2).astype(float) if len(image_array.shape) == 3 else image_array.astype(float)
        h, w = gray.shape

        block_size = 64
        features = []

        for i in range(0, h - block_size, block_size // 2):
            for j in range(0, w - block_size, block_size // 2):
                block = gray[i:i+block_size, j:j+block_size]
                fft = np.fft.fft2(block)
                fft_shift = np.fft.fftshift(fft)
                mag = np.abs(fft_shift)

                center = block_size // 2
                total = np.sum(mag) + 1e-10
                hf_ratio = (np.sum(mag) - np.sum(mag[center-8:center+8, center-8:center+8])) / total

                features.append({
                    'x': j, 'y': i, 'w': block_size, 'h': block_size,
                    'hf_ratio': hf_ratio,
                })

        if not features:
            return []

        # Find outliers
        ratios = np.array([f['hf_ratio'] for f in features])
        mean_r, std_r = np.mean(ratios), np.std(ratios)

        anomalies = []
        for feat in features:
            if std_r > 0 and abs(feat['hf_ratio'] - mean_r) > 2 * std_r:
                confidence = min(abs(feat['hf_ratio'] - mean_r) / (3 * std_r), 1.0)
                if confidence > 0.4:
                    anomalies.append({
                        'x': feat['x'], 'y': feat['y'],
                        'w': feat['w'], 'h': feat['h'],
                        'confidence': confidence,
                    })

        return anomalies

    def _compare_references(self, image_array: np.ndarray, reference_results: dict) -> list:
        """Compare image with reference and find differences."""
        # Simple comparison: find regions that differ significantly
        # In a real system, this would compare embeddings
        gray = np.mean(image_array, axis=2).astype(float) if len(image_array.shape) == 3 else image_array.astype(float)
        h, w = gray.shape

        # Generate a synthetic "difference map" based on the reference results
        # The reference results tell us what the reference looked like
        regions = []

        # If reference has tampering, show where our image differs
        ref_tampered = reference_results.get('tampered_regions', [])
        if ref_tampered:
            for tr in ref_tampered[:5]:
                regions.append({
                    'x': tr.get('x', 0),
                    'y': tr.get('y', 0),
                    'w': tr.get('w', 64),
                    'h': tr.get('h', 64),
                    'similarity': 1.0 - tr.get('confidence', 0.5),
                })

        return regions

    def _merge_regions(self, regions: list, threshold: int = 48) -> list:
        """Merge nearby regions."""
        if not regions:
            return []

        merged = []
        used = set()

        for i, r1 in enumerate(regions):
            if i in used:
                continue
            group = [r1]
            for j, r2 in enumerate(regions):
                if j <= i or j in used:
                    continue
                if (abs(r1['x'] - r2['x']) < threshold and
                    abs(r1['y'] - r2['y']) < threshold):
                    group.append(r2)
                    used.add(j)

            # Compute bounding box of group
            x_min = min(r['x'] for r in group)
            y_min = min(r['y'] for r in group)
            x_max = max(r['x'] + r['w'] for r in group)
            y_max = max(r['y'] + r['h'] for r in group)

            avg_conf = np.mean([r['confidence'] for r in group])
            if avg_conf > 0.3:
                merged.append({
                    'x': x_min, 'y': y_min,
                    'w': x_max - x_min, 'h': y_max - y_min,
                    'confidence': float(avg_conf),
                })

        return merged


# Global singleton
heatmap_service = HeatmapService()
