"""
Document Intelligence Service.
OCR, font analysis, tampering detection, metadata extraction.
"""
import numpy as np
from typing import Optional
import re
import hashlib
from collections import Counter


class DocumentIntelligenceService:
    """Analyze documents for authenticity, tampering, and inconsistencies."""

    def __init__(self):
        self.ocr_engine = None
        self._init_ocr()

    def _init_ocr(self):
        """Initialize OCR engine (EasyOCR preferred, fallback to basic)."""
        try:
            import easyocr
            self.ocr_engine = easyocr.Reader(['en'], gpu=False)
        except ImportError:
            try:
                import pytesseract
                self.ocr_engine = 'tesseract'
            except ImportError:
                self.ocr_engine = 'basic'

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

        doc_type = preprocessed_data.get('document_type', 'image_based')

        if doc_type == 'pdf':
            results = await self._analyze_pdf(preprocessed_data, results)
        else:
            results = await self._analyze_image(preprocessed_data, results)

        # Metadata analysis
        metadata = preprocessed_data.get('metadata', {})
        results['metadata_analysis'] = self._analyze_metadata(metadata)

        # Compute final authenticity score
        results['authenticity_score'] = self._compute_authenticity_score(results)

        return results

    async def _analyze_image(self, data: dict, results: dict) -> dict:
        """Analyze a single image document."""
        original = data.get('original_image')
        if original is None:
            return results

        # OCR
        ocr_result = self._run_ocr(original)
        results['ocr_confidence'] = ocr_result.get('confidence', 0.0)

        # Font analysis via pixel-level statistics
        font_issues = self._detect_font_inconsistencies(original)
        results['font_inconsistencies'] = font_issues
        if font_issues:
            results['tampering_detected'] = True
            results['reasons'].append(f"Detected {len(font_issues)} font inconsistency(ies)")

        # Spacing analysis
        spacing_issues = self._detect_spacing_anomalies(original)
        results['spacing_anomalies'] = spacing_issues
        if spacing_issues:
            results['tampering_detected'] = True
            results['reasons'].append(f"Detected {len(spacing_issues)} spacing anomaly(ies)")

        # Tampered region detection using frequency analysis
        tampered = self._detect_tampered_regions(original)
        results['tampered_regions'] = tampered
        if tampered:
            results['tampering_detected'] = True
            results['reasons'].append(f"Detected {len(tampered)} potentially tampered region(s)")

        return results

    async def _analyze_pdf(self, data: dict, results: dict) -> dict:
        """Analyze a PDF document page by page."""
        pages = data.get('pages', [])
        all_font_issues = []
        all_spacing = []
        all_tampered = []
        all_reasons = []

        for page in pages:
            page_img = page.get('image')
            if page_img is None:
                continue

            font_issues = self._detect_font_inconsistencies(page_img)
            spacing = self._detect_spacing_anomalies(page_img)
            tampered = self._detect_tampered_regions(page_img)

            all_font_issues.extend([{**f, 'page': page['page_num']} for f in font_issues])
            all_spacing.extend([{**s, 'page': page['page_num']} for s in spacing])
            all_tampered.extend([{**t, 'page': page['page_num']} for t in tampered])

        results['font_inconsistencies'] = all_font_issues
        results['spacing_anomalies'] = all_spacing
        results['tampered_regions'] = all_tampered

        if all_font_issues:
            results['tampering_detected'] = True
            all_reasons.append(f"Font inconsistencies across {len(pages)} page(s)")
        if all_spacing:
            results['tampering_detected'] = True
            all_reasons.append(f"Spacing anomalies detected")
        if all_tampered:
            results['tampering_detected'] = True
            all_reasons.append(f"Tampered regions found")

        results['reasons'] = all_reasons

        # PDF metadata analysis
        pdf_meta = data.get('metadata', {})
        results['metadata_analysis'] = self._analyze_metadata(pdf_meta)

        return results

    def _run_ocr(self, image: np.ndarray) -> dict:
        """Run OCR on an image."""
        if self.ocr_engine is None:
            return {'text': '', 'confidence': 0.0}

        if self.ocr_engine == 'basic':
            return self._basic_ocr(image)

        try:
            if hasattr(self.ocr_engine, 'readtext'):
                results = self.ocr_engine.readtext(image)
                text_parts = []
                confidences = []
                for (bbox, text, conf) in results:
                    text_parts.append(text)
                    confidences.append(conf)
                return {
                    'text': ' '.join(text_parts),
                    'confidence': np.mean(confidences) if confidences else 0.0,
                    'details': results,
                }
        except Exception:
            pass

        return self._basic_ocr(image)

    def _basic_ocr(self, image: np.ndarray) -> dict:
        """Fallback basic OCR using image statistics."""
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
        # Simple threshold-based character detection
        threshold = 128
        binary = (gray < threshold).astype(float)
        # Estimate text density
        text_density = np.mean(binary)
        return {
            'text': f'[Basic analysis: text density={text_density:.3f}]',
            'confidence': min(text_density * 5, 1.0),
        }

    def _detect_font_inconsistencies(self, image: np.ndarray) -> list:
        """Detect font inconsistencies via pixel-level analysis."""
        issues = []
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image.astype(float)

        h, w = gray.shape
        if h < 10 or w < 10:
            return issues

        # Divide into grid cells and analyze local statistics
        grid_size = 8
        cell_h, cell_w = h // grid_size, w // grid_size

        local_means = []
        local_stds = []
        local_sharpness = []

        for i in range(grid_size):
            for j in range(grid_size):
                cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                local_means.append(np.mean(cell))
                local_stds.append(np.std(cell))
                # Sharpness via Laplacian variance
                if cell.shape[0] > 2 and cell.shape[1] > 2:
                    laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]])
                    from scipy.ndimage import convolve
                    filtered = convolve(cell, laplacian)
                    local_sharpness.append(np.var(filtered))
                else:
                    local_sharpness.append(0.0)

        if len(local_means) < 4:
            return issues

        # Detect inconsistencies (outliers in local statistics)
        mean_std = np.std(local_means)
        sharpness_std = np.std(local_sharpness) if local_sharpness else 0
        mean_sharpness = np.mean(local_sharpness) if local_sharpness else 0

        if mean_std > 30:
            issues.append({
                'type': 'brightness_variation',
                'severity': 'medium',
                'description': f'Unusual brightness variation (std={mean_std:.1f})',
                'score': min(mean_std / 50.0, 1.0),
            })

        if mean_sharpness > 0 and sharpness_std / mean_sharpness > 0.5:
            issues.append({
                'type': 'sharpness_inconsistency',
                'severity': 'medium',
                'description': 'Inconsistent sharpness suggests editing',
                'score': min(sharpness_std / (mean_sharpness + 1e-6), 1.0),
            })

        return issues

    def _detect_spacing_anomalies(self, image: np.ndarray) -> list:
        """Detect spacing anomalies using horizontal projection analysis."""
        issues = []
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image.astype(float)

        # Horizontal projection (sum of dark pixels per row)
        threshold = 128
        binary = (gray < threshold).astype(float)
        projection = np.sum(binary, axis=1)

        # Analyze gaps (rows with very few dark pixels)
        text_rows = projection > np.max(projection) * 0.05
        gap_sizes = []
        current_gap = 0

        for is_text in text_rows:
            if not is_text:
                current_gap += 1
            else:
                if current_gap > 0:
                    gap_sizes.append(current_gap)
                current_gap = 0

        if len(gap_sizes) > 2:
            gap_std = np.std(gap_sizes)
            gap_mean = np.mean(gap_sizes)
            if gap_mean > 0 and gap_std / gap_mean > 0.8:
                issues.append({
                    'type': 'irregular_spacing',
                    'severity': 'low',
                    'description': f'Irregular line spacing detected (cv={gap_std/gap_mean:.2f})',
                    'score': min(gap_std / (gap_mean + 1e-6) / 2.0, 1.0),
                })

        return issues

    def _detect_tampered_regions(self, image: np.ndarray) -> list:
        """Detect potentially tampered regions using frequency analysis."""
        regions = []
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image.astype(float)

        h, w = gray.shape
        if h < 32 or w < 32:
            return regions

        # Divide into blocks and analyze FFT consistency
        block_size = 64
        num_blocks_h = h // block_size
        num_blocks_w = w // block_size

        if num_blocks_h < 2 or num_blocks_w < 2:
            return regions

        spectral_energies = []
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                fft = np.fft.fft2(block)
                fft_shift = np.fft.fftshift(fft)
                magnitude = np.abs(fft_shift)
                # High-frequency energy ratio
                center = block_size // 2
                total_energy = np.sum(magnitude)
                high_freq_energy = np.sum(magnitude) - np.sum(magnitude[center-8:center+8, center-8:center+8])
                ratio = high_freq_energy / (total_energy + 1e-6)
                spectral_energies.append({
                    'ratio': ratio,
                    'block_i': i,
                    'block_j': j,
                    'x': j * block_size,
                    'y': i * block_size,
                })

        if len(spectral_energies) < 4:
            return regions

        ratios = [e['ratio'] for e in spectral_energies]
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)

        # Blocks that deviate significantly may be tampered
        for energy in spectral_energies:
            if std_ratio > 0 and abs(energy['ratio'] - mean_ratio) > 2 * std_ratio:
                regions.append({
                    'x': energy['x'],
                    'y': energy['y'],
                    'w': block_size,
                    'h': block_size,
                    'confidence': min(abs(energy['ratio'] - mean_ratio) / (3 * std_ratio), 1.0),
                    'type': 'spectral_anomaly',
                })

        return regions

    def _analyze_metadata(self, metadata: dict) -> dict:
        """Analyze document/image metadata for suspicious indicators."""
        analysis = {
            'suspicious_indicators': [],
            'has_exif': metadata.get('has_exif', False),
            'editing_software_detected': False,
            'modification_dates': [],
        }

        exif = metadata.get('exif', {})
        if exif:
            # Check for known editing software
            software_tags = ['Photoshop', 'GIMP', 'Lightroom', 'Snapseed', 'Afterlight']
            software = str(exif.get('Software', ''))
            for tag in software_tags:
                if tag.lower() in software.lower():
                    analysis['editing_software_detected'] = True
                    analysis['suspicious_indicators'].append({
                        'type': 'editing_software',
                        'value': software,
                        'severity': 'info',
                    })

            # Check for date inconsistencies
            dates = {}
            for key in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                if key in exif:
                    dates[key] = exif[key]
            if len(set(dates.values())) > 1:
                analysis['suspicious_indicators'].append({
                    'type': 'date_inconsistency',
                    'value': dates,
                    'severity': 'warning',
                })

        else:
            analysis['suspicious_indicators'].append({
                'type': 'no_metadata',
                'value': 'No EXIF data present',
                'severity': 'info',
            })

        return analysis

    def _compute_authenticity_score(self, results: dict) -> float:
        """Compute final document authenticity score 0-1 (1 = authentic)."""
        score = 1.0

        # Penalize for each type of issue
        for issue in results.get('font_inconsistencies', []):
            severity = issue.get('severity', 'low')
            penalty = {'high': 0.3, 'medium': 0.15, 'low': 0.05}.get(severity, 0.05)
            score -= penalty * issue.get('score', 0.5)

        for issue in results.get('spacing_anomalies', []):
            score -= 0.1 * issue.get('score', 0.5)

        for region in results.get('tampered_regions', []):
            score -= 0.2 * region.get('confidence', 0.5)

        # Penalize for metadata issues
        meta = results.get('metadata_analysis', {})
        if meta.get('editing_software_detected'):
            score -= 0.05
        for indicator in meta.get('suspicious_indicators', []):
            if indicator.get('severity') == 'warning':
                score -= 0.05

        return max(0.0, min(1.0, score))
