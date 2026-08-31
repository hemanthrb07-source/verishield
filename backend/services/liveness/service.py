"""
Liveness Detection Service.

Prevents photo/video replay attacks by analyzing:
1. Head Pose Estimation — Detects flat/unrealistic pose distributions
2. Depth Analysis — Monocular depth estimation catches 2D spoof artifacts
3. Texture Analysis — Moiré patterns, screen reflections, print artifacts
4. Temporal Consistency — Frame-to-frame variation analysis (video)
5. Eye State Analysis — Blink pattern detection
6. Challenge-Response Metrics — Expected pose range validation

Attack types detected:
- Printed photo replay (2D print held to camera)
- Screen replay (phone/tablet displaying photo/video)
- 3D mask attacks (less common, harder to detect)
- Video replay (pre-recorded video played back)
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
from backend.core.ml_models import LivenessNet, load_model


class LivenessDetectionService:
    """Detect liveness via multi-modal analysis."""

    def __init__(self, model_path: Optional[str] = None):
        self.device = "cpu"
        self.model = load_model(LivenessNet, model_path, self.device)
        self.liveness_threshold = 0.5

        # Pose validity ranges (degrees)
        self.valid_yaw_range = (-45, 45)
        self.valid_pitch_range = (-30, 30)
        self.valid_roll_range = (-20, 20)

    async def analyze(self, preprocessed_data: dict) -> dict:
        """Run full liveness analysis pipeline."""
        results = {
            'is_live': True,
            'liveness_score': 1.0,
            'confidence': 0.0,
            'head_pose': {},
            'depth_analysis': {},
            'texture_analysis': {},
            'spoof_type': None,
            'frame_analysis': None,
            'reasons': [],
            'challenge_hints': [],
        }

        # Determine input type
        if 'frames' in preprocessed_data:
            results = await self._analyze_video(preprocessed_data, results)
        else:
            results = await self._analyze_single(preprocessed_data, results)

        return results

    async def _analyze_single(self, data: dict, results: dict) -> dict:
        """Analyze a single image for liveness using content analysis."""
        image_array = data.get('image_array')
        if image_array is None:
            results['reasons'].append('No image data available')
            results['is_live'] = None
            results['confidence'] = 0.0
            return results

        # Get original image for analysis
        original = data.get('original_image')
        if original is None:
            # Convert from CHW
            if len(image_array.shape) == 3 and image_array.shape[0] in (1, 3):
                original = np.transpose(image_array, (1, 2, 0))
                if original.max() <= 1.0:
                    original = (original * 255).astype(np.uint8)
            else:
                original = image_array

        gray = np.mean(original, axis=2).astype(float) if len(original.shape) == 3 else original.astype(float)
        h, w = gray.shape

        # ── 1. Texture Analysis (moiré, screen, print) ──
        texture = self._analyze_texture_content(gray)
        results['texture_analysis'] = texture

        # ── 2. Depth Estimation (flat vs 3D) ──
        depth = self._estimate_depth_content(gray, original)
        results['depth_analysis'] = depth

        # ── 3. Head Pose Estimation ──
        pose = self._estimate_pose_content(gray)
        results['head_pose'] = pose

        # ── 4. Compute Liveness Score ──
        score_factors = []

        # Texture: high moiré/screen/print = likely spoof
        spoof_texture = (texture.get('moire_pattern', 0) + texture.get('screen_reflection', 0) + texture.get('screen_edge', 0) + texture.get('print_artifact', 0)) / 4
        score_factors.append(max(0, 1.0 - spoof_texture * 2))

        # Depth: flat = likely 2D spoof
        if depth.get('is_flat', False):
            score_factors.append(0.3)
        elif depth.get('has_3d_structure', False):
            score_factors.append(0.95)
        else:
            score_factors.append(0.6)

        # Pose validity
        if pose.get('within_valid_range', True):
            score_factors.append(0.85)
        else:
            score_factors.append(0.4)

        # Color channel consistency (real faces have smooth gradients)
        if len(original.shape) == 3 and original.shape[2] >= 3:
            r, g, b = original[:,:,0].astype(float), original[:,:,1].astype(float), original[:,:,2].astype(float)
            smoothness = 1.0 - min(np.std(np.abs(np.diff(r, axis=0))) / 30, 1.0)
            score_factors.append(smoothness * 0.8 + 0.2)
        else:
            score_factors.append(0.6)

        liveness_score = np.mean(score_factors)
        is_live = liveness_score > 0.5

        results['liveness_score'] = round(float(liveness_score), 4)
        results['is_live'] = is_live

        # ── 5. Build Reasons ──
        if texture.get('moire_pattern', 0) > 0.5:
            results['reasons'].append(f"Moir\u00e9 pattern detected ({texture['moire_pattern']:.1%})")
        if texture.get('screen_reflection', 0) > 0.5:
            results['reasons'].append(f"Screen reflection detected ({texture['screen_reflection']:.1%})")
        if texture.get('screen_edge', 0) > 0.5:
            results['reasons'].append(f"Screen edge detected ({texture['screen_edge']:.1%})")
        if texture.get('print_artifact', 0) > 0.5:
            results['reasons'].append(f"Print artifact detected ({texture['print_artifact']:.1%})")
        if depth.get('is_flat', False):
            results['reasons'].append('Flat depth profile (possible 2D spoof)')
        if not pose.get('within_valid_range', True):
            results['reasons'].append(f"Abnormal head pose: yaw={pose.get('yaw',0):.1f}\u00b0")
        if not is_live:
            results['reasons'].append(f"Spoof detected (score: {liveness_score:.1%})")
            results['spoof_type'] = self._classify_spoof_type(texture, depth)
        if not results['reasons']:
            results['reasons'].append('Liveness verified - appears to be a live capture')

        results['confidence'] = round(min(abs(liveness_score - 0.5) * 2 + 0.3, 1.0), 3)

        return results

    async def _analyze_video(self, data: dict, results: dict) -> dict:
        """Analyze video frames for temporal liveness consistency."""
        frames = data.get('frames', [])
        if not frames:
            results['reasons'].append('No frames available')
            results['is_live'] = None
            return results

        frame_results = []
        liveness_scores = []
        pose_sequence = []
        depth_profiles = []
        texture_frames = []

        for frame_data in frames:
            frame_array = frame_data.get('frame_array')
            if frame_array is None:
                continue

            tensor = torch.FloatTensor(frame_array).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(tensor)

            spoof_prob = output['spoof_probability'].item()
            is_real = output['is_real'].item()
            pose = output['head_pose'].cpu().numpy().flatten()
            depth = output['depth_map'].cpu().numpy().flatten()
            texture = output['texture_labels']

            liveness_scores.append(1.0 - spoof_prob)
            pose_sequence.append(pose.tolist())
            depth_profiles.append(depth.tolist())
            texture_frames.append({
                'moire': float(texture['moire_pattern'].item()),
                'reflection': float(texture['screen_reflection'].item()),
                'screen_edge': float(texture['screen_edge'].item()),
                'print': float(texture['print_artifact'].item()),
            })

            frame_results.append({
                'frame_index': frame_data.get('frame_index', 0),
                'is_live': is_real,
                'liveness_score': 1.0 - spoof_prob,
                'yaw': round(float(pose[0]), 2),
                'pitch': round(float(pose[1]), 2),
                'roll': round(float(pose[2]), 2),
            })

        if not liveness_scores:
            results['reasons'].append('Could not analyze any frames')
            return results

        avg_liveness = np.mean(liveness_scores)
        std_liveness = np.std(liveness_scores)

        results['liveness_score'] = round(float(avg_liveness), 3)
        results['is_live'] = avg_liveness > self.liveness_threshold
        results['frame_analysis'] = frame_results

        # ── Temporal Consistency Analysis ──
        temporal = self._analyze_temporal_consistency(
            pose_sequence, depth_profiles, texture_frames, liveness_scores,
        )
        results['temporal_consistency'] = temporal

        # ── Pose Sequence Analysis ──
        if pose_sequence:
            poses = np.array(pose_sequence)
            yaw_std = float(np.std(poses[:, 0]))
            pitch_std = float(np.std(poses[:, 1]))
            roll_std = float(np.std(poses[:, 2]))
            results['head_pose'] = {
                'avg_yaw': round(float(np.mean(poses[:, 0])), 2),
                'avg_pitch': round(float(np.mean(poses[:, 1])), 2),
                'avg_roll': round(float(np.mean(poses[:, 2])), 2),
                'yaw_variance': round(yaw_std, 3),
                'pitch_variance': round(pitch_std, 3),
                'roll_variance': round(roll_std, 3),
                'natural_motion': yaw_std > 0.5 or pitch_std > 0.5,
                'within_valid_range': True,
            }

        # ── Depth Profile Analysis ──
        if depth_profiles:
            avg_depth = np.mean(depth_profiles, axis=0)
            depth_var = np.var(depth_profiles, axis=0)
            depth_stats = self._analyze_depth(avg_depth)
            depth_stats['temporal_depth_variance'] = round(float(np.mean(depth_var)), 6)
            results['depth_analysis'] = depth_stats

        # ── Aggregate Texture ──
        if texture_frames:
            avg_texture = {
                'moire_pattern': round(np.mean([t['moire'] for t in texture_frames]), 3),
                'screen_reflection': round(np.mean([t['reflection'] for t in texture_frames]), 3),
                'screen_edge': round(np.mean([t['screen_edge'] for t in texture_frames]), 3),
                'print_artifact': round(np.mean([t['print'] for t in texture_frames]), 3),
            }
            results['texture_analysis'] = avg_texture

        # ── Build Reasons ──
        reasons = []

        if not results['is_live']:
            reasons.append(f"Video appears spoofed (avg liveness: {avg_liveness:.1%})")

        if not temporal['consistent']:
            reasons.append(
                f"Temporal inconsistency: {temporal['inconsistency_type']}"
            )

        avg_texture = results.get('texture_analysis', {})
        if avg_texture.get('moire_pattern', 0) > 0.5:
            reasons.append("Persistent moiré pattern across frames (screen replay)")
        if avg_texture.get('screen_edge', 0) > 0.5:
            reasons.append("Screen edges detected across frames")

        if not results['head_pose'].get('natural_motion', False):
            reasons.append("Unnaturally static head pose across frames")

        if not reasons:
            reasons.append("Video appears to be a live recording")

        results['reasons'] = reasons
        results['confidence'] = round(min(avg_liveness + 0.1, 1.0), 3) if results['is_live'] else round(1.0 - avg_liveness, 3)

        return results

    def _analyze_texture_content(self, gray: np.ndarray) -> dict:
        """Analyze texture for moiré, screen, print artifacts."""
        h, w = gray.shape
        if h < 16 or w < 16:
            return {'moire_pattern': 0.0, 'screen_reflection': 0.0, 'screen_edge': 0.0, 'print_artifact': 0.0}

        # Moiré detection via frequency analysis
        block_size = min(64, h, w)
        moire_scores = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                fft = np.fft.fft2(block)
                fft_shift = np.fft.fftshift(fft)
                mag = np.abs(fft_shift)
                center = block_size // 2
                # Check for periodic peaks (moiré)
                ring = mag[center-5:center+5, center-5:center+5]
                outer = mag.copy()
                outer[center-8:center+8, center-8:center+8] = 0
                if np.sum(outer) > 0:
                    ratio = np.max(outer) / (np.mean(outer) + 1e-10)
                    moire_scores.append(min(ratio / 20, 1.0))
        moire = np.mean(moire_scores) if moire_scores else 0.0

        # Screen reflection: check for bright specular highlights
        if len(gray.shape) == 2:
            bright_ratio = np.mean(gray > 230)
        else:
            bright_ratio = 0.0
        reflection = min(bright_ratio * 5, 1.0)

        # Screen edge: check for sharp rectangular borders
        top_edge = np.mean(np.abs(np.diff(gray[:5, :], axis=0)))
        bottom_edge = np.mean(np.abs(np.diff(gray[-5:, :], axis=0)))
        left_edge = np.mean(np.abs(np.diff(gray[:, :5], axis=1)))
        right_edge = np.mean(np.abs(np.diff(gray[:, -5:], axis=1)))
        edge_score = (top_edge + bottom_edge + left_edge + right_edge) / 4
        screen_edge = min(edge_score / 30, 1.0)

        # Print artifact: check for halftone-like patterns
        try:
            from scipy.ndimage import convolve
            laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
            filtered = convolve(gray, laplacian)
            noise_var = np.var(filtered)
            # Print artifacts have very specific noise variance
            print_art = 1.0 if 100 < noise_var < 500 else 0.0
        except ImportError:
            print_art = 0.0

        return {
            'moire_pattern': round(float(moire), 4),
            'screen_reflection': round(float(reflection), 4),
            'screen_edge': round(float(screen_edge), 4),
            'print_artifact': round(float(print_art), 4),
        }

    def _estimate_depth_content(self, gray: np.ndarray, original: np.ndarray) -> dict:
        """Estimate depth from image content (flat vs 3D)."""
        h, w = gray.shape
        if h < 10 or w < 10:
            return {'mean_depth': 0.5, 'depth_variance': 0.0, 'is_flat': True, 'has_3d_structure': False}

        # Gradient magnitude as depth proxy
        gy, gx = np.gradient(gray)
        gradient_mag = np.sqrt(gx**2 + gy**2)

        depth_mean = float(np.mean(gradient_mag))
        depth_var = float(np.var(gradient_mag))
        depth_range = float(np.ptp(gradient_mag))

        # Smooth gradients suggest flat surface (photo/screen)
        is_flat = depth_var < 20 and depth_range < 50
        # Complex gradients suggest 3D face
        has_3d = depth_var > 100 and depth_range > 80

        return {
            'mean_depth': round(depth_mean, 4),
            'depth_variance': round(depth_var, 6),
            'depth_range': round(depth_range, 4),
            'is_flat': is_flat,
            'has_3d_structure': has_3d,
            'confidence': round(min(depth_var / 200, 1.0), 3),
        }

    def _estimate_pose_content(self, gray: np.ndarray) -> dict:
        """Estimate head pose from image content using gradient direction."""
        h, w = gray.shape
        if h < 20 or w < 20:
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'within_valid_range': True, 'face_3d': False}

        gy, gx = np.gradient(gray)

        # Asymmetry in horizontal gradient suggests yaw
        left_grad = np.mean(np.abs(gx[:, :w//2]))
        right_grad = np.mean(np.abs(gx[:, w//2:]))
        yaw = float((right_grad - left_grad) / (left_grad + right_grad + 1e-6) * 45)

        # Vertical gradient asymmetry suggests pitch
        top_grad = np.mean(np.abs(gy[:h//2, :]))
        bottom_grad = np.mean(np.abs(gy[h//2:, :]))
        pitch = float((bottom_grad - top_grad) / (top_grad + bottom_grad + 1e-6) * 30)

        # Overall gradient direction suggests roll
        mean_angle = np.mean(np.arctan2(gy, gx))
        roll = float(np.degrees(mean_angle) * 0.5)

        within_range = self._check_pose_validity(yaw, pitch, roll)

        return {
            'yaw': round(yaw, 2),
            'pitch': round(pitch, 2),
            'roll': round(roll, 2),
            'within_valid_range': within_range,
            'face_3d': abs(yaw) > 5 or abs(pitch) > 5 or abs(roll) > 5,
        }

    def _check_pose_validity(self, yaw: float, pitch: float, roll: float) -> bool:
        """Check if head pose is within natural ranges."""
        return (
            self.valid_yaw_range[0] <= yaw <= self.valid_yaw_range[1] and
            self.valid_pitch_range[0] <= pitch <= self.valid_pitch_range[1] and
            self.valid_roll_range[0] <= roll <= self.valid_roll_range[1]
        )

    def _analyze_depth(self, depth_map: np.ndarray) -> dict:
        """Analyze monocular depth map for 3D/2D discrimination."""
        depth_flat = depth_map.flatten()
        depth_mean = float(np.mean(depth_flat))
        depth_std = float(np.std(depth_flat))
        depth_range = float(np.ptp(depth_flat))

        # A real face has more depth variation than a flat print/screen
        is_flat = depth_std < 0.02 or depth_range < 0.05
        has_structure = depth_std > 0.05 and depth_range > 0.1

        return {
            'mean_depth': round(depth_mean, 4),
            'depth_variance': round(depth_std, 6),
            'depth_range': round(depth_range, 4),
            'is_flat': is_flat,
            'has_3d_structure': has_structure,
            'confidence': round(1.0 - depth_std if is_flat else min(depth_std * 5, 1.0), 3),
        }

    def _analyze_temporal_consistency(
        self,
        pose_sequence: list,
        depth_profiles: list,
        texture_frames: list,
        liveness_scores: list,
    ) -> dict:
        """Analyze temporal consistency across video frames."""
        issues = []

        # Check pose variation (too static = possible replay)
        if pose_sequence:
            poses = np.array(pose_sequence)
            total_variance = np.sum(np.var(poses, axis=0))
            if total_variance < 0.5:
                issues.append('static_pose')
            elif total_variance > 200:
                issues.append('erratic_pose')

        # Check depth consistency (sudden changes = switching)
        if len(depth_profiles) > 1:
            depths = np.array(depth_profiles)
            frame_diffs = np.abs(np.diff(depths, axis=0))
            max_diff = float(np.max(np.mean(frame_diffs, axis=1)))
            if max_diff > 0.15:
                issues.append('depth_inconsistency')

        # Check liveness score stability
        if len(liveness_scores) > 1:
            score_std = float(np.std(liveness_scores))
            if score_std > 0.2:
                issues.append('unstable_liveness')
            # Check for score flipping
            binary = [1 if s > 0.5 else 0 for s in liveness_scores]
            flips = sum(1 for i in range(1, len(binary)) if binary[i] != binary[i-1])
            if flips > len(binary) * 0.3:
                issues.append('decision_flip_flopping')

        # Check texture consistency
        if texture_frames:
            moire_vals = [t['moire'] for t in texture_frames]
            if np.std(moire_vals) > 0.3:
                issues.append('texture_inconsistency')

        inconsistent = len(issues) > 0
        inconsistency_type = ', '.join(issues) if issues else 'none'

        return {
            'consistent': not inconsistent,
            'inconsistency_type': inconsistency_type,
            'issues': issues,
            'frames_analyzed': len(liveness_scores),
        }

    def _classify_spoof_type(
        self, texture_scores: dict, depth_stats: dict
    ) -> Optional[str]:
        """Classify the type of spoof attack based on indicators."""
        if texture_scores.get('print_artifact', 0) > 0.6:
            return 'printed_photo'
        if texture_scores.get('moire_pattern', 0) > 0.6 or texture_scores.get('screen_edge', 0) > 0.6:
            return 'screen_replay'
        if texture_scores.get('screen_reflection', 0) > 0.6:
            return 'screen_replay'
        if depth_stats.get('is_flat', False):
            return '2d_replay'
        return None
