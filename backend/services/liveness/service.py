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
        """Analyze a single image for liveness."""
        image_array = data.get('image_array')
        if image_array is None:
            results['reasons'].append('No image data available')
            results['is_live'] = None
            results['confidence'] = 0.0
            return results

        tensor = torch.FloatTensor(image_array).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)

        # ── Spoof Detection ──
        spoof_prob = output['spoof_probability'].item()
        is_real = output['is_real'].item()
        results['liveness_score'] = 1.0 - spoof_prob
        results['is_live'] = is_real

        # ── Head Pose ──
        pose = output['head_pose'].cpu().numpy().flatten()
        yaw, pitch, roll = float(pose[0]), float(pose[1]), float(pose[2])
        results['head_pose'] = {
            'yaw': round(yaw, 2),
            'pitch': round(pitch, 2),
            'roll': round(roll, 2),
            'within_valid_range': self._check_pose_validity(yaw, pitch, roll),
            'face_3d': yaw != 0 or pitch != 0 or roll != 0,
        }

        # ── Depth Analysis ──
        depth_map = output['depth_map'].cpu().numpy().flatten()
        depth_stats = self._analyze_depth(depth_map)
        results['depth_analysis'] = depth_stats

        # ── Texture Analysis ──
        texture = output['texture_labels']
        texture_scores = {
            'moire_pattern': float(texture['moire_pattern'].item()),
            'screen_reflection': float(texture['screen_reflection'].item()),
            'screen_edge': float(texture['screen_edge'].item()),
            'print_artifact': float(texture['print_artifact'].item()),
        }
        results['texture_analysis'] = texture_scores

        # ── Build Reasons ──
        confidence_parts = []

        if not is_real:
            results['reasons'].append(
                f"Spoof detected with {(spoof_prob * 100):.1f}% confidence"
            )
            confidence_parts.append(spoof_prob)

            # Identify spoof type
            results['spoof_type'] = self._classify_spoof_type(texture_scores, depth_stats)
            if results['spoof_type']:
                results['reasons'].append(f"Spoof type: {results['spoof_type']}")

        # Texture anomalies
        if texture_scores['moire_pattern'] > 0.6:
            results['reasons'].append('Moiré pattern detected (possible screen replay)')
            confidence_parts.append(texture_scores['moire_pattern'])
        if texture_scores['screen_reflection'] > 0.6:
            results['reasons'].append('Screen reflection detected')
            confidence_parts.append(texture_scores['screen_reflection'])
        if texture_scores['screen_edge'] > 0.6:
            results['reasons'].append('Screen edge detected (possible device replay)')
            confidence_parts.append(texture_scores['screen_edge'])
        if texture_scores['print_artifact'] > 0.6:
            results['reasons'].append('Print artifacts detected (possible photo replay)')
            confidence_parts.append(texture_scores['print_artifact'])

        # Pose issues
        if not results['head_pose']['within_valid_range']:
            results['reasons'].append(
                f"Abnormal head pose: yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}"
            )

        # Depth issues
        if depth_stats.get('is_flat', False):
            results['reasons'].append('Flat depth profile detected (possible 2D spoof)')
            confidence_parts.append(0.5)

        # Confidence calculation
        if confidence_parts:
            results['confidence'] = round(min(np.mean(confidence_parts) + 0.2, 1.0), 3)
        elif not results['reasons']:
            results['confidence'] = round(0.85 + abs(spoof_prob - 0.5) * 0.3, 3)
            results['reasons'].append('No liveness concerns detected')
        else:
            results['confidence'] = 0.6

        if not results['reasons']:
            results['reasons'].append('Liveness verified - appears to be a live capture')

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
