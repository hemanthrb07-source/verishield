"""
Adversarial Attack Algorithms for Model Robustness Testing.

Implements:
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)
- Universal Patch Attack
- Spatial Transformations (rotation, translation, scaling)
- Gaussian Noise Robustness
- JPEG Compression Robustness
- Brightness/Contrast Perturbation

Reference: Goodfellow et al. "Explaining and Harnessing Adversarial Examples" (2015)
           Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks" (2018)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable


class AdversarialAttacks:
    """Collection of adversarial attack methods."""

    @staticmethod
    def fgsm(
        model: nn.Module,
        image: torch.Tensor,
        label: torch.Tensor,
        epsilon: float = 0.03,
        targeted: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Fast Gradient Sign Method (FGSM).
        Single-step attack that perturbs pixels in the gradient direction.
        
        Args:
            model: Target model (must accept tensor and return dict with 'logits')
            image: Input image tensor [1, C, H, W], requires_grad=False
            label: True label tensor [1]
            epsilon: Perturbation magnitude (0-1 scale, e.g. 8/255 ≈ 0.031)
            targeted: If True, maximize loss (push toward label)
        
        Returns:
            adversarial_image, attack_metadata
        """
        image = image.clone().detach().requires_grad_(True)
        output = model(image)
        logits = output['logits']

        loss = F.cross_entropy(logits, label)

        if targeted:
            loss = -loss

        model.zero_grad()
        loss.backward()

        # Get sign of gradient
        grad_sign = image.grad.data.sign()

        # Create adversarial example
        adv_image = image + epsilon * grad_sign
        adv_image = torch.clamp(adv_image, 0, 1).detach()

        # Compute attack metrics
        with torch.no_grad():
            orig_output = model(image.detach())
            adv_output = model(adv_image)

            orig_pred = orig_output['logits'].argmax(dim=1)
            adv_pred = adv_output['logits'].argmax(dim=1)
            orig_prob = orig_output['probability'].item()
            adv_prob = adv_output['probability'].item()

            perturbation = (adv_image - image.detach()).abs()
            l2_norm = perturbation.norm().item()
            linf_norm = perturbation.max().item()

        metadata = {
            'method': 'FGSM',
            'epsilon': epsilon,
            'original_prediction': orig_pred.item(),
            'adversarial_prediction': adv_pred.item(),
            'original_confidence': orig_prob,
            'adversarial_confidence': adv_prob,
            'prediction_changed': orig_pred.item() != adv_pred.item(),
            'l2_perturbation': round(l2_norm, 6),
            'linf_perturbation': round(linf_norm, 6),
            'success_rate': 1.0 if orig_pred.item() != adv_pred.item() else 0.0,
        }

        return adv_image, metadata

    @staticmethod
    def pgd(
        model: nn.Module,
        image: torch.Tensor,
        label: torch.Tensor,
        epsilon: float = 0.03,
        num_steps: int = 10,
        step_size: float = 0.007,
        random_start: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """
        Projected Gradient Descent (PGD).
        Iterative multi-step FGSM with random initialization.
        Strongest first-order attack.
        
        Args:
            model: Target model
            image: Input image tensor [1, C, H, W]
            label: True label tensor [1]
            epsilon: Maximum perturbation bound
            num_steps: Number of iteration steps
            step_size: Step size per iteration (alpha)
            random_start: Add random noise to initial perturbation
        
        Returns:
            adversarial_image, attack_metadata
        """
        image = image.clone().detach()
        best_adv = image.clone()
        best_loss = float('inf')

        # Random initialization within epsilon ball
        if random_start:
            adv = image + torch.empty_like(image).uniform_(-epsilon, epsilon)
            adv = torch.clamp(adv, 0, 1)
        else:
            adv = image.clone()

        for step in range(num_steps):
            adv = adv.detach().requires_grad_(True)
            output = model(adv)
            logits = output['logits']
            loss = F.cross_entropy(logits, label)

            model.zero_grad()
            loss.backward()

            # PGD step
            adv = adv + step_size * adv.grad.sign()
            # Project back to epsilon ball
            delta = torch.clamp(adv - image, -epsilon, epsilon)
            adv = torch.clamp(image + delta, 0, 1)

            # Track best adversarial example
            with torch.no_grad():
                cur_output = model(adv)
                cur_pred = cur_output['logits'].argmax(dim=1)
                if cur_pred.item() != label.item():
                    cur_loss = F.cross_entropy(cur_output['logits'], label).item()
                    if cur_loss > best_loss:
                        best_loss = cur_loss
                        best_adv = adv.clone()

        # Final evaluation
        with torch.no_grad():
            orig_output = model(image)
            adv_output = model(best_adv)

            orig_pred = orig_output['logits'].argmax(dim=1)
            adv_pred = adv_output['logits'].argmax(dim=1)
            orig_prob = orig_output['probability'].item()
            adv_prob = adv_output['probability'].item()

            perturbation = (best_adv - image).abs()
            l2_norm = perturbation.norm().item()
            linf_norm = perturbation.max().item()

        metadata = {
            'method': 'PGD',
            'epsilon': epsilon,
            'num_steps': num_steps,
            'step_size': step_size,
            'original_prediction': orig_pred.item(),
            'adversarial_prediction': adv_pred.item(),
            'original_confidence': orig_prob,
            'adversarial_confidence': adv_prob,
            'prediction_changed': orig_pred.item() != adv_pred.item(),
            'l2_perturbation': round(l2_norm, 6),
            'linf_perturbation': round(linf_norm, 6),
            'success_rate': 1.0 if orig_pred.item() != adv_pred.item() else 0.0,
        }

        return best_adv.detach(), metadata

    @staticmethod
    def universal_perturbation(
        model: nn.Module,
        images: list[torch.Tensor],
        labels: list[torch.Tensor],
        epsilon: float = 0.03,
        num_steps: int = 200,
        step_size: float = 0.01,
    ) -> tuple[torch.Tensor, dict]:
        """
        Universal Adversarial Perturbation (UAP).
        Finds a single perturbation pattern that fools the model
        on multiple inputs.
        
        Args:
            model: Target model
            images: List of image tensors
            labels: List of label tensors
            epsilon: Maximum perturbation per pixel
            num_steps: Training iterations
            step_size: Learning rate
        
        Returns:
            universal_perturbation, attack_metadata
        """
        if not images:
            return torch.zeros(1, 3, 224, 224), {'success_rate': 0.0}

        channels, h, w = images[0].shape[1], images[0].shape[2], images[0].shape[3]
        universal_pert = torch.zeros(1, channels, h, w)

        success_count = 0
        total_samples = len(images)

        for step in range(num_steps):
            # Sample a random image from the batch
            idx = np.random.randint(0, total_samples)
            img = images[idx].clone().unsqueeze(0)
            lbl = labels[idx].clone().unsqueeze(0)

            # Add current universal perturbation
            perturbed = torch.clamp(img + universal_pert, 0, 1)
            perturbed = perturbed.detach().requires_grad_(True)

            output = model(perturbed)
            logits = output['logits']
            loss = F.cross_entropy(logits, lbl)

            model.zero_grad()
            loss.backward()

            # Update universal perturbation
            grad = perturbed.grad.data
            universal_pert = universal_pert - step_size * grad.sign()

            # Project to epsilon ball
            universal_pert = torch.clamp(universal_pert, -epsilon, epsilon)

        # Evaluate universal perturbation across all samples
        with torch.no_grad():
            fools_count = 0
            for img, lbl in zip(images, labels):
                perturbed = torch.clamp(img.unsqueeze(0) + universal_pert, 0, 1)
                output = model(perturbed)
                pred = output['logits'].argmax(dim=1)
                if pred.item() != lbl.item():
                    fools_count += 1

            success_rate = fools_count / total_samples

            # Perturbation statistics
            l2_norm = universal_pert.norm().item()
            linf_norm = universal_pert.abs().max().item()

        metadata = {
            'method': 'Universal Perturbation',
            'epsilon': epsilon,
            'num_steps': num_steps,
            'total_samples': total_samples,
            'fool_count': fools_count,
            'success_rate': round(success_rate, 4),
            'l2_perturbation': round(l2_norm, 6),
            'linf_perturbation': round(linf_norm, 6),
        }

        return universal_pert.detach(), metadata

    @staticmethod
    def gaussian_noise(
        image: torch.Tensor,
        std: float = 0.1,
        num_samples: int = 10,
    ) -> tuple[torch.Tensor, list[dict]]:
        """
        Gaussian noise robustness test.
        Applies random Gaussian noise at varying levels and measures
        prediction stability.
        """
        results = []
        noisy_images = []

        for i in range(num_samples):
            noise = torch.randn_like(image) * std
            noisy = torch.clamp(image + noise, 0, 1)
            noisy_images.append(noisy)

        # Stack for batch processing if possible
        noisy_batch = torch.cat(noisy_images, dim=0)

        return noisy_images, [{'method': 'Gaussian Noise', 'std': std, 'num_samples': num_samples}]

    @staticmethod
    def spatial_attack(
        model: nn.Module,
        image: torch.Tensor,
        label: torch.Tensor,
        rotation_range: float = 15.0,
        translation_range: float = 0.1,
        scale_range: float = 0.1,
    ) -> tuple[list[torch.Tensor], dict]:
        """
        Spatial transformation attack.
        Applies geometric transformations that are common in real-world
        scenarios but can fool models.
        """
        perturbed_images = []
        metadata_list = []

        for _ in range(10):
            angle = np.random.uniform(-rotation_range, rotation_range)
            tx = np.random.uniform(-translation_range, translation_range) * image.shape[2]
            ty = np.random.uniform(-translation_range, translation_range) * image.shape[3]
            scale = 1.0 + np.random.uniform(-scale_range, scale_range)

            # Apply affine transformation
            angle_rad = angle * np.pi / 180
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

            # Build affine grid
            theta = torch.tensor([
                [scale * cos_a, -scale * sin_a, tx / image.shape[2] * 2],
                [scale * sin_a, scale * cos_a, ty / image.shape[3] * 2],
            ], dtype=torch.float32).unsqueeze(0)

            grid = F.affine_grid(theta, image.size(), align_corners=False)
            transformed = F.grid_sample(image, grid, align_corners=False, mode='bilinear')
            perturbed_images.append(transformed)

        return perturbed_images, [{'method': 'Spatial Attack', 'angles': rotation_range, 'translations': translation_range}]

    @staticmethod
    def brightness_contrast_attack(
        image: torch.Tensor,
        brightness_range: tuple = (-0.3, 0.3),
        contrast_range: tuple = (0.7, 1.3),
        num_samples: int = 10,
    ) -> list[torch.Tensor]:
        """
        Brightness and contrast perturbation.
        Tests model resilience to common image variations.
        """
        results = []

        for _ in range(num_samples):
            # Brightness
            brightness = np.random.uniform(*brightness_range)
            adjusted = image + brightness

            # Contrast
            contrast = np.random.uniform(*contrast_range)
            mean = image.mean()
            adjusted = (adjusted - mean) * contrast + mean

            adjusted = torch.clamp(adjusted, 0, 1)
            results.append(adjusted)

        return results

    @staticmethod
    def compression_attack(
        image: torch.Tensor,
        quality_levels: list[int] = None,
    ) -> list[tuple[torch.Tensor, int]]:
        """
        JPEG compression robustness test.
        Simulates the effect of lossy compression.
        """
        if quality_levels is None:
            quality_levels = [90, 70, 50, 30, 10]

        from PIL import Image
        import io

        results = []
        # Convert tensor to PIL
        img_np = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        for quality in quality_levels:
            # JPEG compress and decompress
            buf = io.BytesIO()
            pil_img.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            decompressed = Image.open(buf)

            # Convert back to tensor
            decompressed_np = np.array(decompressed).astype(np.float32) / 255.0
            tensor = torch.from_numpy(decompressed_np).permute(2, 0, 1).unsqueeze(0)

            # Resize if dimensions differ
            if tensor.shape != image.shape:
                tensor = F.interpolate(tensor, size=image.shape[2:], mode='bilinear', align_corners=False)

            results.append((tensor, quality))

        return results


class RobustnessEvaluator:
    """Evaluates model robustness across multiple attack types."""

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.attacks = AdversarialAttacks()

    def evaluate(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        epsilons: list[float] = None,
    ) -> dict:
        """
        Run a comprehensive robustness evaluation.
        
        Args:
            image: Input image [1, C, H, W]
            label: True label [1]
            epsilons: List of epsilon values to test
        
        Returns:
            Comprehensive robustness report
        """
        if epsilons is None:
            epsilons = [0.01, 0.03, 0.05, 0.1, 0.2]

        image = image.to(self.device)
        label = label.to(self.device)

        report = {
            'fgsm_results': [],
            'pgd_results': [],
            'noise_results': [],
            'spatial_results': [],
            'brightness_results': [],
            'compression_results': [],
            'overall_robustness_score': 0.0,
            'vulnerabilities': [],
            'recommendations': [],
        }

        # ── FGSM at multiple epsilons ──
        for eps in epsilons:
            adv_image, metadata = self.attacks.fgsm(self.model, image, label, epsilon=eps)
            metadata['epsilon'] = eps
            report['fgsm_results'].append(metadata)

        # ── PGD at multiple epsilons ──
        for eps in epsilons:
            adv_image, metadata = self.attacks.pgd(
                self.model, image, label,
                epsilon=eps, num_steps=10, step_size=eps / 4,
            )
            report['pgd_results'].append(metadata)

        # ── Gaussian noise robustness ──
        for std in [0.05, 0.1, 0.15, 0.2]:
            noisy_images, _ = self.attacks.gaussian_noise(image, std=std, num_samples=5)
            stable_count = 0
            for noisy in noisy_images:
                with torch.no_grad():
                    orig_out = self.model(image)
                    noisy_out = self.model(noisy.to(self.device))
                    if orig_out['logits'].argmax() == noisy_out['logits'].argmax():
                        stable_count += 1
            report['noise_results'].append({
                'std': std,
                'stability_rate': stable_count / len(noisy_images),
                'method': 'Gaussian Noise',
            })

        # ── Spatial robustness ──
        spatial_images, _ = self.attacks.spatial_attack(self.model, image, label)
        spatial_stable = 0
        with torch.no_grad():
            orig_out = self.model(image)
            orig_pred = orig_out['logits'].argmax()
            for sp_img in spatial_images:
                sp_out = self.model(sp_img.to(self.device))
                if sp_out['logits'].argmax() == orig_pred:
                    spatial_stable += 1
        report['spatial_results'] = {
            'stability_rate': spatial_stable / len(spatial_images) if spatial_images else 1.0,
            'num_transforms': len(spatial_images),
            'method': 'Spatial Transform',
        }

        # ── Brightness/Contrast ──
        bc_images = self.attacks.brightness_contrast_attack(image)
        bc_stable = 0
        with torch.no_grad():
            for bc_img in bc_images:
                bc_out = self.model(bc_img.to(self.device))
                if bc_out['logits'].argmax() == orig_pred:
                    bc_stable += 1
        report['brightness_results'] = {
            'stability_rate': bc_stable / len(bc_images) if bc_images else 1.0,
            'num_variations': len(bc_images),
            'method': 'Brightness/Contrast',
        }

        # ── Compression ──
        comp_images = self.attacks.compression_attack(image)
        comp_stable = 0
        for comp_img, quality in comp_images:
            with torch.no_grad():
                comp_out = self.model(comp_img.to(self.device))
                if comp_out['logits'].argmax() == orig_pred:
                    comp_stable += 1
        report['compression_results'] = {
            'stability_rate': comp_stable / len(comp_images) if comp_images else 1.0,
            'num_qualities': len(comp_images),
            'method': 'JPEG Compression',
        }

        # ── Compute Overall Robustness Score ──
        report['overall_robustness_score'] = self._compute_robustness_score(report)
        report['vulnerabilities'] = self._identify_vulnerabilities(report)
        report['recommendations'] = self._generate_recommendations(report)

        return report

    def _compute_robustness_score(self, report: dict) -> float:
        """
        Compute an overall robustness score (0-100).
        
        Weights:
        - FGSM resilience: 25%
        - PGD resilience: 30%
        - Noise resilience: 15%
        - Spatial resilience: 10%
        - Brightness/contrast: 10%
        - Compression: 10%
        """
        scores = []

        # FGSM: average resistance across epsilons
        if report['fgsm_results']:
            fgsm_resistance = np.mean([
                1.0 - r.get('success_rate', 0) for r in report['fgsm_results']
            ])
            scores.append(('fgsm', fgsm_resistance, 0.25))

        # PGD: average resistance across epsilons
        if report['pgd_results']:
            pgd_resistance = np.mean([
                1.0 - r.get('success_rate', 0) for r in report['pgd_results']
            ])
            scores.append(('pgd', pgd_resistance, 0.30))

        # Noise stability
        if report['noise_results']:
            noise_stability = np.mean([r['stability_rate'] for r in report['noise_results']])
            scores.append(('noise', noise_stability, 0.15))

        # Spatial
        if report['spatial_results']:
            scores.append(('spatial', report['spatial_results']['stability_rate'], 0.10))

        # Brightness/contrast
        if report['brightness_results']:
            scores.append(('bc', report['brightness_results']['stability_rate'], 0.10))

        # Compression
        if report['compression_results']:
            scores.append(('comp', report['compression_results']['stability_rate'], 0.10))

        if not scores:
            return 50.0

        total_weight = sum(w for _, _, w in scores)
        weighted_sum = sum(s * w for _, s, w in scores)

        return round((weighted_sum / total_weight) * 100, 1) if total_weight > 0 else 50.0

    def _identify_vulnerabilities(self, report: dict) -> list[dict]:
        """Identify specific vulnerabilities."""
        vulns = []

        # Check FGSM at small epsilon
        for r in report['fgsm_results']:
            if r.get('epsilon', 0) <= 0.05 and r.get('success_rate', 0) > 0.5:
                vulns.append({
                    'type': 'gradient_vulnerability',
                    'severity': 'high',
                    'description': f"Model fooled by FGSM (eps={r['epsilon']}) with {r['success_rate']:.0%} success rate",
                    'attack': 'FGSM',
                    'epsilon': r['epsilon'],
                })

        # Check PGD (strongest attack)
        for r in report['pgd_results']:
            if r.get('success_rate', 0) > 0.3:
                vulns.append({
                    'type': 'iterative_vulnerability',
                    'severity': 'critical' if r.get('success_rate', 0) > 0.7 else 'high',
                    'description': f"Model fooled by PGD (eps={r['epsilon']}) with {r['success_rate']:.0%} success rate",
                    'attack': 'PGD',
                    'epsilon': r['epsilon'],
                })

        # Noise
        if report['noise_results']:
            low_noise_stability = [r for r in report['noise_results']
                                   if r['std'] <= 0.1 and r['stability_rate'] < 0.8]
            for r in low_noise_stability:
                vulns.append({
                    'type': 'noise_vulnerability',
                    'severity': 'medium',
                    'description': f"Unstable under noise (std={r['std']}): {r['stability_rate']:.0%} stability",
                    'attack': 'Gaussian Noise',
                    'std': r['std'],
                })

        # Spatial
        if report['spatial_results'] and report['spatial_results']['stability_rate'] < 0.7:
            vulns.append({
                'type': 'spatial_vulnerability',
                'severity': 'medium',
                'description': f"Spatial transforms cause {1 - report['spatial_results']['stability_rate']:.0%} prediction changes",
                'attack': 'Spatial',
            })

        return vulns

    def _generate_recommendations(self, report: dict) -> list[str]:
        """Generate recommendations based on vulnerabilities."""
        recs = []
        score = report['overall_robustness_score']

        if score < 30:
            recs.append("Critical: Model is highly vulnerable to adversarial attacks. Consider adversarial training.")
            recs.append("Implement input preprocessing defenses (JPEG compression, spatial smoothing).")
        elif score < 60:
            recs.append("Model has moderate robustness. Consider adding adversarial training with PGD.")
            recs.append("Add input validation and anomaly detection before model inference.")

        # Specific recommendations
        fgsm_vulns = [v for v in report['vulnerabilities'] if v['attack'] == 'FGSM']
        if fgsm_vulns:
            recs.append("Gradient-based attacks are effective. Implement gradient masking or defensive distillation.")

        pgd_vulns = [v for v in report['vulnerabilities'] if v['attack'] == 'PGD']
        if pgd_vulns:
            recs.append("Iterative attacks succeed frequently. Add PGD adversarial training to the training pipeline.")

        noise_vulns = [v for v in report['vulnerabilities'] if v['attack'] == 'Gaussian Noise']
        if noise_vulns:
            recs.append("Add Gaussian noise augmentation during training to improve noise robustness.")

        spatial_vulns = [v for v in report['vulnerabilities'] if v['attack'] == 'Spatial']
        if spatial_vulns:
            recs.append("Spatial transforms degrade performance. Add spatial augmentation to training data.")

        if not recs:
            recs.append("Model shows good robustness across all tested attack vectors.")

        return recs
