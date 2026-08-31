"""
ML Models for the verification system.
CNN-based classifiers using PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class DeepfakeCNN(nn.Module):
    """
    CNN-based deepfake detector.
    Detects: face artifacts, abnormal blinking, GAN fingerprints.
    """

    def __init__(self, num_classes: int = 2, in_channels: int = 3):
        super().__init__()
        # Feature extractor - lightweight ResNet-style
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        # Artifact detection head
        self.artifact_head = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
        )
        # Blinking analysis head
        self.blink_head = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
        )
        # GAN fingerprint head
        self.gan_head = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
        )
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict:
        features = self.features(x)
        flat = features.view(features.size(0), -1)

        artifact_emb = self.artifact_head(flat)
        blink_emb = self.blink_head(flat)
        gan_emb = self.gan_head(flat)

        combined = torch.cat([artifact_emb, blink_emb, gan_emb], dim=1)
        logits = self.classifier(combined)

        return {
            "logits": logits,
            "probability": F.softmax(logits, dim=1)[:, 1],
            "artifact_embedding": artifact_emb,
            "blink_embedding": blink_emb,
            "gan_embedding": gan_emb,
        }


class DocumentCNN(nn.Module):
    """
    Document authenticity analyzer.
    Detects font inconsistencies, spacing anomalies, tampered regions.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        # Tampering localization (segmentation-style output)
        self.tamper_localizer = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict:
        features = self.features(x)
        tamper_map = self.tamper_localizer(features)
        flat = features.view(features.size(0), -1)
        logits = self.classifier(flat)

        return {
            "logits": logits,
            "probability": F.softmax(logits, dim=1)[:, 1],
            "tamper_map": tamper_map,
        }


class FaceEmbedder(nn.Module):
    """
    Face embedding network for face matching.
    Generates 512-dimensional face embeddings for comparison.
    """

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = features.view(features.size(0), -1)
        embedding = self.embedding(features)
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class LivenessNet(nn.Module):
    """
    Multi-task liveness detection network.
    
    Detects:
    - Spoof vs Real (binary classification)
    - Head pose estimation (yaw, pitch, roll angles)
    - Depth map estimation (monocular depth)
    - Texture analysis (moiré patterns, screen reflections)
    
    Architecture: Shared backbone with 3 task-specific heads.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Shared backbone - efficient feature extractor
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        backbone_out = 256 * 4 * 4

        # Head 1: Spoof classification (real vs fake)
        self.spoof_head = nn.Sequential(
            nn.Linear(backbone_out, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        # Head 2: Head pose estimation (yaw, pitch, roll)
        self.pose_head = nn.Sequential(
            nn.Linear(backbone_out, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),  # yaw, pitch, roll in degrees
        )

        # Head 3: Depth estimation (coarse depth map)
        self.depth_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

        # Head 4: Texture analysis (moiré, reflections, screen artifacts)
        self.texture_head = nn.Sequential(
            nn.Linear(backbone_out, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),  # moire, reflection, screen_edge, print_artifact
        )

    def forward(self, x: torch.Tensor) -> dict:
        backbone_features = self.backbone(x)
        flat = backbone_features.view(backbone_features.size(0), -1)

        spoof_logits = self.spoof_head(flat)
        pose = self.pose_head(flat)
        depth = self.depth_head(backbone_features)
        texture = self.texture_head(flat)

        return {
            'spoof_logits': spoof_logits,
            'spoof_probability': F.softmax(spoof_logits, dim=1)[:, 1],
            'is_real': spoof_logits.argmax(dim=1) == 1,
            'head_pose': pose,  # [yaw, pitch, roll]
            'depth_map': depth,
            'texture_scores': torch.sigmoid(texture),
            'texture_labels': {
                'moire_pattern': texture[:, 0],
                'screen_reflection': texture[:, 1],
                'screen_edge': texture[:, 2],
                'print_artifact': texture[:, 3],
            },
        }


def load_model(model_class, model_path: Optional[str] = None, device: str = "cpu"):
    """Load a model, optionally from a checkpoint."""
    model = model_class()
    if model_path:
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            print(f"Warning: Model file not found at {model_path}, using random weights")
    model.eval()
    model.to(device)
    return model
