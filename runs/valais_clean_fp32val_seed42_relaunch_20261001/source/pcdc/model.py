"""Shared ResNet-18 + FPN-style absolute-difference decoder, without PSE/PCDC."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .config import Config

Features = list[Tensor]


class ResNet18Encoder(nn.Module):
    channels = (64, 128, 256, 512)

    def __init__(self, weights: ResNet18_Weights | None = None):
        super().__init__()
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.stages = nn.ModuleList([backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4])

    def forward(self, image: Tensor) -> Features:
        features = []
        image = self.stem(image)
        for stage in self.stages:
            image = stage(image)
            features.append(image)
        return features


def conv_block(in_channels: int, out_channels: int, groups: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                         nn.GroupNorm(groups, out_channels), nn.ReLU(inplace=True))


class FPNDifferenceDecoder(nn.Module):
    def __init__(self, width: int, groups: int):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(channels, width, 1)
                                      for channels in ResNet18Encoder.channels])
        self.refine = nn.ModuleList([conv_block(width, width, groups) for _ in range(4)])
        self.half_resolution = conv_block(width, width // 2, groups)
        self.full_resolution = conv_block(width // 2, width // 4, groups)
        self.classifier = nn.Conv2d(width // 4, 1, 1)

    def forward(self, differences: Features, output_size: tuple[int, int]) -> Tensor:
        pyramid = [projection(feature) for projection, feature in zip(self.lateral, differences)]
        fused = self.refine[3](pyramid[3])
        for level in (2, 1, 0):
            fused = self.refine[level](pyramid[level] + F.interpolate(
                fused, size=pyramid[level].shape[-2:], mode="nearest"))
        # Nearest + learned refinement avoids CUDA bilinear-backward nondeterminism.
        half_size = tuple((side + 1) // 2 for side in output_size)
        fused = self.half_resolution(F.interpolate(fused, size=half_size, mode="nearest"))
        fused = self.full_resolution(F.interpolate(fused, size=output_size, mode="nearest"))
        return self.classifier(fused)


class SiameseChangeDetector(nn.Module):
    """One encoder instance; correction is an explicit future insertion point."""

    def __init__(self, width: int = 64, groups: int = 8,
                 weights: ResNet18_Weights | None = None):
        super().__init__()
        self.encoder = ResNet18Encoder(weights)
        self.decoder = FPNDifferenceDecoder(width, groups)

    def encode_pair(self, image_a: Tensor, image_b: Tensor) -> tuple[Features, Features]:
        if image_a.shape != image_b.shape or image_a.ndim != 4 or image_a.shape[1] != 3:
            raise ValueError("Expected aligned A/B tensors with shape [N,3,H,W]")
        if image_a.shape[0] < 1 or min(image_a.shape[-2:]) < 32:
            raise ValueError("Need a nonempty batch and spatial dimensions >= 32")
        batch = image_a.shape[0]
        # Both dates share weights AND one BatchNorm update; no date-order bias.
        features = self.encoder(torch.cat((image_a, image_b), dim=0))
        return [feature[:batch] for feature in features], [feature[batch:] for feature in features]

    def correct_features(self, features_a: Features, features_b: Features) -> tuple[Features, Features]:
        """Phase 4/5 seam. Phase 1 returns features unchanged; no fake expert exists."""
        return features_a, features_b

    @staticmethod
    def difference_features(features_a: Features, features_b: Features) -> Features:
        if len(features_a) != 4 or len(features_b) != 4:
            raise ValueError("Expected four feature stages per date")
        if any(a.shape != b.shape for a, b in zip(features_a, features_b)):
            raise ValueError("Bi-temporal feature dimensions must match")
        return [torch.abs(a - b) for a, b in zip(features_a, features_b)]

    def forward(self, image_a: Tensor, image_b: Tensor, *, return_features: bool = False) -> dict:
        features_a, features_b = self.encode_pair(image_a, image_b)
        corrected_a, corrected_b = self.correct_features(features_a, features_b)
        differences = self.difference_features(corrected_a, corrected_b)
        output = {"logits": self.decoder(differences, tuple(image_a.shape[-2:]))}
        if return_features:
            output.update(features_a=features_a, features_b=features_b,
                          corrected_a=corrected_a, corrected_b=corrected_b, differences=differences)
        return output


def build_model(config: Config, *, initialize_encoder: bool = True) -> SiameseChangeDetector:
    settings = config.values["model"]
    weights = None
    if initialize_encoder and settings["encoder_weights"] is not None:
        torch.hub.set_dir(str(config.path(settings["weights_cache"])))
        weights = ResNet18_Weights.IMAGENET1K_V1
        # Download errors intentionally propagate; never substitute random weights.
    return SiameseChangeDetector(settings["decoder_channels"], settings["normalization_groups"], weights)

