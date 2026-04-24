"""
Minimal PatchCore implementation.

Pipeline:
  image (3, 224, 224)
    -> WideResNet50 layer2 (512, 28, 28) + layer3 (1024, 14, 14 -> upsample 28, 28)
    -> concat (1536, 28, 28)
    -> 3x3 avg-pool neighborhood aggregation (stride 1)
    -> flatten -> (784, 1536) patch embeddings per image

Training: stack embeddings across good images, greedy coreset subsample.
Inference: for each test patch, min distance to memory bank -> 28x28 anomaly map
           -> upsample to input size. Global image score = max patch score.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torchvision.models import Wide_ResNet50_2_Weights


INPUT_SIZE = 224


class PatchFeatureExtractor(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
        backbone = tvm.wide_resnet50_2(weights=weights)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # Normalization (ImageNet stats)
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        self.to(device)
        self.device = device
        self.transforms = weights.transforms()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) in [0, 1]
        returns: (B, 1536, 28, 28) feature map
        """
        x = (x - self.mean) / self.std
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)          # (B, 512, 28, 28)
        f3 = self.layer3(f2)         # (B, 1024, 14, 14)
        f3 = F.interpolate(
            f3, size=f2.shape[-2:], mode='bilinear', align_corners=False
        )
        feat = torch.cat([f2, f3], dim=1)  # (B, 1536, 28, 28)
        # 3x3 neighborhood average (as in PatchCore paper)
        feat = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
        return feat

    @torch.inference_mode()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B*H*W, C) flat patch embeddings + spatial shape."""
        feat = self.forward(x)
        B, C, H, W = feat.shape
        flat = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
        return flat, (B, H, W)


def coreset_subsample(
    features: torch.Tensor,
    ratio: float = 0.1,
    seed: int = 0,
) -> torch.Tensor:
    """
    Greedy k-center (farthest-point) coreset subsampling.
    features: (N, C) on any device.
    returns: (M, C) where M = max(1, int(N * ratio)).
    """
    N = features.shape[0]
    M = max(1, int(N * ratio))
    device = features.device

    g = torch.Generator(device='cpu').manual_seed(seed)
    start = torch.randint(0, N, (1,), generator=g).item()

    selected = torch.zeros(M, dtype=torch.long, device=device)
    selected[0] = start
    min_dists = torch.cdist(features[start:start + 1], features).squeeze(0)

    for i in range(1, M):
        idx = int(torch.argmax(min_dists).item())
        selected[i] = idx
        new_dists = torch.cdist(features[idx:idx + 1], features).squeeze(0)
        min_dists = torch.minimum(min_dists, new_dists)

    return features[selected]


class MemoryBank:
    def __init__(self, features: torch.Tensor, device: torch.device):
        self.features = features.to(device)
        self.device = device

    @torch.inference_mode()
    def score(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: (N, C) query patch embeddings.
        returns: (N,) min-distance anomaly scores.
        """
        patches = patches.to(self.device)
        # Chunked cdist to avoid blowing up memory for large banks
        chunk = 512
        mins = torch.empty(patches.shape[0], device=self.device)
        for i in range(0, patches.shape[0], chunk):
            d = torch.cdist(patches[i:i + chunk], self.features)
            mins[i:i + chunk] = d.min(dim=1).values
        return mins

    def save(self, path: str, meta: dict | None = None) -> None:
        payload = {'features': self.features.cpu()}
        if meta:
            payload['meta'] = meta
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: torch.device) -> tuple['MemoryBank', dict]:
        obj = torch.load(path, map_location='cpu')
        meta = obj.get('meta', {}) if isinstance(obj, dict) else {}
        return cls(obj['features'], device), meta


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
