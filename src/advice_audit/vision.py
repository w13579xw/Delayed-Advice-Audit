"""Visual baseline components with explicit RGB/proprioception-only inputs.

Optional Torch environment only. No simulator or native object/goal state is
imported here. Encoder parameters and normalization are frozen ACT components.
"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from torchvision.models import resnet18
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d


class VisualEncoder(nn.Module):
    def __init__(self, assets):
        super().__init__()
        assets = Path(assets)
        self.backbone = IntermediateLayerGetter(
            resnet18(weights=None, norm_layer=FrozenBatchNorm2d),
            return_layers={'layer4': 'feature_map'})
        self.backbone.load_state_dict(load_file(str(assets/'encoder.safetensors')), strict=True)
        stats = json.loads((assets/'normalization.json').read_text())
        self.register_buffer('mean', torch.tensor(stats['image_mean']).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(stats['image_std']).reshape(1, 3, 1, 1))
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def encode(self, images):
        if isinstance(images, np.ndarray):
            if images.dtype != np.uint8:
                raise ValueError('RGB must be uint8')
            images = torch.from_numpy(np.ascontiguousarray(images)).to(self.mean.device)
        if images.ndim != 4 or tuple(images.shape[1:]) != (512, 512, 3) or images.dtype != torch.uint8:
            raise ValueError('Need Bx512x512x3 uint8 RGB; no native state accepted')
        x = images.permute(0, 3, 1, 2).float()/255.
        return self.backbone((x-self.mean)/self.std)['feature_map']


class VisualHead(nn.Module):
    """Trainable small head; the same class can predict actions or image poses."""
    def __init__(self, channels=32, hidden=256, output_dim=4):
        super().__init__()
        self.projection = nn.Sequential(nn.Conv2d(512, channels, 1), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(channels*16*16+4, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, output_dim))

    def forward(self, features, robot):
        if features.ndim != 4 or tuple(features.shape[1:]) != (512, 16, 16):
            raise ValueError('Need Bx512x16x16 visual features')
        if robot.ndim != 2 or tuple(robot.shape) != (len(features), 4):
            raise ValueError('Only robot4D is a policy input')
        x = self.projection(features).flatten(1)
        return self.mlp(torch.cat([x, robot], dim=1))
