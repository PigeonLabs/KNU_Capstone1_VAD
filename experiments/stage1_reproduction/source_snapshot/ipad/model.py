"""Paper-first reconstruction; upstream encoder and decoder remain untouched."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .vendor.VST_block import SwinTransformer3D
from .vendor.reconstruction_model import VST3DDecoder


class PeriodMemory(nn.Module):
    def __init__(self, slots=2000, channels=768, classes=200, axis='tokens', shrink=0., radius=0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(slots, channels))
        nn.init.uniform_(self.weight, -1/math.sqrt(channels), 1/math.sqrt(channels))
        self.slots, self.classes, self.axis, self.shrink, self.radius = slots, classes, axis, shrink, radius

    def forward(self, x, logits):
        b,c,t,h,w = x.shape
        features = x.flatten(2).transpose(1,2)  # [B, tokens, C]; NEVER merge B and tokens.
        addresses = features @ self.weight.T
        confidence, phase = logits.softmax(-1).max(-1)
        slot = torch.div(phase*self.slots, self.classes, rounding_mode='floor').clamp_max(self.slots-1)
        selected = (torch.arange(self.slots, device=x.device)[None,:]-slot[:,None]).abs() <= self.radius
        addresses = addresses * (1 + selected[:,None,:] * confidence[:,None,None])
        dim = 1 if self.axis == 'tokens' else 2
        attention = addresses.float().softmax(dim)
        if self.shrink:
            attention = F.relu(attention-self.shrink)*attention / ((attention-self.shrink).abs()+1e-12)
            attention = F.normalize(attention, p=1, dim=dim)
        output = (attention.to(features.dtype) @ self.weight.to(features.dtype)).transpose(1,2).reshape(b,c,t,h,w)
        entropy = -(attention*attention.clamp_min(1e-12).log()).sum(dim).mean()
        return output, entropy


class IPAD(nn.Module):
    def __init__(self, memory=True, axis='tokens', shrink=0., radius=0, checkpoint=False):
        super().__init__()
        self.options = dict(memory=memory, axis=axis, shrink=shrink, radius=radius, checkpoint=checkpoint)
        self.encoder = SwinTransformer3D(use_checkpoint=checkpoint)
        # Preserve the published source architecture, including its large period head.
        self.period = nn.Sequential(
            nn.Conv3d(768,768,3,stride=(1,2,2),padding=1), nn.BatchNorm3d(768),
            nn.LeakyReLU(.2,inplace=True), nn.Flatten(1), nn.Linear(768*4*4*4,4096),
            nn.ReLU(), nn.Linear(4096,2048), nn.ReLU(), nn.Linear(2048,200))
        self.memory = PeriodMemory(axis=axis, shrink=shrink, radius=radius) if memory else None
        self.decoder = VST3DDecoder(chnum_out=3)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.period(features)
        entropy = features.new_zeros(())
        if self.memory is not None:
            features, entropy = self.memory(features, logits)
        return {'reconstruction': self.decoder(features), 'phase_logits': logits, 'entropy': entropy}


def objective(output, x, phase):
    reconstruction = F.mse_loss(output['reconstruction'].float(), x.float())
    period = F.cross_entropy(output['phase_logits'].float(), phase)
    entropy = output['entropy']
    return reconstruction + .02*period + .0002*entropy, {'reconstruction': reconstruction, 'period': period, 'entropy': entropy}
