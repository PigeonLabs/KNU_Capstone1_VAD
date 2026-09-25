"""Q/V-only low-rank updates for DINOv2's fused qkv linear layers."""
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F
from .stage5 import Extractor


class QVLoRA(nn.Module):
    def __init__(self, base, rank=4, alpha=8):
        super().__init__()
        if base.out_features != 3 * base.in_features:
            raise ValueError('Expected fused qkv with three equal output slices')
        self.base = base.requires_grad_(False)
        self.channels = base.in_features
        self.scale = alpha / rank
        self.enabled = True
        for key in ['q', 'v']:
            a = nn.Parameter(torch.empty(rank, self.channels, device=base.weight.device))
            b = nn.Parameter(torch.zeros(self.channels, rank, device=base.weight.device))
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.register_parameter(f'{key}_a', a)
            self.register_parameter(f'{key}_b', b)

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        q = F.linear(F.linear(x, self.q_a), self.q_b) * self.scale
        v = F.linear(F.linear(x, self.v_a), self.v_b) * self.scale
        return y + torch.cat([q, torch.zeros_like(q), v], dim=-1)

    @torch.no_grad()
    def merged(self):
        result = nn.Linear(self.channels, 3*self.channels, bias=self.base.bias is not None,
                           device=self.base.weight.device, dtype=self.base.weight.dtype)
        result.load_state_dict(self.base.state_dict())
        result.weight[:self.channels].add_((self.q_b @ self.q_a) * self.scale)
        result.weight[2*self.channels:].add_((self.v_b @ self.v_a) * self.scale)
        return result.requires_grad_(False)


class LoRAExtractor(Extractor):
    def __init__(self, adapted=True):
        super().__init__('B')
        if adapted:
            for block in self.backbone.blocks[8:12]:
                block.attn.qkv = QVLoRA(block.attn.qkv)
        self.eval()

    def adapters(self):
        return [m for m in self.modules() if isinstance(m, QVLoRA)]

    def adapter_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_adapter(self, state):
        params = dict(self.named_parameters())
        expected = {n for n,p in params.items() if p.requires_grad}
        if set(state) != expected:
            raise ValueError('Adapter parameter names do not match')
        with torch.no_grad():
            for n,v in state.items():
                params[n].copy_(v)

    @contextmanager
    def teacher(self):
        modules = self.adapters()
        old = [m.enabled for m in modules]
        try:
            for m in modules: m.enabled = False
            with torch.no_grad(): yield
        finally:
            for m,v in zip(modules,old): m.enabled = v

    def merge(self):
        for block in self.backbone.blocks:
            if isinstance(block.attn.qkv, QVLoRA):
                block.attn.qkv = block.attn.qkv.merged()
        self.requires_grad_(False)
        return self

    def forward(self, x):
        rgb = (x[:,[2,1,0]].float()+1)/2
        rgb = F.interpolate(rgb, size=(252,252), mode='bilinear', align_corners=False)
        rgb = ((rgb-self.mean)/self.std).to(next(self.backbone.parameters()).dtype)
        patch, cls = self.backbone.get_intermediate_layers(rgb, n=[11], return_class_token=True, norm=True)[0]
        return {'cls': F.normalize(cls.float(),dim=-1), 'patch12': F.normalize(patch.float(),dim=-1)}


def photometric(x, generator=None):
    """Input/output BGR [-1,1]; identical spatial support; per-image scalars."""
    image = (x+1)/2
    scalars = .9 + .2*torch.rand(len(x),2,1,1,1,device=x.device,generator=generator)
    center = image.mean((2,3),keepdim=True)
    return (((image-center)*scalars[:,1]+center)*scalars[:,0]).clamp(0,1)*2-1


def cosine_loss(a,b):
    return (1-(F.normalize(a.float(),dim=-1)*F.normalize(b.float(),dim=-1)).sum(-1)).mean()
