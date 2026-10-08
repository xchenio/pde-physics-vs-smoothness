"""Exactly parameter-matched spectral and finite-kernel POD encoders."""
import torch
from src.fno import SpectralConv1d
from src.utils import set_seed
from scripts.architecture_core import DeterministicPool


class SpectralBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spectral=SpectralConv1d(32,16)
        self.local=torch.nn.Conv1d(32,32,1)
    def forward(self,x):return self.spectral(x)+self.local(x)


class EncoderPOD(torch.nn.Module):
    def __init__(self,kind,rank):
        super().__init__()
        if kind not in ('spectral_pod','local_pod'):raise ValueError(kind)
        self.lift=torch.nn.Conv1d(2,32,1)
        self.blocks=torch.nn.ModuleList([
            SpectralBlock() if kind=='spectral_pod' else
            torch.nn.Conv1d(32,32,33,padding=16,padding_mode='circular') for _ in range(4)])
        self.pool=DeterministicPool()
        self.head=torch.nn.Linear(32*24,rank)
    def forward(self,x):
        coord=((torch.arange(x.shape[-1],device=x.device,dtype=x.dtype)+.5)/x.shape[-1]).expand(x.shape[0],1,-1)
        h=self.lift(torch.cat([x[:,None],coord],1))
        for block in self.blocks:h=torch.nn.functional.gelu(block(h))
        return self.head(self.pool(h).flatten(1))


def make_model(c,in_dim,device):
    if c['mapping']!='direct' or c['penalty']!=0:raise ValueError('direct-only protocol')
    set_seed(c['seed']+2000)
    model=EncoderPOD(c['architecture'],c['rank'])
    torch.nn.init.zeros_(model.head.weight);torch.nn.init.zeros_(model.head.bias)
    return model.to(device)
