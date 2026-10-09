"""Same FNO layers/parameters, optional single periodic position feature."""
import torch
from src.time_fno import TimeFNO1d


class PositionTimeFNO1d(TimeFNO1d):
    def __init__(self,width=16,modes=16,layers=4,position='linear',length=1.,position_center=.5,position_scale=.4):
        super().__init__(width,modes,layers)
        if position not in ('linear','periodic'):raise ValueError('unknown position feature')
        self.position=position;self.length=length;self.position_center=position_center;self.position_scale=position_scale

    def encode_position(self,x):
        if self.position=='linear':return x
        return self.position_center+self.position_scale*torch.sin(2*torch.pi*x/self.length)

    def forward(self,u0,times,coordinates):
        if self.position=='linear':return super().forward(u0,times,coordinates)
        x=torch.as_tensor(coordinates,device=u0.device,dtype=u0.dtype)
        return super().forward(u0,times,self.encode_position(x))
