"""Native-pad residual 3-D encoder/decoder for conditional-K charge partition.

The implemented architecture is not a trained model. Inference through a
checkpoint requires an explicit training-completed flag and a compatible data
contract. Coordinates and true pad widths are channels; pads are never expanded
into invented sub-pixels. Slots denote arbitrary components, not MC track IDs.
"""
from pathlib import Path
import hashlib
import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:
    raise ImportError('waveform_models requires PyTorch; classical finding does not') from exc

from .finding import PAD_U_MM, PAD_WIDTH_MM


def features_numpy(positive_waveform, tb_bin=4):
    q = np.asarray(positive_waveform, dtype=np.float32)
    if q.shape != (12,64,512) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError('Expected nonnegative finite native-pad waveform (12,64,512)')
    if isinstance(tb_bin, bool) or int(tb_bin) != tb_bin or tb_bin < 1 or 512 % int(tb_bin):
        raise ValueError('tb_bin must divide 512')
    b = q.reshape(12,64,512//int(tb_bin),int(tb_bin))
    mean, peak = b.mean(-1), b.max(-1)
    log_mean = np.log1p(mean/4095.)
    derivative = np.diff(log_mean,axis=-1,prepend=log_mean[...,:1])
    return np.stack((log_mean,np.log1p(peak/4095.),derivative)).astype(np.float32)


def _groups(channels):
    return next(g for g in (8,4,2,1) if channels%g == 0)


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=(1,1,1), dropout=.05):
        super().__init__()
        self.body=nn.Sequential(nn.Conv3d(incoming,outgoing,3,stride=stride,padding=1,bias=False),
            nn.GroupNorm(_groups(outgoing),outgoing),nn.SiLU(),nn.Dropout3d(dropout),
            nn.Conv3d(outgoing,outgoing,3,padding=1,bias=False),nn.GroupNorm(_groups(outgoing),outgoing))
        self.skip=nn.Identity() if incoming==outgoing and stride==(1,1,1) else nn.Conv3d(incoming,outgoing,1,stride=stride,bias=False)
        self.activation=nn.SiLU()
    def forward(self,x):
        return self.activation(self.body(x)+self.skip(x))


class AxialContext(nn.Module):
    """Attention separately along layer and TB axes at the coarse bottleneck."""
    def __init__(self,channels,heads=4):
        super().__init__()
        self.layer_attention=nn.MultiheadAttention(channels,heads,dropout=0.,batch_first=True)
        self.time_attention=nn.MultiheadAttention(channels,heads,dropout=0.,batch_first=True)
        self.layer_norm=nn.LayerNorm(channels)
        self.time_norm=nn.LayerNorm(channels)
        self.mix=nn.Conv3d(channels,channels,1)
    def forward(self,x):
        b,c,d,h,w=x.shape
        tokens=x.permute(0,3,4,2,1).reshape(b*h*w,d,c)
        norm=self.layer_norm(tokens)
        tokens=tokens+self.layer_attention(norm,norm,norm,need_weights=False)[0]
        x=tokens.reshape(b,h,w,d,c).permute(0,4,3,1,2)
        tokens=x.permute(0,2,3,4,1).reshape(b*d*h,w,c)
        norm=self.time_norm(tokens)
        tokens=tokens+self.time_attention(norm,norm,norm,need_weights=False)[0]
        return self.mix(tokens.reshape(b,d,h,w,c).permute(0,4,1,2,3))


class GatedDecoder(nn.Module):
    def __init__(self,incoming,skip_channels,outgoing,dropout):
        super().__init__()
        self.project=nn.Conv3d(incoming,outgoing,1)
        self.skip_project=nn.Conv3d(skip_channels,outgoing,1)
        self.attention=nn.Sequential(nn.Conv3d(outgoing,1,1),nn.Sigmoid())
        self.body=ResidualBlock(outgoing+skip_channels,outgoing,dropout=dropout)
    def forward(self,x,skip):
        x=F.interpolate(self.project(x),size=skip.shape[-3:],mode='trilinear',align_corners=False)
        gate=self.attention(F.silu(x+self.skip_project(skip)))
        return self.body(torch.cat((x,skip*gate),dim=1))


class NativeWaveformResUNet(nn.Module):
    """Residual multiscale + axial attention + learned skip gates, K+1 outputs.

    Input B,3,12,64,nt features; output B,K+1,12,64,nt probabilities.
    This model has three encoder downsamplings and three learned decoder skips;
    layer resolution is preserved in the first downsampling. GroupNorm supports
    batch size one without introducing BatchNorm running-statistic dependence.
    """
    def __init__(self,k=3,tb_bin=4,width=16,dropout=.05):
        super().__init__()
        if not 1<=int(k)<=6 or 512%int(tb_bin) or int(width)<4 or int(width)%4:
            raise ValueError('k 1..6; tb_bin divisor512; width>=4 divisible4')
        self.k,self.tb_bin,self.width=int(k),int(tb_bin),int(width)
        self.dropout=float(dropout)
        self.register_buffer('pad_u',torch.tensor(PAD_U_MM,dtype=torch.float32))
        self.register_buffer('pad_width',torch.tensor(PAD_WIDTH_MM,dtype=torch.float32))
        w=self.width
        self.stem=ResidualBlock(7,w,dropout=dropout)
        self.enc1=ResidualBlock(w,2*w,(1,2,2),dropout)
        self.enc2=ResidualBlock(2*w,4*w,(2,2,2),dropout)
        self.enc3=ResidualBlock(4*w,6*w,(2,2,2),dropout)
        self.context=AxialContext(6*w,heads=4)
        self.dec2=GatedDecoder(6*w,4*w,4*w,dropout)
        self.dec1=GatedDecoder(4*w,2*w,2*w,dropout)
        self.dec0=GatedDecoder(2*w,w,w,dropout)
        self.head=nn.Conv3d(w,self.k+1,1)
        for module in self.modules():
            if isinstance(module,nn.Conv3d):
                nn.init.kaiming_normal_(module.weight,nonlinearity='relu')
                if module.bias is not None:nn.init.zeros_(module.bias)
    def geometry(self,x):
        b,_,d,h,t=x.shape
        if (d,h,t)!=(12,64,512//self.tb_bin):
            raise ValueError('Model input geometry differs from checkpoint contract')
        u=(self.pad_u/37.)[None,:,None].expand(d,h,t)
        v=torch.linspace(-1.,1.,d,device=x.device,dtype=x.dtype)[:,None,None].expand(d,h,t)
        time=torch.linspace(-1.,1.,t,device=x.device,dtype=x.dtype)[None,None,:].expand(d,h,t)
        width=(self.pad_width/1.9)[None,:,None].expand(d,h,t)
        return torch.stack((u,v,time,width),dim=0)[None].expand(b,-1,-1,-1,-1)
    def logits(self,x):
        if x.ndim!=5 or x.shape[1]!=3:raise ValueError('Expected B,3,12,64,nt')
        a=self.stem(torch.cat((x,self.geometry(x)),dim=1))
        b=self.enc1(a);c=self.enc2(b);d=self.enc3(c)
        d=d+self.context(d)
        return self.head(self.dec0(self.dec1(self.dec2(d,c),b),a))
    def forward(self,x):
        return torch.softmax(self.logits(x),dim=1)
    def contract(self):
        return dict(architecture='NativeWaveformResUNet-v1',k=self.k,tb_bin=self.tb_bin,width=self.width,
                    dropout=self.dropout,native_pads=64,layers=12,raw_tb=512,
                    features='log_mean_positive,log_peak_positive,temporal_difference + u,v,time,pad_width',
                    background_slot='last',physical_w_calibrated=False)


def load_finding_checkpoint(path,device='cpu',require_trained=True,trusted_checkpoint=False):
    path=Path(path).expanduser().resolve(strict=True)
    # Only tensor/primitive checkpoints produced by this package are supported.
    # On pre-weights_only PyTorch, require the caller to load in a modern runtime
    # rather than silently enabling pickle execution for an arbitrary file.
    try:
        checkpoint=torch.load(str(path),map_location='cpu',weights_only=True)
    except TypeError as exc:
        if not trusted_checkpoint:
            raise RuntimeError('Legacy PyTorch: set trusted_checkpoint=True ONLY for your own verified training checkpoint; otherwise use restricted-load-capable PyTorch') from exc
        checkpoint=torch.load(str(path),map_location='cpu')
    if not isinstance(checkpoint,dict) or not {'contract','model','training_completed'}<=set(checkpoint):
        raise ValueError('Not a reconstruction finding checkpoint')
    if require_trained and checkpoint['training_completed'] is not True:
        raise ValueError('Checkpoint contains an untrained model; inference refused')
    c=checkpoint['contract']
    if c.get('architecture')!='NativeWaveformResUNet-v1':raise ValueError('Unsupported model architecture')
    # Fixed buffers are detector geometry, not freely replaceable learned weights.
    with torch.random.fork_rng(devices=[]):
        model=NativeWaveformResUNet(c['k'],c['tb_bin'],c['width'],c.get('dropout',.05))
    for key in ('pad_u','pad_width'):
        if not torch.equal(checkpoint['model'][key],model.state_dict()[key]):
            raise ValueError('Checkpoint native-pad geometry mismatch')
    model.load_state_dict(checkpoint['model'],strict=True)
    return model.to(device).eval(),checkpoint,hashlib.sha256(path.read_bytes()).hexdigest()


def predict_checkpoint(path,positive_waveform,device='cpu',expected_k=None,expected_tb_bin=None,trusted_checkpoint=False):
    model,checkpoint,digest=load_finding_checkpoint(path,device,trusted_checkpoint=trusted_checkpoint)
    if expected_k is not None and model.k!=int(expected_k):raise ValueError('Checkpoint K mismatch')
    if expected_tb_bin is not None and model.tb_bin!=int(expected_tb_bin):raise ValueError('Checkpoint time bin mismatch')
    f=features_numpy(positive_waveform,model.tb_bin)
    with torch.no_grad():p=model(torch.from_numpy(f[None]).to(device))[0].cpu().numpy()
    if not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(0),1.,atol=2e-5):
        raise RuntimeError('CNN returned invalid fractions; compare CPU precision and checkpoint contract')
    entropy=-(p*np.log(np.maximum(p,1e-12))).sum(0)
    return dict(fractions=p,entropy=entropy,checkpoint_sha256=digest,training_completed=True,
                uncertainty_note='Softmax entropy is a diagnostic, not calibrated error probability',
                contract=model.contract(),training_provenance=checkpoint.get('training_provenance',{}))
