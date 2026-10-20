"""诚实基线：ICC 分离 + 标准 FS 误差扩散（不依赖任何网络）vs NN 硬阈值 vs gs RIP。"""
import sys; sys.path.insert(0,'.')
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from pathlib import Path
import eval as E
from halftone.model import UNet, hard_levels
from halftone.cmyk import (find_icc, load_or_build_lut256, neugebauer_lab_2bit,
                           srgb_to_lab, srgb_to_cov, LEVELS)
from train import delta_e_2000
torch.set_num_threads(20); dev='cpu'
ck=torch.load('runs/cmyk_sd/best.pt',map_location=dev,weights_only=False)
a=ck['args']
m=UNet(base=a.get('base',32),up=a.get('up','nearest'),in_ch=a.get('in_ch',3),
       out_ch=a.get('out_ch',4),levels=a.get('levels',4)).to(dev)
m.load_state_dict(ck['model']); m.eval()
icc=find_icc(a.get('icc')); intent=a.get('icc_intent',0)
lut=load_or_build_lut256(Path('runs/cmyk_sd'),icc,intent)

def fs_plane(x):
    """标准 Floyd-Steinberg 蛇形，单通道覆盖率域。"""
    H,W=x.shape; buf=x.astype(np.float32).copy()
    for y in range(H):
        rng=range(W) if y%2==0 else range(W-1,-1,-1); d=1 if y%2==0 else -1
        for xx in rng:
            old=buf[y,xx]; new=1.0 if old>0.5 else 0.0
            buf[y,xx]=new; err=old-new
            for dx,dy,wt in ((d,0,7/16),(-d,1,3/16),(0,1,5/16),(d,1,1/16)):
                nx,ny=xx+dx,y+dy
                if 0<=nx<W and ny<H: buf[ny,nx]+=err*wt
    return buf

rows=[]
for f in sorted(Path('data/photos_val').glob('*.png'))[:8]:
    rgb=Image.open(f).convert('RGB'); w,h=rgb.size
    pw,ph=(8-w%8)%8,(8-h%8)%8
    t=torch.from_numpy(np.asarray(rgb,np.float32)/255.).permute(2,0,1)[None]
    t=F.pad(t,(0,pw,0,ph),mode='reflect')
    lab_tg=srgb_to_lab(t.permute(0,2,3,1)).permute(0,3,1,2)
    cov_tg=srgb_to_cov(t.permute(0,2,3,1),icc,intent).permute(0,3,1,2)
    with torch.no_grad():
        nh=hard_levels(m(t)/0.01)
        lab_nn=neugebauer_lab_2bit(nh,lut)
        # 经典 ED：对 ICC 连续调逐通道标准 FS，再量化到 4 档
        fs_cov=torch.from_numpy(np.stack([fs_plane(cov_tg[0,c].numpy()) for c in range(4)]))[None]
        lab_ed=neugebauer_lab_2bit(E.one_hot_from_cov(fs_cov),lut)
        # gs RIP
        rip=E.rip_cmyk(Image.fromarray((t[0].permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8),'RGB'),icc,intent)
        lab_rip=neugebauer_lab_2bit(E.one_hot_from_cov(torch.from_numpy(rip)[None]),lut)
        dv=lambda L: E.delta_e_visual(L,lab_tg,2.0).mean().item()
        r=(f.stem,dv(lab_nn),dv(lab_ed),dv(lab_rip))
    rows.append(r)
    print(f"{r[0]}  NN硬 {r[1]:5.2f} | 经典ED {r[2]:5.2f} | gsRIP {r[3]:5.2f}")
import statistics as st
g=lambda i: st.mean(x[i] for x in rows)
print(f"\n均值（σ=2px 视觉口径）NN硬 {g(1):.2f} | 经典ED {g(2):.2f} | gsRIP {g(3):.2f}")
print(f"经典ED 比 NN 好 {g(1)-g(2):+.2f} | 经典ED 比 gsRIP 好 {g(3)-g(2):+.2f}")
