"""独立实现经典「多档误差扩散」基线（不经任何网络），验证 2-bit 能力的真实价值。
   与 ed_fs 无关的纯 numpy 实现：对 ICC 连续覆盖率按最近档量化 + FS 扩散余量。
"""
import sys; sys.path.insert(0,'.')
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from pathlib import Path
import eval as E
from halftone.model import UNet, hard_levels
from halftone.cmyk import (find_icc, load_or_build_lut256, neugebauer_lab_2bit,
                           srgb_to_lab, srgb_to_cov, LEVELS)
torch.set_num_threads(20); dev='cpu'
ck=torch.load('runs/cmyk_sd/best.pt',map_location=dev,weights_only=False)
a=ck['args']
m=UNet(base=a.get('base',32),up=a.get('up','nearest'),in_ch=a.get('in_ch',3),
       out_ch=a.get('out_ch',4),levels=a.get('levels',4)).to(dev)
m.load_state_dict(ck['model']); m.eval()
icc=find_icc(a.get('icc')); intent=a.get('icc_intent',0)
lut=load_or_build_lut256(Path('runs/cmyk_sd'),icc,intent)
LV=np.array([0.,1/3,2/3,1.],np.float64)

def ml_ed_plane(x):
    """最近档量化 + FS 扩散（真·多档误差扩散）"""
    H,W=x.shape; buf=x.astype(np.float64).copy(); out=np.zeros((H,W),np.float64)
    for y in range(H):
        rng=range(W) if y%2==0 else range(W-1,-1,-1); d=1 if y%2==0 else -1
        for xx in rng:
            old=buf[y,xx]
            idx=int(np.argmin(np.abs(LV-old))); new=LV[idx]
            out[y,xx]=idx; buf[y,xx]=new; err=old-new
            for dx,dy,wt in ((d,0,7/16),(-d,1,3/16),(0,1,5/16),(d,1,1/16)):
                nx,ny=xx+dx,y+dy
                if 0<=nx<W and ny<H: buf[ny,nx]+=err*wt
    return out.astype(np.int64)

rows=[]
for f in sorted(Path('data/photos_val').glob('*.png'))[:8]:
    rgb=Image.open(f).convert('RGB'); w,h=rgb.size
    pw,ph=(8-w%8)%8,(8-h%8)%8
    t=torch.from_numpy(np.asarray(rgb,np.float32)/255.).permute(2,0,1)[None]
    t=F.pad(t,(0,pw,0,ph),mode='reflect')
    lab_tg=srgb_to_lab(t.permute(0,2,3,1)).permute(0,3,1,2)
    cov_tg=srgb_to_cov(t.permute(0,2,3,1),icc,intent).permute(0,3,1,2)
    with torch.no_grad():
        idx=np.stack([ml_ed_plane(cov_tg[0,c].numpy()) for c in range(4)])
        oh=F.one_hot(torch.from_numpy(idx),4).float()[None]
        lab_ml=neugebauer_lab_2bit(oh,lut)
        lab_nn=neugebauer_lab_2bit(hard_levels(m(t)/0.01),lut)
        rip=E.rip_cmyk(Image.fromarray((t[0].permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8),'RGB'),icc,intent)
        lab_rip=neugebauer_lab_2bit(E.one_hot_from_cov(torch.from_numpy(rip)[None]),lut)
        dv=lambda L: E.delta_e_visual(L,lab_tg,2.0).mean().item()
        r=(f.stem,dv(lab_ml),dv(lab_nn),dv(lab_rip))
    rows.append(r); print(f"{r[0]}  多档ED {r[1]:5.2f} | NN硬 {r[2]:5.2f} | RIP {r[3]:5.2f}")
import statistics as st
g=lambda i: st.mean(x[i] for x in rows)
print(f"\n均值  经典多档ED {g(1):.2f} | NN硬 {g(2):.2f} | gsRIP(1bit) {g(3):.2f}")
print(f"多档ED 优于 NN {g(2)-g(1):+.2f} | 多档ED 优于 RIP {g(3)-g(1):+.2f}")
