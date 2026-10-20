"""σ 扫描：低通半径从 0.5px 到 4px，看 NN vs RIP 的 ΔE00 差距如何随观察尺度变化。"""
import sys; sys.path.insert(0,'.')
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from pathlib import Path
import eval as E
from halftone.model import UNet, hard_levels
from halftone.cmyk import (find_icc, load_or_build_lut256, neugebauer_lab_2bit,
                           srgb_to_lab, LEVELS)
from train import delta_e_2000
from eval import one_hot_from_cov
torch.set_num_threads(20); dev='cpu'
ck=torch.load('runs/cmyk_sd/best.pt',map_location=dev,weights_only=False)
a=ck['args']
m=UNet(base=a.get('base',32),up=a.get('up','nearest'),in_ch=a.get('in_ch',3),
       out_ch=a.get('out_ch',4),levels=a.get('levels',4)).to(dev)
m.load_state_dict(ck['model']); m.eval()
icc=find_icc(a.get('icc')); intent=a.get('icc_intent',0)
lut=load_or_build_lut256(Path('runs/cmyk_sd'),icc,intent)

def lowpass(x,sig):
    if sig<=0: return x
    r=max(1,int(3*sig)+1); xs=torch.arange(-r,r+1,dtype=torch.float32)
    k=torch.exp(-xs**2/(2*sig*sig)); k=k/k.sum(); k2=(k[:,None]@k[None,:])[None,None]
    C=x.shape[1]
    return F.conv2d(x.reshape(-1,1,x.shape[2],x.shape[3]),k2,
                    padding=k2.shape[-1]//2).reshape(1,C,x.shape[2],x.shape[3])

SIGS=[0.0,0.5,1.0,2.0,3.0,4.0]
res={s:{"nn":[], "rip":[]} for s in SIGS}
for f in sorted(Path('data/photos_val').glob('*.png'))[:8]:
    rgb=Image.open(f).convert('RGB'); w,h=rgb.size
    pw,ph=(8-w%8)%8,(8-h%8)%8
    t=torch.from_numpy(np.asarray(rgb,np.float32)/255.).permute(2,0,1)[None]
    t=F.pad(t,(0,pw,0,ph),mode='reflect')
    lab_tg=srgb_to_lab(t.permute(0,2,3,1)).permute(0,3,1,2)
    with torch.no_grad():
        lab_nn=neugebauer_lab_2bit(hard_levels(m(t)/0.01),lut)
        rip=E.rip_cmyk(Image.fromarray((t[0].permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8),'RGB'),icc,intent)
        lab_rp=neugebauer_lab_2bit(one_hot_from_cov(torch.from_numpy(rip)[None]),lut)
        for s in SIGS:
            res[s]["nn"].append(delta_e_2000(lowpass(lab_nn,s),lowpass(lab_tg,s)).mean().item())
            res[s]["rip"].append(delta_e_2000(lowpass(lab_rp,s),lowpass(lab_tg,s)).mean().item())
import statistics as st
print(f"{'σ(px)':>6} {'NN ΔE00':>9} {'RIP ΔE00':>9} {'NN优势':>9}")
for s in SIGS:
    n=st.mean(res[s]["nn"]); r=st.mean(res[s]["rip"])
    print(f"{s:6.1f} {n:9.2f} {r:9.2f} {r-n:+9.2f}")
