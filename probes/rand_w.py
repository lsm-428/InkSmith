"""关键对照：训练过的权重 vs 随机权重，同一硬阈值管线。训练若无效，两者应接近。"""
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
def mk():
    return UNet(base=a.get('base',32),up=a.get('up','nearest'),in_ch=a.get('in_ch',3),
        out_ch=a.get('out_ch',4),levels=a.get('levels',4)).to(dev)
trained=mk(); trained.load_state_dict(ck['model']); trained.eval()
torch.manual_seed(0); rand=mk(); rand.eval()
icc=find_icc(a.get('icc')); intent=a.get('icc_intent',0)
lut=load_or_build_lut256(Path('runs/cmyk_sd'),icc,intent)
rows=[]
for f in sorted(Path('data/photos_val').glob('*.png'))[:8]:
    rgb=Image.open(f).convert('RGB'); w,h=rgb.size
    pw,ph=(8-w%8)%8,(8-h%8)%8
    t=torch.from_numpy(np.asarray(rgb,np.float32)/255.).permute(2,0,1)[None]
    t=F.pad(t,(0,pw,0,ph),mode='reflect')
    lab_tg=srgb_to_lab(t.permute(0,2,3,1)).permute(0,3,1,2)
    with torch.no_grad():
        dv=lambda L: E.delta_e_visual(L,lab_tg,2.0).mean().item()
        d_tr=dv(neugebauer_lab_2bit(hard_levels(trained(t)/0.01),lut))
        d_ra=dv(neugebauer_lab_2bit(hard_levels(rand(t)/0.01),lut))
        r=(f.stem,d_tr,d_ra)
    rows.append(r); print(f"{r[0]}  trained {r[1]:5.2f} | random {r[2]:5.2f}")
import statistics as st
g=lambda i: st.mean(x[i] for x in rows)
print(f"\n均值（σ=2px）trained {g(1):.2f} | random {g(2):.2f} | 训练收益 {g(2)-g(1):+.2f}")
