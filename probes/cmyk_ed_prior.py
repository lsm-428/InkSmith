"""CMYK+ED 视觉口径下，模型 logits 到底贡献多少？
   A 硬 argmax(模型) / B ED(模型 logits) / C ED(零 logits) / D ED(打乱 spatial 的 logits)
"""
import sys; sys.path.insert(0,'.')
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from pathlib import Path
import eval as E
from halftone.model import UNet, hard_levels
from halftone.cmyk import (find_icc, load_or_build_lut256, neugebauer_lab_2bit,
                           srgb_to_lab, srgb_to_cov, LEVELS)
from halftone.ed_fs import ed_fs
from train import delta_e_2000
torch.set_num_threads(20); dev='cpu'
ck=torch.load('runs/cmyk_sd/best.pt',map_location=dev,weights_only=False)
a=ck['args']
m=UNet(base=a.get('base',32),up=a.get('up','nearest'),in_ch=a.get('in_ch',3),
       out_ch=a.get('out_ch',4),levels=a.get('levels',4)).to(dev)
m.load_state_dict(ck['model']); m.eval()
icc=find_icc(a.get('icc')); intent=a.get('icc_intent',0)
lut=load_or_build_lut256(Path('runs/cmyk_sd'),icc,intent)
LV=LEVELS.to(dev)

def de_v(a_,b_,sig=2.0):
    return E.delta_e_visual(a_,b_,sig).mean().item()

rows=[]
for f in sorted(Path('data/photos_val').glob('*.png'))[:8]:
    rgb=Image.open(f).convert('RGB'); w,h=rgb.size
    pw,ph=(8-w%8)%8,(8-h%8)%8
    t=torch.from_numpy(np.asarray(rgb,np.float32)/255.).permute(2,0,1)[None]
    t=F.pad(t,(0,pw,0,ph),mode='reflect')
    lab_tg=srgb_to_lab(t.permute(0,2,3,1)).permute(0,3,1,2)
    cov_tg=srgb_to_cov(t.permute(0,2,3,1),icc,intent).permute(0,3,1,2)
    with torch.no_grad():
        lg=m(t)
        A=hard_levels(lg/0.01)
        B=ed_fs(lg,cov_tg,LV)
        C=ed_fs(torch.zeros_like(lg),cov_tg,LV)
        # D: 把 logits 在空间上打乱（保持数值分布，破坏结构）
        B_,C_,H_,W_,L_ = lg.shape
        perm=torch.randperm(H_*W_)
        flat=lg.reshape(B_,C_,H_*W_,L_)
        D=ed_fs(flat[:,:,perm,:].reshape(B_,C_,H_,W_,L_), cov_tg, LV)
        r=(f.stem, de_v(neugebauer_lab_2bit(A,lut),lab_tg),
              de_v(neugebauer_lab_2bit(B,lut),lab_tg),
              de_v(neugebauer_lab_2bit(C,lut),lab_tg),
              de_v(neugebauer_lab_2bit(D,lut),lab_tg))
    rows.append(r)
    print(f"{r[0]} A_硬 {r[1]:5.2f} | B_ED模型 {r[2]:5.2f} | C_ED零 {r[3]:5.2f} | D_ED乱 {r[4]:5.2f}")
import statistics as st
g=lambda i: st.mean(x[i] for x in rows)
print(f"\n均值 A_硬 {g(1):.2f} | B_ED模型 {g(2):.2f} | C_ED零 {g(3):.2f} | D_ED打乱 {g(4):.2f}")
print(f"模型贡献 B-C = {g(2)-g(3):+.2f} | B-D = {g(2)-g(4):+.2f}")
print("\nlogits 幅度:", lg.min().item(), lg.max().item(), lg.std().item())
print("cov_tg vs logits 相关（量化忽略程度）")
