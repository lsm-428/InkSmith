"""判决性对照：同一 target/同一 printer/同一 psnr 口径下，
   A eval.floyd_steinberg（现有基线）
   B 手写标准 FS（蛇形，直接在覆盖域抖动，无 PIL resize/convert）
   C ed_fs(零先验)
   D ed_fs(模型 logits)
"""
import sys; sys.path.insert(0,'.')
import torch, numpy as np, torch.nn.functional as F
from PIL import Image
from pathlib import Path
import eval as E
from halftone.model import UNet
from halftone.printer import FilmSimulator
from halftone.ed_fs import ed_fs
torch.set_num_threads(20)
dev='cpu'
ck=torch.load('runs/onebit/best.pt',map_location=dev,weights_only=False)
a=ck['args']; m=UNet(base=a.get('base',32),up=a.get('up','nearest'),
   in_ch=a.get('in_ch',1),out_ch=a.get('out_ch',1),levels=a.get('levels',1)).to(dev)
m.load_state_dict(ck['model']); m.eval()
pr=FilmSimulator(sigma_um=a['sigma_um'], dpi=a['dpi']).to(dev)
LV=torch.tensor([0.,1.],device=dev)

def fs_pure(tgt):
    H,W=tgt.shape; buf=tgt.copy()
    for y in range(H):
        rng=range(W) if y%2==0 else range(W-1,-1,-1); d=1 if y%2==0 else -1
        for x in rng:
            old=buf[y,x]; new=1.0 if old>0.5 else 0.0
            buf[y,x]=new; err=old-new
            for dx,dy,wt in ((d,0,7/16),(-d,1,3/16),(0,1,5/16),(d,1,1/16)):
                nx,ny=x+dx,y+dy
                if 0<=nx<W and ny<H: buf[ny,nx]+=err*wt
    return (buf>0.5).astype(np.float32)

rows=[]
for f in sorted(Path('data/photos_val').glob('*.png')):
    img=Image.open(f).convert('L')
    target, base = E.film_target_and_base(img, True, dev)
    size=(target.shape[3],target.shape[2])
    A=E.floyd_steinberg(base,size).to(dev)
    B=torch.from_numpy(fs_pure(target[0,0].numpy()))[None,None]
    with torch.no_grad():
        lg=m(target)
        C=ed_fs(torch.stack([torch.zeros_like(lg)]*2,-1), target, LV, bias_gain=4.0)[...,1]
        D=ed_fs(torch.stack([torch.zeros_like(lg), lg],-1), target, LV, bias_gain=4.0)[...,1]
        r=(f.stem, E.psnr(pr(A),target), E.psnr(pr(B),target),
           E.psnr(pr(C),target), E.psnr(pr(D),target))
    rows.append(r)
    print(f"{r[0]}  A_evalFS {r[1]:6.2f} | B_手写FS {r[2]:6.2f} | C_ed零 {r[3]:6.2f} | D_ed模型 {r[4]:6.2f}")
import statistics as st
print("\n均值  A_evalFS %.2f | B_手写FS %.2f | C_ed零 %.2f | D_ed模型 %.2f"
      % tuple(st.mean(x[i] for x in rows) for i in (1,2,3,4)))
print("D-B = %+.2f dB（ED 相对公平 FS 基线的真实差距）" % st.mean(x[4]-x[2] for x in rows))
