"""1-bit 对照：trained vs random 硬阈值 / ED(模型) / ED(零先验)。"""
import sys; sys.path.insert(0,'.')
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from pathlib import Path
import eval as E
from halftone.model import UNet
from halftone.printer import FilmSimulator
from halftone.ed_fs import ed_fs
torch.set_num_threads(20); dev='cpu'
ck=torch.load('runs/onebit/best.pt',map_location=dev,weights_only=False)
a=ck['args']
def mk():
    return UNet(base=a.get('base',32),up=a.get('up','nearest'),in_ch=1,out_ch=1,levels=1).to(dev)
tr=mk(); tr.load_state_dict(ck['model']); tr.eval()
torch.manual_seed(0); rd=mk(); rd.eval()
pr=FilmSimulator(sigma_um=a['sigma_um'], dpi=a['dpi']).to(dev)
LV=torch.tensor([0.,1.],device=dev)
def edc(lg,tgt,model_prior=True):
    z=torch.stack([torch.zeros_like(lg), lg if model_prior else torch.zeros_like(lg)],-1)
    return ed_fs(z,tgt,LV,bias_gain=4.0)[...,1]
rows=[]
for f in sorted(Path('data/photos_val').glob('*.png'))[:8]:
    img=Image.open(f).convert('L')
    tgt,base=E.film_target_and_base(img,True,dev)
    with torch.no_grad():
        lg_tr=tr(tgt); lg_rd=rd(tgt)
        r=(f.stem, E.psnr(pr((lg_tr>0).float()),tgt), E.psnr(pr((lg_rd>0).float()),tgt),
           E.psnr(pr(edc(lg_tr,tgt,True)),tgt), E.psnr(pr(edc(lg_tr,tgt,False)),tgt))
    rows.append(r); print(f"{r[0]}  trained硬 {r[1]:5.2f} | random硬 {r[2]:5.2f} | ED(训练) {r[3]:5.2f} | ED(零) {r[4]:5.2f}")
import statistics as st
g=lambda i: st.mean(x[i] for x in rows)
print(f"\n均值 trained硬 {g(1):.2f} | random硬 {g(2):.2f} | ED(训练) {g(3):.2f} | ED(零) {g(4):.2f}")
print(f"训练收益(硬) {g(2)-g(1):+.2f}dB | ED后训练收益 {g(3)-g(4):+.2f}dB")
