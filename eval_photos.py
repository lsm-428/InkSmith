"""批量客场评测：真实照片文件夹上 模型 vs FS vs GS 筛网。

用法：uv run python eval_photos.py [文件夹] [--ckpt runs/halftone/best.pt]
默认文件夹 data/photos_val（CBSD68 真实照片，未参与训练）。
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from halftone.model import UNet
from halftone.printer import PrinterSimulator
from verify import floyd_steinberg, gs_halftone


def psnr(x, y):
    return 10 * math.log10(1.0 / (F.mse_loss(x, y).item() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="data/photos_val")
    ap.add_argument("--ckpt", default="runs/halftone/best.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = UNet(base=a["base"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    printer = PrinterSimulator(a["sigma"], a["gain"]).to(device)

    print(f"{'照片':>12} {'模型':>7} {'FS':>7} {'GS':>7} {'vs FS':>7} {'vs GS':>7}")
    rows = []
    for p in sorted(Path(args.folder).glob("*.png")):
        img = Image.open(p).convert("L")
        w, h = img.size
        pw, ph = (8 - w % 8) % 8, (8 - h % 8) % 8
        t = torch.from_numpy(np.asarray(img, np.float32) / 255.)[None, None].to(device)
        t = F.pad(t, (0, pw, 0, ph), mode="reflect")
        with torch.no_grad():
            ink_nn = (model(t) > 0).float()
        size = (t.shape[3], t.shape[2])
        ink_fs = floyd_steinberg(img, size).to(device)
        ink_gs = gs_halftone(img, size, Path("/tmp")).to(device)
        r = [psnr(printer(x), t) for x in (ink_nn, ink_fs, ink_gs)]
        rows.append(r)
        print(f"{p.stem:>12} {r[0]:7.2f} {r[1]:7.2f} {r[2]:7.2f} "
              f"{r[0]-r[1]:+7.2f} {r[0]-r[2]:+7.2f}")
    rows = np.array(rows)
    print(f"{'均值':>12} {rows[:,0].mean():7.2f} {rows[:,1].mean():7.2f} "
          f"{rows[:,2].mean():7.2f} {rows[:,0].mean()-rows[:,1].mean():+7.2f} "
          f"{rows[:,0].mean()-rows[:,2].mean():+7.2f}")


if __name__ == "__main__":
    main()
