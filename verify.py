"""验证：模型 vs Floyd-Steinberg 基线，同一打印模拟器下比 PSNR。

产出对比图：原图 | 模型位图 | 模型模拟打印 | FS位图 | FS模拟打印
用法：uv run python verify.py 图片.jpg [--ckpt runs/halftone/best.pt]
"""
import argparse
import math
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image

from halftone.model import UNet
from halftone.printer import PrinterSimulator


def psnr(a, b):
    return 10 * math.log10(1.0 / (F.mse_loss(a, b).item() + 1e-12))


def floyd_steinberg(img_pil, size):
    """PIL 自带 FS 抖动，作为传统基线。返回墨点图（1=墨）。"""
    bw = img_pil.resize(size).convert("1")  # 1-bit，FS 抖动
    arr = np.asarray(bw).astype(np.float32)  # mode "1" 是 bool：True=白，勿除 255
    return torch.from_numpy(1.0 - arr)[None, None]  # 转成 1=墨


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--ckpt", default="runs/halftone/best.pt")
    p.add_argument("--out", default="out/verify")
    args = p.parse_args()

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(base=a["base"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    printer = PrinterSimulator(a["sigma"], a["gain"]).to(device)

    img = Image.open(args.input).convert("L")
    w, h = img.size
    pw, ph = (8 - w % 8) % 8, (8 - h % 8) % 8
    arr = np.asarray(img, dtype=np.float32) / 255.0
    target = torch.from_numpy(arr)[None, None].to(device)
    target = F.pad(target, (0, pw, 0, ph), mode="reflect")

    with torch.no_grad():
        ink_nn = (model(target) > 0).float()
        printed_nn = printer(ink_nn)
    ink_fs = floyd_steinberg(img, (target.shape[3], target.shape[2])).to(device)
    printed_fs = printer(ink_fs)

    p_nn, p_fs = psnr(printed_nn, target), psnr(printed_fs, target)
    print(f"模拟打印 PSNR：模型 {p_nn:.2f} dB | Floyd-Steinberg {p_fs:.2f} dB "
          f"| 差 {p_nn - p_fs:+.2f} dB")

    # 落盘位图（墨=黑）并用 ghostscript inkcov 独立审计墨覆盖率
    cov = {}
    for name, ink in [("model", ink_nn), ("fs", ink_fs)]:
        path = f"{args.out}_{name}_bitmap.png"
        pil = Image.fromarray(((1 - ink[0, 0].cpu().numpy()) * 255).astype(np.uint8),
                              mode="L")
        pil.save(path)
        pil.save(f"{args.out}_{name}_bitmap.pdf", "PDF")  # gs 不认 PNG 输入
        cov[name] = gs_inkcov(f"{args.out}_{name}_bitmap.pdf")
        warn = " ⚠️ 覆盖率过高，实心填充风险" if cov[name] > 0.85 else ""
        print(f"[inkcov] {name}: K 覆盖率 {cov[name]:.1%} "
              f"(张量自检 {ink.mean().item():.1%}){warn} -> {path}")

    grid = torch.cat([target, ink_nn, printed_nn, ink_fs, printed_fs])
    save_image(grid, f"{args.out}.png", nrow=1)
    print(f"对比图：{args.out}.png（原图|模型位图|模型打印|FS位图|FS打印）")


def gs_inkcov(bitmap_path):
    """ghostscript inkcov 设备审计：返回 K 通道墨覆盖率。gs 失败则回退 None。"""
    try:
        out = subprocess.run(
            ["gs", "-o", "-", "-sDEVICE=inkcov", bitmap_path],
            capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[inkcov] ghostscript 调用失败：{e}")
        return float("nan")
    for line in out.splitlines():
        if "CMYK OK" in line:
            return float(line.split()[3])  # C M Y K 中的 K
    return float("nan")


if __name__ == "__main__":
    main()
