"""评测公共工具：模型 vs 传统基线在同一打印模拟器下比 PSNR。

把各入口脚本（eval.py / webapp.py / train.py）里重复的公共函数集中到这里，
各入口脚本只 import，不改行为。
"""
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from halftone.model import UNet
from halftone.printer import build_printer


def psnr(a, b):
    return 10 * math.log10(1.0 / (F.mse_loss(a, b).item() + 1e-12))


def load_model_and_printer(ckpt_path, dpi=None, device=None):
    """从 ckpt 重建 UNet + printer；film 模式下可用 dpi 覆盖预览/评估分辨率。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    if dpi and a.get("printer") == "film":
        a = {**a, "dpi": dpi}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(base=a["base"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    printer = build_printer(a).to(device)
    return model, printer, a, device


def film_target_and_base(img, film, device):
    """灰度图 -> pad(8 倍数, reflect) 后的 target；film 时同时返回反相 FS/GS 输入。"""
    w, h = img.size
    pw, ph = (8 - w % 8) % 8, (8 - h % 8) % 8
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)[None, None].to(device)
    t = F.pad(t, (0, pw, 0, ph), mode="reflect")
    base = Image.eval(img, lambda p: 255 - p) if film else img
    return t, base


def floyd_steinberg(img_pil, size):
    """PIL 自带 FS 抖动，作为传统基线。返回墨点图（1=墨）。"""
    bw = img_pil.resize(size).convert("1")  # 1-bit，FS 抖动
    arr = np.asarray(bw).astype(np.float32)  # mode "1" 是 bool：True=白，勿除 255
    return torch.from_numpy(1.0 - arr)[None, None]  # 转成 1=墨


def gs_halftone(img_pil, size, workdir):
    """Ghostscript 工业 RIP 基线：走它自家 halftone 筛网渲成 1-bit。

    gs 不吃 PNG：PIL 包 PDF（resolution=72 使 1px=1pt），再用 pbmraw
    1-bit 设备在 72dpi 下渲染，输出与原图像素一一对应。
    """
    tmp_pdf = Path(workdir) / "_gs_in.pdf"
    tmp_pbm = Path(workdir) / "_gs_out.pbm"
    img_pil.resize(size).save(tmp_pdf, "PDF", resolution=72)
    subprocess.run(["gs", "-dBATCH", "-dNOPAUSE", "-q",
                    "-sDEVICE=pbmraw", "-r72", "-o", str(tmp_pbm), str(tmp_pdf)],
                   check=True, capture_output=True)
    bw = Image.open(tmp_pbm)
    if bw.size != size:  # gs 页面舍入可能导致 1px 偏差，对齐
        bw = bw.resize(size, Image.NEAREST)
    arr = np.asarray(bw.convert("L"), dtype=np.float32) / 255.0  # 1=白
    tmp_pdf.unlink()
    tmp_pbm.unlink()
    return torch.from_numpy(1.0 - arr)[None, None]  # 转成 1=墨


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
