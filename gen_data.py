"""生成合成训练图：渐变/斑块/分形噪声/几何形状/条纹/混合。

半色调模型需要见过各种频率成分：低频渐变、中频纹理、锐利边缘、高频周期图案。
输出 PNG 到 data/images_gray/。
"""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

rng = np.random.default_rng(42)


def to_pil(a):
    """[0,1] float -> L 模式 PIL 图。30% 概率压缩动态范围，让模型也练中间调。"""
    a = a - a.min()
    a = a / (a.max() + 1e-9)
    if rng.random() < 0.3:
        lo, hi = sorted(rng.uniform(0.0, 1.0, 2))
        if hi - lo < 0.3:
            hi = min(1.0, lo + 0.3)
        a = lo + (hi - lo) * a
    return Image.fromarray((a * 255).astype(np.uint8), mode="L")


def gradient(s):
    """随机角度线性渐变或径向渐变。"""
    x = np.linspace(0, 1, s)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    if rng.random() < 0.5:
        theta = rng.uniform(0, math.pi)
        a = xx * math.cos(theta) + yy * math.sin(theta)
    else:
        cx, cy = rng.uniform(0.2, 0.8, 2)
        a = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return a


def blobs(s):
    """高斯斑块叠加（平滑低中频内容）。"""
    x = np.linspace(0, 1, s)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    a = np.zeros((s, s))
    for _ in range(rng.integers(3, 10)):
        cx, cy = rng.uniform(0, 1, 2)
        sigma = rng.uniform(0.03, 0.3)
        a += rng.uniform(-1, 1) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    return a


def fbm(s):
    """分形噪声：多倍频随机网格双三次上采样叠加（纹理）。"""
    a = np.zeros((s, s))
    amp, total = 1.0, 0.0
    for octave in range(rng.integers(3, 6)):
        n = 4 * 2**octave
        coarse = Image.fromarray((rng.random((n, n)) * 255).astype(np.uint8))
        up = np.asarray(coarse.resize((s, s), Image.BICUBIC), dtype=np.float64) / 255.0
        a += amp * up
        total += amp
        amp *= 0.5
    return a / total


def shapes(s):
    """几何形状：圆/矩形/多边形/圆环，随机明暗 + 可选边缘模糊。"""
    img = Image.new("L", (s, s), int(rng.uniform(0, 255)))
    d = ImageDraw.Draw(img)
    for _ in range(rng.integers(2, 7)):
        v = int(rng.uniform(0, 255))
        x0, y0 = rng.uniform(0, s, 2)
        r = rng.uniform(0.05, 0.4) * s
        kind = rng.integers(0, 4)
        if kind == 0:
            d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=v)
        elif kind == 1:
            d.rectangle([x0 - r, y0 - r * rng.uniform(0.3, 1), x0 + r, y0 + r], fill=v)
        elif kind == 2:
            n = rng.integers(3, 7)
            pts = [(x0 + r * math.cos(2 * math.pi * k / n + rng.uniform(0, 1)),
                    y0 + r * math.sin(2 * math.pi * k / n)) for k in range(n)]
            d.polygon(pts, fill=v)
        else:
            w = max(2, int(r * rng.uniform(0.1, 0.3)))
            d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], outline=v, width=w)
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(1, 4)))
    return np.asarray(img, dtype=np.float64) / 255.0


def stripes(s):
    """正弦光栅：随机频率/方向，可两张叠加（高频周期内容）。"""
    x = np.linspace(0, 1, s)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    a = np.zeros((s, s))
    for _ in range(rng.integers(1, 3)):
        theta = rng.uniform(0, math.pi)
        freq = rng.uniform(5, 40) * 2 * math.pi
        phase = rng.uniform(0, 2 * math.pi)
        a += 0.5 + 0.5 * np.sin(freq * (xx * math.cos(theta) + yy * math.sin(theta)) + phase)
    return a


def mixed(s):
    """两类随机混合，模拟真实照片的复合频率。"""
    gens = [gradient, blobs, fbm, shapes, stripes]
    g1, g2 = rng.choice(gens, size=2, replace=False)
    alpha = rng.uniform(0.3, 0.7)
    return alpha * g1(s) + (1 - alpha) * g2(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--out", default="data/images_gray")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    global rng
    rng = np.random.default_rng(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gens = [gradient, blobs, fbm, shapes, stripes, mixed]

    for i in range(args.n):
        gen = gens[i % len(gens)]
        img = to_pil(gen(args.size))
        img.save(out / f"gen_{i:04d}_{gen.__name__}.png")
    print(f"已生成 {args.n} 张 {args.size}x{args.size} 灰度图到 {out}/")


if __name__ == "__main__":
    main()
