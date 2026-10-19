"""生成合成训练图：渐变/斑块/分形噪声/几何形状/条纹/混合。

半色调模型需要见过各种频率成分：低频渐变、中频纹理、锐利边缘、高频周期图案。
输出 PNG 到 data/images/。

`--rgb` 输出彩色图（RGB 模式）供 CMYK 多档墨量路线训练用：
  - 复用同一批生成器，把标量场映射到 3 个通道（各自独立相位/偏移）；
  - 另加 `cmyk_patch`：随机 4 通道连续覆盖率 -> ICC LUT -> sRGB，
    保证训练目标一定在 CMYK 色域内且 4 档可精确表达。
"""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFilter

rng = np.random.default_rng(42)

#: 4 档覆盖率（2-bit）
LEVELS = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])


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


# ---------------------------------------------------------------- CMYK 路线


def _icc_xf(icc, intent=0):
    from PIL import ImageCms
    src = ImageCms.getOpenProfile(icc)
    return ImageCms.buildTransform(src, ImageCms.createProfile("sRGB"),
                                   "CMYK", "RGB", renderingIntent=intent)


def _cov_to_srgb(cov, icc, intent=0):
    """cov: (s,s,4) 连续覆盖率 [0,1] -> sRGB uint8 (s,s,3)。"""
    s = cov.shape[0]
    ink = np.round(np.clip(cov, 0, 1) * 255).astype(np.uint8)
    img = Image.merge("CMYK", [Image.fromarray(ink[:, :, i], "L") for i in range(4)])
    return np.asarray(ImageCms.applyTransform(img, _icc_xf(icc, intent)).convert("RGB"))


def _srgb_to_lab(rgb01):
    m = rgb01 <= 0.04045
    lin = np.where(m, rgb01 / 12.92, ((rgb01 + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def _lab_to_srgb(lab):
    fy = (lab[..., 0] + 16) / 116
    fx = lab[..., 1] / 500 + fy
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787)
    xyz = xyz * np.array([0.95047, 1.0, 1.08883])
    M = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689, 1.8758, 0.0415],
                  [0.0557, -0.2040, 1.0570]])
    lin = xyz @ M.T
    a = np.clip(lin, 0, 1)
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055)


def cmyk_patch(s, icc, intent=0):
    """随机 4 通道连续覆盖率 -> LUT -> sRGB：色域内且 4 档可精确表达。

    覆盖率先量化到 4 档（LEVELS），再过 ICC 得到 sRGB；因此该样本的
    「sRGB -> CMYK 分离 -> 最近档」往返误差接近 0，2-bit 网络可精确表达。
    """
    cov = np.zeros((s, s, 4))
    for ch in range(4):
        kind = rng.integers(0, 3)
        if kind == 0:                      # 平滑渐变
            a = gradient(s)
        elif kind == 1:                    # 斑块
            a = blobs(s)
        else:                              # 常数 + 局部扰动
            a = 0.5 + 0.5 * np.ones((s, s))
            for _ in range(rng.integers(1, 5)):
                a += rng.uniform(-0.5, 0.5) * np.exp(
                    -(((np.meshgrid(np.linspace(0, 1, s), np.linspace(0, 1, s),
                                    indexing="ij")[0] - rng.uniform(0, 1)) ** 2
                       + (np.meshgrid(np.linspace(0, 1, s), np.linspace(0, 1, s),
                                      indexing="ij")[1] - rng.uniform(0, 1)) ** 2)
                      / (2 * rng.uniform(0.02, 0.2) ** 2)))
        a = (a - a.min()) / (a.max() - a.min() + 1e-9)
        if rng.random() < 0.4:             # 通道几乎不用（模拟真实照片的稀疏墨）
            a = a * rng.uniform(0.0, 0.25)
        cov[:, :, ch] = np.clip(a, 0, 1)
    # TAC 限制：四通道和 > 3.0 时按比例缩放，然后**量化到 4 档**
    tac = cov.sum(-1, keepdims=True)
    over = tac > 3.0
    cov = np.where(over, cov * (3.0 / np.maximum(tac, 1e-6)), cov)
    cov = LEVELS[np.clip(np.round(cov * 3), 0, 3).astype(int)]
    rgb = _cov_to_srgb(cov, icc, intent)
    return rgb.astype(np.float64) / 255.0


def colorize(s, icc=None, sat=0.85, gamma=1.0):
    """把标量场映射成 RGB：3 通道用不同相位/偏移，避免退化成灰度。

    `sat<1` 压低通道间差异（降饱和度）、`gamma>1` 整体压暗——用于把
    合成图的分布往真实照片靠（照片饱和度均值 ~0.25、线性均值 0.209，
    而默认参数生成的合成图 OOG 率 33% vs 照片 16.9%，PITFALLS #22）。
    """
    a = gradient(s) if rng.random() < 0.5 else blobs(s)
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    x = np.linspace(0, 1, s)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    out = np.stack([a, a, a], -1)
    for ch in range(3):
        theta = rng.uniform(0, math.pi)
        out[:, :, ch] = np.clip(
            0.5 + 0.45 * np.sin(6.0 * (xx * math.cos(theta) + yy * math.sin(theta))
                                + rng.uniform(0, 2 * math.pi)), 0, 1)
        out[:, :, ch] = out[:, :, ch] * (0.3 + 0.7 * a)
    # 向灰度收缩 = 降饱和度；gamma 压暗。都作用在 [0,1] 上，不引入 OOG。
    gray = out.mean(-1, keepdims=True)
    out = gray + (out - gray) * sat
    if gamma != 1.0:
        out = np.clip(out, 0, 1) ** (1.0 / gamma)
    return np.clip(out, 0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--out", default="data/images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rgb", action="store_true",
                   help="输出彩色 RGB 图（CMYK 多档墨量路线训练用）")
    p.add_argument("--icc", default=None,
                   help="CMYK ICC 路径；--rgb 时必填（或用内置候选表）")
    p.add_argument("--intent", type=int, default=0,
                   help="ICC rendering intent：0=relative colorimetric（默认，自带 OOG 裁剪）")
    p.add_argument("--cmyk-frac", type=float, default=0.4,
                   help="--rgb 时 cmyk_patch（色域内保证）样本占比。"
                        "★真实照片训练时建议 0.15：40% 会把分布拉离现实"
                        "（合成 OOG 33% vs 照片 16.9%，PITFALLS #22）")
    p.add_argument("--sat", type=float, default=0.85,
                   help="--rgb 时彩色样本的饱和度缩放（1=原样，向照片靠用 ~0.6）")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="--rgb 时彩色样本的压暗 gamma（>1 更暗，照片线性均值 0.209）")
    args = p.parse_args()

    global rng
    rng = np.random.default_rng(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gens = [gradient, blobs, fbm, shapes, stripes, mixed]

    icc = None
    if args.rgb:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from halftone.cmyk import find_icc
        icc = find_icc(args.icc)

    for i in range(args.n):
        if args.rgb:
            if rng.random() < args.cmyk_frac:
                a = cmyk_patch(args.size, icc, args.intent)
                tag = "cmyk"
            else:
                a = colorize(args.size, icc, sat=args.sat, gamma=args.gamma)
                tag = "color"
            img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), mode="RGB")
            img.save(out / f"gen_{i:04d}_{tag}.png")
        else:
            gen = gens[i % len(gens)]
            img = to_pil(gen(args.size))
            img.save(out / f"gen_{i:04d}_{gen.__name__}.png")
    if args.rgb:
        print(f"已生成 {args.n} 张 {args.size}x{args.size} 彩色图到 {out}/ "
              f"(cmyk_patch 占比 {args.cmyk_frac}, ICC={icc}, intent={args.intent}, "
              f"sat={args.sat}, gamma={args.gamma})")
    else:
        print(f"已生成 {args.n} 张 {args.size}x{args.size} 灰度图到 {out}/")


if __name__ == "__main__":
    main()
