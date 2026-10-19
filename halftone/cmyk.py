"""CMYK 多档墨量 + Neugebauer 叠印：ICC 色标定、LUT 构建、Demichel 面积模型。

与 1-bit 灰度路线的关系：
  - 本模块不依赖也不修改 halftone/model.py / printer.py / losses.py 的既有路径；
  - 灰度 1-bit 的 film/paper 物理模拟器保持单通道语义，CMYK 下不用可微打印模拟器
    （网点扩大的物理标定未做），直接用 ICC LUT 精确叠印；
  - Lab 目标一律取「原图 sRGB -> Lab」，不取「sRGB -> CMYK -> Lab」：
    ICC 往返有损（实测 intent=0 MAE 0.65/255，intent=1/2 MAE 3.07），往返会把
    目标色一起污染。

档位约定（2-bit = 4 档）：levels = [0, 1/3, 2/3, 1]（0/85/170/255 墨量），
与 WORKFLOW.md 阶段 5 的「无/小/中/大墨滴」一致。

LUT 索引约定（务必保持一致，多处依赖）：
  - 16 项 on/off 组合：itertools.product([0, 1], repeat=4) 顺序，K 变化最快，
    n = lC*8 + lM*4 + lY*2 + lK；
  - 256 项 4 档组合：n = lC*64 + lM*16 + lY*4 + lK，
    等价于 LUT.reshape(4, 4, 4, 4, 3)[lC, lM, lY, lK]。
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import torch
from PIL import Image, ImageCms

# ---------------------------------------------------------------- ICC 资产

ICC_CANDIDATES = [
    "/usr/share/color/icc/colord/FOGRA39L_coated.icc",
    "/usr/share/color/icc/colord/GRACoL_TR006_coated.icc",
    "/usr/share/color/icc/colord/FOGRA47L_coated.icc",
    "/usr/share/color/icc/colord/SWOP_TR003_coated.icc",
]

#: 2-bit 档位值（连续覆盖率）。多档模型输出 (B,4,4,H,W)，需要
#: `LEVELS.view(1, 4, 4, 1, 1)` 广播（前两个 4 = 通道数/档数）。
LEVELS = torch.tensor((0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0))


def find_icc(explicit=None):
    """返回可用的 CMYK ICC 路径；explicit 优先，其次候选表。"""
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"ICC 不存在: {explicit}")
        return explicit
    for p in ICC_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "找不到 CMYK ICC 配置文件；请用 --icc 指定，或安装 icc-profiles"
    )


def lab_to_srgb(lab):
    """Lab -> sRGB [0,1]（torch，可微）。用于 CMYK 路线训练预览。"""
    fy = (lab[..., 0] + 16) / 116
    fx = lab[..., 1] / 500 + fy
    fz = fy - lab[..., 2] / 200
    f = torch.stack([fx, fy, fz], dim=-1)
    xyz = torch.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787)
    xyz = xyz * torch.tensor([0.95047, 1.0, 1.08883],
                             dtype=lab.dtype, device=lab.device)
    M = torch.tensor([[3.2406, -1.5372, -0.4986],
                      [-0.9689, 1.8758, 0.0415],
                      [0.0557, -0.2040, 1.0570]],
                     dtype=lab.dtype, device=lab.device)
    lin = xyz @ M.transpose(-1, -2)
    a = lin.clamp(0, 1)
    return torch.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055)


def srgb_to_lab(rgb01):
    """sRGB [0,1] (…,H,W,3) -> Lab（torch，可微）。"""
    lin = torch.where(rgb01 <= 0.04045, rgb01 / 12.92,
                      ((rgb01 + 0.055) / 1.055) ** 2.4)
    M = torch.tensor([[0.4124, 0.3576, 0.1805],
                      [0.2126, 0.7152, 0.0722],
                      [0.0193, 0.1192, 0.9505]],
                     dtype=rgb01.dtype, device=rgb01.device)
    xyz = lin @ M.transpose(-1, -2)
    xyz = xyz / torch.tensor([0.95047, 1.0, 1.08883],
                             dtype=rgb01.dtype, device=rgb01.device)
    f = torch.where(xyz > 0.008856, xyz.clamp(min=1e-8) ** (1 / 3),
                    7.787 * xyz + 16 / 116)
    return torch.stack([116 * f[..., 1] - 16,
                        500 * (f[..., 0] - f[..., 1]),
                        200 * (f[..., 1] - f[..., 2])], dim=-1)


def srgb_to_cov(rgb01, icc, intent=0):
    """sRGB [0,1] (B,H,W,3) 或 (H,W,3) -> CMYK 连续覆盖率 (B,H,W,4)（ICC 分离）。

    非可微（PIL 路径），只作 l_cov 的软目标（实测 ICC 8bit 往返
    MAE 37.6/255，不能当精确档位标签）。
    """
    single = rgb01.dim() == 3
    if single:
        rgb01 = rgb01[None]
    flat = np.clip(rgb01.detach().cpu().numpy(), 0, 1)
    xf = _xf(icc, intent, "RGB", "CMYK")
    outs = []
    for k in range(flat.shape[0]):      # PIL 需要整幅 HxWx3 图
        img = Image.fromarray((flat[k] * 255).round().astype(np.uint8), "RGB")
        # PIL CMYK 语义：255 = 满墨；白 = 0，黑 = 大量 CMY+K
        ink = np.asarray(ImageCms.applyTransform(img, xf),
                         dtype=np.float32) / 255.0
        outs.append(ink)
    out = torch.from_numpy(np.stack(outs, 0))
    return out[0] if single else out


def _cmyk_to_lab(cmyk_u8, icc, intent=0):
    """cmyk_u8: (N,4) uint8 -> Lab (N,3) float64。

    走 PIL ImageCms：CMYK -> sRGB -> (自写) Lab。
    单张 1xN 图像批量过 transform，避免逐像素调用。
    """
    n = cmyk_u8.shape[0]
    img = Image.merge("CMYK", [Image.fromarray(
        cmyk_u8[:, i].reshape(1, n), "L") for i in range(4)])
    rgb = np.asarray(ImageCms.applyTransform(img, _xf(icc, intent)).convert("RGB"),
                     dtype=np.float64).reshape(n, 3) / 255.0
    return _srgb_to_lab_np(rgb)


_XF_CACHE = {}


def _xf(icc, intent, input_mode="CMYK", output_mode="RGB"):
    """双向 transform 缓存。

    `ImageCms.buildTransform(inProfile, outProfile, inMode, outMode)` ——
    inProfile/outProfile 决定方向：默认 CMYK->RGB 用 (icc, sRGB)；
    RGB->CMYK 分离路径需要把两者**对调**成 (sRGB, icc)。
    """
    key = (icc, intent, input_mode, output_mode)
    if key not in _XF_CACHE:
        srgb = ImageCms.createProfile("sRGB")
        prof = ImageCms.getOpenProfile(icc)
        src, dst = (srgb, prof) if input_mode == "RGB" else (prof, srgb)
        _XF_CACHE[key] = ImageCms.buildTransform(
            src, dst, input_mode, output_mode, renderingIntent=intent)
    return _XF_CACHE[key]


def _srgb_to_lab_np(rgb01):
    """sRGB([0,1]) -> Lab(D65)。白点校验：白 vs 灰50 ΔE=33.415。"""
    m = rgb01 <= 0.04045
    lin = np.where(m, rgb01 / 12.92, ((rgb01 + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


# ---------------------------------------------------------------- LUT 构建

IDX16 = torch.tensor(list(itertools.product([0, 1], repeat=4)), dtype=torch.float32)


def build_lut16(icc, intent=0):
    """16 个 on/off 组合的 Lab -> (16,3) float32，索引 n = lC*8+lM*4+lY*2+lK。"""
    combos = np.array(list(itertools.product([0, 255], repeat=4)), dtype=np.uint8)
    return torch.tensor(_cmyk_to_lab(combos, icc, intent), dtype=torch.float32)


def build_lut256(icc, intent=0):
    """256 个 4 档组合的 Lab -> (256,3) float32，索引 n = lC*64+lM*16+lY*4+lK。

    档位值 LEVELS=[0,1/3,2/3,1] 映射到墨量 [0,85,170,255]。
    """
    combos = np.array(list(itertools.product(range(4), repeat=4)), dtype=np.uint8)
    ink = np.round(LEVELS.numpy()[combos] * 255).astype(np.uint8)
    return torch.tensor(_cmyk_to_lab(ink, icc, intent), dtype=torch.float32)


def lut_path(out_dir):
    return os.path.join(out_dir, "lut256.npy")


def load_or_build_lut256(out_dir, icc, intent=0, force=False):
    """磁盘缓存 runs/<exp>/lut256.npy；元数据不匹配则重建。

    返回 lut256 (256,3) float32。
    """
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "lut256_meta.json")
    meta = {"icc": os.path.abspath(icc), "intent": intent,
            "levels": LEVELS.tolist(),
            "index": "n = lC*64 + lM*16 + lY*4 + lK"}
    if os.path.exists(lut_path(out_dir)) and not force:
        import json
        try:
            with open(meta_path) as f:
                old = json.load(f)
        except Exception:
            old = None
        if old == meta:
            return torch.tensor(np.load(lut_path(out_dir)), dtype=torch.float32)
    lut = build_lut256(icc, intent)
    np.save(lut_path(out_dir), lut.numpy())
    with open(meta_path, "w") as f:
        import json
        json.dump(meta, f, indent=2)
    return lut


# ---------------------------------------------------------------- Neugebauer


def areas16(cov4, idx16=IDX16):
    """Demichel 16 项乘积面积。

    cov4: (B,4,H,W) 连续覆盖率（0..1）
    返回 (B,16,H,W)；每像素 16 项和 = 1（数值误差 ~3e-7）。
    """
    b = cov4.shape[0]
    a = torch.ones(b, 16, *cov4.shape[2:], device=cov4.device, dtype=cov4.dtype)
    idx = idx16.to(cov4.device)
    for ch in range(4):
        sel = cov4[:, ch][:, None]                      # (B,1,H,W)
        on = idx[:, ch][None, :, None, None]            # (1,16,1,1)
        a = a * (sel * on + (1 - sel) * (1 - on))
    return a


def neugebauer_lab16(cov4, lut16, idx16=IDX16):
    """16 项精确叠印 -> Lab (B,3,H,W)。"""
    a = areas16(cov4, idx16)
    return torch.einsum("bnhw,nc->bhwc", a, lut16.to(a.device)).permute(0, 3, 1, 2)


def neugebauer_lab_2bit(p4, lut256):
    """4 档概率张量链收缩 -> Lab (B,3,H,W)。

    p4:   (B,4,H,W,4) 每像素每通道 4 档概率（softmax 输出；硬域时是 one-hot）
    lut256: (256,3)，索引 n = lC*64 + lM*16 + lY*4 + lK

    实现要点（PITFALLS：einsum 标签会静默算错，已实测复现）：
      - einsum 中「出现在 operand 但不出现在输出」的标签会被**静默求和**：
        实测 `bchw,bhwle->bhwe` 把 lC 维当求和维吞掉（结果 1572 = 4 项之和，不报错）；
      - 因此每条 einsum 的收缩标签必须**显式**作为 operand1 的标签出现，
        且输出保留其余全部自由维；
      - 内建 one-hot 单点断言 assert_lut256_consistency() 防回归。
    """
    T = lut256.to(p4.device).reshape(4, 4, 4, 4, 3)      # [lC,lM,lY,lK,e]
    # K 收缩：p4[:,3] (b,hw,k) 对 T 的 lK 加权 -> (b,hw,lC,lM,lY,e)
    g = torch.einsum("bhwk,lmyke->bhwlmye", p4[:, 3], T)
    # Y 收缩：p4[:,2] (b,hw,y) 对 g 的 lY 加权 -> (b,hw,lC,lM,e)
    g = torch.einsum("bhwy,bhwlmye->bhwlme", p4[:, 2], g)
    # M 收缩：p4[:,1] (b,hw,m) 对 g 的 lM 加权 -> (b,hw,lC,e)
    g = torch.einsum("bhwm,bhwlme->bhwle", p4[:, 1], g)
    # C 收缩：p4[:,0] (b,hw,c) 的 c 与 g 的 lC **同名匹配** -> (b,hw,e)
    #     两个 operand 的收缩标签必须同名（都写 c 或都写 l），否则 einsum
    #     把 lC 当求和维静默吞掉（实测白点算出 321 = 4 项之和，不报错）。
    return torch.einsum("bhwc,bhwce->bhwe", p4[:, 0], g).permute(0, 3, 1, 2)


def assert_lut256_consistency(lut256, tol=1e-5):
    """one-hot 单点自测：p one-hot -> 输出必须 == lut[n]。

    同时校验 Demichel 16 项与 256 项在 on/off 子网格上一致。
    """
    lut = lut256 if lut256.dim() == 2 else lut256.reshape(256, 3)
    # 单张 (1,4,1,1,4) 逐组合设 one-hot，逐组合核对（4 个通道各取各自档位）
    p = torch.zeros(1, 4, 1, 1, 4)
    for lc in range(4):
        for lm in range(4):
            for ly in range(4):
                for lk in range(4):
                    p.zero_()
                    p[0, 0, 0, 0, lc] = 1.0
                    p[0, 1, 0, 0, lm] = 1.0
                    p[0, 2, 0, 0, ly] = 1.0
                    p[0, 3, 0, 0, lk] = 1.0
                    n = lc * 64 + lm * 16 + ly * 4 + lk
                    got = neugebauer_lab_2bit(p, lut)[0, :, 0, 0]
                    want = lut[n]
                    assert torch.allclose(got, want.to(got.device), atol=tol), \
                        f"LUT256 索引不一致: combo({lc},{lm},{ly},{lk}) n={n} " \
                        f"got {got.tolist()} want {want.tolist()}"
    # 16 项一致性：on/off 子网格（档位 0 与 3）上 16 项与 256 项应相同
    cov = torch.ones(2, 4, 2, 2)
    lab16 = neugebauer_lab16(cov, build_lut16_from(lut))
    p2 = torch.zeros(2, 4, 2, 2, 4)
    p2[..., 3] = 1.0           # 全通道 level3 = 全墨
    lab256 = neugebauer_lab_2bit(p2, lut)
    assert torch.allclose(lab16, lab256, atol=1e-4), \
        f"16 项与 256 项在 on/off 子网格不一致: max diff " \
        f"{(lab16 - lab256).abs().max().item()}"
    return True


def build_lut16_from(lut256):
    """从 256 项 LUT 抽 on/off 子网格 -> (16,3)，索引 n = lC*8+lM*4+lY*2+lK。"""
    t = lut256.reshape(4, 4, 4, 4, 3)
    sub = t[0, 0, 0, 0], t[0, 0, 0, 3], t[0, 0, 3, 0], t[0, 0, 3, 3], \
        t[0, 3, 0, 0], t[0, 3, 0, 3], t[0, 3, 3, 0], t[0, 3, 3, 3], \
        t[3, 0, 0, 0], t[3, 0, 0, 3], t[3, 0, 3, 0], t[3, 0, 3, 3], \
        t[3, 3, 0, 0], t[3, 3, 0, 3], t[3, 3, 3, 0], t[3, 3, 3, 3]
    return torch.stack([s[0] for s in sub], 0) if False else torch.stack(
        [s.reshape(3) for s in sub], 0)
