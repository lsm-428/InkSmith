"""训练：单次训练 + 自动多轮编排（两种模式，同一份实现）。

模式（默认自动编排，这是推荐入口）：
  uv run python train.py --printer film --sigma-um 25 --dpi 1440 --out runs/film

  自动跑"退火多轮 -> 硬 STE 微调"，每轮后验证并按规则调参：
    1. 孤立点过多 -> w_iso 加倍（封顶 0.2）
    2. psnr 仍在涨 -> 直接续训一轮
    3. psnr 停滞   -> lr 减半
  温度逐轮几何退火 1.0 -> 0.3 -> 0.09 -> 0.03，w_bin 逐轮加强。

模式（--cmyk，2-bit CMYK 多档墨量路线）：
  uv run python train.py --cmyk --rgb --out runs/cmyk
  输出 4 通道 × 4 档互斥概率 -> LUT256 -> Lab -> ΔE 损失；
  详见 halftone/cmyk.py。该模式不加可微打印模拟器（网点扩大未标定）。

模式（--rounds 1，手工单段）：
  uv run python train.py --rounds 1 --epochs 10 --t-start 1.0 --t-end 0.3 ...
  uv run python train.py --rounds 1 --epochs 20 --hard-ste ...   # 只做硬 STE 微调

冒烟（改代码后先跑，~10 秒）：
  uv run python train.py --smoke

两段式顺序不能颠倒（见 PITFALLS #1/#4）：--hard-ste 只能用于退火后的微调，
从零硬 STE 必死（flatness degradation）。自动模式已内置这个顺序。

产物（out/ 下）：last.pt / best.pt / args.json / train.log（逐 epoch）
                pipeline_log.jsonl（逐轮）、epoch_*.png（逐 epoch 对比图）
"""
import argparse
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from halftone.cmyk import (LEVELS, assert_lut256_consistency, find_icc,
                           lab_to_srgb, load_or_build_lut256,
                           neugebauer_lab_2bit, srgb_to_cov, srgb_to_lab)
from halftone.data import RgbCrops
from halftone.losses import HVSLoss, isolated_dot_penalty, striping_penalty
from halftone.model import (UNet, hard_binarize, hard_levels, soft_binarize,
                            soft_levels)
from halftone.printer import build_printer, gaussian_blur


# ---------------------------------------------------------------- 验证


def evaluate(ckpt_path, device, n_val=30, size=256, seed=1234):
    """验证集硬阈值指标：psnr（模拟打印 vs 目标）、孤立点率、明暗偏差。

    这是唯一可信的指标（PITFALLS #4、#14）。注意 train.log 里退火轮记的
    psnr 是**软前向**口径（T=0.03 时 ink 仍是 0.05~0.95 的软值），
    与这里的硬阈值口径能差 20dB，两者不可直接比较。
    seed 固定验证集的随机裁剪/翻转：否则同 ckpt 两次评测能差 3dB 以上，
    pipeline 里"psnr 是否上涨"的逐轮策略会被噪声牵着走（PITFALLS #12）。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = UNet(base=a["base"], up=a.get("up", "nearest")).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    printer = build_printer(a).to(device)
    ds = RgbCrops("data/val", size=size, length=n_val)
    k = torch.ones(1, 1, 3, 3, device=device)
    psnrs, isos, biases = [], [], []
    # 固定裁剪/翻转随机源；只影响验证，不碰训练用的全局 random 状态
    rng = random.Random(seed)
    state = random.getstate()
    random.setstate((state[0], tuple(rng.getstate()[1]), state[2]))
    with torch.no_grad():
        for i in range(len(ds)):
            t = ds[i][None].to(device)
            if a.get("in_ch", 1) == 1:
                t = t[:, :1]              # 1-bit 老路只吃灰度通道
            ink = (model(t) > 0).float()
            printed = printer(ink)
            mse = F.mse_loss(printed, t).item()
            psnrs.append(10 * np.log10(1.0 / (mse + 1e-12)))
            neighbors = F.conv2d(ink, k, padding=1) - ink
            isos.append((ink * (neighbors < 0.5)).sum().item()
                        / (ink.sum().item() + 1e-9))
            biases.append((printed.mean() - t.mean()).item())
    random.setstate(state)
    return {"psnr": float(np.mean(psnrs)), "iso_ratio": float(np.mean(isos)),
            "bias": float(np.mean(biases))}


def evaluate_cmyk(ckpt_path, device, n_val=30, size=256, seed=1234):
    """CMYK 多档墨量路线的验证：硬阈值 Lab ΔE（唯一可信指标，PITFALLS #4/#14）。

    硬域 = 每通道取 argmax 档位（one-hot），过 LUT256 -> Lab 与目标
    sRGB->Lab 比较。指标：
      - de    : 全图平均 ΔE2000
      - de_p95: 95 分位 ΔE（最差 5% 的像素）
      - cov   : 四通道平均覆盖率（防全 0/全 1 塌缩）
      - tac   : 单像素四通道墨量和（TAC）均值
    seed 固定随机裁剪/翻转：否则同 ckpt 两次评测能差 1 ΔE 以上（PITFALLS #12）。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = UNet(base=a["base"], up=a.get("up", "nearest"),
                 in_ch=a.get("in_ch", 1), out_ch=a.get("out_ch", 1),
                 levels=a.get("levels", 1)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    icc = find_icc(a.get("icc"))
    lut256 = load_or_build_lut256(Path(ckpt_path).parent, icc,
                                  a.get("icc_intent", 0))
    ds = RgbCrops("data/val", size=size, length=n_val)
    k = torch.ones(1, 1, 3, 3, device=device)
    des, des95, covs, tacs, isos = [], [], [], [], []
    rng = random.Random(seed)
    state = random.getstate()
    random.setstate((state[0], tuple(rng.getstate()[1]), state[2]))
    with torch.no_grad():
        for i in range(len(ds)):
            t = ds[i][None].to(device)
            logits = model(t)
            hard = hard_levels(logits / 0.01)
            cov = (hard * LEVELS.view(1, 1, 1, 4).to(device)).sum(-1)          # (B,4,H,W)
            lab_nn = neugebauer_lab_2bit(hard, lut256)         # (B,3,H,W)
            lab_tg = srgb_to_lab(t.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            de = delta_e_2000(lab_nn, lab_tg)
            des.append(de.mean().item())
            des95.append(torch.quantile(de.flatten(), 0.95).item())
            covs.append(cov.mean().item())
            tacs.append(cov.sum(1).mean().item())
            # 孤立点率：联合掩码（任一通道有墨）判定，与训练惩罚同口径
            # （PITFALLS #21：逐通道判定下四通道叠加会互相"洗白"，惩罚失效）
            ink = (cov.sum(1, keepdim=True) > 0.05).float()
            nb = F.conv2d(ink, k, padding=1) - ink
            isos.append((ink * (nb < 0.5)).sum().item()
                        / (ink.sum().item() + 1e-9))
    random.setstate(state)
    return {"de": float(np.mean(des)), "de_p95": float(np.mean(des95)),
            "cov": float(np.mean(covs)), "tac": float(np.mean(tacs)),
            "iso_ratio": float(np.mean(isos))}


def delta_e_2000(lab1, lab2):
    """CIEDE2000 色差。

    lab1/lab2 接受 (…,3) 或 (B,3,…) 两种排布：按最后一维是不是 3 自动判定。
    """
    if lab1.shape[-1] != 3:
        lab1 = lab1.movedim(1, -1)
    if lab2.shape[-1] != 3:
        lab2 = lab2.movedim(1, -1)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    kL = kC = kH = 1.0
    # 0/0 型反传（实测：单像素中性目标让全图 a/b 梯度 nan）：
    #   - sqrt(0) 反向 = 1/(2*0) = inf，乘上前向 0 得 nan；
    #   - Cbarp**7 在 Cbarp=0 处导数 0，与分母 25**7 组成 0/0。
    # 统一在所有 sqrt / **7 处加 eps 保底：值不变（eps=1.2e-7 << 典型量级），
    # 梯度有限。**这是 CIEDE2000 可微化的必要条件，不要回退。**
    eps = torch.finfo(lab1.dtype).eps
    C1 = torch.sqrt(a1 ** 2 + b1 ** 2 + eps)
    C2 = torch.sqrt(a2 ** 2 + b2 ** 2 + eps)
    Cbar = 0.5 * (C1 + C2)
    G = 0.5 * (1 - torch.sqrt(Cbar ** 7 / (Cbar ** 7 + 25 ** 7 + eps)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = torch.sqrt(a1p ** 2 + b1 ** 2 + eps)
    C2p = torch.sqrt(a2p ** 2 + b2 ** 2 + eps)
    # atan2(0,0) 反传 nan（PyTorch 在该点无定义）。两侧都中性色时 hue 无意义，
    # 用 torch.where 选出后用 eps 化归避开：atan2(eps, 1) ≈ 0 且可微。
    h1p = torch.atan2(b1, a1p + eps) % (2 * math.pi)
    h2p = torch.atan2(b2, a2p + eps) % (2 * math.pi)
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = torch.where(dhp > math.pi, dhp - 2 * math.pi,
                      torch.where(dhp < -math.pi, dhp + 2 * math.pi, dhp))
    # C1p*C2p=0（某一侧中性色）时 sqrt 反向 = inf -> nan。clamp 到 eps：
    # 该项在任一侧中性时本就 ~0，钳制不改变有效值。
    dHp = 2 * torch.sqrt(torch.clamp(C1p * C2p, min=eps)) * torch.sin(dhp / 2)
    Lbarp = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)
    hsum = h1p + h2p
    # hbarp：C1p*C2p=0 时取 hsum；否则取两者均值（跨 180° 时 ±2π 修正）。
    # 注意 torch.where 的**两个分支都会求值**：旧写法把 python if 嵌在
    # where 内部，`.all()` 判据在 batch>1 时把整图当一个判据，数值虽对但
    # 反向会把已废弃分支的 inf/nan 传播回来（实测 a/b 梯度 nan）。
    hbarp = torch.where(C1p * C2p == 0, hsum, hsum / 2)
    hbarp = torch.where((C1p * C2p != 0) & (torch.abs(h1p - h2p) > math.pi),
                        hbarp + math.pi, hbarp) % (2 * math.pi)
    T = (1 - 0.17 * torch.cos(hbarp - math.pi / 6)
         + 0.24 * torch.cos(2 * hbarp) + 0.32 * torch.cos(3 * hbarp + math.pi / 30)
         - 0.20 * torch.cos(4 * hbarp - 63 * math.pi / 180))
    dtheta = 30 * math.pi / 180 * torch.exp(-((hbarp * 180 / math.pi - 275) / 25) ** 2)
    # Cbarp=0 时 Cbarp**7 的导数是 0，与分母里的 25**7 组合成 0/0 -> nan。
    RC = 2 * torch.sqrt(Cbarp ** 7 / (Cbarp ** 7 + 25 ** 7 + eps))
    SL = 1 + 0.015 * (Lbarp - 50) ** 2 / torch.sqrt(20 + (Lbarp - 50) ** 2)
    SC = 1 + 0.045 * Cbarp
    SH = 1 + 0.015 * Cbarp * T
    RT = -torch.sin(2 * dtheta) * RC
    # 完全相等时 de=0，sqrt 反传 1/(2*0)=inf -> nan。加 eps 保底。
    return torch.sqrt((dLp / (kL * SL)) ** 2 + (dCp / (kC * SC)) ** 2
                      + (dHp / (kH * SH)) ** 2
                      + RT * (dCp / (kC * SC)) * (dHp / (kH * SH)) + eps)


# ---------------------------------------------------------------- 单段训练

def train_once(args, device, out, log, *, epochs, lr, w_iso, w_bin,
               t_start, t_end, hard_ste, resume=True, cmyk=None,
               seed_noise=0.0):
    """跑一段训练并返回（模型留在显存上供下一段续训）。

    cmyk: None = 1-bit 灰度老路；dict = CMYK 多档墨量路线配置
          （icc / icc_intent / w_lab / w_cov / w_tac / tac_max / levels），
          key 存在即走 CMYK。
    seed_noise: Nyquist 种子噪声幅度（0 = 关）；见下方用法。
          此时目标改为 Lab ΔE，不加可微打印模拟器。
    """
    cmyk = cmyk or {}
    model = UNet(base=args.base, up=args.up,
                 in_ch=3 if cmyk else 1,
                 out_ch=4 if cmyk else 1,
                 levels=cmyk.get("levels", 4) if cmyk else 1).to(device)
    if resume and (out / "last.pt").exists():
        model.load_state_dict(torch.load(out / "last.pt", map_location=device,
                                         weights_only=False)["model"])
        log(f"[resume] 从 {out/'last.pt'} 继续")
    if cmyk:
        icc = find_icc(cmyk.get("icc"))
        lut256 = load_or_build_lut256(out, icc, cmyk.get("icc_intent", 0))
        assert_lut256_consistency(lut256)
        log(f"[cmyk] ICC={icc} intent={cmyk.get('icc_intent', 0)} "
            f"levels={cmyk.get('levels', 4)} LUT256={lut256.shape}")
    else:
        printer = build_printer(vars(args)).to(device)
        # film + dpi 范围：每 batch 随机 dpi 算 σ[px]，一个模型覆盖连续打印分辨率
        rand_dpi = (args.printer == "film" and args.dpi_min and args.dpi_max
                    and args.dpi_max > args.dpi_min)
        if args.printer == "film":
            if rand_dpi:
                log(f"[printer] film 连续dpi：每batch对数随机 dpi∈"
                    f"[{args.dpi_min},{args.dpi_max}]，σ={args.sigma_um}µm "
                    f"= {args.sigma_um*args.dpi_min/25400:.2f}~"
                    f"{args.sigma_um*args.dpi_max/25400:.2f}px")
            else:
                log(f"[printer] film σ={printer.sigma_px:.3f}px "
                    f"(σ={args.sigma_um}µm @ {args.dpi}dpi)，线性体积模型，无饱和")
        hvs = HVSLoss(sigma=2.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ds = RgbCrops(args.data, size=args.size, length=args.length)
    if ds.synthetic:
        log(f"[data] {args.data} 为空，使用合成灰度图（仅冒烟验证流程）")
    else:
        log(f"[data] {len(ds.paths)} 张图片，随机裁 {args.size}x{args.size}")
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=2, drop_last=True)

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "l1": 0.0, "hvs": 0.0, "cov": 0.0, "iso": 0.0,
                  "stripe": 0.0, "tac": 0.0, "psnr": 0.0, "de": 0.0}
        n = 0
        # 温度线性退火：早期软（学覆盖率），后期硬（结晶出抖动纹理）
        temp = t_start + (t_end - t_start) * (epoch - 1) / max(1, epochs - 1)
        mode = "hard" if hard_ste else f"T={temp:.3f}"
        for target in tqdm(loader, desc=f"epoch {epoch}/{epochs} {mode}"):
            target = target.to(device)
            if not cmyk and target.shape[1] == 3:
                target = target[:, :1]   # 1-bit 老路只吃灰度通道
            # Nyquist 种子噪声：给输入叠加高频（棋盘）扰动，逼模型把抖动纹理
            # 学出来（PITFALLS #13 的 1-bit 待试项；#27 实测 CMYK 下邻域档位
            # 多样性仅 1.1，模型在做逐像素独立决策，没利用空间自由度）。
            # 只加在输入上，不加在目标上——网络必须自己决定如何响应。
            if seed_noise > 0:
                nyq = torch.stack(
                    torch.meshgrid(
                        torch.arange(target.shape[-2], device=device),
                        torch.arange(target.shape[-1], device=device),
                        indexing="ij"), -1).sum(-1) % 2
                nyq = (nyq.float() * 2 - 1)[None, None]      # (1,1,H,W) ±1 棋盘
                amp = seed_noise * (0.5 + 0.5 * torch.rand(
                    target.shape[0], 1, 1, 1, device=device))
                target = (target + nyq * amp).clamp(0, 1)
            logits = model(target)
            if cmyk:
                # 多档：软域 softmax 覆盖率 -> LUT256 -> Lab -> ΔE 目标
                # cmyk_ste_train=True 时把硬 argmax（STE 反传）放进训练前向，
                # 让损失直接作用在硬域——这是唯一没被证伪的方案（PITFALLS
                # #24：w_cov/降T/gap penalty 的梯度在 gap≈0 区域全部趋近 0）。
                if cmyk.get("ed_fs"):
                    # FS 式逐行串行误差扩散（PITFALLS #32）：七个方案里第一个
                    # 同时做到覆盖率正确 + 有 dithering 的。代价是行内 W 步
                    # python 循环，256² 下训练慢，需配小 size 或少 epoch。
                    from halftone.ed_fs import ed_fs
                    p = ed_fs(logits, srgb_to_cov(
                        target.permute(0, 2, 3, 1), icc,
                        cmyk.get("icc_intent", 0)).permute(0, 3, 1, 2),
                        LEVELS.to(device))
                else:
                    p = (hard_levels(logits / 0.01)
                         if (hard_ste or cmyk.get("ste_train"))
                         else soft_levels(logits, temp))
                cov = (p * LEVELS.view(1, 1, 1, 4).to(device)).sum(-1)   # (B,4,H,W)
                lab_nn = neugebauer_lab_2bit(p, lut256)         # (B,3,H,W)
                lab_tg = srgb_to_lab(target.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                l_lab = delta_e_2000(lab_nn, lab_tg).mean()
                # l_win：邻域窗口平均色损失（#28 之后唯一没试过的维度）。
                # 逐像素 ΔE 的天花板是 2-bit 量化误差 3-5 ΔE00（#26），而工业
                # RIP 靠误差扩散把误差摊到空间频率上、被视觉平均掉。这里显式
                # 建模：对 win×win 窗口，用窗口内**档位概率的平均**过
                # Neugebauer 得到"混合色"，与窗口平均目标比 ΔE。
                # 数学上等于允许邻域像素选不同档位、只要求组合平均对——
                # 这正是 dithering 的定义。
                # 只在软域有意义（硬 argmax 下窗口平均 = 逐像素，无增益）。
                win = int(cmyk.get("win", 0))
                if win >= 3 and not (hard_ste or cmyk.get("ste_train")):
                    Bc, Cc, Hc, Wc, Lc = p.shape
                    # p (B,4,H,W,L) -> (B,L,4,H,W) 展平成 (B*L*4,1,H,W)，
                    # 逐档位做窗口平均，再还原回 (B,4,H,W,L)
                    pf = p.permute(0, 4, 1, 2, 3).reshape(-1, 1, Hc, Wc)
                    pf = F.avg_pool2d(pf, win, stride=1, padding=0)
                    p_win = pf.view(Bc, Lc, Cc, Hc - win + 1, Wc - win + 1
                                    ).permute(0, 2, 3, 4, 1)
                    lab_win = neugebauer_lab_2bit(p_win, lut256)
                    lab_tg_win = F.avg_pool2d(lab_tg, win, stride=1,
                                              padding=0)
                    l_lab = l_lab + cmyk.get("w_win", 1.0) * delta_e_2000(
                        lab_win, lab_tg_win).mean()
                # l_cov：把每通道覆盖率拉向 ICC 分离的软目标（w_cov=0 会跨通道作弊）
                cov_tg = srgb_to_cov(target.permute(0, 2, 3, 1), icc,
                                     cmyk.get("icc_intent", 0))
                cov_tg = cov_tg.permute(0, 3, 1, 2).to(device)  # (B,4,H,W)
                l_cov = ((cov - cov_tg) ** 2).mean()
                # TAC：四通道墨量和超 tac_max 的部分按平方惩罚
                l_tac = F.relu(cov.sum(1) - cmyk.get("tac_max", 3.2)).pow(2).mean()
                l_iso = isolated_dot_penalty(cov)
                # l_bin：软/硬域桥。p*(1-p) 只在 p 居中(0.5)时最大，压不住
                # "4 档 logit 全相等 -> p=[0.25,0.25,0.25,0.25] -> 期望墨量
                # 由跨档凑数凑出来"这个退化解（PITFALLS #23：实测 top1-top2
                # gap 中位 0.006，软 ΔE 2.3 vs 硬 6.1）。改用 logit 间距项：
                # 直接惩罚 gap<gap_min 的像素，把 logit 推开。
                s = torch.sort(logits, dim=-1).values
                gap = s[..., -1] - s[..., -2]          # (B,4,H,W) top1-top2
                l_gap = F.relu(cmyk.get("gap_min", 0.5) - gap).mean()
                l_bin = l_gap
                loss = (cmyk["w_lab"] * l_lab + cmyk["w_cov"] * l_cov
                        + cmyk["w_tac"] * l_tac + w_iso * l_iso
                        + w_bin * l_bin)
                b = target.shape[0]
                totals["loss"] += loss.item() * b
                totals["de"] += l_lab.item() * b
                totals["cov"] += l_cov.item() * b
                totals["iso"] += l_iso.item() * b
                totals["tac"] += l_tac.item() * b
                n += b
                opt.zero_grad(); loss.backward(); opt.step()
                continue
            ink = (hard_binarize(logits) if hard_ste
                   else soft_binarize(logits, temp))
            if rand_dpi:
                # 对数均匀采样：dpi 跨数量级时每个倍频程等权（σ 与 dpi 线性）
                dpi = math.exp(torch.empty(()).uniform_(
                    math.log(args.dpi_min), math.log(args.dpi_max)).item())
                printed = printer(ink, sigma_px=args.sigma_um * dpi / 25400.0)
                # 保真项统一在物理尺度比较（100µm = 2×墨滴直径）：低于该尺度的
                # 抖动噪声物理上不可避免；像素尺度 l1 在低 dpi 端会攻击抖动
                # 纹理本身，跨 dpi 梯度互相打架（PITFALLS #11）
                s_view = 100.0 * dpi / 25400.0
                pb, tb = gaussian_blur(printed, s_view), gaussian_blur(target, s_view)
                l1 = F.l1_loss(pb, tb)
                l_hvs = F.mse_loss(pb, tb)
            else:
                printed = printer(ink)
                pb, tb = printed, target
                l1 = F.l1_loss(printed, target)
                l_hvs = hvs(printed, target)
            l_iso = isolated_dot_penalty(ink)
            l_ink = ink.mean()
            l_bin = (ink * (1 - ink)).mean()
            l_stripe = striping_penalty(ink)
            loss = (l1 + args.w_hvs * l_hvs + w_iso * l_iso
                    + args.w_ink * l_ink + w_bin * l_bin
                    + args.w_stripe * l_stripe)

            opt.zero_grad()
            loss.backward()
            opt.step()

            b = target.shape[0]
            totals["loss"] += loss.item() * b
            totals["l1"] += l1.item() * b
            totals["hvs"] += l_hvs.item() * b
            totals["iso"] += l_iso.item() * b
            totals["stripe"] += l_stripe.item() * b
            totals["psnr"] += 10 * np.log10(
                1.0 / (F.mse_loss(pb.detach(), tb).item() + 1e-12)) * b
            n += b

        log(f"[epoch {epoch}] {mode} " + " ".join(f"{k}={v/n:.4f}"
                                                   for k, v in totals.items()))
        model.eval()
        with torch.no_grad():
            sample = next(iter(loader)).to(device)[:8]
            if cmyk:
                # 预览：目标 sRGB | 4 档覆盖率灰度化 | ICC 合成的近似 sRGB
                hard = hard_levels(model(sample) / 0.01)
                cov = (hard * LEVELS.view(1, 1, 1, 4).to(device)).sum(-1)        # (B,4,H,W)
                lab_nn = neugebauer_lab_2bit(hard, lut256)
                rgb_nn = lab_to_srgb(lab_nn.permute(0, 2, 3, 1))
                prev = torch.cat([sample, cov[:, :1].repeat(1, 3, 1, 1),
                                  cov[:, 3:4].repeat(1, 3, 1, 1),
                                  rgb_nn.permute(0, 3, 1, 2).clamp(0, 1)])
                save_image(prev, out / f"epoch_{epoch:03d}.png", nrow=8)
            else:
                s8 = sample[:, :1]
                ink = (model(s8) > 0).float()
                grid = torch.cat([s8, ink, printer(ink)])
                save_image(grid, out / f"epoch_{epoch:03d}.png", nrow=8)
        torch.save({"model": model.state_dict(),
                    "args": {**vars(args), "in_ch": 3 if cmyk else 1,
                             "out_ch": 4 if cmyk else 1,
                             "levels": cmyk.get("levels", 4) if cmyk else 1,
                             **({"icc": cmyk.get("icc"),
                                 "icc_intent": cmyk.get("icc_intent", 0)} if cmyk else {})}},
                   out / "last.pt")
    return model


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rounds", type=int, default=4,
                   help="自动编排轮数；1=只跑一次 --epochs（手工单段）")
    p.add_argument("--epochs", type=int, default=20, help="rounds=1 时的 epoch 数")
    p.add_argument("--epochs-per-round", type=int, default=10)
    p.add_argument("--finetune-rounds", type=int, default=1, help="退火后硬 STE 微调轮数")
    p.add_argument("--finetune-epochs", type=int, default=20)
    p.add_argument("--target-psnr", type=float, default=18.0)
    p.add_argument("--target-de", type=float, default=3.0,
                   help="--cmyk：达标 ΔE2000 阈值（全图平均）")
    p.add_argument("--gap-min", type=float, default=0.5,
                   help="--cmyk：档位 logit 间距下限（PITFALLS #23）。"
                        "l_bin 改为惩罚 gap<该值 的像素，把 4 档 logit 推开；"
                        "0 = 关掉（回到 p*(1-p) 项）")
    p.add_argument("--ed-fs", action="store_true",
                   help="--cmyk：训练前向用 FS 式逐行串行误差扩散（PITFALLS #32，"
                        "第一个同时做到覆盖率正确+dithering 的方案）。代价：行内 "
                        "W 步 python 循环，256² 下慢，建议配 --size 128")
    p.add_argument("--win", type=int, default=0,
                   help="--cmyk：邻域窗口平均色损失的窗口边长（0=关，3 或 4）。"
                        "窗口内档位概率平均过 Neugebauer 与窗口平均目标比 ΔE，"
                        "逼出 dithering（PITFALLS #28：输入种子噪声无效，"
                        "必须在损失里显式建模）")
    p.add_argument("--w-win", type=float, default=1.0,
                   help="--cmyk：邻域窗口损失权重（--win 开启时生效）")
    p.add_argument("--cmyk-ste-train", action="store_true",
                   help="--cmyk：训练前向就用硬 argmax（STE 反传），损失直接作用"
                        "硬域（PITFALLS #24 唯一未证伪方案）。⚠ 1-bit 下同类设置"
                        "必死（#1 flatness degradation），多档未验证")
    p.add_argument("--seed-noise", type=float, default=0.0,
                   help="Nyquist 种子噪声幅度（0-1，默认 0=关）。给训练输入叠加"
                        "±棋盘高频扰动，逼模型学出抖动纹理而非逐像素独立决策"
                        "（PITFALLS #13 待试项 / #27 实测邻域多样性仅 1.1）。"
                        "建议先试 0.3")
    p.add_argument("--iso-threshold", type=float, default=0.02,
                   help="孤立点率阈值；CMYK 走联合掩码口径（PITFALLS #21），"
                        "比逐通道严 2-5 倍，同阈值下更容易触发 w_iso 加倍")

    p.add_argument("--data", default="data/images", help="图片文件夹，空则合成数据冒烟")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--length", type=int, default=2000, help="每 epoch 样本数")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=32, help="U-Net 通道基数")
    p.add_argument("--up", choices=["nearest", "bilinear"], default="nearest",
                   help="U-Net 解码上采样；bilinear 抑制条纹退化解（见 PITFALLS #12）")
    p.add_argument("--printer", choices=["paper", "film"], default="paper",
                   help="paper=吸收基材(exp饱和)；film=非吸收膜基+100%固含(线性体积)")
    p.add_argument("--sigma", type=float, default=1.0, help="paper 模式：墨滴扩散 σ（像素）")
    p.add_argument("--gain", type=float, default=1.5, help="paper 模式：网点扩大系数")
    p.add_argument("--sigma-um", type=float, default=12.5,
                   help="film 模式：铺展高斯 σ（µm）；50µm 实测直径 → 12.5")
    p.add_argument("--dpi", type=int, default=720, help="film 模式：预览/验证用 dpi")
    p.add_argument("--dpi-min", type=int, default=None,
                   help="film 模式：训练时每 batch 随机 dpi 的下限（连续 dpi 支持）")
    p.add_argument("--dpi-max", type=int, default=None,
                   help="film 模式：训练时每 batch 随机 dpi 的上限（配合 --dpi-min）")
    p.add_argument("--w-hvs", type=float, default=2.0)
    p.add_argument("--w-iso", type=float, default=0.05)
    p.add_argument("--cmyk", action="store_true",
                   help="2-bit CMYK 多档墨量路线：Lab ΔE 损失 + 互斥档位 softmax；"
                        "不加可微打印模拟器（网点扩大未标定）")
    p.add_argument("--icc", default=None,
                   help="CMYK ICC 路径；--cmyk 时必填（或用内置候选表）")
    p.add_argument("--icc-intent", type=int, default=0,
                   help="ICC rendering intent（--cmyk）：0=relative colorimetric")
    p.add_argument("--levels", type=int, default=4,
                   help="每通道墨量档数（--cmyk）：4 = 2-bit")
    p.add_argument("--w-lab", type=float, default=1.0, help="--cmyk：Lab ΔE 权重")
    p.add_argument("--w-cov", type=float, default=0.3,
                   help="--cmyk：覆盖率软对齐权重（实测 0 会跨通道作弊，0.3 已够）")
    p.add_argument("--w-tac", type=float, default=1.0, help="--cmyk：TAC 超限惩罚权重")
    p.add_argument("--tac-max", type=float, default=3.2,
                   help="--cmyk：TAC 上限（四通道墨量和；240-320% 常规区间）")
    p.add_argument("--cmyk-finetune", action="store_true",
                   help="--cmyk：强行启用在多档下会毁掉退火成果的硬 STE 微调"
                        "（默认跳过，见 PITFALLS #19）")
    p.add_argument("--w-ink", type=float, default=0.02, help="省墨惩罚，防实心填充锁死")
    p.add_argument("--w-bin", type=float, default=0.05, help="二值化惩罚 soft*(1-soft)")
    p.add_argument("--w-stripe", type=float, default=0.0,
                   help="条纹惩罚；默认 0=关闭（实验证明会误伤真实低频，见 PITFALLS #12）")
    p.add_argument("--t-start", type=float, default=1.0, help="退火起始温度（rounds=1 用）")
    p.add_argument("--t-end", type=float, default=0.05, help="退火结束温度（rounds=1 用）")
    p.add_argument("--hard-ste", action="store_true",
                   help="硬前向+截断STE（仅退火后微调用，勿从零训练）")
    p.add_argument("--out", default="runs/halftone",
                   help="产物目录（不同物理模型用不同目录，勿混）")
    p.add_argument("--smoke", action="store_true", help="小数据快速冒烟")
    p.add_argument("--resume", action="store_true", help="从 out/last.pt 续训")
    args = p.parse_args()

    if args.smoke:
        args.size, args.length, args.batch = 128, 100, 8
        args.epochs, args.epochs_per_round, args.finetune_epochs = 1, 1, 1
        args.rounds, args.finetune_rounds = 1, 1
        args.out = "runs/smoke"
        args.resume = False

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))
    log_file = open(out / "train.log", "a", buffering=1)  # 行缓冲，append 续训不断档

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # CMYK 多档墨量路线配置（train_once / evaluate 共用）
    cmyk_cfg = None
    if args.cmyk:
        cmyk_cfg = {"icc": args.icc, "icc_intent": args.icc_intent,
                    "levels": args.levels, "w_lab": args.w_lab,
                    "w_cov": args.w_cov, "w_tac": args.w_tac,
                    "tac_max": args.tac_max, "gap_min": args.gap_min,
                    "win": args.win, "w_win": args.w_win,
                    "ed_fs": args.ed_fs,
                    "ste_train": args.cmyk_ste_train}
        if args.icc is None:
            args.icc = find_icc(None)
    # 数据自备：目录空则生成（train.py 直接跑时才有意义，编排模式统一在这里保证）
    if not args.smoke:
        gen_args = ["--rgb"] if args.cmyk else []
        if args.cmyk and args.icc:
            gen_args += ["--icc", args.icc]
        if not list(Path("data/images").glob("*.png")):
            log("[data] data/images 为空，生成合成训练图")
            subprocess.run([sys.executable, "gen_data.py", "--n", "500"] + gen_args,
                           check=True)
        if not list(Path("data/val").glob("*.png")):
            log("[data] data/val 为空，生成验证集（种子 123）")
            subprocess.run([sys.executable, "gen_data.py", "--n", "30",
                            "--seed", "123", "--out", "data/val"] + gen_args,
                           check=True)

    # 单段模式：只跑一次 --epochs
    if args.rounds <= 1:
        train_once(args, device, out, log, epochs=args.epochs, lr=args.lr,
                   w_iso=args.w_iso, w_bin=args.w_bin, t_start=args.t_start,
                   t_end=args.t_end, hard_ste=args.hard_ste,
                   resume=args.resume, cmyk=cmyk_cfg,
                   seed_noise=args.seed_noise)
        return

    # 编排模式：退火多轮 -> 硬 STE 微调
    hp = {"w_iso": args.w_iso, "lr": args.lr}
    history, best = [], {"psnr": -1.0, "de": 1e9}
    # 退火终点：1-bit 用 0.03（1-bit 下软 argmax 与硬 argmax 本就一致）；
    # CMYK 多档必须更低——T=0.03 时 softmax 没真正结晶，软域仍比硬域好
    # 3.8 ΔE（PITFALLS #19 的"软域作弊空间"）。0.003 时两者才追上。
    t_floor = 0.003 if args.cmyk else 0.03
    for rnd in range(1, args.rounds + 1):
        # 温度随轮次几何退火：1.0 -> 0.3 -> 0.09 -> t_floor
        t_hi = max(t_floor, 1.0 * 0.3 ** (rnd - 1))
        t_lo = max(t_floor, t_hi * 0.3)
        w_bin = 0.05 * rnd  # 二值化惩罚随退火逐步加强
        # w_iso/w_bin 一律经 train_once 形参传入（PITFALLS #47：曾长期被
        # args.w_iso/args.w_bin 在函数体内覆盖，逐轮加强从未生效）
        args.lr = hp["lr"]
        log(f"\n===== 第 {rnd}/{args.rounds} 轮 | {hp} | T {t_hi:.2f}->{t_lo:.2f} "
            f"| w_bin={w_bin:.2f} =====")
        train_once(args, device, out, log, epochs=args.epochs_per_round,
                   lr=hp["lr"], w_iso=hp["w_iso"], w_bin=w_bin,
                   t_start=t_hi, t_end=t_lo, hard_ste=False, resume=True,
                   cmyk=cmyk_cfg, seed_noise=args.seed_noise)
        m = ((evaluate_cmyk if args.cmyk else evaluate)(out / "last.pt", device))
        record = {"round": rnd, **hp, "t_hi": t_hi, "t_lo": t_lo, **m}
        history.append(record)
        # 逐轮落盘：中途被杀也不丢已完成轮次
        with open(out / "pipeline_log.jsonl", "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if args.cmyk:
            log(f"[验证] ΔE={m['de']:.2f} p95={m['de_p95']:.2f} "
                f"覆盖率={m['cov']:.3f} TAC={m['tac']:.2f} "
                f"孤立点率={m['iso_ratio']:.3%}")
            if m["de"] < best["de"]:
                best = m
                shutil.copy(out / "last.pt", out / "best.pt")
            if m["de"] <= args.target_de and m["iso_ratio"] <= args.iso_threshold:
                log(f"达标：ΔE <= {args.target_de} 且 孤立点率 "
                    f"<= {args.iso_threshold:.0%}")
                break
            prev = history[-2]["de"] if len(history) > 1 else 1e9
            if m["iso_ratio"] > args.iso_threshold:
                hp["w_iso"] = min(hp["w_iso"] * 2, 0.2)
                log(f"[策略] 孤立点过多，w_iso -> {hp['w_iso']}")
            elif prev - m["de"] > 0.1:
                log("[策略] ΔE 仍在下降，直接续训")
            else:
                hp["lr"] /= 2
                log(f"[策略] ΔE 停滞，lr -> {hp['lr']}")
            continue
        log(f"[验证] psnr={m['psnr']:.2f}dB 孤立点率={m['iso_ratio']:.3%} "
            f"明暗偏差={m['bias']:+.3f}")
        if m["psnr"] > best["psnr"]:
            best = m
            shutil.copy(out / "last.pt", out / "best.pt")
        if m["psnr"] >= args.target_psnr and m["iso_ratio"] <= args.iso_threshold:
            log(f"达标：psnr >= {args.target_psnr} 且 孤立点率 "
                f"<= {args.iso_threshold:.0%}")
            break
        # 策略：每轮只改一个变量
        prev = history[-2]["psnr"] if len(history) > 1 else -1.0
        if m["iso_ratio"] > args.iso_threshold:
            hp["w_iso"] = min(hp["w_iso"] * 2, 0.2)  # 封顶，防硬化期失控
            log(f"[策略] 孤立点过多，w_iso -> {hp['w_iso']}")
        elif m["psnr"] - prev > 0.3:
            log("[策略] psnr 仍在上涨，直接续训")
        else:
            hp["lr"] /= 2
            log(f"[策略] psnr 停滞，lr -> {hp['lr']}")

    # 硬 STE 微调：消灭软值作弊，让训练指标=推理指标（退火后才安全）
    # ★ CMYK 多档下这一段会**毁掉退火成果**（实测 ΔE 6.38 -> 42.04）：
    #   根因是 best.pt（退火末态）的硬 argmax 覆盖率 0.336，而硬 STE 训完
    #   涨到 0.562——`w_bin=0.2` 的 `p*(1-p)` 项在硬域恒为 0（one-hot），
    #   但反传的 STE 窗口会把梯度集中到 logit 最大处，等于放开 l_cov 的约束，
    #   覆盖率被 l_lab 单独拉高。且 best.pt 已由 evaluate_cmyk 证明比
    #   finetune 后的 last.pt 好（4 张图 15.58 vs 37.82）。
    #   => CMYK 默认**跳过**硬 STE 微调，保留退火末态；1-bit 老路不变。
    if args.cmyk and not args.cmyk_finetune:
        log("[cmyk] 跳过硬 STE 微调（多档下会毁掉退火成果，见 PITFALLS #19）；"
            "要强行启用加 --cmyk-finetune")
    else:
        for ft in range(1, args.finetune_rounds + 1):
            args.lr = hp["lr"] / 2
            log(f"\n===== 硬 STE 微调 {ft}/{args.finetune_rounds} | {hp} lr={hp['lr']/2} =====")
            train_once(args, device, out, log, epochs=args.finetune_epochs,
                       lr=hp["lr"] / 2, w_iso=hp["w_iso"], w_bin=0.2,
                       t_start=t_floor, t_end=t_floor, hard_ste=True,
                       resume=True, cmyk=cmyk_cfg,
                       seed_noise=args.seed_noise)
            m = ((evaluate_cmyk if args.cmyk else evaluate)(out / "last.pt", device))
            record = {"round": f"finetune-{ft}", **hp, **m}
            with open(out / "pipeline_log.jsonl", "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.cmyk:
                log(f"[验证] ΔE={m['de']:.2f} p95={m['de_p95']:.2f} "
                    f"覆盖率={m['cov']:.3f} TAC={m['tac']:.2f} "
                    f"孤立点率={m['iso_ratio']:.3%}")
                if m["de"] < best["de"]:
                    best = m
                    shutil.copy(out / "last.pt", out / "best.pt")
                continue
            log(f"[验证] psnr={m['psnr']:.2f}dB 孤立点率={m['iso_ratio']:.3%} "
                f"明暗偏差={m['bias']:+.3f}")
            if m["psnr"] > best["psnr"]:
                best = m
                shutil.copy(out / "last.pt", out / "best.pt")

    (out / "pipeline_log.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False))
    key = "de" if args.cmyk else "psnr"
    log(f"\n完成。最佳 {key}={best[key]:.2f}，最佳权重 {out}/best.pt")
    log(f"轮次日志：{out}/pipeline_log.json")


if __name__ == "__main__":
    main()