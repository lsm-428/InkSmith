"""训练：单次训练 + 自动多轮编排（两种模式，同一份实现）。

模式（默认自动编排，这是推荐入口）：
  uv run python train.py --printer film --sigma-um 25 --dpi 1440 --out runs/film

  自动跑"退火多轮 -> 硬 STE 微调"，每轮后验证并按规则调参：
    1. 孤立点过多 -> w_iso 加倍（封顶 0.2）
    2. psnr 仍在涨 -> 直接续训一轮
    3. psnr 停滞   -> lr 减半
  温度逐轮几何退火 1.0 -> 0.3 -> 0.09 -> 0.03，w_bin 逐轮加强。

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

from halftone.data import GrayCrops
from halftone.losses import HVSLoss, isolated_dot_penalty, striping_penalty
from halftone.model import UNet, hard_binarize, soft_binarize
from halftone.printer import build_printer, gaussian_blur


# ---------------------------------------------------------------- 验证

def evaluate(ckpt_path, device, n_val=30, size=256):
    """验证集硬阈值指标：psnr（模拟打印 vs 目标）、孤立点率、明暗偏差。

    这是唯一可信的指标（train.log 里的 psnr 是软前向值，会骗人，PITFALLS #4）。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = UNet(base=a["base"], up=a.get("up", "nearest")).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    printer = build_printer(a).to(device)
    ds = GrayCrops("data/val", size=size, length=n_val)
    k = torch.ones(1, 1, 3, 3, device=device)
    psnrs, isos, biases = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            t = ds[i][None].to(device)
            ink = (model(t) > 0).float()
            printed = printer(ink)
            mse = F.mse_loss(printed, t).item()
            psnrs.append(10 * np.log10(1.0 / (mse + 1e-12)))
            neighbors = F.conv2d(ink, k, padding=1) - ink
            isos.append((ink * (neighbors < 0.5)).sum().item()
                        / (ink.sum().item() + 1e-9))
            biases.append((printed.mean() - t.mean()).item())
    return {"psnr": float(np.mean(psnrs)), "iso_ratio": float(np.mean(isos)),
            "bias": float(np.mean(biases))}


# ---------------------------------------------------------------- 单段训练

def train_once(args, device, out, log, *, epochs, lr, w_iso, w_bin,
               t_start, t_end, hard_ste, resume=True):
    """跑一段训练并返回（模型留在显存上供下一段续训）。"""
    model = UNet(base=args.base, up=args.up).to(device)
    if resume and (out / "last.pt").exists():
        model.load_state_dict(torch.load(out / "last.pt", map_location=device,
                                         weights_only=False)["model"])
        log(f"[resume] 从 {out/'last.pt'} 继续")
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
    ds = GrayCrops(args.data, size=args.size, length=args.length)
    if ds.synthetic:
        log(f"[data] {args.data} 为空，使用合成灰度图（仅冒烟验证流程）")
    else:
        log(f"[data] {len(ds.paths)} 张图片，随机裁 {args.size}x{args.size}")
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=2, drop_last=True)

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "l1": 0.0, "hvs": 0.0, "iso": 0.0,
                  "stripe": 0.0, "psnr": 0.0}
        n = 0
        # 温度线性退火：早期软（学覆盖率），后期硬（结晶出抖动纹理）
        temp = t_start + (t_end - t_start) * (epoch - 1) / max(1, epochs - 1)
        mode = "hard" if hard_ste else f"T={temp:.3f}"
        for target in tqdm(loader, desc=f"epoch {epoch}/{epochs} {mode}"):
            target = target.to(device)
            logits = model(target)
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
            loss = (l1 + args.w_hvs * l_hvs + args.w_iso * l_iso
                    + args.w_ink * l_ink + args.w_bin * l_bin
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
            ink = (model(sample) > 0).float()
            grid = torch.cat([sample, ink, printer(ink)])
            save_image(grid, out / f"epoch_{epoch:03d}.png", nrow=8)
        torch.save({"model": model.state_dict(), "args": vars(args)},
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
    p.add_argument("--iso-threshold", type=float, default=0.02)

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
    # 数据自备：目录空则生成（train.py 直接跑时才有意义，编排模式统一在这里保证）
    if not args.smoke:
        if not list(Path("data/images").glob("*.png")):
            log("[data] data/images 为空，生成合成训练图")
            subprocess.run([sys.executable, "gen_data.py", "--n", "500"], check=True)
        if not list(Path("data/val").glob("*.png")):
            log("[data] data/val 为空，生成验证集（种子 123）")
            subprocess.run([sys.executable, "gen_data.py", "--n", "30",
                            "--seed", "123", "--out", "data/val"], check=True)

    # 单段模式：只跑一次 --epochs
    if args.rounds <= 1:
        train_once(args, device, out, log, epochs=args.epochs, lr=args.lr,
                   w_iso=args.w_iso, w_bin=args.w_bin, t_start=args.t_start,
                   t_end=args.t_end, hard_ste=args.hard_ste,
                   resume=args.resume)
        return

    # 编排模式：退火多轮 -> 硬 STE 微调
    hp = {"w_iso": args.w_iso, "lr": args.lr}
    history, best = [], {"psnr": -1.0}
    for rnd in range(1, args.rounds + 1):
        # 温度随轮次几何退火：1.0 -> 0.3 -> 0.09 -> 0.03
        t_hi = max(0.03, 1.0 * 0.3 ** (rnd - 1))
        t_lo = max(0.03, t_hi * 0.3)
        w_bin = 0.05 * rnd  # 二值化惩罚随退火逐步加强
        args.w_iso, args.lr = hp["w_iso"], hp["lr"]
        log(f"\n===== 第 {rnd}/{args.rounds} 轮 | {hp} | T {t_hi:.2f}->{t_lo:.2f} "
            f"| w_bin={w_bin:.2f} =====")
        train_once(args, device, out, log, epochs=args.epochs_per_round,
                   lr=hp["lr"], w_iso=hp["w_iso"], w_bin=w_bin,
                   t_start=t_hi, t_end=t_lo, hard_ste=False, resume=True)
        m = evaluate(out / "last.pt", device)
        record = {"round": rnd, **hp, "t_hi": t_hi, "t_lo": t_lo, **m}
        history.append(record)
        # 逐轮落盘：中途被杀也不丢已完成轮次
        with open(out / "pipeline_log.jsonl", "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
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
    for ft in range(1, args.finetune_rounds + 1):
        args.w_iso, args.lr = hp["w_iso"], hp["lr"] / 2
        log(f"\n===== 硬 STE 微调 {ft}/{args.finetune_rounds} | {hp} lr={hp['lr']/2} =====")
        train_once(args, device, out, log, epochs=args.finetune_epochs,
                   lr=hp["lr"] / 2, w_iso=hp["w_iso"], w_bin=0.2,
                   t_start=0.03, t_end=0.03, hard_ste=True, resume=True)
        m = evaluate(out / "last.pt", device)
        record = {"round": f"finetune-{ft}", **hp, **m}
        with open(out / "pipeline_log.jsonl", "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log(f"[验证] psnr={m['psnr']:.2f}dB 孤立点率={m['iso_ratio']:.3%} "
            f"明暗偏差={m['bias']:+.3f}")
        if m["psnr"] > best["psnr"]:
            best = m
            shutil.copy(out / "last.pt", out / "best.pt")

    (out / "pipeline_log.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False))
    log(f"\n完成。最佳 psnr={best['psnr']:.2f}dB，最佳权重 {out}/best.pt")
    log(f"轮次日志：{out}/pipeline_log.json")


if __name__ == "__main__":
    main()