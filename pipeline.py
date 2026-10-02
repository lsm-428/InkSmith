"""按 WORKFLOW 自动执行：数据 -> 训练 -> 验证 -> 不达标按规则调参续训。

策略（对应 WORKFLOW.md 的调参表，每轮只改一个变量）：
  1. 孤立点过多  -> w_iso 加倍
  2. psnr 仍在涨 -> 直接续训一轮
  3. psnr 停滞   -> 学习率减半后续训

用法：uv run python pipeline.py
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from halftone.data import GrayCrops
from halftone.model import UNet
from halftone.printer import PrinterSimulator

RUNS = Path("runs/halftone")


def sh(cmd):
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def ensure_data(regen=False):
    if regen or not list(Path("data/images").glob("*.png")):
        sh([sys.executable, "gen_data.py", "--n", "500"])
    if not list(Path("data/val").glob("*.png")):
        sh([sys.executable, "gen_data.py", "--n", "30", "--seed", "123",
            "--out", "data/val"])


def evaluate(ckpt_path, device, n_val=30, size=256):
    """验证集指标：psnr（模拟打印 vs 目标）、孤立点率、整体明暗偏差。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = UNet(base=a["base"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    printer = PrinterSimulator(a["sigma"], a["gain"]).to(device)
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
    return {"psnr": float(np.mean(psnrs)),
            "iso_ratio": float(np.mean(isos)),
            "bias": float(np.mean(biases))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-psnr", type=float, default=18.0)
    p.add_argument("--iso-threshold", type=float, default=0.02)
    p.add_argument("--max-rounds", type=int, default=4)
    p.add_argument("--epochs-per-round", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--sigma", type=float, default=1.0, help="墨滴扩散 σ")
    p.add_argument("--gain", type=float, default=1.5, help="网点扩大系数")
    p.add_argument("--finetune-rounds", type=int, default=1, help="退火后硬 STE 微调轮数")
    p.add_argument("--finetune-epochs", type=int, default=20)
    p.add_argument("--regen-data", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_data(regen=args.regen_data)

    # 训练超参（会被策略修改）
    hp = {"w_iso": 0.05, "lr": args.lr}
    history = []
    best = {"psnr": -1.0}

    for rnd in range(1, args.max_rounds + 1):
        # 温度随轮次几何退火：1.0 -> 0.3 -> 0.09 -> 0.03
        t_hi = max(0.03, 1.0 * 0.3 ** (rnd - 1))
        t_lo = max(0.03, t_hi * 0.3)
        w_bin = 0.05 * rnd  # 二值化惩罚随退火逐步加强
        print(f"\n===== 第 {rnd}/{args.max_rounds} 轮 | {hp} | T {t_hi:.2f}->{t_lo:.2f} | w_bin={w_bin:.2f} =====")
        sh([sys.executable, "train.py", "--resume",
            "--epochs", str(args.epochs_per_round),
            "--w-iso", str(hp["w_iso"]),
            "--lr", str(hp["lr"]),
            "--sigma", str(args.sigma), "--gain", str(args.gain),
            "--w-bin", str(w_bin),
            "--t-start", str(t_hi), "--t-end", str(t_lo)])

        m = evaluate(RUNS / "last.pt", device)
        record = {"round": rnd, **hp, "t_hi": t_hi, "t_lo": t_lo, **m}
        history.append(record)
        # 逐轮落盘：中途被杀也不丢已完成轮次
        with open(RUNS / "pipeline_log.jsonl", "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[验证] psnr={m['psnr']:.2f}dB 孤立点率={m['iso_ratio']:.3%} "
              f"明暗偏差={m['bias']:+.3f}")

        if m["psnr"] > best["psnr"]:
            best = m
            shutil.copy(RUNS / "last.pt", RUNS / "best.pt")

        if m["psnr"] >= args.target_psnr and m["iso_ratio"] <= args.iso_threshold:
            print(f"达标：psnr >= {args.target_psnr} 且 孤立点率 <= "
                  f"{args.iso_threshold:.0%}")
            break

        # 策略：每轮只改一个变量
        prev = history[-2]["psnr"] if len(history) > 1 else -1.0
        if m["iso_ratio"] > args.iso_threshold:
            hp["w_iso"] = min(hp["w_iso"] * 2, 0.2)  # 封顶，防硬化期失控
            print(f"[策略] 孤立点过多，w_iso -> {hp['w_iso']}")
        elif m["psnr"] - prev > 0.3:
            print("[策略] psnr 仍在上涨，直接续训")
        else:
            hp["lr"] /= 2
            print(f"[策略] psnr 停滞，lr -> {hp['lr']}")

    # 硬 STE 微调：消灭软值作弊，让训练指标=推理指标（退火后才安全）
    for ft in range(1, args.finetune_rounds + 1):
        print(f"\n===== 硬 STE 微调 {ft}/{args.finetune_rounds} | {hp} =====")
        sh([sys.executable, "train.py", "--resume",
            "--epochs", str(args.finetune_epochs),
            "--w-iso", str(hp["w_iso"]),
            "--lr", str(hp["lr"] / 2),
            "--sigma", str(args.sigma), "--gain", str(args.gain),
            "--w-bin", "0.2", "--hard-ste"])
        m = evaluate(RUNS / "last.pt", device)
        record = {"round": f"finetune-{ft}", **hp, **m}
        with open(RUNS / "pipeline_log.jsonl", "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[验证] psnr={m['psnr']:.2f}dB 孤立点率={m['iso_ratio']:.3%} "
              f"明暗偏差={m['bias']:+.3f}")
        if m["psnr"] > best["psnr"]:
            best = m
            shutil.copy(RUNS / "last.pt", RUNS / "best.pt")

    (RUNS / "pipeline_log.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False))
    print(f"\n完成。最佳 psnr={best['psnr']:.2f}dB，最佳权重 runs/halftone/best.pt")
    print(f"轮次日志：runs/halftone/pipeline_log.json")


if __name__ == "__main__":
    main()
