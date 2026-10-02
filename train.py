"""训练：灰度图 -> U-Net -> STE 二值墨点 -> 可微打印模拟 -> 与目标算损失。"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from halftone.data import GrayCrops
from halftone.losses import HVSLoss, isolated_dot_penalty
from halftone.model import UNet, hard_binarize, soft_binarize
from halftone.printer import PrinterSimulator


def psnr(a, b):
    return 10 * math.log10(1.0 / (F.mse_loss(a, b).item() + 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/images", help="图片文件夹，空则合成数据冒烟")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--length", type=int, default=2000, help="每 epoch 样本数")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=32, help="U-Net 通道基数")
    p.add_argument("--sigma", type=float, default=1.0, help="墨滴扩散 σ（像素）")
    p.add_argument("--gain", type=float, default=1.5, help="网点扩大系数")
    p.add_argument("--w-hvs", type=float, default=2.0)
    p.add_argument("--w-iso", type=float, default=0.05)
    p.add_argument("--w-ink", type=float, default=0.02, help="省墨惩罚，防实心填充锁死")
    p.add_argument("--w-bin", type=float, default=0.05, help="二值化惩罚 soft*(1-soft)")
    p.add_argument("--t-start", type=float, default=1.0, help="退火起始温度")
    p.add_argument("--t-end", type=float, default=0.05, help="退火结束温度")
    p.add_argument("--hard-ste", action="store_true",
                   help="硬前向+截断STE（仅退火后微调用，勿从零训练）")
    p.add_argument("--out", default="runs/halftone")
    p.add_argument("--smoke", action="store_true", help="小数据快速冒烟")
    p.add_argument("--resume", action="store_true", help="从 out/last.pt 续训")
    args = p.parse_args()

    if args.smoke:
        args.size, args.length, args.batch, args.epochs = 128, 100, 8, 1
        args.out = "runs/smoke"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))
    log_file = open(out / "train.log", "a", buffering=1)  # 行缓冲，append 续训不断档

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(base=args.base).to(device)
    ckpt_file = out / "last.pt"
    if args.resume and ckpt_file.exists():
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        log(f"[resume] 从 {ckpt_file} 继续训练")
    printer = PrinterSimulator(sigma=args.sigma, gain=args.gain).to(device)
    hvs = HVSLoss(sigma=2.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ds = GrayCrops(args.data, size=args.size, length=args.length)
    if ds.synthetic:
        log(f"[data] {args.data} 为空，使用合成灰度图（仅冒烟验证流程）")
    else:
        log(f"[data] {len(ds.paths)} 张图片，随机裁 {args.size}x{args.size}")
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=2, drop_last=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "l1": 0.0, "hvs": 0.0, "iso": 0.0, "psnr": 0.0}
        n = 0
        # 温度线性退火：早期软（学覆盖率），后期硬（结晶出抖动纹理）
        temp = args.t_start + (args.t_end - args.t_start) * (epoch - 1) / max(1, args.epochs - 1)
        mode = "hard" if args.hard_ste else f"T={temp:.3f}"
        for target in tqdm(loader, desc=f"epoch {epoch}/{args.epochs} {mode}"):
            target = target.to(device)
            logits = model(target)
            ink = (hard_binarize(logits) if args.hard_ste
                   else soft_binarize(logits, temp))
            printed = printer(ink)

            l1 = F.l1_loss(printed, target)
            l_hvs = hvs(printed, target)
            l_iso = isolated_dot_penalty(ink)
            l_ink = ink.mean()
            l_bin = (ink * (1 - ink)).mean()
            loss = (l1 + args.w_hvs * l_hvs + args.w_iso * l_iso
                    + args.w_ink * l_ink + args.w_bin * l_bin)

            opt.zero_grad()
            loss.backward()
            opt.step()

            b = target.shape[0]
            totals["loss"] += loss.item() * b
            totals["l1"] += l1.item() * b
            totals["hvs"] += l_hvs.item() * b
            totals["iso"] += l_iso.item() * b
            totals["psnr"] += psnr(printed.detach(), target) * b
            n += b

        msg = " ".join(f"{k}={v / n:.4f}" for k, v in totals.items())
        log(f"[epoch {epoch}] {mode} {msg}")

        model.eval()
        with torch.no_grad():
            sample = next(iter(loader)).to(device)[:8]
            ink = (model(sample) > 0).float()
            printed = printer(ink)
            grid = torch.cat([sample, ink, printed])
            save_image(grid, out / f"epoch_{epoch:03d}.png", nrow=8)
        torch.save({"model": model.state_dict(), "args": vars(args)}, out / "last.pt")
        log(f"[epoch {epoch}] 样例与权重已存到 {out}")


if __name__ == "__main__":
    main()
