"""推理：一张灰度图 -> 二值打印位图 + 模拟打印预览。"""
import argparse

import numpy as np
import torch
from PIL import Image

from halftone.model import UNet
from halftone.printer import PrinterSimulator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="输入图片")
    p.add_argument("--ckpt", default="runs/halftone/last.pt")
    p.add_argument("--out", default="out", help="输出前缀")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    train_args = ckpt["args"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(base=train_args["base"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    printer = PrinterSimulator(sigma=train_args["sigma"],
                               gain=train_args["gain"]).to(device)

    img = Image.open(args.input).convert("L")
    w, h = img.size
    # 补边到 8 的倍数，推理后裁回
    pw, ph = (8 - w % 8) % 8, (8 - h % 8) % 8
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr)[None, None].to(device)
    x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")

    with torch.no_grad():
        ink = (model(x) > 0).float()[..., :h, :w]
        printed = printer(ink)

    bitmap = (ink[0, 0].cpu().numpy() * 255).astype(np.uint8)
    preview = (printed[0, 0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(bitmap, mode="L").save(f"{args.out}_bitmap.png")
    Image.fromarray(preview, mode="L").save(f"{args.out}_preview.png")
    print(f"二值位图: {args.out}_bitmap.png（1=墨，直接可打）")
    print(f"模拟打印: {args.out}_preview.png（经扩散模型后的纸上效果）")


if __name__ == "__main__":
    main()
