"""AI 生成图像：用 Stable Diffusion 出照片级彩色图（历史属 CMYK 训练数据，见 PITFALLS #50；
NN 训练已于 #52 整体移除，本脚本保留用于给 ED 引擎造真实照片域的内容图/测试图）。

原用途：替代脚本合成解决 CMYK 的分布裂口（#22/#39）。现主线（无 NN）的
评测集是 data/photos_val；本脚本保留供未来需要照片域纹理/图案内容时使用。

动机（见 PITFALLS #22）：原 `gen_data.py`（已随 NN 一并删除）脚本合成的图
饱和度 0.619，而真实照片只有 0.276——合成图「太艳」，与 photos_val 存在
分布裂口（合成验证集 ΔE 7.36 vs 真实照片 15.05）。SD 出的是真实照片分布。

用法：
    uv run python gen_ai.py --n 100 --out data/images_ai          # 文生图（目录自动创建）
    uv run python gen_ai.py --n 100 --init <种子图目录>            # 图生图（保留结构）。
    #   原示例 data/images 已随 CMYK 数据于 2026-10-21 删除；任意存在的图片目录均可作种子
    uv run python gen_ai.py --n 20 --steps 15 --size 512 --dry-run # 小批试跑

注意：
- 首次运行需联网下载 SD1.5 权重（约 4GB，走 hf-mirror）；之后纯本地。
- Quadro P2000 上 512² 20 步约 28 s/张（FP16 + attention slicing）。
  100 张 ≈ 47 min，500 张 ≈ 3.9 h，建议后台跑。
- 输出是 sRGB PNG，不做色彩管理（原 ICC 分离链路已随 CMYK 路线删除）。
- 自动过滤 SD 失败模式（近黑/近白/极低方差），失败的 seed 会重试。
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# 提示词池：覆盖真实照片的内容多样性，避开文字/水印/纯肖像。
# 每条 = (主体, 场景)。随机组合出「a photo of X in Y, natural lighting」。
SUBJECTS = [
    "a wooden table surface", "a close-up of fabric texture",
    "a garden with flowers", "a city street at dusk", "a plate of food",
    "clouds over mountains", "a pile of autumn leaves", "a sandy beach",
    "a close-up of tree bark", "a vintage camera on a desk",
    "a cup of coffee on a saucer", "a row of books on a shelf",
    "a close-up of water droplets", "a field of grass in wind",
    "a bicycle leaning on a wall", "a close-up of rusted metal",
    "a basket of fruit", "a winding river from above",
    "a close-up of pebbles", "a lantern-lit night market",
    "a snowy forest path", "a close-up of woven basket",
    "a vineyard in summer", "a close-up of cracked paint",
    "a seashell collection", "a close-up of fern leaves",
    "a stone wall with moss", "a hot air balloon in the sky",
    "a close-up of knitted wool", "a desert with sand dunes",
    "a puddle reflection", "a close-up of bird feathers",
    "a market stall with vegetables", "a foggy morning lake",
    "a close-up of ceramic glaze", "a close-up of soap bubbles",
    "a wheat field at sunset", "a close-up of stained glass",
]
SCENES = [
    "natural daylight", "soft overcast light", "warm sunset light",
    "cool morning light", "indoor window light", "golden hour",
]
STYLES = [
    "sharp focus", "shallow depth of field", "highly detailed",
    "documentary photography", "film photography",
]


def build_prompt(rng):
    return (f"a photo of {rng.choice(SUBJECTS)}, "
            f"{rng.choice(SCENES)}, {rng.choice(STYLES)}, 4k")


def is_degenerate(img):
    """过滤 SD 失败模式：近黑/近白/极低方差/大面积纯色。"""
    a = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    if a.std() < 0.03:
        return "std<0.03"
    if a.mean() < 0.04:
        return "近黑"
    if a.mean() > 0.96:
        return "近白"
    # 单色占比 > 85% 视为失败（SD 有时出纯色块）
    mx, mn = a.max(-1), a.min(-1)
    flat = ((mx - mn) < 0.02).mean()
    if flat > 0.85:
        return f"单色{flat:.0%}"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="data/images_ai")
    ap.add_argument("--init", default=None,
                    help="图生图：从该目录取图做 init（保留结构多样性）")
    ap.add_argument("--strength", type=float, default=0.55,
                    help="图生图重绘强度（低=更贴近 init 结构）")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--max-retry", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true",
                    help="只出提示词和前 2 张，不落盘，用来验证配置")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False, variant="fp16",
        local_files_only=True).to("cuda")
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    print(f"[sd] {args.model} 加载完成，{args.steps} 步 / {args.size}² / "
          f"FP16 + attention slicing", flush=True)

    init_imgs = []
    if args.init:
        init_imgs = sorted(Path(args.init).glob("*.png"))
        print(f"[init] 图生图模式：{len(init_imgs)} 张 init 图，"
              f"strength={args.strength}", flush=True)

    rng = random.Random(args.seed)
    ok = fail = 0
    t_start = time.time()
    # 从已有文件数接着编号，避免重跑时覆盖（seed 不同也会从 0 开始）。
    idx = len(list(out.glob("ai_*.png")))
    print(f"[resume] 目录已有 {idx} 张，从 ai_{idx:04d} 继续", flush=True)
    while ok < args.n:
        seed = rng.randrange(2 ** 31)
        prompt = build_prompt(rng)
        gen = torch.Generator("cuda").manual_seed(seed)
        try:
            kw = dict(prompt=prompt, num_inference_steps=args.steps,
                      guidance_scale=args.guidance, generator=gen,
                      height=args.size, width=args.size)
            if init_imgs:
                init = Image.open(rng.choice(init_imgs)).convert("RGB")
                init = init.resize((args.size, args.size), Image.LANCZOS)
                kw.update(image=init, strength=args.strength)
                tag = "img2img"
            else:
                tag = "txt2img"
            img = pipe(**kw).images[0]
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"[oom] seed={seed} 显存不足，清理后重试", flush=True)
                torch.cuda.empty_cache()
                fail += 1
                if fail > args.max_retry:
                    print("[abort] 连续显存不足，放弃", flush=True)
                    break
                continue
            print(f"[err] seed={seed}: {e}", flush=True)
            fail += 1
            continue

        bad = is_degenerate(img)
        if bad:
            print(f"[skip] seed={seed} {bad}", flush=True)
            continue

        if not args.dry_run:
            p = out / f"ai_{idx:04d}_{tag}.png"
            img.save(p)
        ok += 1
        idx += 1
        dt = time.time() - t_start
        print(f"[{ok}/{args.n}] seed={seed} {tag} "
              f"均速 {dt/max(ok,1):.1f}s/张 | {prompt[:60]}", flush=True)
        if args.dry_run and ok >= 2:
            break

    dt = time.time() - t_start
    print(f"\n[done] {ok} 张成功 / {fail} 次失败，总耗时 {dt/60:.1f} min"
          f"（{dt/max(ok,1):.1f} s/张）")
    if not args.dry_run:
        print(f"[out] {out}")


if __name__ == "__main__":
    sys.exit(main())
