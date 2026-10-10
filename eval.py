"""评测与诊断入口：跑模型/基线，出 PSNR 对照与"哪里不对"的量化诊断。

评测（产出结果）
  uv run python eval.py infer <图>                        -> 打印位图 + 模拟预览
  uv run python eval.py one   <图>  --ckpt <ckpt>         -> 单图三方对照 + inkcov 审计
  uv run python eval.py batch [文件夹] --ckpt <ckpt>      -> 批量三方对照（默认 data/photos_val）

诊断（查问题，改配方前先跑）
  uv run python eval.py wedge  --ckpt <ckpt>              -> 灰度尺分段偏差 + 条纹检测
  uv run python eval.py data                             -> 训练集平坦区/灰阶分布
  uv run python eval.py norm                             -> 网络能否区分绝对灰阶
  uv run python eval.py compare --a A.pt --b B.pt         -> 两个 ckpt 逐一对比

三条路线（模型 / FS / GS 筛网）共用同一套物理语义：film 模型下灰度=相对膜厚，
FS/GS 按"越暗墨越多"加网，必须反相输入才能对齐（film_target_and_base 负责）。
"""
import argparse
import glob
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image

from halftone.model import UNet
from halftone.printer import build_printer

GS_WORKDIR = Path("out/gs")


# ---------------------------------------------------------------- 底层工具

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
    """ghostscript inkcov 设备审计：返回 K 通道墨覆盖率。gs 失败则回退 NaN。"""
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


# ---------------------------------------------------------------- 共用

def triplet(img, ckpt, dpi=None):
    """同一张图跑 模型/FS/GS 三条路线，返回 (inks, prints, target, ckpt_args)。"""
    model, printer, a, device = load_model_and_printer(ckpt, dpi=dpi)
    film = a.get("printer", "paper") == "film"
    target, base_img = film_target_and_base(img, film, device)
    size = (target.shape[3], target.shape[2])
    GS_WORKDIR.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        inks = {"nn": (model(target) > 0).float(),
                "fs": floyd_steinberg(base_img, size).to(device),
                "gs": gs_halftone(base_img, size, GS_WORKDIR).to(device)}
        prints = {k: printer(v) for k, v in inks.items()}
    return inks, prints, target, a


def texture_stats(ink):
    """位图纹理质量。像素级抖动应局部密度均匀、行墨量方差≈0。

    row_std 高 => 用整行条纹表达灰阶（退化解），会被打印模糊横向平均掉，
    导致覆盖率偏离目标（见 PITFALLS #12）。列方差不作为判据：测试图本身
    可能是竖条结构（如灰度尺），每列覆盖率天然不同。
    """
    x = ink[0, 0].cpu().numpy().astype(np.float64)
    row = x.mean(axis=1)
    from numpy.lib.stride_tricks import sliding_window_view
    loc = sliding_window_view(x, (9, 9)).mean(axis=(-1, -2))
    return {"cover": x.mean(), "row_std": row.std(), "loc_std": loc.std(),
            "row_range": row.max() - row.min()}


def print_texture(tag, t):
    print(f"  {tag:3s} 覆盖 {t['cover']:.3f} | 行墨量std {t['row_std']:.4f} "
          f"| 局部密度std {t['loc_std']:.4f} | 行极差 {t['row_range']:.3f}")


# ---------------------------------------------------------------- infer

def cmd_infer(args):
    """单图推理：出打印位图 + 模拟预览。位图不跑基线，最快。"""
    model, printer, a, device = load_model_and_printer(args.ckpt, dpi=args.dpi)
    img = Image.open(args.input).convert("L")
    w, h = img.size
    target, _ = film_target_and_base(img, False, device)  # 推理不需基线反相
    with torch.no_grad():
        ink = (model(target) > 0).float()[..., :h, :w]   # 裁回原尺寸
        printed = printer(ink)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    bitmap = (ink[0, 0].cpu().numpy() * 255).astype(np.uint8)      # 1=墨
    preview = (printed[0, 0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(bitmap, mode="L").save(f"{args.out}_bitmap.png")
    Image.fromarray(preview, mode="L").save(f"{args.out}_preview.png")
    print(f"二值位图: {args.out}_bitmap.png（1=墨，直接可打）")
    print(f"模拟打印: {args.out}_preview.png（经扩散模型后的纸上效果）")


# ---------------------------------------------------------------- one

def cmd_one(args):
    """单图三方对照 + inkcov 审计 + 七行对比图。"""
    inks, prints, target, a = triplet(Image.open(args.input).convert("L"),
                                      args.ckpt, args.dpi)
    if a.get("printer") == "film":
        print("[film] 膜基模式：模拟输出=相对膜厚，基线用反相图加网")
    outdir = Path(args.out).parent
    outdir.mkdir(parents=True, exist_ok=True)
    p = {k: psnr(v, target) for k, v in prints.items()}
    print(f"模拟打印 PSNR：模型 {p['nn']:.2f} | FS {p['fs']:.2f} | "
          f"GS筛网 {p['gs']:.2f} dB | 模型 vs FS {p['nn']-p['fs']:+.2f} | "
          f"模型 vs GS {p['nn']-p['gs']:+.2f}")

    # 落盘位图（墨=黑）并用 ghostscript inkcov 独立审计墨覆盖率
    for name, ink in inks.items():
        path = f"{args.out}_{name}_bitmap.png"
        pil = Image.fromarray(((1 - ink[0, 0].cpu().numpy()) * 255).astype(np.uint8),
                              mode="L")
        pil.save(path)
        pil.save(f"{args.out}_{name}_bitmap.pdf", "PDF")  # gs 不认 PNG 输入
        cov = gs_inkcov(f"{args.out}_{name}_bitmap.pdf")
        warn = " ⚠️ 覆盖率过高，实心填充风险" if cov > 0.85 else ""
        print(f"[inkcov] {name}: K 覆盖率 {cov:.1%} "
              f"(张量自检 {ink.mean().item():.1%}){warn} -> {path}")

    grid = torch.cat([target, inks["nn"], prints["nn"], inks["fs"], prints["fs"],
                      inks["gs"], prints["gs"]])
    save_image(grid, f"{args.out}.png", nrow=1)
    print(f"对比图：{args.out}.png（原图|模型位图|模型打印|FS位图|FS打印|GS位图|GS打印）")


# ---------------------------------------------------------------- batch

def cmd_batch(args):
    """真实照片文件夹批量三方对照，末尾给均值。"""
    folder = Path(args.folder)
    files = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
    if not files:
        raise SystemExit(f"{folder} 里没有图片")
    print(f"{'照片':>12} {'模型':>7} {'FS':>7} {'GS':>7} {'vs FS':>7} {'vs GS':>7}")
    rows = []
    for f in files:
        _, prints, target, a = triplet(Image.open(f).convert("L"),
                                       args.ckpt, args.dpi)
        if rows == [] and a.get("printer") == "film":
            print("[film] 膜基模式：模拟输出=相对膜厚，基线用反相图加网")
        r = [psnr(prints[m], target) for m in ("nn", "fs", "gs")]
        rows.append(r)
        print(f"{f.stem[:12]:>12} {r[0]:7.2f} {r[1]:7.2f} {r[2]:7.2f} "
              f"{r[0]-r[1]:+7.2f} {r[0]-r[2]:+7.2f}")
    m = np.array(rows).mean(axis=0)
    print(f"{'均值':>12} {m[0]:7.2f} {m[1]:7.2f} {m[2]:7.2f} "
          f"{m[0]-m[1]:+7.2f} {m[0]-m[2]:+7.2f}")


# ---------------------------------------------------------------- wedge

def cmd_wedge(args):
    """灰度尺：分段覆盖偏差 + 位图纹理质量。查灰阶 S 形畸变与条纹退化解。"""
    lv = np.arange(args.levels) / (args.levels - 1)
    arr = np.concatenate([np.full((args.seg, args.seg), v, np.float32) for v in lv],
                        axis=1)
    inks, prints, target, a = triplet(Image.fromarray((arr * 255).astype("uint8"), "L"),
                                      args.ckpt, args.dpi)
    print(f"灰度尺 {args.levels} 阶 @ {args.dpi or a.get('dpi')} dpi "
          f"(ckpt={args.ckpt}, {a.get('printer')})")
    print(f"\n全图 PSNR: 模型 {psnr(prints['nn'], target):.2f} | "
          f"FS {psnr(prints['fs'], target):.2f} | "
          f"GS {psnr(prints['gs'], target):.2f} dB")
    print("\n分段覆盖偏差（实际覆盖率 - 目标覆盖率，|x|>0.02 才有意义）")
    print(f"{'灰阶':>6} {'目标':>6} | {'模型':>6} {'偏差':>7} | {'FS':>6} {'偏差':>7} "
          f"| {'GS':>6} {'偏差':>7}")
    worst = (0.0, 0.0)
    for k, v in enumerate(lv):
        seg = lambda t: t[0, 0, :, k * args.seg:(k + 1) * args.seg].mean().item()
        row = [seg(prints[m]) for m in ("nn", "fs", "gs")]
        d = [row[0] - v, row[1] - v, row[2] - v]
        if abs(d[0]) > abs(worst[1]):
            worst = (v, d[0])
        print(f"{v:6.3f} {v:6.3f} | {row[0]:6.3f} {d[0]:+7.3f} | {row[1]:6.3f} "
              f"{d[1]:+7.3f} | {row[2]:6.3f} {d[2]:+7.3f}")
    print(f"\n模型最大偏差: 灰阶 {worst[0]:.3f} 处 {worst[1]:+.3f}")
    print(f"形态: {'S 形（两端外鼓/中间内塌或反之）' if worst[1] else '无明显畸变'}")

    print("\n位图纹理质量（像素级抖动应各项方差≈0）")
    for m in ("nn", "fs", "gs"):
        print_texture(m, texture_stats(inks[m]))
    t = texture_stats(inks["nn"])
    if t["row_std"] > 0.05:
        print(f"  ⚠ 模型行墨量std {t['row_std']:.3f} 过高 => 用整行条纹表达灰阶，"
              f"这是覆盖率失真的主因（见 PITFALLS #12）")


# ---------------------------------------------------------------- data

def cmd_data(args):
    """训练集平坦区占比与灰阶分布：查"数据够不够"。"""
    from halftone.data import GrayCrops
    paths = sorted(glob.glob(f"{args.dir}/*.png"))
    if not paths:
        raise SystemExit(f"{args.dir} 里没有图，先生成：uv run python gen_data.py --n 500")
    print(f"训练集统计：{args.dir}，{len(paths)} 张，按 size={args.size} 采样 "
          f"{args.n} 个裁块（模型实际看到的）")
    ds = GrayCrops(args.dir, size=args.size, length=args.n)
    k = torch.ones(1, 1, 3, 3) / 9.0
    fracs, levels, hist = [], [], torch.zeros(10)
    for i in range(len(ds)):
        x = ds[i][None]
        var = F.conv2d(F.pad(x ** 2, (1, 1, 1, 1), mode="reflect"), k) - \
              F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k) ** 2
        flat = (var < args.flat_std ** 2).float()
        fracs.append(flat.mean().item())
        if flat.sum() > 200:
            lv = ((x * flat).sum() / flat.sum()).item()
            levels.append(lv)
            hist[int(lv * 9 + 0.5)] += 1
    print(f"\n平坦像素占比: 均值 {np.mean(fracs):.1%} 中位 {np.median(fracs):.1%} "
          f"最小 {np.min(fracs):.1%}  (std<{args.flat_std} 视为平坦)")
    print(f"平坦区样本 {len(levels)}/{len(ds)}，灰阶中位 {np.median(levels):.3f}")
    print("\n平坦区灰阶分布（10 档）——分布太窄 = 模型没见过某些绝对灰阶")
    for j in range(10):
        n = hist[j].item()
        print(f"  {j/9:.2f}  {n:5.0f}  {'#' * int(n / max(1, hist.max().item()) * 40)}")
    print("\n判读：两端（<0.1 / >0.9）样本 <5% 就要怀疑暗/亮端覆盖不足。")


# ---------------------------------------------------------------- norm

def cmd_norm(args):
    """架构探针：网络能否区分不同绝对灰阶（查信息有没有被归一化抹掉）。"""
    from halftone.model import UNet
    torch.manual_seed(0)
    net = UNet(base=args.base).eval()
    lv = [0.0, 0.133, 0.267, 0.5, 0.733, 0.867, 1.0]
    print(f"架构探针：冻结随机 UNet(base={args.base}) 喂空间常数图")
    print("若不同灰阶的输出差异随灰阶差单调增大 => 信息保留正常")
    print(f"{'灰阶':>6} {'logits均值':>12} {'logits标准差':>13}")
    outs = []
    for c in lv:
        with torch.no_grad():
            y = net(torch.full((1, 1, 128, 128), float(c)))
        outs.append(y)
        print(f"{c:6.3f} {y.mean().item():12.5f} {y.std().item():13.5f}")
    pair = sorted((abs(lv[i] - lv[j]), (outs[i] - outs[j]).abs().mean().item())
                  for i in range(len(lv)) for j in range(i + 1, len(lv)))
    print("\n灰阶差 -> 输出差")
    for i in range(0, len(pair), max(1, len(pair) // 8)):
        print(f"  {pair[i][0]:.3f} -> {pair[i][1]:.6f}")
    ok = pair[-1][1] > pair[0][1] * 3
    verdict = "正常，灰阶信息保留" if ok else \
        "异常！网络无法区分绝对灰阶，应怀疑 GroupNorm/BatchNorm 抹掉了均值信息"
    print(f"\n判读: {verdict}")


# ---------------------------------------------------------------- compare

def cmd_compare(args):
    """两个 ckpt 在同一批图上逐一对比：查新配方有没有变好。"""
    imgs = sorted(glob.glob(f"{args.dir}/*.png"))[: args.limit]
    if not imgs:
        raise SystemExit(f"{args.dir} 里没有图片")
    print(f"对比 {len(imgs)} 张图：{args.a}  vs  {args.b}\n")
    print(f"{'图片':>16} | {'A模型':>7} {'A vs FS':>8} | {'B模型':>7} {'B vs FS':>8} "
          f"| {'B-A':>7}")
    tot = {"a": [], "b": []}
    for p in imgs:
        img = Image.open(p).convert("L")
        _, pa, ta, _ = triplet(img, args.a, args.dpi)
        _, pb, tb, _ = triplet(img, args.b, args.dpi)
        sa, sb = psnr(pa["nn"], ta), psnr(pb["nn"], tb)
        fa, fb = psnr(pa["fs"], ta), psnr(pb["fs"], tb)
        tot["a"].append(sa - fa)
        tot["b"].append(sb - fb)
        print(f"{Path(p).stem[:16]:>16} | {sa:7.2f} {sa-fa:+8.2f} | "
              f"{sb:7.2f} {sb-fb:+8.2f} | {sb-sa:+7.2f}")
    ma, mb = np.mean(tot["a"]), np.mean(tot["b"])
    print(f"\n均值 vs FS:  A {ma:+.2f} dB   B {mb:+.2f} dB   "
          f"B-A {mb-ma:+.2f} dB  {'← B 更好' if mb > ma else '← A 更好或持平'}")


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def ckpt_arg(sp, default="runs/halftone/best.pt"):
        sp.add_argument("--ckpt", default=default)
        sp.add_argument("--dpi", type=int, default=None,
                        help="film 模式：按此 dpi 评估（默认 ckpt 里的）")
        return sp

    i = ckpt_arg(sub.add_parser("infer", help="单图推理 -> 打印位图 + 预览"),
                 "runs/halftone/last.pt")
    i.add_argument("input")
    i.add_argument("--out", default="out/photo")
    i.set_defaults(func=cmd_infer)

    o = ckpt_arg(sub.add_parser("one", help="单图三方对照 + inkcov 审计"))
    o.add_argument("input")
    o.add_argument("--out", default="out/verify")
    o.set_defaults(func=cmd_one)

    b = ckpt_arg(sub.add_parser("batch", help="批量三方对照"))
    b.add_argument("folder", nargs="?", default="data/photos_val")
    b.set_defaults(func=cmd_batch)

    w = ckpt_arg(sub.add_parser("wedge", help="灰度尺分段偏差 + 条纹检测"))
    w.add_argument("--levels", type=int, default=16)
    w.add_argument("--seg", type=int, default=128, help="每阶块边长")
    w.set_defaults(func=cmd_wedge)

    d = sub.add_parser("data", help="训练集平坦区与灰阶分布")
    d.add_argument("--dir", default="data/images")
    d.add_argument("--size", type=int, default=256)
    d.add_argument("--n", type=int, default=256)
    d.add_argument("--flat-std", type=float, default=0.02)
    d.set_defaults(func=cmd_data)

    m = sub.add_parser("norm", help="网络能否区分绝对灰阶")
    m.add_argument("--base", type=int, default=32)
    m.set_defaults(func=cmd_norm)

    c = ckpt_arg(sub.add_parser("compare", help="两个 ckpt 逐一对比"))
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--dir", default="data/photos_val")
    c.add_argument("--limit", type=int, default=12)
    c.set_defaults(func=cmd_compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()