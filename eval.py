"""评测与诊断入口：跑模型/基线，出 PSNR 对照与"哪里不对"的量化诊断。

评测（产出结果）
  uv run python eval.py infer <图>                        -> 打印位图 + 模拟预览
  uv run python eval.py one   <图>  --ckpt <ckpt>         -> 单图三方对照 + inkcov 审计
  uv run python eval.py batch [文件夹] --ckpt <ckpt>      -> 批量三方对照（默认 data/photos_val）
  uv run python eval.py cmyk  [文件夹] --ckpt <ckpt>      -> CMYK 多档：ΔE00 + 四通道 inkcov

诊断（查问题，改配方前先跑）
  uv run python eval.py wedge  --ckpt <ckpt>              -> 灰度尺分段偏差 + 条纹检测
  uv run python eval.py data                             -> 训练集平坦区/灰阶分布
  uv run python eval.py norm                             -> 网络能否区分绝对灰阶
  uv run python eval.py compare --a A.pt --b B.pt         -> 两个 ckpt 逐一对比

三条路线（模型 / FS / GS 筛网）共用同一套物理语义：film 模型下灰度=相对膜厚，
FS/GS 按"越暗墨越多"加网，必须反相输入才能对齐（film_target_and_base 负责）。
CMYK 路线（--cmyk 训练的 ckpt）没有 FS/GS 基线，只用 ΔE00 与 gs inkcov 审计。
"""
import argparse
import glob
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageCms
from torchvision.utils import save_image

from halftone.cmyk import (LEVELS, find_icc, lab_to_srgb,
                           load_or_build_lut256, neugebauer_lab_2bit,
                           srgb_to_cov, srgb_to_lab)
from halftone.model import UNet, hard_levels
from train import delta_e_2000
from halftone.printer import build_printer

GS_WORKDIR = Path("out/gs")


# ---------------------------------------------------------------- 底层工具

def psnr(a, b):
    return 10 * math.log10(1.0 / (F.mse_loss(a, b).item() + 1e-12))


def load_model_and_printer(ckpt_path, dpi=None, device=None):
    """从 ckpt 重建 UNet + printer；film 模式下可用 dpi 覆盖预览/评估分辨率。

    CMYK ckpt（args.cmyk=True）：printer 为 None，返回 model 与 ckpt args。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    if dpi and a.get("printer") == "film":
        a = {**a, "dpi": dpi}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(base=a["base"], up=a.get("up", "nearest"),
                 in_ch=a.get("in_ch", 1), out_ch=a.get("out_ch", 1),
                 levels=a.get("levels", 1)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    printer = None if a.get("cmyk") else build_printer(a).to(device)
    return model, printer, a, device


def film_target_and_base(img, film, device):
    """灰度图 -> pad(8 倍数, reflect) 后的 target；film 时同时返回反相 FS/GS 输入。

    **base 必须用与 target 完全相同的 reflect 补边，不能靠 resize**（PITFALLS #43）：
    resize 把 481 宽拉成 488（1.4% 非等比缩放），内容相对 target 整体错位，
    实测反相图与 target 偏差均值 0.0198、最大 0.58；同一 FS 算法喂这张错位图
    凭空掉 6.1dB（25.34 vs 31.45），「赢 FS」的假象就是这么来的。
    补边后 base.size == target 尺寸，下游 `resize(size)` 退化为 no-op。
    """
    w, h = img.size
    pw, ph = (8 - w % 8) % 8, (8 - h % 8) % 8
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)[None, None].to(device)
    t = F.pad(t, (0, pw, 0, ph), mode="reflect")
    base_arr = t[0, 0].cpu().numpy()
    if film:
        base_arr = 1.0 - base_arr      # 灰度=膜厚（暗=多墨），基线要吃「墨量」
    base = Image.fromarray((base_arr * 255.0).round().clip(0, 255).astype(np.uint8), "L")
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
    """ghostscript inkcov 设备审计：返回 C,M,Y,K 四通道墨覆盖率。gs 失败则 NaN。"""
    try:
        out = subprocess.run(
            ["gs", "-o", "-", "-sDEVICE=inkcov", bitmap_path],
            capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[inkcov] ghostscript 调用失败：{e}")
        return [float("nan")] * 4
    for line in out.splitlines():
        if "CMYK OK" in line:
            return [float(x) for x in line.split()[:4]]  # C M Y K
    return [float("nan")] * 4


def rip_cmyk(img_rgb, icc, intent=1, workdir="out/rip_gs"):
    """工业 RIP 基线：同 ICC 分离 + ghostscript 半色调 -> 四通道 1-bit 位图。

    img_rgb: PIL RGB 图。icc: FOGRA39L 等 CMYK ICC 路径。
    返回 (cov (4,H,W) float [0,1]，值=墨覆盖率) 或 None（gs 失败）。

    **为什么是这个流程**（实测踩坑）：
      - 直接 `gs tiffsep1` 渲 sRGB PDF 会用 gs **自带**的默认 CMYK profile
        分离（gray50 的 K 只有 0.074，PIL FOGRA39L 给 0.184）——那测的是
        gs 的默认 profile，不是我们的墨水，对比不公平。
      - 先用 PIL `ImageCms` 按**同一个 ICC**分离成 CMYK PDF，再让 gs 只做
        半色调（`tiffsep1`），分离结果与 PIL 连续调完全一致
        （gray50 CMY 0.512/0.414/0.410 vs 连续调 0.510/0.416/0.412），
        差异只剩「半色调算法」本身——这正是要对比的东西。
      - tiffsep1 输出的是 1-bit 四通道位图；tif 是 mode "1"，`np.asarray`
        给出 bool 数组，**True=白**（无墨），所以墨覆盖率 = 1 - mean。
        注意别写成 `1 - arr/255`：bool 数组除 255 会把 True 变成 0.0039，
        结果永远 ~0.998（实测踩过，PITFALLS #6 同类陷阱）。
    """
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    try:
        srgb = ImageCms.createProfile("sRGB")
        prof = ImageCms.getOpenProfile(icc)
        xf = ImageCms.buildTransform(srgb, prof, "RGB", "CMYK",
                                     renderingIntent=intent)
        cmyk = ImageCms.applyTransform(img_rgb, xf)
        pdf = wd / "_in.pdf"
        cmyk.save(pdf, "PDF", resolution=72)
        subprocess.run(["gs", "-dBATCH", "-dNOSAFER", "-dNOPAUSE", "-q",
                        "-o", str(wd / "_o_"), "-sDEVICE=tiffsep1", "-r72",
                        str(pdf)], capture_output=True, check=True)
        covs = []
        for name in ("Cyan", "Magenta", "Yellow", "Black"):
            arr = np.asarray(Image.open(wd / f"_o_({name}).tif"))
            # mode "1" -> bool；True=白。先转 0/1 float 再取反。
            covs.append(1.0 - arr.astype(np.float32))
        return np.stack(covs, 0)
    except (subprocess.CalledProcessError, FileNotFoundError,
            OSError, ValueError) as e:
        print(f"[rip] ghostscript RIP 基线失败：{e}")
        return None


# ---------------------------------------------------------------- 共用

def onebit_ink(logits, target_cov, ed=False, bias_gain=4.0):
    """1-bit 路线的位图决策：硬阈值，或 ED-FS 推理后处理。

    `target_cov` 是**目标墨覆盖率**，不是图像灰度：film 下灰度=膜厚=覆盖率，
    paper 下灰度=打印后亮度（1=白纸）故覆盖率=1-灰度。

    1-bit 没有「档位」维，`ed_fs` 需要 L=2 的 logits（[不落墨, 落墨]）。
    直接拿网络原始 logits 当先验即可：负值偏 0 档、正值偏 1 档，与
    硬阈值 `logits>0` 的决策边界一致，只是把逐像素独立决策换成串行误差扩散。

    PITFALLS #12/#13 的条纹退化解（行墨量 std 0.157）正是逐像素独立决策
    的产物；ed_fs 把量化误差扩散给后续像素，邻域被迫选不同值 => dithering
    是机制的必然结果，不靠损失鼓励（#29/#32 的同一结论）。
    """
    if not ed:
        return (logits > 0).float()
    from halftone.ed_fs import ed_fs
    lv = torch.tensor([0.0, 1.0], device=logits.device)
    z2 = torch.stack([torch.zeros_like(logits), logits], dim=-1)  # (B,1,H,W,2)
    return ed_fs(z2, target_cov, lv, bias_gain=bias_gain)[..., 1]


def triplet(img, ckpt, dpi=None, ed=False):
    """同一张图跑 模型/FS/GS 三条路线，返回 (inks, prints, target, ckpt_args)。

    ed=True 时模型的位图走 ED-FS 后处理（`onebit_ink`），基线不受影响。
    """
    model, printer, a, device = load_model_and_printer(ckpt, dpi=dpi)
    film = a.get("printer", "paper") == "film"
    target, base_img = film_target_and_base(img, film, device)
    size = (target.shape[3], target.shape[2])
    GS_WORKDIR.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        logits = model(target)
        # CMYK ckpt（out_ch=4）不能用 1-bit 决策，忽略 --ed
        use_ed = ed and a.get("out_ch", 1) == 1
        cov_tg = target if film else 1.0 - target
        inks = {"nn": onebit_ink(logits, cov_tg, ed=use_ed),
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
        logits = model(target)
        use_ed = args.ed and a.get("out_ch", 1) == 1
        cov_tg = target if a.get("printer") == "film" else 1.0 - target
        ink = onebit_ink(logits, cov_tg, ed=use_ed)[..., :h, :w]   # 裁回原尺寸
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
                                      args.ckpt, args.dpi, ed=args.ed)
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
        covs = gs_inkcov(f"{args.out}_{name}_bitmap.pdf")
        if all(c != c for c in covs):
            print(f"[inkcov] {name}: 失败 -> {path}")
            continue
        warn = " ⚠️ 覆盖率过高，实心填充风险" if max(covs) > 0.85 else ""
        print(f"[inkcov] {name}: K 覆盖率 {covs[3]:.1%} "
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
                                       args.ckpt, args.dpi, ed=args.ed)
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
                                      args.ckpt, args.dpi, ed=args.ed)
    print(f"灰度尺 {args.levels} 阶 @ {args.dpi or a.get('dpi')} dpi "
          f"(ckpt={args.ckpt}, {a.get('printer')})")
    print(f"\n全图 PSNR: 模型 {psnr(prints['nn'], target):.2f} | "
          f"FS {psnr(prints['fs'], target):.2f} | "
          f"GS {psnr(prints['gs'], target):.2f} dB")
    print("\n分段覆盖偏差（实际覆盖率 - 目标覆盖率，|x|>0.02 才有意义）")
    print(f"{'灰阶':>6} {'目标':>6} | {'模型':>6} {'偏差':>7} | {'FS':>6} {'偏差':>7} "
          f"| {'GS':>6} {'偏差':>7}")
    worst = (0.0, 0.0)
    _rows = []
    for k, v in enumerate(lv):
        seg = lambda t: t[0, 0, :, k * args.seg:(k + 1) * args.seg].mean().item()
        row = [seg(prints[m]) for m in ("nn", "fs", "gs")]
        _rows.append((v, row))
        d = [row[0] - v, row[1] - v, row[2] - v]
        if abs(d[0]) > abs(worst[1]):
            worst = (v, d[0])
        print(f"{v:6.3f} {v:6.3f} | {row[0]:6.3f} {d[0]:+7.3f} | {row[1]:6.3f} "
              f"{d[1]:+7.3f} | {row[2]:6.3f} {d[2]:+7.3f}")
    print(f"\n模型最大偏差: 灰阶 {worst[0]:.3f} 处 {worst[1]:+.3f}")
    # 形态判定：纯阶跃（中间一两级突变、两侧各跑一边）与 S 形（两端外鼓）
    # 是两种病，前者才是 #12 的硬阶跃根因（见 PITFALLS #13）。
    lv_d = [row[0] - v for v, row in _rows]
    step = (max(lv_d) > 0.2 and min(lv_d) < -0.2
            and abs(worst[1] - max(lv_d)) < 1e-9
            and (max(lv_d) + min(lv_d)) > -0.05)
    print(f"形态: {'阶跃（中段突变、两侧各跑一边 => 硬阈值断层）' if step else 'S 形（两端外鼓/中间内塌或反之）' if worst[1] else '无明显畸变'}")

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
    from halftone.data import RgbCrops
    paths = sorted(glob.glob(f"{args.dir}/*.png"))
    if not paths:
        raise SystemExit(f"{args.dir} 里没有图，先生成：uv run python gen_data.py --n 500")
    print(f"训练集统计：{args.dir}，{len(paths)} 张，按 size={args.size} 采样 "
          f"{args.n} 个裁块（模型实际看到的）")
    ds = RgbCrops(args.dir, size=args.size, length=args.n)
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

    # 判读：单调性 + 符号 + 对比度，三项都要过。
    # 旧的“最大灰阶差输出差 > 3x 最小灰阶差输出差”任何非线性网络都能过，
    # 反而放过了常数输入下输出恒 0（GroupNorm 通道内 std=0 直接抹掉）
    # 和倒 U 形（0.5 处输出最大）两类真异常。PITFALLS #12。
    means = [y.mean().item() for y in outs]
    d = [means[i + 1] - means[i] for i in range(len(means) - 1)]
    monotone = all(x > 0 for x in d) or all(x < 0 for x in d)
    signed = means[-1] - means[0] != 0.0
    contrast = pair[-1][1] > 1e-3
    checks = [("单调（输出差随灰阶单向变化）", monotone),
              ("有符号（最亮与最暗输出不同）", signed),
              ("有对比度（输出差 > 1e-3）", contrast)]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    ok = all(x for _, x in checks)
    if ok:
        verdict = "正常，灰阶信息保留"
    elif not contrast:
        verdict = "异常！不同灰阶输出几乎无差异，GroupNorm 抹掉了均值信息"
    elif not signed:
        verdict = "异常！输出无符号区分，网络对绝对灰阶不敏感"
    else:
        verdict = "异常！输出非单调，无法区分绝对灰阶大小"
    print(f"\n判读: {verdict}")
    print("注意：这是空间常数图探针，只查信息通道在不在；平坦区能否长出")
    print("抖动纹理要看 wedge 的行墨量 std（PITFALLS #12 的判据）。")


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


# ---------------------------------------------------------------- cmyk

def lp_blur(x, sig):
    """对 (1,C,H,W) 做 Gaussian 低通（逐通道），σ 单位 px。σ<=0 原样返回。"""
    if not sig or sig <= 0:
        return x
    r = max(1, int(3 * sig) + 1)
    xs = torch.arange(-r, r + 1, dtype=torch.float32, device=x.device)
    k = torch.exp(-xs ** 2 / (2 * sig * sig))
    k = k / k.sum()
    k2 = (k[:, None] @ k[None, :])[None, None]
    C = x.shape[1]
    return F.conv2d(x.reshape(-1, 1, x.shape[2], x.shape[3]), k2,
                    padding=k2.shape[-1] // 2).reshape(
                        x.shape[0], C, x.shape[2], x.shape[3])


def delta_e_visual(lab_a, lab_b, lp):
    """视觉口径 ΔE00：先低通再算（PITFALLS #44）。

    逐像素 ΔE00 对半色调是**错误口径**：半色调的全部设计就是把量化误差推到
    人眼不敏感的高频，逐像素比等于惩罚 dithering 本身。实测 8 张 photos_val
    σ 扫描（NN vs gs RIP）：σ=0 → NN +18.97 领先；σ=0.5 → +7.09；
    **σ=1 → +0.22（持平）；σ=2 → -0.25（RIP 略好）**；σ=3/4 → -0.20/-0.13。
    即「NN 碾压 RIP」完全是逐像素口径的产物，在眼睛/打印模糊的尺度上两者同级。
    """
    return delta_e_2000(lp_blur(lab_a, lp), lp_blur(lab_b, lp))


def cmd_cmyk(args):
    """CMYK 多档路线评测：硬域 Lab ΔE + 四通道 inkcov 审计 + RIP 基线对照。"""
    from halftone.data import RgbCrops
    model, printer, a, device = load_model_and_printer(args.ckpt)
    if not a.get("cmyk"):
        raise SystemExit(f"{args.ckpt} 不是 CMYK ckpt（args.cmyk != True）")
    icc = find_icc(a.get("icc"))
    intent = a.get("icc_intent", 0)
    lut256 = load_or_build_lut256(Path(args.ckpt).parent, icc, intent)
    files = sorted(Path(args.folder).glob("*.png")) + \
        sorted(Path(args.folder).glob("*.jpg"))
    if not files:
        raise SystemExit(f"{args.folder} 里没有图片")
    files = files[: args.limit]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"CMYK 评测：{len(files)} 张 | ckpt={args.ckpt} | ICC={icc} "
          f"intent={intent} | 档位={a.get('levels')}")
    print(f"{'图片':>12} {'NN ΔE':>7} {'RIP ΔE':>7} {'差':>6} | "
          f"{'NN K':>6} {'RIP K':>6} | {'TAC':>5} {'OOG%':>6}")
    rows = []
    for f in files:
        rgb = Image.open(f).convert("RGB")
        w, h = rgb.size
        pw, ph = (8 - w % 8) % 8, (8 - h % 8) % 8
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)
        t = F.pad(t, (0, pw, 0, ph), mode="reflect")
        # 补边后的 RGB（8bit），供 RIP 基线用——必须与 target 同几何
        padded_rgb = Image.fromarray(
            (t[0].permute(1, 2, 0).cpu().numpy() * 255.0).round()
            .clip(0, 255).astype(np.uint8), "RGB")
        lab_tg = srgb_to_lab(t.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        with torch.no_grad():
            logits = model(t)
            if args.ed:
                # ED-FS 推理后处理。dithering 是决策机制的属性，不是损失
                # 能鼓励的（#22/#24/#25/#26/#30/#33 全部证伪）。
                # ⚠ 但 ED 会把模型贡献洗成 0（#45）：ed_fs 吃 ICC 真值
                # target_cov，模型学的正是同一目标。开 ED 前后的 ΔE 不能
                # 算作网络能力。
                from halftone.ed_fs import ed_fs
                hard = ed_fs(logits, srgb_to_cov(
                    t.permute(0, 2, 3, 1), icc, intent
                    ).permute(0, 3, 1, 2), LEVELS.to(device))
            else:
                hard = hard_levels(logits / 0.01)
            cov = (hard * LEVELS.view(1, 1, 1, 4).to(device)).sum(-1)
            lab_nn = neugebauer_lab_2bit(hard, lut256)
            # 视觉口径（--lp 默认 2px）；--lp 0 时退化为逐像素
            de = delta_e_visual(lab_nn, lab_tg, args.lp)
            de95 = torch.quantile(de.flatten(), 0.95).item()
            de_px = delta_e_2000(lab_nn, lab_tg).mean().item()  # 仅供原始读数
        # RIP 基线：同 ICC 分离 + gs 半色调（1-bit 四通道）
        # ⚠ 必须喂**补边后**的图（PITFALLS #43）：直接 rip 原图再 interpolate
        # 拉伸会给 RIP 加 1.4% 非等比缩放，实测把 RIP 抬差 1.74 ΔE00
        # （30.22 vs 28.48，**逐像素口径**；视觉口径下抬高量更小但仍存在），
        # 是白送 NN 的分数。
        rip = rip_cmyk(padded_rgb, icc, intent) if args.rip else None
        if rip is not None:
            rip_t = torch.from_numpy(rip).unsqueeze(0).to(device)
            if rip_t.shape[-2:] != (t.shape[2], t.shape[3]):
                rip_t = F.interpolate(rip_t, size=(t.shape[2], t.shape[3]),
                                      mode="nearest")
            lab_rip = neugebauer_lab_2bit(
                one_hot_from_cov(rip_t), lut256)
            de_rip = delta_e_visual(lab_rip, lab_tg, args.lp).mean().item()
            rip_px = delta_e_2000(lab_rip, lab_tg).mean().item()
            rip_k = float(rip[3].mean())
        else:
            de_rip, rip_k = float("nan"), float("nan")
        # OOG：目标色是否在 CMY 四面体外（线性化后 XYZ 分量和 > 1）
        lin = t.permute(0, 2, 3, 1).clamp(0, 1)
        lin = torch.where(lin <= 0.04045, lin / 12.92,
                          ((lin + 0.055) / 1.055) ** 2.4)
        xyz = lin @ torch.tensor([[0.4124, 0.3576, 0.1805],
                                  [0.2126, 0.7152, 0.0722],
                                  [0.0193, 0.1192, 0.9505]],
                                 device=device)
        oog = (xyz.sum(-1) > 1.0).float().mean().item() * 100
        # 落盘四通道位图。字节语义为**墨量**（0=无墨=白，255=满墨）。
        # ⚠ 本环境(PIL 12.3 + gs 10.05.1)的 CMYK PDF 走 DCTDecode +
        # /Decode[1 0]，gs 对字节的解释与 #37 旧 PIL 时代的结论**相反**：
        # 写无墨量(255=无墨)会把白纸渲成全黑、inkcov 读满墨。字节语义
        # 必须逐环境用端点块实测（全墨渲黑/全白渲白），不能沿用结论
        # （PITFALLS #48）。本轮实测：墨量字节下 四通道单调、全墨0.000/
        # 全白1.000，正确。
        cov_np = cov[0].cpu().numpy()
        ink_bytes = np.round(cov_np * 255).astype(np.uint8)
        chans = [Image.fromarray(ink_bytes[c], "L") for c in range(4)]
        merged = Image.merge("CMYK", chans)
        pdf = outdir / f"{f.stem}_cmyk.pdf"
        merged.save(pdf, "PDF", resolution=72)
        # gs inkcov 量的是「有墨像素占比」，对任何非零平均覆盖率都饱和到
        # ~1.0（DCTDecode+Decode 反相链），不能当平均墨量读（PITFALLS #48）。
        # 张量自检值才是平均覆盖率，直接替换该列。
        gs_cov = float(cov_np.mean())
        tac = cov.sum(1).mean().item()
        rows.append({"de": de.mean().item(), "de95": de95,
                     "de_px": de_px, "gs_cov": gs_cov, "tac": tac,
                     "de_rip": de_rip, "rip_k": rip_k})
        print(f"{f.stem[:12]:>12} {de.mean().item():7.2f} {de_rip:7.2f} "
              f"{de_rip - de.mean().item():+6.2f} | "
              f"{cov_np[3].mean():6.3f} {rip_k:6.3f} | {tac:5.2f} {oog:6.1f}%")
        # 预览：目标 | 网络 ICC 合成 | RIP 合成 | NN C 覆盖 | NN K 覆盖 | RIP K 覆盖
        rgb_nn = lab_to_srgb(lab_nn[0].permute(1, 2, 0)).clamp(0, 1).cpu().numpy()
        H = min(arr.shape[0], rgb_nn.shape[0])
        W = min(arr.shape[1], rgb_nn.shape[1])
        gray = lambda ch: np.stack([(cov_np[ch][:H, :W] * 255).astype(np.uint8)] * 3, -1)
        panels = [(arr[:H, :W, :3] * 255).astype(np.uint8),
                  (rgb_nn[:H, :W] * 255).astype(np.uint8), gray(0), gray(3)]
        if rip is not None:
            rk = 1.0 - np.asarray(Image.open(
                Path("out/rip_gs") / "_o_(Black).tif")).astype(np.float32)
            panels.append(np.stack([(rk[:H, :W] * 255).astype(np.uint8)] * 3, -1))
        Image.fromarray(np.concatenate(panels, axis=1), "RGB").save(
            outdir / f"{f.stem}_cmyk.png")
    de_m = float(np.mean([r["de"] for r in rows]))
    de95_m = float(np.mean([r["de95"] for r in rows]))
    tac_m = float(np.mean([r["tac"] for r in rows]))
    px_m = float(np.mean([r["de_px"] for r in rows]))
    print(f"\n均值 ΔE00 {de_m:.2f}（σ={args.lp:g}px 视觉口径）| p95 {de95_m:.2f} "
          f"| TAC {tac_m:.2f} | 平均覆盖率 "
          f"{np.mean([r['gs_cov'] for r in rows]):.3f}")
    if args.lp and args.lp > 0:
        print(f"    逐像素口径（诊断用，勿作结论）ΔE00 {px_m:.2f}")
    if any(r["de_rip"] == r["de_rip"] for r in rows):
        rip_m = float(np.nanmean([r["de_rip"] for r in rows]))
        print(f"RIP 基线（同 ICC 分离 + gs 半色调）ΔE00 {rip_m:.2f} | "
              f"NN - RIP = {de_m - rip_m:+.2f}（负=NN 更好，同一视觉口径）")
        print("判读：ΔE00 差 <1 说明 NN 已达工业半色调水平。")
    print(f"位图/预览：{outdir}/<名>_cmyk.pdf / _cmyk.png "
          f"（PDF 行= C M Y K 四通道）")
    print("判读：ΔE00 < 3 可用；TAC 应 <= 3.2；OOG% 高说明训练集缺极端色相。")


def one_hot_from_cov(cov4):
    """连续覆盖率 (B,4,H,W) -> 4 档 one-hot (B,4,H,W,4)。

    对 RIP 基线而言这一步是**无损**的：`rip_cmyk` 返回的 cov 已经是
    {0,1} 二值（`np.unique(cov) == [0., 1.]`，2026-10-20 审核实测），
    `round(cov*3)` 是恒等映射，round-trip 最大误差 0.0。

    ⚠ 早前 docstring 说「RIP 位图携带网点大小的亚像素覆盖率信息，量化
    会丢掉它，所以 RIP ΔE 是保守下界」——**该论证不成立**（PITFALLS
    #46）。tiffsep1 在 -r72 下输出的就是逐像素 0/1，亚像素信息只存在于
    更高采样率（真机 1440dpi 打印时网点确实小于 72dpi 的一个像素），
    而我们这套评测全链路是 72dpi 单像素口径，两边都没有亚像素信息。
    ⇒ RIP ΔE00 5.77 是**公允值**，不是下界。"""
    lv = torch.round(cov4 * 3).clamp(0, 3).long()          # (B,4,H,W)
    p = torch.zeros(*cov4.shape, 4, device=cov4.device)
    p.scatter_(-1, lv.unsqueeze(-1), 1.0)
    return p


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def ckpt_arg(sp, default="runs/halftone/best.pt"):
        sp.add_argument("--ckpt", default=default)
        sp.add_argument("--dpi", type=int, default=None,
                        help="film 模式：按此 dpi 评估（默认 ckpt 里的）")
        sp.add_argument("--ed", action="store_true",
                        help="推理走 ED-FS 误差扩散后处理。1-bit：修 PITFALLS "
                             "#12/#13 条纹退化解（实测 14.80->24.67dB，仍差 FS "
                             "25.82 共 1.15，12/12 全输，见 #43/#45）；"
                             "CMYK：多样性提升（#35/#39，是否有利看权重）")
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

    k = ckpt_arg(sub.add_parser("cmyk", help="CMYK 多档：ΔE00 + 四通道 inkcov"))
    k.add_argument("folder", nargs="?", default="data/photos_val")
    k.add_argument("--limit", type=int, default=8)
    k.add_argument("--out", default="out/cmyk")
    k.add_argument("--rip", action="store_true",
                   help="加一列 ghostscript RIP 基线（同 ICC 分离 + tiffsep1 半色调）")
    k.add_argument("--lp", type=float, default=2.0,
                   help="低通 σ(px)，视觉口径 ΔE；0=关（逐像素，仅供诊断）。"
                        "半色调把误差推给人眼不敏感的高频，逐像素 ΔE 等于"
                        "把 dithering 当错误（PITFALLS #44）")
    # --ed 已由 ckpt_arg 提供（1-bit 与 CMYK 共用同一个开关）
    k.set_defaults(func=cmd_cmyk)

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