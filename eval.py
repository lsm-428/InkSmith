"""评测与推理入口（无 NN 主线，2026-10-21 起）：ED-FS / FS / GS 三路线对照 + 诊断。

推理（产出可打印位图）
  uv run python eval.py infer <图>                -> ED-FS 位图 + 模拟预览
  uv run python eval.py one   <图>                -> 单图三方对照 + inkcov 审计
  uv run python eval.py batch [文件夹]            -> 批量三方对照（默认 data/photos_val）

诊断（改引擎/配方前必跑）
  uv run python eval.py wedge                     -> 灰度尺分段偏差 + 条纹检测

物理参数是 CLI 参数（原从 ckpt 重建，NN 移除后恒用下述默认——已验证配方）：
  --printer film --sigma-um 25 --dpi 1440（σ=1.42px，#11 结晶下限之上）。

三条路线共用同一套物理语义：film 模型下灰度=相对膜厚，FS/GS 按"越暗墨越多"
加网，必须反相输入才能对齐（film_target_and_base 负责）。

引擎现状（photos_val 12 张，同一几何口径）：ED-FS 24.67 | FS 25.82 | GS 24.67。
ED-FS 仍差 FS 1.15——差距在机制（逐列并行把 2D 误差流压成 1D），不是缺网络
（模型贡献实测 +0.02dB，PITFALLS #45/#52）；引擎升级（蛇形/2D 核）是下一步。
"""
import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image

from printer import build_printer

GS_WORKDIR = Path("out/gs")


# ---------------------------------------------------------------- 底层工具

def psnr(a, b):
    return 10 * math.log10(1.0 / (F.mse_loss(a, b).item() + 1e-12))


def build_sim(args):
    """CLI 物理参数 -> FilmSimulator/PrinterSimulator（复用 printer.build_printer）。"""
    return build_printer(vars(args)).to(
        "cuda" if torch.cuda.is_available() else "cpu")


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


# ---------------------------------------------------------------- 主引擎：ED-FS

def ed_halftone(target_cov, device, bias_gain=4.0, drops=1):
    """ED-FS 逐列串行误差扩散（本项目主引擎，根目录 `ed_fs.py`）。

    `target_cov` 是**相对膜厚 ∈ [0,1]**（film 下 = 灰度；paper 下 = 1-灰度）。

    `drops`=N（N=1 即现役 1-bit）= **多档液滴 film**：档位表 lv=[0,1/N,..,1]，
    物理模型 = 压电喷头**波形分级变体积墨滴**（如 ⅓/⅔/全量滴）——one-pass 下
    同一像素只喷一次，"多液滴"只能以档位实现（多滴叠打=多 pass，已排除）。
    步进体积 = drop_pl/N ⇒ **绝对膜厚颗粒度随 N 变细**（这与 dpi 相反：dpi 管
    量程、档位管颗粒，PITFALLS #52）。若实测顶档是叠滴 regime，把 --drop-pl
    调大即可，ED 侧档位表通用（真机非线性档位直接传实测 lv 表——printer 的
    `ink` 就是 lv 加权和）。
    返回相对膜厚图 ∈ [0,1]（可直接喂 printer）。

    logits 恒零（无先验——模型先验实测为负资产，#45/#52），决策完全由 buf
    偏置引导；量化误差按 FS 权重扩散给后续列，权重和严格为 1（#42）。
    """
    from ed_fs import ed_fs
    lv = torch.arange(0, drops + 1, dtype=torch.float32, device=device) / drops
    z = torch.zeros(*target_cov.shape, drops + 1, device=device)
    out = ed_fs(z, target_cov, lv, bias_gain=bias_gain)
    return (out * lv.view(1, 1, 1, -1)).sum(-1)     # one-hot -> 相对膜厚图
    # drops=1 时 lv=[0,1]，与旧实现 out[...,1] 逐位等价（回归安全）


def tmax_um(dpi, drop_pl=6.0):
    """顶档满覆盖膜厚：T = V_top/p²，V_top=drop_pl×1e3 µm³，p=25400/dpi µm。
    与档位数无关——顶档体积定标在 drop_pl 上（6pL 全量滴）。"""
    return drop_pl * 1e3 * (dpi / 25400.0) ** 2


def thickness_rms_um(printed01, target01, dpi, tmax, win_um=100.0):
    """物理尺度均匀性（多档 film 的验收指标，AGENTS 路线图）：
    win_um 滑窗均值偏差 RMS，单位 µm。场为相对膜厚 [0,1]，×tmax 换算绝对厚度。"""
    w = max(2, round(win_um * dpi / 25400.0))
    a, b = w // 2, w - 1 - w // 2          # pad 后池化尺寸与原图一致
    ma = F.avg_pool2d(F.pad(printed01, (a, b, a, b), mode="reflect"), w)
    mt = F.avg_pool2d(F.pad(target01, (a, b, a, b), mode="reflect"), w)
    return math.sqrt(F.mse_loss(ma, mt).item()) * tmax


# ---------------------------------------------------------------- 共用

def triplet(img, args, device):
    """同一张图跑 ED/FS/GS 三条路线，返回 (inks, prints, target)。

    drops>1 时只有 ED 路线吃多档（FS/GS 是 1-bit 工业基线，保持单档口径
    ——这正是"多档 film 相对 1-bit 的增量"要在同一 PSNR/RMS 尺上显出来的原因）。
    """
    film = args.printer == "film"
    target, base_img = film_target_and_base(img, film, device)
    size = (target.shape[3], target.shape[2])
    printer = build_sim(args)
    GS_WORKDIR.mkdir(parents=True, exist_ok=True)
    cov_tg = target if film else 1.0 - target
    with torch.no_grad():
        inks = {"ed": ed_halftone(cov_tg, device, args.bias_gain, args.drops),
                "fs": floyd_steinberg(base_img, size).to(device),
                "gs": gs_halftone(base_img, size, GS_WORKDIR).to(device)}
        prints = {k: printer(v) for k, v in inks.items()}
    return inks, prints, target


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

def ink_bitmap(ink_rel):
    """相对膜厚图（档位图 ∈[0,1]）-> 打印位图，口径**墨=黑/0**：
    像素 = 255×(1−档值)。drops=1 退化为旧式 (1-ink)*255，逐位一致。"""
    return np.uint8(np.clip((1 - ink_rel[0, 0].cpu().numpy()) * 255, 0, 255))


def cmd_infer(args):
    """单图推理：ED-FS 出打印位图 + 模拟预览。drops>1 时位图是体积档位图。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    printer = build_sim(args)
    img = Image.open(args.input).convert("L")
    w, h = img.size
    film = args.printer == "film"
    target, _ = film_target_and_base(img, film, device)
    cov_tg = target if film else 1.0 - target
    with torch.no_grad():
        ink = ed_halftone(cov_tg, device, args.bias_gain, args.drops)[..., :h, :w]
        printed = printer(ink)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # 墨=黑（0=墨）：与 cmd_one/webapp 落盘口径一致，可直接喂 RIP/inkcov（#52）
    bitmap = ink_bitmap(ink)
    preview = (printed[0, 0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(bitmap, mode="L").save(f"{args.out}_bitmap.png")
    Image.fromarray(preview, mode="L").save(f"{args.out}_preview.png")
    print(f"打印位图: {args.out}_bitmap.png（墨=黑/0，{args.drops+1} 档体积级，直接可打）")
    print(f"模拟打印: {args.out}_preview.png（经铺展模拟的膜厚效果）")


# ---------------------------------------------------------------- one

def cmd_one(args):
    """单图三方对照 + inkcov 审计 + 七行对比图。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inks, prints, target = triplet(Image.open(args.input).convert("L"), args, device)
    if args.printer == "film":
        print(f"[film] 膜基模式：模拟输出=相对膜厚，基线用反相图加网"
              f"（ED={args.drops+1} 档体积级，FS/GS 保持 1-bit 口径）")
    outdir = Path(args.out).parent
    outdir.mkdir(parents=True, exist_ok=True)
    p = {k: psnr(v, target) for k, v in prints.items()}
    print(f"模拟打印 PSNR：ED-FS {p['ed']:.2f} | FS {p['fs']:.2f} | "
          f"GS筛网 {p['gs']:.2f} dB | ED vs FS {p['ed']-p['fs']:+.2f} | "
          f"ED vs GS {p['ed']-p['gs']:+.2f}")

    # 落盘位图（墨=黑）并用 ghostscript inkcov 独立审计墨覆盖率
    # inkcov 读 L 灰度 PDF 报告的是平均色覆盖率 = 相对膜厚（多滴档位图同样可比）
    for name, ink in inks.items():
        path = f"{args.out}_{name}_bitmap.png"
        pil = Image.fromarray(ink_bitmap(ink),
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

    grid = torch.cat([target, inks["ed"], prints["ed"], inks["fs"], prints["fs"],
                      inks["gs"], prints["gs"]])
    save_image(grid, f"{args.out}.png", nrow=1)
    print(f"对比图：{args.out}.png（原图|ED位图|ED打印|FS位图|FS打印|GS位图|GS打印）")


# ---------------------------------------------------------------- batch

def cmd_batch(args):
    """真实照片文件夹批量三方对照，末尾给均值。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    folder = Path(args.folder)
    files = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
    if not files:
        raise SystemExit(f"{folder} 里没有图片")
    tmax = tmax_um(args.dpi, args.drop_pl) if args.printer == "film" else 0.0
    print(f"{'照片':>12} {'ED-FS':>7} {'FS':>7} {'GS':>7} {'vs FS':>7} {'vs GS':>7} "
          f"{'ED@100µm RMS':>12}")
    rows, rmss = [], []
    for f in files:
        inks, prints, target = triplet(Image.open(f).convert("L"), args, device)
        if args.printer == "film" and rows == []:
            print(f"[film] 膜基模式：模拟输出=相对膜厚，基线用反相图加网"
                  f"（ED={args.drops+1} 档，T_max={tmax:.1f}µm；FS/GS 1-bit 口径）")
        r = [psnr(prints[m], target) for m in ("ed", "fs", "gs")]
        rms = (thickness_rms_um(prints["ed"], target, args.dpi, tmax)
               if args.printer == "film" else float("nan"))
        rows.append(r); rmss.append(rms)
        print(f"{f.stem[:12]:>12} {r[0]:7.2f} {r[1]:7.2f} {r[2]:7.2f} "
              f"{r[0]-r[1]:+7.2f} {r[0]-r[2]:+7.2f} {rms:9.2f}µm")
    m = np.array(rows).mean(axis=0)
    print(f"{'均值':>12} {m[0]:7.2f} {m[1]:7.2f} {m[2]:7.2f} "
          f"{m[0]-m[1]:+7.2f} {m[0]-m[2]:+7.2f} {np.mean(rmss):9.2f}µm")


# ---------------------------------------------------------------- wedge

def cmd_wedge(args):
    """灰度尺：分段覆盖偏差 + 位图纹理质量。查灰阶 S 形畸变与条纹退化解。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lv = np.arange(args.levels) / (args.levels - 1)
    arr = np.concatenate([np.full((args.seg, args.seg), v, np.float32) for v in lv],
                         axis=1)
    inks, prints, target = triplet(
        Image.fromarray((arr * 255).astype("uint8"), "L"), args, device)
    sigma_px = args.sigma_um * args.dpi / 25400.0 if args.printer == "film" \
        else args.sigma
    tmax = tmax_um(args.dpi, args.drop_pl) if args.printer == "film" else 0.0
    print(f"灰度尺 {args.levels} 阶 @ {args.dpi} dpi（{args.printer} "
          f"σ={sigma_px:.2f}px，ED={args.drops+1} 档体积级，"
          f"T_max={tmax:.1f}µm）")
    rms = {m: (thickness_rms_um(prints[m], target, args.dpi, tmax)
               if args.printer == "film" else float("nan"))
           for m in ("ed", "fs", "gs")}
    print(f"\n全图 PSNR: ED-FS {psnr(prints['ed'], target):.2f} | "
          f"FS {psnr(prints['fs'], target):.2f} | "
          f"GS {psnr(prints['gs'], target):.2f} dB | "
          f"100µm 膜厚RMS: ED {rms['ed']:.2f} | FS {rms['fs']:.2f} | "
          f"GS {rms['gs']:.2f} µm")
    print("\n分段覆盖偏差（实际覆盖率 - 目标覆盖率，|x|>0.02 才有意义）")
    print(f"{'灰阶':>6} {'目标':>6} | {'ED':>6} {'偏差':>7} | {'FS':>6} {'偏差':>7} "
          f"| {'GS':>6} {'偏差':>7}")
    worst = (0.0, 0.0)
    _rows = []
    for k, v in enumerate(lv):
        seg = lambda t: t[0, 0, :, k * args.seg:(k + 1) * args.seg].mean().item()
        row = [seg(prints[m]) for m in ("ed", "fs", "gs")]
        _rows.append((v, row))
        d = [row[0] - v, row[1] - v, row[2] - v]
        if abs(d[0]) > abs(worst[1]):
            worst = (v, d[0])
        print(f"{v:6.3f} {v:6.3f} | {row[0]:6.3f} {d[0]:+7.3f} | {row[1]:6.3f} "
              f"{d[1]:+7.3f} | {row[2]:6.3f} {d[2]:+7.3f}")
    print(f"\nED 最大偏差: 灰阶 {worst[0]:.3f} 处 {worst[1]:+.3f}")
    # 形态判定：纯阶跃（中间一两级突变、两侧各跑一边）与 S 形（两端外鼓）
    # 是两种病，前者才是硬阈值断层（见 PITFALLS #12/#13）。
    lv_d = [row[0] - v for v, row in _rows]
    step = (max(lv_d) > 0.2 and min(lv_d) < -0.2
            and abs(worst[1] - max(lv_d)) < 1e-9
            and (max(lv_d) + min(lv_d)) > -0.05)
    print(f"形态: {'阶跃（中段突变、两侧各跑一边 => 硬阈值断层）' if step else 'S 形（两端外鼓/中间内塌或反之）' if worst[1] else '无明显畸变'}")

    print("\n位图纹理质量（像素级抖动应各项方差≈0）")
    for m in ("ed", "fs", "gs"):
        print_texture(m, texture_stats(inks[m]))
    t = texture_stats(inks["ed"])
    if t["row_std"] > 0.05:
        print(f"  ⚠ ED行墨量std {t['row_std']:.3f} 过高 => 用整行条纹表达灰阶，"
              f"这是覆盖率失真的主因（见 PITFALLS #12）")


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # 物理参数（原随 ckpt 存权重里；NN 移除后为 CLI 参数，默认=已验证配方）
    phys = argparse.ArgumentParser(add_help=False)
    phys.add_argument("--printer", choices=["paper", "film"], default="film",
                      help="paper=吸收基材(exp饱和)；film=非吸收膜基(线性体积)，主线")
    phys.add_argument("--sigma-um", type=float, default=25.0,
                      help="film：铺展高斯 σ（µm）；25=1.42px@1440dpi（#11 铺展下限）")
    phys.add_argument("--dpi", type=int, default=1440, help="film：打印分辨率")
    phys.add_argument("--drops", type=int, default=1,
                      help="ED 引擎体积档位数 N（档位表 [0,1/N..1]×顶档体积）；"
                           "N=1 现役 1-bit，N>1 多档 film（细颗粒度，#52）。"
                           "FS/GS 基线恒 1-bit 口径")
    phys.add_argument("--drop-pl", type=float, default=6.0,
                      help="顶档单滴体积 pL（µm³=pL×1000），T_max 换算用；"
                           "KM1024iSHE 全量滴=6；多档波形分级顶档实测值替换")
    phys.add_argument("--sigma", type=float, default=1.0,
                      help="paper：墨滴扩散 σ（像素）")
    phys.add_argument("--gain", type=float, default=1.5,
                      help="paper：网点扩大系数")
    phys.add_argument("--bias-gain", type=float, default=4.0,
                      help="ED 偏置增益（|buf-档位| 的权重，越大越贴覆盖率）")

    i = sub.add_parser("infer", parents=[phys],
                       help="单图推理 -> ED-FS 打印位图 + 预览")
    i.add_argument("input")
    i.add_argument("--out", default="out/photo")
    i.set_defaults(func=cmd_infer)

    o = sub.add_parser("one", parents=[phys],
                       help="单图三方对照 + inkcov 审计")
    o.add_argument("input")
    o.add_argument("--out", default="out/verify")
    o.set_defaults(func=cmd_one)

    b = sub.add_parser("batch", parents=[phys], help="批量三方对照")
    b.add_argument("folder", nargs="?", default="data/photos_val")
    b.set_defaults(func=cmd_batch)

    w = sub.add_parser("wedge", parents=[phys],
                       help="灰度尺分段偏差 + 条纹检测")
    w.add_argument("--levels", type=int, default=16)
    w.add_argument("--seg", type=int, default=128, help="每阶块边长")
    w.set_defaults(func=cmd_wedge)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
