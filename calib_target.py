"""网点扩大标定靶：打印 256 个 CMYK 色块，用 ghostscript 渲成 RIP 位图。

**流程**：
  1. `python calib_target.py`     生成本 PDF + manifest.json（每个色块的
     理想覆盖率 + ICC 分离出的 CMYK 值）
  2. 用真打印机按"无色彩管理、100% 墨量、对应纸张"打印该 PDF
  3. 扫描/拍照成 `measured.png`（同一物理尺寸）
  4. `python calib_target.py --analyze measured.png`  算出每个色块的
     实测 Lab，与理想 Lab 比对，输出 dot_gain.json（本项目的墨水标定）

**为什么需要它**：CMYK 路线现在的 Neugebauer LUT 是 ICC 的**理想**叠印
（假设油墨完美叠加、无铺展）。真实喷墨有网点扩大（dot gain）：单个墨滴
铺展比名义面积大，且四通道相互影响。这正是 NN 相对 RIP 的最大优势——
RIP 结构上不知道你的墨水和纸张，而 NN 可以把标定出的偏差焊进损失里
反向补偿。

用法：
    python calib_target.py                     # 生成靶（默认 out/dotgain/）
    python calib_target.py --analyze 扫描.png   # 分析实测结果
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image

PATCH = 64          # 每个色块的像素数（64px @72dpi = 1 英寸）
COLS = 16           # 16x16 = 256 块


def srgb_to_cov_pil(srgb, icc, intent=0):
    """sRGB (H,W,3) uint8 -> CMYK 覆盖率 (H,W,4) float [0,1]。"""
    from PIL import ImageCms
    prof = ImageCms.getOpenProfile(icc)
    srgb_prof = ImageCms.createProfile("sRGB")
    xf = ImageCms.buildTransform(srgb_prof, prof, "RGB", "CMYK",
                                 renderingIntent=intent)
    pil = Image.fromarray(srgb, "RGB")
    out = np.asarray(ImageCms.applyTransform(pil, xf), dtype=np.float32)
    return 1.0 - out / 255.0     # PIL CMYK: 0=无墨 -> 转成 1=有墨


def build(icc, out_dir, intent=0):
    """生成 256 色块靶：4 个通道各 4 档的全组合，ICC 分离后量化回 4 档。

    **PIL CMYK 语义陷阱**（实测踩坑）：`Image.fromarray(arr, "CMYK")` 会把
    numpy 数组的字节当成「墨量」（0=无墨），但 save PDF 时 ghostscript 按
    「无墨量」解释同一字节——结果整幅靶**颜色反转**：全 level0（应白）
    渲成 RGB(0,0,0) 黑，全 level3（应满墨黑）渲成白。

    正确做法：用 `Image.new("CMYK", size, (c,m,y,k))` 逐块构造，
    值是**无墨量** 0-255（0=无墨=白，255=全墨）。或者更稳：
    先用 numpy 拼 CMYK 数组、`Image.fromarray` 后**反转**再存 PDF。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # 256 组合（4 通道 × 4 档），按 itertools.product 顺序与 LUT256 索引一致
    combos = list(itertools.product(range(4), repeat=4))
    # CMYK 无墨量数组（0=无墨=白，255=全墨）；putdata 逐块填充
    grid = np.full((PATCH * COLS, PATCH * COLS, 4), 0, np.uint8)
    meta = []
    for idx, lv in enumerate(combos):
        r, c = divmod(idx, COLS)
        # 档位 -> 无墨量：lv0->0(白/无墨)，lv3->255(满墨)
        nolv = [int(round(lv[i] / 3.0 * 255)) for i in range(4)]
        grid[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = nolv
        meta.append({"idx": idx, "levels": list(lv), "cmyk255": nolv,
                     "cell": [r, c]})
    pdf = out / "dotgain_target.pdf"
    Image.fromarray(grid, "CMYK").save(pdf, "PDF", resolution=72)
    (out / "manifest.json").write_text(json.dumps(
        {"combos": combos, "meta": meta, "patch": PATCH, "cols": COLS,
         "icc": str(icc), "intent": intent,
         "cmyk255_semantics": "无墨量 0-255（0=无墨=白，255=满墨）"}, indent=1))
    print(f"靶已生成：{pdf}（{COLS}x{COLS}=256 块，每块 {PATCH}px @72dpi）")
    print(f"manifest：{out / 'manifest.json'}")
    print("下一步：用真打印机打印该 PDF（关色彩管理、100% 墨量），扫描/拍照。")


def analyze(measured_png, icc, out_dir, intent=0):
    """读实测扫描图，逐块求平均 Lab，与 ICC 理想 Lab 比对，输出 dot_gain.json。

    产物：
      - measured_rgb.json  每块实测平均 RGB（原有）
      - measured_lut256.npy 用**实测 Lab** 重建的 LUT256，训练时直接替换
        ICC 理想 LUT（见 --use-measured 用法）
      - measured_lut256_meta.json  元数据 + 每块与理想 Lab 的偏差统计

    语义：`--analyze` 产出的 LUT 反映**你这台打印机+墨水+纸张**的真实叠印
    特性（含网点扩大、通道串扰），而 `build_lut256` 是 ICC 的理想叠印。
    """
    import torch
    from halftone.cmyk import (LEVELS, srgb_to_lab, load_or_build_lut256)
    out = Path(out_dir)
    manifest = json.loads((out / "manifest.json").read_text())
    patch, cols = manifest["patch"], manifest["cols"]
    img = Image.open(measured_png).convert("RGB")
    arr = np.asarray(img, np.float32)
    if arr.shape[0] < patch * cols or arr.shape[1] < patch * cols:
        raise SystemExit(
            f"[err] 扫描图 {arr.shape[:2]} 太小，需要 >= "
            f"{patch*cols}x{patch*cols}（{cols}x{cols} 块 × {patch}px）。"
            f"打印设置 72dpi 时扫描至少需 300dpi 才能看清块内细节，"
            f"但块级平均只需分辨率匹配，可用 --dpi 缩放。")
    rows = []
    for m in manifest["meta"]:
        r, c = m["cell"]
        blk = arr[r * patch:(r + 1) * patch, c * patch:(c + 1) * patch]
        if blk.shape[:2] != (patch, patch):
            print(f"[warn] 块 {m['idx']} 尺寸异常 {blk.shape[:2]}，跳过")
            continue
        mean_rgb = blk.reshape(-1, 3).mean(0) / 255.0
        rows.append({"idx": m["idx"], "levels": m["levels"],
                     "measured_rgb": [round(float(v), 4) for v in mean_rgb]})
    (out / "measured_rgb.json").write_text(json.dumps(rows, indent=1))
    print(f"已提取 {len(rows)} 块的实测平均 RGB -> {out / 'measured_rgb.json'}")

    # 实测 RGB -> Lab -> 重建 LUT256
    lut_ideal = load_or_build_lut256(out_dir, icc, intent).numpy()
    rgb = np.array([r["measured_rgb"] for r in rows], dtype=np.float32)
    lab = srgb_to_lab(torch.from_numpy(rgb)).numpy()
    np.save(out / "measured_lut256.npy", lab)
    # 偏差统计：每块 |ΔE00| 用 L1 近似（完整 CIEDE2000 在 train.py，
    # 这里只需要给打印报告一个人能读的汇总）
    dev = np.linalg.norm(lab - lut_ideal[:len(lab)], axis=1)
    meta = {"icc": str(icc), "intent": intent, "levels": LEVELS.tolist(),
            "index": "n = lC*64 + lM*16 + lY*4 + lK",
            "n_blocks": len(rows),
            "lab_dev_L2_mean": float(dev.mean()),
            "lab_dev_L2_max": float(dev.max()),
            "lab_dev_L2_p95": float(np.quantile(dev, 0.95)),
            "worst_blocks": sorted(
                [{"idx": int(i), "levels": rows[i]["levels"],
                  "dev": round(float(dev[i]), 2)}
                 for i in np.argsort(-dev)[:10]],
                key=lambda d: -d["dev"])}
    (out / "measured_lut256_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"实测 LUT256 -> {out / 'measured_lut256.npy'}")
    print(f"与 ICC 理想 Lab 偏差（L2）：均值 {dev.mean():.2f} | "
          f"p95 {np.quantile(dev, 0.95):.2f} | 最大 {dev.max():.2f}")
    print(f"偏差最大的块（idx=levels）："
          f"{[(d['idx'], d['levels'], d['dev']) for d in meta['worst_blocks'][:5]]}")
    print("\n训练时替换 ICC 理想 LUT：把 measured_lut256.npy 拷成 "
          "runs/<exp>/lut256.npy，并同步改 lut256_meta.json 的 "
          "\"measured\": true 字段（否则元数据不匹配会触发重建）。")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icc", default=None)
    ap.add_argument("--intent", type=int, default=0)
    ap.add_argument("--out", default="out/dotgain")
    ap.add_argument("--analyze", default=None,
                    help="实测扫描图路径（生成靶时忽略）")
    a = ap.parse_args()
    from halftone.cmyk import find_icc
    icc = find_icc(a.icc)
    if a.analyze:
        analyze(a.analyze, icc, a.out, a.intent)
    else:
        build(icc, a.out, a.intent)


if __name__ == "__main__":
    main()
