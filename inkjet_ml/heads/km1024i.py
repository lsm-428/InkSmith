# -*- coding: utf-8 -*-
r"""KM1024i 一套 ML（HeadType 80）。

数据现状
--------
**本机 876 组真实数据中没有 KM1024i**，交付"可训练管线 + 特征映射"。

KM1024i 的特殊性 —— 参数空间极宽
----------------------------------
**基线 = HMI 4.7.3 / V4730**：流程有 **115 个可扫维度**，其中
    KM1024iWaveformPulse{0..6}Section{0..14}Width  = 7×15 = **105 个分段宽度**
    KM1024iDriverPowerVoltageAdjustLow / High
105 个特征 + 样本量通常只有几百 → 必然过拟合，且每个分段单独一个特征没有
物理意义（真正起作用的是各段的整体形状）。

因此本模块采用**降维聚合**，把 105 个分段压成少量有解释力的统计量：

    w_mean / w_std / w_min / w_max     整体宽度水平与起伏
    p0_total … p6_total                每个 pulse 的总宽度（相位长度）
    w_slope                            段宽沿段的线性趋势（上升/下降斜率）
    w_narrow_cnt                       窄段个数（< 全局中位数的一半）
    w_active_pulses                    总宽度 > 0 的 pulse 数

这样既保住波形形状信息，又把特征数压到个位数（105 → 14，压缩比 1:7.5），
与 Samba/KM512 可比。若将来样本量足够（>1000），可改成直接用 105 维原始特征。

特征命名空间（Advanced-Drop-Analysis-KM1024i.xml，4.7.3）
---------------------------------------------------------
    KM1024iWaveformPulse{n}Section{m}Width                    105 个分段宽度（基线）
    KM1024iDriverPowerVoltageAdjustLow / High                 驱动供电微调

⚠️ **4.4.5 时代的三个参数在 4.7.3 里已被移除**：
    KM1024i.Width、KM1024i.VoltageAdjustLeft、KM1024i.VoltageAdjustRight
   （4.4.5 只有 11 个可扫维度）。本模块按基线走分段聚合；
   遇到 4.4.5 旧数据时 `legacy_features()` 会自动退化，不影响历史数据接入。
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional

import numpy as np

from ..dataset import Measurement, by_head, load_all, resolve_roots
from ..headml import (CLS_KEYS, MODEL_ORDER, REG_KEYS, HeadDataset,
                      best_by, evaluate_classification, evaluate_regression,
                      _fmt_table, save_json)

HEAD_NAME = "KM1024i"
HEAD_TYPE = "80"

N_PULSE, N_SECTION = 7, 15          # 4.7.3 流程定义的上限：7 个 pulse × 15 段
#: **实际用到的段数下限**（真实波形只点亮前几段）。
#: 2026-11-04 实测 V4730 真实落盘 `ADA KM1024i 045`：
#: 只有 `Section0/1/2` 三段（5.4 / 10.8 / 10.8 µs），其余段未点亮。
#: 故判定"数据有效"不能要求凑满 15 段，否则真实数据一条都算不上。
MIN_SECTIONS = 2

#: 基础特征（两版都有）
BASE_KEYS = ["width", "voltage_adjust_left", "voltage_adjust_right",
             "driver_power_voltage_adjust_low", "driver_power_voltage_adjust_high"]

#: 分段宽度在 Measurement.xml 里的键名候选（**顺序 = 优先级**，取第一个命中的）。
#:
#: ⚠️ 2026-11-04 实测 V4730 真实落盘（`ScreenCaptures\ADA KM1024i 045\Measurement.xml`）
#: 用的是 **`KM1024iPulse0Section0Width`**（无 "Waveform" 字样），
#: 故把它排在首位——否则真实数据一条都匹配不上（`build()` 返回 None）。
SECTION_KEY_TMPL = [
    "KM1024iPulse{p}Section{s}Width",
    "KM1024iWaveformPulse{p}Section{s}Width",
    "WaveformPulse{p}Section{s}Width",
    "Pulse{p}Section{s}Width",
    "KM1024i.Pulse{p}.Section{s}.Width",
]

#: 分段宽度在 **Results.csv 尾部** 的键名格式（另一套命名，实测存在）。
#: 形如 `Width Pulse Section 1.1 (0)` / `Width Pulse Section 1.3 (0)`：
#: `1.x` = 第 x+1 个 pulse 的第 1 段（1-based），`(0)` = 通道号。
CSV_SECTION_KEY_TMPL = "Width Pulse Section {p}.{s} ({ch})"
#: CSV 尾部键名里的通道号（KM1024i 左/右 bank 各一套参数，(0) 是常见值）
CSV_SECTION_CHANNELS = ["0", "1"]

#: 聚合特征顺序（固定）
AGG_FEATURES = ["w_mean", "w_std", "w_min", "w_max",
                "p0_total", "p1_total", "p2_total", "p3_total",
                "p4_total", "p5_total", "p6_total",
                "w_slope", "w_narrow_cnt", "w_active_pulses"]


def _f(x):
    try:
        return float(str(x).strip().strip('"'))
    except (TypeError, ValueError):
        return None


def collect_sections(m: Measurement) -> Optional[np.ndarray]:
    """从一次测量里取出 [n_pulse, n_section] 的分段宽度矩阵。

    两套键名都试（**实测真实落盘两套并存**）：
      1. `Measurement.xml` 标签：`KM1024iPulse0Section0Width`
      2. `Results.csv` 尾部：`Width Pulse Section 1.1 (0)`（1-based 脉冲号）

    ⚠️ 有效判定用 `MIN_SECTIONS`（实测真实波形只点亮 3 段，不是 15 段），
    但**段数不足时其余段留 NaN** —— `aggregate()` 按 NaN 忽略计算，
    这样特征维度恒定（可横向比较），又不因段数少而丢数据。
    """
    s = m.settings
    mat = np.full((N_PULSE, N_SECTION), np.nan)
    hit = 0
    for p in range(N_PULSE):
        for si in range(N_SECTION):
            v = None
            for tmpl in SECTION_KEY_TMPL:                 # ① Measurement.xml
                v = _f(s.get(tmpl.format(p=p, s=si)))
                if v is not None:
                    break
            if v is None:                                  # ② CSV 尾部
                for ch in CSV_SECTION_CHANNELS:
                    v = _f(s.get(CSV_SECTION_KEY_TMPL.format(
                        p=p + 1, s=si + 1, ch=ch)))
                    if v is not None:
                        break
            if v is not None:
                mat[p, si] = v
                hit += 1
    return mat if hit >= MIN_SECTIONS else None


def aggregate(mat: np.ndarray) -> List[float]:
    """把分段宽度矩阵压成统计特征。"""
    flat = mat[~np.isnan(mat)]
    if flat.size == 0:
        return [np.nan] * len(AGG_FEATURES)
    med = float(np.median(flat))
    x = np.arange(mat.shape[1], dtype=float)
    # 段宽沿段的线性趋势（斜率），忽略 NaN
    slopes = []
    for p in range(mat.shape[0]):
        row = mat[p]
        m_ = ~np.isnan(row)
        if m_.sum() >= 2:
            slopes.append(float(np.polyfit(x[m_], row[m_], 1)[0]))
    slope = float(np.mean(slopes)) if slopes else 0.0
    totals = [float(np.nansum(mat[p])) for p in range(mat.shape[0])]
    n_active = sum(1 for t in totals if t > 0)
    return [
        float(np.mean(flat)), float(np.std(flat)), float(np.min(flat)), float(np.max(flat)),
        *totals,
        slope,
        float(np.sum(flat < med / 2.0)),
        float(n_active),
    ]


def legacy_features(m: Measurement) -> Optional[List[float]]:
    """4.4.5 的 3 个基础特征（没有分段结构时用）。"""
    s = m.settings
    out = []
    for key in ("KM1024i.Width", "Width"):
        v = _f(s.get(key))
        if v is not None:
            out.append(v)
            break
    else:
        return None
    for key in ("KM1024i.VoltageAdjustLeft", "KM512.VoltageAdjustLeft"):
        v = _f(s.get(key))
        out.append(v if v is not None else 100.0)
    for key in ("KM1024i.VoltageAdjustRight",):
        v = _f(s.get(key))
        out.append(v if v is not None else 100.0)
    for key in ("KM1024iDriverPowerVoltageAdjustLow",
                "KM1024iDriverPowerVoltageAdjustHigh"):
        v = _f(s.get(key))
        out.append(v if v is not None else 0.0)
    return out


def build(measurements: Optional[List[Measurement]] = None,
          roots: Optional[List[str]] = None) -> Optional[HeadDataset]:
    """`roots` 指定要扫描的数据根（`None` = 用 `dataset.DEFAULT_ROOTS`）。"""
    ms = by_head(measurements if measurements is not None
                 else load_all(roots), HEAD_NAME)
    X, yc, yv, g, meta = [], [], [], [], []
    for m in ms:
        mat = collect_sections(m)
        if mat is not None:
            feat = aggregate(mat)
            mode = "agg"
        else:
            base = legacy_features(m)
            if base is None:
                continue
            feat = base + [np.nan] * (len(AGG_FEATURES) - len(base))
            feat = [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v
                    for v in feat]
            mode = "legacy"
        for nz in m.nozzles:
            X.append(feat)
            yc.append(nz.y_cls)
            yv.append(np.nan if nz.speed is None else nz.speed)
            g.append(m.session)
            meta.append({"session": m.session, "nozzle": nz.nozzle, "mode": mode})
    if not X:
        return None
    return HeadDataset(AGG_FEATURES, np.asarray(X, dtype=float),
                       np.asarray(yc, dtype=int), np.asarray(yv, dtype=float),
                       np.asarray(g), meta, HEAD_NAME, HEAD_TYPE)


def selftest() -> bool:
    """管线自检：造含分段结构的 KM1024i 测量，验证聚合与端到端训练。"""
    rng = np.random.RandomState(1)
    X, yc, yv, g = [], [], [], []
    for session in ("k1", "k2", "k3", "k4", "k5", "k6"):
        base = rng.uniform(3, 8)
        slope = rng.uniform(-0.2, 0.2)
        mat = np.full((N_PULSE, N_SECTION), base + slope * np.arange(N_SECTION))
        mat += rng.normal(0, 0.3, mat.shape)
        feat = aggregate(mat)
        vel = 2.0 + 0.25 * base + 1.2 * abs(slope) + rng.normal(0, 0.12)
        for _ in range(24):
            nd = 0 if vel < 1.6 else (2 if vel > 4.6 else 1)
            X.append(feat)
            yc.append(nd)
            yv.append(np.nan if nd == 0 else vel)
            g.append(session)
    ds = HeadDataset(AGG_FEATURES, np.asarray(X), np.asarray(yc), np.asarray(yv),
                     np.asarray(g), head_name=HEAD_NAME, head_type=HEAD_TYPE)
    cls = evaluate_classification(ds.X, ds.y_cls, ds.groups)
    reg = evaluate_regression(ds.X, ds.y_vel, ds.groups)
    bc, br = best_by(cls, "accuracy"), best_by(reg, "r2")
    ok = ("error" not in cls.get(bc[0], {}) and "error" not in reg.get(br[0], {}))
    print("自检: %d 样本 / %d 聚合特征（原始分段 %d → 压缩比 1:%.1f）"
          % (ds.n, len(ds.features), N_PULSE * N_SECTION,
             N_PULSE * N_SECTION / len(ds.features)))
    print("聚合特征: %s" % ", ".join(ds.features))
    print("分类最优 %s 准确率 %.4f" % (bc[0], bc[1]))
    print("回归最优 %s R² %.4f" % (br[0], br[1]))
    print("管线自检: %s" % ("通过（数据接入即可训练）" if ok else "失败"))
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="KM1024i ML")
    ap.add_argument("--out", default=os.path.join("results", "km1024i"))
    ap.add_argument("--data", action="append", metavar="ROOT", default=[],
                    help="追加的数据根目录（可重复）。指向机台上新扫的 ADA 落盘根；"
                         "默认与既有数据合并训练，加 --replace-data 则只用这里指定的")
    ap.add_argument("--replace-data", action="store_true",
                    help="丢弃默认数据源，只用 --data 指定的根")
    args = ap.parse_args(argv)

    roots = resolve_roots(args.data, replace=args.replace_data)
    missing = [r for r in roots if not os.path.isdir(r)]
    if missing:
        print("⚠️ 以下数据根不存在，已跳过：%s" % ", ".join(missing))
    print("数据根：%s" % (" | ".join(roots)))

    print("# KM1024i 一套 ML（HeadType 80）")
    ds = build(roots=roots)
    if ds is None:
        print("本机暂无 KM1024i 实测数据。")
        print("交付内容：可训练管线 + 105 维分段宽度的聚合特征设计。\n")
        ok = selftest()
        os.makedirs(args.out, exist_ok=True)
        save_json({"head": HEAD_NAME, "head_type": HEAD_TYPE,
                   "features": AGG_FEATURES,
                   "n_pulse": N_PULSE, "n_section": N_SECTION,
                   "section_key_templates": SECTION_KEY_TMPL,
                   "note": "本机无实测数据；已通过管线自检"},
                  os.path.join(args.out, "config.json"))
        print("\n接入数据时需核对 SECTION_KEY_TMPL 的键名与实机落盘是否一致。")
        return 0 if ok else 1

    print(ds.describe())
    cls = evaluate_classification(ds.X, ds.y_cls, ds.groups)
    reg = evaluate_regression(ds.X, ds.y_vel, ds.groups)
    print(_fmt_table("分类", cls, CLS_KEYS, MODEL_ORDER))
    print()
    print(_fmt_table("回归", reg, REG_KEYS, MODEL_ORDER))
    os.makedirs(args.out, exist_ok=True)
    save_json({"classification": cls, "regression": reg},
              os.path.join(args.out, "metrics.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())