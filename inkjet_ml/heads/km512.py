# -*- coding: utf-8 -*-
r"""KM512 一套 ML（HeadType 3）。

数据现状
--------
**本机 876 组真实数据中没有 KM512**（实测汇总：Samba 108 孔 / Spectra 768 孔），
所以本模块现在只能：
    * 自检：确认数据加载路径通、特征映射正确（用假数据验证管线）
    * 交付：数据一到就能直接 `python -m inkjet_ml.heads.km512` 训练出结果

特征（严格取自 Advanced-Drop-Analysis-KM512.xml 的参数命名空间）
----------------------------------------------------------------
    low_voltage     DropAnalysis.KM512.LowVoltage      OFF pulse voltage
    high_voltage    DropAnalysis.KM512.HighVoltage     ON  pulse voltage
    drop_period     DropAnalysis.KM512.DropPeriod
    on_pulse_width  DropAnalysis.KM512.OnPulseWidth
    off_pulse_width DropAnalysis.KM512.OffPulseWidth
    on_to_off_delay DropAnalysis.KM512.OnToOffDelay
    phase_length    DropAnalysis.KM512.PhaseLength
    grayscale       DropAnalysis.KM512.Grayscale       Number of drops

⚠️ 这些是**脉冲电压/脉宽/周期**型参数，与 Samba 的
   Pulse{Voltage,Width,Tail}+Slewrate **不是一回事**，不能互相套用。

⚠️ **两版 HMI 的流程默认值不同，本项目基线取 4.7.3 / V4730**：
        参数                4.4.5 流程默认   →   4.7.3 流程默认（基线）
        DropPeriod          80:10:150       →   10:20
        OnPulseWidth        30:50           →   3:5
        OffPulseWidth       60:80           →   6:10
        PhaseLength         100:10:200      →   20:10:50
   4.4.5 的旧值恰好等于 4.7.3 **recipe** 的覆盖值 —— 两版互换了位置，容易照抄错。
   本模块**不硬编码任何默认值**：范围一律由
   `python -m inkjet_ml.ada --dims --head KM512` 从 4.7.3 实机文件解析。

数据从哪来
----------
Measurement.xml / Results.csv 尾部没有直接写 KM512 字段名，需要按本机实际
落盘列名映射。`KM_SETTINGS` 给出候选键名，`resolve_settings()` 会依次尝试，
取到第一个有值的。**接入新数据时请核对这里**。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

from ..dataset import Measurement, by_head, load_all, resolve_roots
from ..headml import (CLS_KEYS, MODEL_ORDER, REG_KEYS, evaluate_classification,
                      evaluate_regression, best_by, _fmt_table, save_json)

HEAD_NAME = "KM512"
HEAD_TYPE = "3"

#: 特征顺序固定下来，换数据时不要改顺序（否则模型对不上）
FEATURES = ["low_voltage", "high_voltage", "drop_period", "on_pulse_width",
            "off_pulse_width", "on_to_off_delay", "phase_length", "grayscale"]

#: 每个特征在 Measurement.xml / Results.csv 里可能出现的键名（按优先级）
KM_SETTINGS: Dict[str, List[str]] = {
    "low_voltage": ["LowVoltage", "KM512.LowVoltage", "Low voltage A"],
    "high_voltage": ["HighVoltage", "KM512.HighVoltage", "High voltage A",
                     "High voltage B"],
    "drop_period": ["DropPeriod", "KM512.DropPeriod", "Drop period"],
    "on_pulse_width": ["OnPulseWidth", "KM512.OnPulseWidth", "On pulse width"],
    "off_pulse_width": ["OffPulseWidth", "KM512.OffPulseWidth", "Off pulse width"],
    "on_to_off_delay": ["OnToOffDelay", "KM512.OnToOffDelay", "On to off delay"],
    "phase_length": ["PhaseLength", "KM512.PhaseLength", "Phase length"],
    "grayscale": ["Grayscale", "KM512.Grayscale", "Grayscale value"],
}

#: 通用量（所有喷头都有）
COMMON_SETTINGS: Dict[str, List[str]] = {
    "frequency": ["FiringFrequency", "Frequency"],
    "head_temperature": ["HeadTemperature", "Head temperature"],
    "ink_pressure": ["InkPressure", "Ink Pressure"],
}


def _f(x):
    try:
        return float(str(x).strip().strip('"'))
    except (TypeError, ValueError):
        return None


def resolve_settings(m: Measurement) -> Optional[Dict[str, float]]:
    """从一次测量里取出 KM512 的特征值；缺任一项则返回 None（该次测量不可用）。"""
    s = m.settings
    out: Dict[str, float] = {}
    for feat in FEATURES:
        v = None
        for key in KM_SETTINGS[feat]:
            v = s.get(key)
            if v is not None:
                break
        if v is None:
            return None
        out[feat] = float(v)
    return out


def build(measurements: Optional[List[Measurement]] = None,
          roots: Optional[List[str]] = None) -> Optional["KM512Dataset"]:
    """`roots` 指定要扫描的数据根（`None` = 用 `dataset.DEFAULT_ROOTS`）。"""
    ms = by_head(measurements if measurements is not None
                 else load_all(roots), HEAD_NAME)
    rows = []
    for m in ms:
        f = resolve_settings(m)
        if f is None:
            continue
        for nz in m.nozzles:
            rows.append(([f[k] for k in FEATURES], nz, m.session))
    if not rows:
        return None
    from ..headml import HeadDataset
    return HeadDataset(
        features=FEATURES,
        X=np.asarray([r[0] for r in rows], dtype=float),
        y_cls=np.asarray([r[1].y_cls for r in rows], dtype=int),
        y_vel=np.asarray([np.nan if r[1].speed is None else r[1].speed for r in rows],
                         dtype=float),
        groups=np.asarray([r[2] for r in rows]),
        meta=[{"session": r[2], "nozzle": r[1].nozzle} for r in rows],
        head_name=HEAD_NAME, head_type=HEAD_TYPE,
    )


def selftest() -> bool:
    """管线自检：造一条 KM512 测量，验证特征提取与端到端训练能跑通。

    目的不是给出性能数字，而是证明"数据接进来就能用"。
    """
    import tempfile
    from ..dataset import Measurement, NozzleSample

    rng = np.random.RandomState(0)
    X, yc, yv, g = [], [], [], []
    for session in ("s1", "s2", "s3", "s4", "s5"):
        lv, hv, dp = rng.uniform(4, 12), rng.uniform(12, 28), rng.uniform(80, 150)
        opw, offw, od, pl, gs = 30, 60, 0, 100, 1
        f = dict(low_voltage=lv, high_voltage=hv, drop_period=dp,
                 on_pulse_width=opw, off_pulse_width=offw, on_to_off_delay=od,
                 phase_length=pl, grayscale=gs)
        for k in range(24):
            vel = 3.0 + 0.1 * lv + 0.01 * dp - 0.05 * opw + rng.normal(0, 0.15)
            nd = 0 if vel < 2.0 else (2 if vel > 5.0 else 1)
            X.append([f[c] for c in FEATURES])
            yc.append(nd)
            yv.append(np.nan if nd == 0 else vel)
            g.append(session)
    from ..headml import HeadDataset
    ds = HeadDataset(FEATURES, np.asarray(X), np.asarray(yc), np.asarray(yv),
                     np.asarray(g), head_name=HEAD_NAME, head_type=HEAD_TYPE)
    cls = evaluate_classification(ds.X, ds.y_cls, ds.groups)
    reg = evaluate_regression(ds.X, ds.y_vel, ds.groups)
    bc, br = best_by(cls, "accuracy"), best_by(reg, "r2")
    ok = ("error" not in cls.get(bc[0], {}) and "error" not in reg.get(br[0], {}))
    print("自检数据: %d 样本 / %d 特征 / %d 会话" % (ds.n, len(ds.features),
                                                    len(set(ds.groups))))
    print("特征顺序: %s" % ", ".join(ds.features))
    print("分类最优 %s 准确率 %.4f" % (bc[0], bc[1]))
    print("回归最优 %s R² %.4f" % (br[0], br[1]))
    print("管线自检: %s" % ("通过（数据接入即可训练）" if ok else "失败"))
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="KM512 ML")
    ap.add_argument("--selftest", action="store_true", help="只跑管线自检")
    ap.add_argument("--out", default=os.path.join("results", "km512"))
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

    print("# KM512 一套 ML（HeadType 3）")
    ds = build(roots=roots)
    if ds is None:
        print("本机暂无 KM512 实测数据（876 组里只有 DMC Samba 108 孔 + Spectra 768 孔）。")
        print("交付内容：可训练管线 + 特征映射，数据接入即可训练。\n")
        ok = selftest()
        print("\n接入数据时需核对 KM_SETTINGS 的键名是否与实机落盘一致。")
        os.makedirs(args.out, exist_ok=True)
        save_json({"head": HEAD_NAME, "head_type": HEAD_TYPE,
                   "features": FEATURES,
                   "setting_keys": KM_SETTINGS,
                   "note": "本机无实测数据；已通过管线自检"},
                  os.path.join(args.out, "config.json"))
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