# -*- coding: utf-8 -*-
"""ADA 实测数据的统一读取层 —— 三套喷头 ML 共用。

为什么需要一个统一层
--------------------
每个喷头在 ADA 里的参数命名空间完全不同（已逐个流程文件核实）：

    DMC Samba    DropAnalysis.DMC.Samba.Pulse{0..3}.{Voltage,Width,Tail} + Slewrate
                 + VoltageSupply                       → HeadType 1117
    KM512        DropAnalysis.KM512.{LowVoltage,HighVoltage,DropPeriod,
                 OnPulseWidth,OffPulseWidth,OnToOffDelay,PhaseLength,Grayscale}
                                                      → HeadType 3
    KM1024i      DropAnalysis.KM1024iWaveformPulse{n}Section{m}Width  (HMI 4.7.3 上
                 共 105 个分段宽度) + KM1024i.Width / VoltageAdjustLeft|Right
                 + KM1024iDriverPowerVoltageAdjustLow|High          → HeadType 80

所以**特征不能共用**，但"读 Results.csv / Measurement.xml"这段完全共用。
本模块只负责把一次 ADA 测量还原成结构化记录，具体特征映射交给各喷头模块。

标签口径（全部来自 ADA 自己落盘的读数，不引入任何测不到的物理量）
--------------------------------------------------------------
    y_cls  : 0 = 不喷(NrDroplets==0) / 1 = 单滴无卫星(==1) / 2 = 多滴含卫星(>=2)
    y_vel  : 第一个液滴的速度 [m/s]（无液滴时为 NaN）
    y_vol  : Total volume [pL]
    y_ang  : 第一个液滴的 Angle [deg]

数据来源
--------
    <根目录>\\**\\Results.csv      逐孔读数（含喷头元信息与波形快照）
    <根目录>\\**\\Measurement.xml  该次测量的真实设定值（最可靠）
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# HeadType → 喷头名（来自各 HeadAssy\<head>.xml 的 PrintHead.Type，
# 与 ADA 流程的 DropAnalysis.HeadType 一致）
HEADTYPE2NAME = {
    "1117": "DMC Samba", "1116": "DMC", "90": "Spectra Samba",
    "80": "KM1024i", "6": "KM204", "3": "KM512", "2": "Spectra S-class",
    "4": "OCE CP (CanonC29)", "5": "Trident",
}

DEFAULT_ROOTS = [r"C:\LP50\ScreenCaptures", r"D:\Gitee\LP50\AutoADA\share"]


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
def _f(x) -> Optional[float]:
    """ADA 用 '???' 表示无效值（如视野外的液滴），统一转 None。"""
    if x is None:
        return None
    s = str(x).strip().strip('"').strip()
    if s in ("", "???"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _read_csv_header_and_rows(path: str):
    """返回 (表头列表, 数据行列表)。数据行 = 首格是 .png 路径的行。"""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 3:
        return None, []
    rd = lambda s: next(csv.reader([s], delimiter=";", quotechar='"'), [])
    header = rd(lines[0])
    rows = []
    for line in lines[2:]:
        r = rd(line)
        if r and r[0] and ".png" in r[0]:
            rows.append(r)
    return header, rows


def _csv_tail_meta(path: str) -> Dict[str, str]:
    """Results.csv 尾部元信息区（Start time / IDPI Head Type / ...）。"""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    out: Dict[str, str] = {}
    for line in text.splitlines():
        cells = next(csv.reader([line], delimiter=";", quotechar='"'), [])
        if len(cells) >= 2:
            k = cells[0].strip().strip('"')
            if k and not k.endswith(".png") and k not in ("Filename",):
                out[k] = cells[1].strip().strip('"')
    return out


def _measurement(path: str) -> Dict[str, str]:
    """读 Measurement.xml 的真实设定值（比 CSV 尾部更完整）。"""
    import xml.etree.ElementTree as ET
    if not os.path.isfile(path):
        return {}
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return {}
    out: Dict[str, str] = {}
    for parent in list(root.iter()):
        for child in list(parent):
            tag = child.tag
            txt = (child.text or "").strip()
            if txt and tag not in out:
                out[tag] = txt
    return out


# --------------------------------------------------------------------------- #
# 记录
# --------------------------------------------------------------------------- #
@dataclass
class NozzleSample:
    """单孔结果。"""
    nozzle: Optional[int]
    n_droplets: int
    speed: Optional[float]
    volume: Optional[float]
    angle: Optional[float]
    pressure: Optional[float] = None
    head_temp: Optional[float] = None

    @property
    def y_cls(self) -> int:
        if self.n_droplets <= 0:
            return 0
        if self.n_droplets == 1:
            return 1
        return 2


@dataclass
class Measurement:
    """一次 ADA 测量 = 一个配方设置下的一批喷嘴结果。"""
    folder: str
    head_type: str
    head_name: str
    root: str
    session: str
    settings: Dict[str, Optional[float]] = field(default_factory=dict)
    nozzles: List[NozzleSample] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.nozzles)


def _setting(mm: Dict[str, str], *names) -> Optional[float]:
    """按优先级从 Measurement.xml 取设定值。"""
    for nm in names:
        v = _f(mm.get(nm))
        if v is not None:
            return v
    return None


def load_measurement(folder: str, root: str) -> Optional[Measurement]:
    """把一个含 Results.csv 的目录还原成 Measurement。"""
    csvp = os.path.join(folder, "Results.csv")
    header, rows = _read_csv_header_and_rows(csvp)
    if not header or not rows:
        return None

    idx = {h.strip(): i for i, h in enumerate(header)}
    i_nr = idx.get("Nr. of droplets")
    if i_nr is None:
        return None
    i_speed = None
    i_ang = None
    for key in ("Speed", "Angle"):
        for i, h in enumerate(header):
            if h.strip() == key:
                if key == "Speed" and i_speed is None:
                    i_speed = i
                if key == "Angle" and i_ang is None:
                    i_ang = i
                break
    i_vol = idx.get("Total volume")
    i_noz = idx.get("Nozzle")
    i_prs = idx.get("Ink Pressure")
    i_tmp = idx.get("Head temperature")

    g = lambda r, i: (_f(r[i]) if i is not None and i < len(r) else None)

    nozzles = [
        NozzleSample(
            nozzle=int(g(r, i_noz)) if g(r, i_noz) is not None else None,
            n_droplets=int(g(r, i_nr) or 0),
            speed=g(r, i_speed),
            volume=g(r, i_vol),
            angle=g(r, i_ang),
            pressure=g(r, i_prs),
            head_temp=g(r, i_tmp),
        )
        for r in rows
    ]

    meta = _csv_tail_meta(csvp)
    mm = _measurement(os.path.join(folder, "Measurement.xml"))
    ht = str(meta.get("IDPI Head Type") or mm.get("Printhead") or "?").strip()

    # 波形设定：优先 Measurement.xml（真实设定），回落到 CSV 尾部快照
    s: Dict[str, Optional[float]] = {
        "frequency": _setting(mm, "FiringFrequency"),
        "head_temperature": _setting(mm, "HeadTemperature"),
        "ink_pressure": _setting(mm, "InkPressure"),
        "led_delay": _setting(mm, "LEDDelay"),
        "voltage_supply": _setting(mm, "VoltageSupply", "High voltage B"),
        "slewrate": _setting(mm, "Slewrate"),
        "pulse0_voltage": _setting(mm, "Pulse0Voltage", "High voltage A"),
        "pulse0_width": _setting(mm, "Pulse0Width"),
        "pulse0_tail": _setting(mm, "Pulse0Tail"),
        "spectra_high_voltage": _setting(mm, "SpectraHighVoltage"),
        "standard_pulse": _setting(mm, "StandardPulse"),
        "rising": _setting(mm, "Rising"),
        "peak": _setting(mm, "Peak"),
        "falling": _setting(mm, "Falling"),
    }
    for k, v in s.items():
        if v is None:                       # CSV 尾部兜底
            src = {"frequency": "Firing Frequency",
                   "voltage_supply": "High voltage B",
                   "pulse0_voltage": "High voltage A",
                   "slewrate": "Slew Rate",
                   "pulse0_width": "Pulse 1 Width",
                   "pulse0_tail": "Pulse 1 Space"}.get(k)
            if src:
                s[k] = _f(meta.get(src))
        if s[k] is None and nozzles and nozzles[0].pressure is not None:
            if k == "ink_pressure":
                s[k] = nozzles[0].pressure
            if k == "head_temperature":
                s[k] = nozzles[0].head_temp

    return Measurement(
        folder=folder, head_type=ht,
        head_name=HEADTYPE2NAME.get(ht, meta.get("Headtype", "?")),
        root=root, session=os.path.relpath(folder, root),
        settings=s, nozzles=nozzles,
    )


def load_all(roots: Optional[List[str]] = None) -> List[Measurement]:
    """遍历所有含 Results.csv 的测量目录。"""
    out: List[Measurement] = []
    for root in (roots or DEFAULT_ROOTS):
        if not os.path.isdir(root):
            continue
        for dp, dn, fns in os.walk(root):
            if "Results.csv" in fns and any(f.startswith("noz") for f in fns):
                m = load_measurement(dp, root)
                if m and m.n:
                    out.append(m)
    return out


def by_head(measurements: List[Measurement], head: str) -> List[Measurement]:
    """筛出某个喷头的测量（head 传名字或 HeadType）。"""
    return [m for m in measurements
            if m.head_name == head or m.head_type == str(head)]


def summary(measurements: List[Measurement]) -> None:
    """打印按喷头分组的汇总。"""
    from collections import defaultdict
    g = defaultdict(list)
    for m in measurements:
        g[(m.head_type, m.head_name)].append(m)
    print("%-10s %-18s %6s %8s %8s %8s" %
          ("HeadType", "喷头", "会话", "样本", "不喷", "卫星"))
    print("-" * 64)
    for (ht, hn), ms in sorted(g.items()):
        tot = sum(m.n for m in ms)
        nj = sum(sum(1 for x in m.nozzles if x.y_cls == 0) for m in ms)
        ns = sum(sum(1 for x in m.nozzles if x.y_cls == 2) for m in ms)
        print("%-10s %-18s %6d %8d %8d %8d" % (ht, hn, len(ms), tot, nj, ns))