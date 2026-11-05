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

    扫描根见 `DEFAULT_ROOTS`。
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

#: 实测数据根目录。**只放本项目认可的资产**——
#: `ML\share`（DMC Samba 108 孔 / 9 会话，§4.1）。
#: 旧机台 `C:\LP50\ScreenCaptures` 的 768 组已从本项目的记录中移除：
#: 那批是 Spectra S-class（HeadType 2）数据，与本项目的三个喷头（1117/3/80）无关，
#: 且其 28 例卫星滴不属于 Samba 口径（§4.1 的 Samba 卫星滴仍为 0 例）。
#: 磁盘上的原始文件保持原样，不在本模块的扫描范围内。
#:
#: ⚠️ **实测数据不入库**（`.gitignore` 排除 `share/`），故这里是**候选列表**：
#: ① 本机绝对路径；② 仓库内相对路径（`inkjet_ml/` 的上一级）。
#: `load_all()` 会遍历**所有存在**的候选——两者指向同一份数据时不会重复加载
#: （同一文件只会被 walk 一次，因路径不同故按 root 分别计入，见 `load_all` 的 multi 判定）。
#: 都找不到时训练报"样本不足"，这是预期：**实测数据必须自行采集**。
DEFAULT_ROOTS = [
    r"D:\Gitee\LP50\ML\share",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "share"),
]


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
        "slewrate": _setting(mm, "PulseSlewRate", "Slewrate"),
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

    # ---- 保留**全部**原始键（Measurement.xml 标签 + CSV 尾部条目）----
    # 各 head 的特征键名五花八门（KM1024i 是 `KM1024iPulse0Section0Width`
    # 这种分段宽度、Spectra 是 `SpectraRising`…），
    # 若只保留上面 14 个固定键，各 head 模块就取不到自己的参数
    # （表现为 `build()` 返回 None / 数据被静默跳过）。
    # 这里把原始键**原样**并入 settings（数值化，保留原标签名），
    # 固定键在上层仍可覆盖同名项。
    for k, v in list(mm.items()):
        if k not in s:
            fv = _f(v)
            if fv is not None:
                s[k] = fv
    for k, v in list(meta.items()):
        if k not in s:
            fv = _f(v)
            if fv is not None:
                s[k] = fv

    return Measurement(
        folder=folder, head_type=ht,
        head_name=HEADTYPE2NAME.get(ht, meta.get("Headtype", "?")),
        root=root, session=os.path.relpath(folder, root),
        settings=s, nozzles=nozzles,
    )


def load_all(roots: Optional[List[str]] = None) -> List[Measurement]:
    """遍历所有含 Results.csv 的测量目录。

    `roots` 省略时用 `DEFAULT_ROOTS`；给了就**只**用这些根（不合并默认值）。

    ⚠️ **多根时的 session 名冲突**：session = 测量目录相对 root 的路径。
    两个根下若有同名目录（如两个 root 各有一个 `NOA89-test-01`），
    session 会撞名 → 分组交叉验证会把**不同来源的数据当成同一组**，
    导致交叉验证分组泄漏、指标虚高。
    故多根时给 session 加**根目录名前缀**（`_root_tag`），保证全局唯一。

    ⚠️ **同一目录被多个根覆盖时不得重复计入**：`DEFAULT_ROOTS` 里
    「本机绝对路径」与「仓库相对路径」在开发机上指向**同一份数据**，
    若按 root 分别 walk，108 孔会被算成 216 孔（样本翻倍、指标失真）。
    故按**规范化绝对路径**去重根，并记录已加载的目录。
    """
    use = list(roots) if roots else list(DEFAULT_ROOTS)
    # 根去重（规范化绝对路径）：绝对路径与相对路径指向同一目录时只留一个
    seen_roots = set()
    uniq_roots = []
    for r in use:
        key = os.path.normcase(os.path.abspath(r))
        if key in seen_roots:
            continue
        seen_roots.add(key)
        uniq_roots.append(r)
    use = uniq_roots

    multi = len([r for r in use if os.path.isdir(r)]) > 1
    out: List[Measurement] = []
    loaded_dirs = set()          # 已计入的测量目录（绝对路径），防重复
    for root in use:
        if not os.path.isdir(root):
            continue
        tag = os.path.basename(os.path.normpath(root))
        for dp, dn, fns in os.walk(root):
            if "Results.csv" not in fns or not any(f.startswith("noz") for f in fns):
                continue
            dkey = os.path.normcase(os.path.abspath(dp))
            if dkey in loaded_dirs:
                continue
            loaded_dirs.add(dkey)
            m = load_measurement(dp, root)
            if m and m.n:
                if multi:
                    m.session = "%s/%s" % (tag, m.session.replace(os.sep, "/"))
                out.append(m)
    return out


def by_head(measurements: List[Measurement], head: str) -> List[Measurement]:
    """筛出某个喷头的测量（head 传名字或 HeadType）。"""
    return [m for m in measurements
            if m.head_name == head or m.head_type == str(head)]


def resolve_roots(extra: Optional[List[str]] = None,
                  replace: bool = False) -> List[str]:
    """把 CLI 的 `--data` 参数解析成最终要扫描的数据根列表。

    - 不给 `--data` → 用 `DEFAULT_ROOTS`（本项目默认认可的资产）。
    - 给了 `--data` → **默认与 `DEFAULT_ROOTS` 合并**（新扫的数据与既有数据一起训练）。
    - `--replace-data` → 只用 `--data` 指定的根，丢弃 `DEFAULT_ROOTS`
      （用于"只看新数据"或"旧数据工作点不同、不能混"的场景）。

    合并时做**路径去重**（同一目录被指两次不会重复加载），
    并保持顺序稳定（默认根在前，新增根在后），使输出可复现。

    ⚠️ 去重按**规范化绝对路径**判定：`DEFAULT_ROOTS` 里「本机绝对路径」与
    「仓库相对路径」在开发机上指向同一份数据，只按 `normpath` 去重会漏判
    （绝对 vs 相对仍算不同），导致同一批数据被计入两次。
    """
    extra = [r for r in (extra or []) if r]
    if not extra:
        return _dedup_roots(DEFAULT_ROOTS)
    if replace:
        return _dedup_roots(extra)
    return _dedup_roots(list(DEFAULT_ROOTS) + list(extra))


def _dedup_roots(roots: List[str]) -> List[str]:
    """按规范化绝对路径去重，保持首次出现的顺序。"""
    seen = set()
    out = []
    for r in roots:
        key = os.path.normcase(os.path.abspath(r))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


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