# -*- coding: utf-8 -*-
"""从 ADA 的 `Results.csv` 还原 DropQCS 规则变量。

变量清单与语义来自平台规则集自带的说明块
（"NB: Parameter names are case sensitive"那一段），逐条对齐：

    [Present] bool            孔是否出滴
    [NrDroplets] int          找到的液滴/卫星数
    [Nozzle] int              孔号
    [Printhead] int           喷头号（双头时 1/2）
    [SpeedAverage] [SpeedMin] [SpeedMax]      速度 [m/s]，无效为负
    [AngleAverage] [AngleMin] [AngleMax]      角度 [deg]，无效为 -180
    [VolumeTotal] [VolumeMin] [VolumeMax]     体积 [pL]
    [XCenterAverage] [XCenterMin] [XCenterMax]   X 位置 [um]
    [YCenterAverage] [YCenterMin] [YCenterMax]   Y 位置 [um]
    [XMin] [XMax] [YMin] [YMax]               液滴包围盒

⚠️ **哨兵值**：规则引擎约定速度无效时为**负**、角度无效时为 **-180**。
0 滴孔的 CSV 行**只有 10 列**（后续液滴列直接省略，不是空串）——解析必须按
"组"读，不能按固定下标读，否则会错位。

⚠️ **逐滴列**：CSV 里 `Speed/Volume/Angle/X center/Y center/X left/Y top/
X right/Y bottom` 这 9 列**重复 5 组**，对应第 1..5 个液滴（实测多滴样本确认：
主滴 + 卫星滴各占一组）。因此 `[SpeedMin]/[SpeedMax]` 等"所有液滴"统计量
可以直接从这 5 组算出，不必读 `noz*.xml`。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["Droplet", "NozzleVars", "read_results_csv", "nozzle_vars",
           "INVALID_SPEED", "INVALID_ANGLE"]

#: 规则引擎的哨兵值约定
INVALID_SPEED = -1.0
INVALID_ANGLE = -180.0

#: 每组液滴占 9 列
_DROP_WIDTH = 9
#: 第 1 组液滴列的起始下标（前 9 列是元信息）
_FIRST_DROP = 9


def _f(x: Optional[str]) -> Optional[float]:
    """ADA 用 '???' 表示无效值；空串同理。"""
    if x is None:
        return None
    s = str(x).strip().strip('"').strip()
    if s in ("", "???"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


@dataclass
class Droplet:
    """一个液滴的读数。"""
    speed: Optional[float] = None
    volume: Optional[float] = None
    angle: Optional[float] = None
    x_center: Optional[float] = None
    y_center: Optional[float] = None
    x_min: Optional[float] = None
    y_min: Optional[float] = None
    x_max: Optional[float] = None
    y_max: Optional[float] = None


@dataclass
class NozzleVars:
    """一个喷嘴的全部规则变量。"""
    nozzle: Optional[int] = None
    printhead: int = 1
    nr_droplets: int = 0
    total_volume: float = 0.0
    droplets: List[Droplet] = field(default_factory=list)

    # ---- 规则变量构造 ----
    def variables(self) -> Dict[str, Any]:
        """生成 DropQCS 规则可用的变量字典（键不带方括号）。"""
        sp = [d.speed for d in self.droplets if d.speed is not None]
        vol = [d.volume for d in self.droplets if d.volume is not None]
        ang = [d.angle for d in self.droplets if d.angle is not None]
        xc = [d.x_center for d in self.droplets if d.x_center is not None]
        yc = [d.y_center for d in self.droplets if d.y_center is not None]

        def avg(v):
            return sum(v) / len(v) if v else None

        def mn(v, invalid):
            return min(v) if v else invalid

        def mx(v, invalid):
            return max(v) if v else invalid

        v: Dict[str, Any] = {
            "Present": self.nr_droplets > 0,
            "NrDroplets": self.nr_droplets,
            "Nozzle": self.nozzle,
            "Printhead": self.printhead,
            "VolumeTotal": self.total_volume,
            "VolumeMin": mn(vol, 0.0),
            "VolumeMax": mx(vol, 0.0),
            "SpeedAverage": avg(sp) if sp else INVALID_SPEED,
            "SpeedMin": mn(sp, INVALID_SPEED),
            "SpeedMax": mx(sp, INVALID_SPEED),
            "AngleAverage": avg(ang) if ang else INVALID_ANGLE,
            "AngleMin": mn(ang, INVALID_ANGLE),
            "AngleMax": mx(ang, INVALID_ANGLE),
            "XCenterAverage": avg(xc) if xc else 0.0,
            "XCenterMin": mn(xc, 0.0),
            "XCenterMax": mx(xc, 0.0),
            "YCenterAverage": avg(yc) if yc else 0.0,
            "YCenterMin": mn(yc, 0.0),
            "YCenterMax": mx(yc, 0.0),
        }
        # 包围盒：所有液滴的并集
        xs = [d.x_min for d in self.droplets if d.x_min is not None]
        ys = [d.y_min for d in self.droplets if d.y_min is not None]
        xr = [d.x_max for d in self.droplets if d.x_max is not None]
        yb = [d.y_max for d in self.droplets if d.y_max is not None]
        v["XMin"] = min(xs) if xs else 0.0
        v["XMax"] = max(xr) if xr else 0.0
        v["YMin"] = min(ys) if ys else 0.0
        v["YMax"] = max(yb) if yb else 0.0
        return v


def _parse_drops(row: List[str]) -> List[Droplet]:
    """从一行里读出逐滴数据。

    行长度不固定（0 滴孔只有 10 列），故按"组"迭代：每组 9 列，
    一组里若 Speed 为空则该组不存在。
    """
    out: List[Droplet] = []
    g = 0
    while True:
        b = _FIRST_DROP + g * _DROP_WIDTH
        if b + _DROP_WIDTH > len(row):
            break
        sp = _f(row[b])
        if sp is None:
            break
        out.append(Droplet(
            speed=sp,
            volume=_f(row[b + 1]),
            angle=_f(row[b + 2]),
            x_center=_f(row[b + 3]),
            y_center=_f(row[b + 4]),
            x_min=_f(row[b + 5]),
            y_min=_f(row[b + 6]),
            x_max=_f(row[b + 7]),
            y_max=_f(row[b + 8]),
        ))
        g += 1
    return out


def read_results_csv(path: str) -> List[NozzleVars]:
    """读一份 ADA `Results.csv` → 逐孔变量列表。"""
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig", errors="replace")
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 3:
        return []

    def cells(line):
        return next(csv.reader([line], delimiter=";", quotechar='"'))

    header = cells(lines[0])
    idx = {h.strip(): i for i, h in enumerate(header)}

    def col(row, name, default=None):
        i = idx.get(name)
        if i is None or i >= len(row):
            return default
        return row[i]

    out: List[NozzleVars] = []
    for line in lines[2:]:
        row = cells(line)
        # 数据行 = 首格是 .png 路径；尾部元信息区到此为止
        if not row or not row[0] or ".png" not in row[0]:
            break
        nr = _f(col(row, "Nr. of droplets"))
        if nr is None:
            continue
        drops = _parse_drops(row)
        out.append(NozzleVars(
            nozzle=int(_f(col(row, "Nozzle"))) if _f(col(row, "Nozzle")) is not None else None,
            printhead=1,
            nr_droplets=int(nr),
            total_volume=_f(col(row, "Total volume")) or 0.0,
            droplets=drops,
        ))
    return out


def nozzle_vars(nozzles: List[NozzleVars]) -> List[Dict[str, Any]]:
    """便捷函数：直接返回变量字典列表。"""
    return [n.variables() for n in nozzles]
