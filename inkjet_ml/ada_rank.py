# -*- coding: utf-8 -*-
"""ADA 扫描结果读取 + 波形组合排序（把 `ada.py` 的"扫"接上"判"，闭环最后一环）。

职责
----
`ada.py` 负责"要填什么范围、会得到多少张图"，本模块负责**反过来读**：
把一次 ADA 扫描落盘的目录树，还原成"每个波形组合一行"的汇总表，
按与 DropQCS 相同的判据排序，直接指出**哪个波形组合最优**。

为什么要写它
------------
ADA 一次全析因扫描会落 `成百上千` 张 `noz*.png` / `Results.csv`，
散在 `<BaseFolder>\\<Directory>\\<参数前缀+取值>\\...\\` 的目录树里。
DropQCS 的规则引擎是对**单次测量（一套波形、固定一孔→全孔）**做判据的，
它不知道"哪个波形组合更好"。这一环就是：**按波形组合聚类 → 算合格率 → 排序**。

判据口径（与 `ada.py --rules` 生成的 DropQCS 规则逐条一致）
-----------------------------------------------------------
    [NrDroplets] == 1            单滴、无卫星
    [SpeedAverage] ∈ [lo, hi]    速度落在目标邻域
    |[AngleAverage]| <= angle_max 飞行方向正
→ 三条全过的孔记为「合格孔」，组合的 `pass_rate` = 合格孔 / 测孔。

用法
----
    python -m inkjet_ml.ada_rank "C:\\LP50\\ScreenCaptures\\Measurement 1"
    python -m inkjet_ml.ada_rank <扫描根目录> --target-speed 5 --speed-margin 0.2
    python -m inkjet_ml.ada_rank <扫描根目录> --top 20 --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# 目录前缀 → ADA 参数名（证据：Tools\DropAnalysis\Viewer\Parameters.xml
# 与各喷头 Parameters_<Head>.xml 的 <FolderPrefix>）
# --------------------------------------------------------------------------- #
PREFIX_TO_PARAM: Dict[str, str] = {
    "P0V": "Pulse0.Voltage", "P0W": "Pulse0.Width", "P0T": "Pulse0.Tail",
    "P1V": "Pulse1.Voltage", "P1W": "Pulse1.Width", "P1T": "Pulse1.Tail",
    "P2V": "Pulse2.Voltage", "P2W": "Pulse2.Width", "P2T": "Pulse2.Tail",
    "P3V": "Pulse3.Voltage", "P3W": "Pulse3.Width", "P3T": "Pulse3.Tail",
    "Slewrate": "Slewrate", "Vsupply": "VoltageSupply",
    "noz": "Nozzle", "fnoz": "FastScanSeries",
    "freq": "Frequency", "Pi": "InkPressure",
    "temp": "HeadTemperature", "hotmelt": "HotmeltTemperature",
    "Tled": "LEDDelay", "try": "LongTermStability", "custom": "Custom",
}
# 排序优先级：先按路径从深到浅还原（即 ADA 扫描时最外层→最内层的逆序）
WAVEFORM_KEYS = ["Pulse%d.%s" % (i, s)
                 for i in range(4) for s in ("Voltage", "Width", "Tail")] + \
                ["Slewrate", "VoltageSupply"]


def split_combo(dir_path: str, root: str) -> Dict[str, str]:
    """把目录树相对 root 的各层目录名，还原成 {参数名: 取值}。

    ADA 生成路径时（ADA.cs:5433-5437）按 m_oParams 顺序拼接
    `Prefix + value`，所以**越靠外层 = 越早被扫描的参数**；
    反过来还原时逐层识别即可得到完整的组合。

    目录名 = `FolderPrefix` 与取值的直接拼接（如 `P0V24`），
    两者之间没有分隔符，因此必须按**已知前缀最长优先**匹配
    （否则 `P0V24` 会被误认成前缀 `P0` + 值 `V24`）。
    """
    rel = os.path.relpath(dir_path, root)
    combo: Dict[str, str] = {}
    if rel == ".":
        return combo
    parts = [p for p in rel.split(os.sep) if p and p != "."]
    for name in parts:
        for prefix in _PREFIXES_BY_LEN:            # 长前缀优先
            if name.startswith(prefix):
                combo[PREFIX_TO_PARAM[prefix]] = name[len(prefix):]
                break
    return combo


# 按长度降序的前缀列表（保证 P0V 优先于 P0 这类前缀自包含的情况）
_PREFIXES_BY_LEN = sorted(PREFIX_TO_PARAM, key=len, reverse=True)


# --------------------------------------------------------------------------- #
# Results.csv 解析（按列名匹配，不写死下标）
# --------------------------------------------------------------------------- #
def _f(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    s = x.strip().strip('"').strip()
    if s == "" or s == "???":
        return None
    try:
        return float(s)
    except ValueError:
        return None


@dataclass
class NozzleRow:
    nozzle: Optional[int]
    nr_droplets: int
    speed: Optional[float]          # SpeedAverage（多滴时 ADA 给的是第一个滴）
    volume: Optional[float]         # VolumeTotal（pl）
    angle: Optional[float]


@dataclass
class ComboResult:
    combo: Dict[str, str] = field(default_factory=dict)
    folder: str = ""
    nozzles: List[NozzleRow] = field(default_factory=list)
    scanned_params: Dict[str, str] = field(default_factory=dict)

    # 统计
    @property
    def n_nozzles(self) -> int:
        return len(self.nozzles)

    @property
    def n_nojet(self) -> int:
        return sum(1 for r in self.nozzles if r.nr_droplets == 0)

    @property
    def n_single(self) -> int:
        return sum(1 for r in self.nozzles if r.nr_droplets == 1)

    @property
    def n_satellite(self) -> int:
        return sum(1 for r in self.nozzles if r.nr_droplets >= 2)

    def _vals(self, attr):
        return [getattr(r, attr) for r in self.nozzles if getattr(r, attr) is not None]

    def speed_stats(self):
        v = self._vals("speed")
        return _stats(v)

    def volume_stats(self):
        return _stats(self._vals("volume"))

    def pass_rate(self, lo: float, hi: float, angle_max: float) -> float:
        ok = 0
        for r in self.nozzles:
            if r.nr_droplets != 1:
                continue
            if r.speed is None or not (lo <= r.speed <= hi):
                continue
            if r.angle is not None and abs(r.angle) > angle_max:
                continue
            ok += 1
        return ok / self.n_nozzles if self.n_nozzles else 0.0

    def describe(self) -> str:
        keys = [k for k in WAVEFORM_KEYS if k in self.combo]
        if not keys:
            return "(单波形)"
        return " ".join("%s=%s" % (k, self.combo[k]) for k in keys)


def _stats(v: List[float]):
    if not v:
        return (None, None, None, None, None)
    n = len(v)
    mean = sum(v) / n
    var = sum((x - mean) ** 2 for x in v) / n
    s = sorted(v)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return (mean, var ** 0.5, min(v), max(v), median)


def parse_results_csv(path: str) -> Optional[ComboResult]:
    """解析一个 Results.csv → 该目录（= 一个波形组合）的逐孔读数。"""
    try:
        text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return None
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 3:
        return None

    def cells(line):
        # ADA 用 ';' 分隔并给每格加引号；用 csv 解析器最稳
        return next(csv.reader([line], delimiter=";", quotechar='"'))

    header = cells(lines[0])

    def idx_all(name):
        return [i for i, h in enumerate(header) if h.strip() == name]

    i_nr = idx_all("Nr. of droplets")
    if not i_nr:
        return None
    i_nr = i_nr[0]
    res = ComboResult(folder=os.path.dirname(path))

    for line in lines[2:]:
        row = cells(line)
        # 数据行 = 首格是 .png 路径；尾部元数据区（Start time / IDPI Head Type ...）到此为止
        first = row[0].strip() if row else ""
        if ".png" not in first:
            break
        nr = _f(row[i_nr]) if i_nr < len(row) else None
        if nr is None:
            continue

        def g(name):
            idx = idx_all(name)
            return _f(row[idx[0]]) if idx and idx[0] < len(row) else None

        i_noz = idx_all("Nozzle")
        res.nozzles.append(NozzleRow(
            nozzle=int(_f(row[i_noz[0]])) if i_noz and i_noz[0] < len(row) else None,
            nr_droplets=int(nr),
            speed=g("Speed"),
            volume=g("Total volume"),
            angle=g("Angle"),
        ))
    return res if res.nozzles else None


def scan_tree(root: str) -> List[ComboResult]:
    """递归收集 root 下所有 Results.csv，还原成按波形组合聚合的结果。"""
    out: List[ComboResult] = []
    for dp, _dn, fns in os.walk(root):
        if "Results.csv" not in fns:
            continue
        r = parse_results_csv(os.path.join(dp, "Results.csv"))
        if r is None:
            continue
        r.combo = split_combo(dp, root)
        # 其它扫描参数（CSV 表头里的已扫参数列）也记下来，便于核对
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
def rank(combos: List[ComboResult], lo: float, hi: float,
         angle_max: float, target: float) -> List[ComboResult]:
    """按「合格率 → 速度贴近目标 → 卫星最少」排序（判读优先级）。"""
    def key(c: ComboResult):
        pr = c.pass_rate(lo, hi, angle_max)
        smean = c.speed_stats()[0]
        speed_dev = abs(smean - target) if smean is not None else 9e9
        return (-round(pr, 6), speed_dev, c.n_satellite)

    return sorted(combos, key=key)


def report(combos: List[ComboResult], lo: float, hi: float,
           angle_max: float, target: float, top: int = 20,
           csv_out: Optional[str] = None) -> None:
    ranked = rank(combos, lo, hi, angle_max, target)
    print("共解析 %d 个扫描组合（每个组合 = 一个目录下的一个 Results.csv）" % len(ranked))
    print("判据：NrDroplets==1 且 %.2f<=Speed<=%.2f m/s 且 |Angle|<=%.1f°\n"
          % (lo, hi, angle_max))
    hdr = ("%-52s %6s %6s %6s %6s %7s %7s %7s %8s"
           % ("波形组合", "孔数", "不喷", "单滴", "卫星", "合格率",
              "速度均", "速度σ", "体积均(pl)"))
    print(hdr)
    print("-" * len(hdr))
    rows_csv = []
    for c in ranked[:top]:
        pr = c.pass_rate(lo, hi, angle_max)
        sm, ss, _, _, _ = c.speed_stats()
        vm, _, _, _, _ = c.volume_stats()
        print("%-52s %6d %6d %6d %6d %6.1f%% %7.2f %7.3f %8.2f"
              % (c.describe()[:52], c.n_nozzles, c.n_nojet, c.n_single,
                 c.n_satellite, pr * 100, sm if sm is not None else -1,
                 ss if ss is not None else -1, vm if vm is not None else -1))
        row = dict(c.combo)
        row.update(n_nozzles=c.n_nozzles, n_nojet=c.n_nojet,
                   n_single=c.n_single, n_satellite=c.n_satellite,
                   pass_rate=round(pr, 4), speed_mean=sm, speed_std=ss,
                   volume_mean=vm, folder=c.folder)
        rows_csv.append(row)
    if csv_out and rows_csv:
        keys = sorted({k for r in rows_csv for k in r})
        with open(csv_out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows_csv)
        print("\n已写出排序表：%s" % csv_out)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="读取 ADA 扫描结果，按 DropQCS 判据给波形组合排序")
    ap.add_argument("root", help="扫描根目录（含 Results.csv 的那次测量目录）")
    ap.add_argument("--target-speed", type=float, default=5.0, help="目标速度 m/s")
    ap.add_argument("--speed-margin", type=float, default=0.20, help="速度容差")
    ap.add_argument("--angle-max", type=float, default=2.0, help="方向容差(度)")
    ap.add_argument("--top", type=int, default=20, help="显示前 N 名")
    ap.add_argument("--csv", help="把完整排序表写到 CSV")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.root):
        print("目录不存在：%s" % args.root, file=sys.stderr)
        return 2
    combos = scan_tree(args.root)
    if not combos:
        print("该目录下没找到含逐孔读数的 Results.csv。\n"
              "提示：一次 ADA 扫描的每个波形组合各有一份 Results.csv，"
              "请把 root 指向那次测量的目录。", file=sys.stderr)
        return 1
    lo = args.target_speed * (1 - args.speed_margin)
    hi = args.target_speed * (1 + args.speed_margin)
    report(combos, lo, hi, args.angle_max, args.target_speed, args.top, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())