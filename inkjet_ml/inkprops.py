# -*- coding: utf-8 -*-
"""墨水物性 → Z 数（Ohnesorge 数倒数）计算器。

论文（Kim et al., Sci Rep 2022, doi:10.1038/s41598-022-08784-y）Eq.4：

    Z = 1/Oh = Re/√We = √(ργD) / η

其中（论文 Eq.1–3）：

    Re = ρvD/η      惯性力 / 黏性力
    We = ρv²D/γ     惯性力 / 表面张力
    Oh = η/√(ργD)   黏性力 / 表面张力

注意 Re/√We 里 **液滴速度 v 被约掉了** —— Z 只由流体物性和一个几何量决定，
与波形、驱动电压、喷射速度都无关。这正是论文能把它当作"墨水维度"单一特征的原因。

## D 到底是滴直径还是喷孔直径？

论文正文写 "D is the drop diameter"，但**用论文自己 Table 1 的 ρ/η/γ 反解**
（令 Z 完全吻合），12 组墨水给出的等效 D 全部落在 **39.7–41.1 µm**，
与论文自述的喷嘴规格 **40 µm**（MicroFab MJ-ATP-01-040）一致 —— 说明公式里的 D
是**喷孔直径（orifice diameter）**，不是滴直径。这是 DoD 领域的标准取法
（喷孔直径才是墨水流经的几何约束），论文措辞不严谨。

    python -m inkjet_ml.inkprops --table1     # 用 Table 1 原始数据自检本实现

## 单位：这是最容易出错的地方

    Z = 0.0316228 · √(ρ[kg/m³] · γ[mN/m] · D[µm]) / η[mPa·s]

SI 原式是 `√(ργD)/η`（ρ kg/m³、γ N/m、η Pa·s、D m）。工程单位换算见 `z_practical()`。

另：1 pL = 1e-15 m³（因为 1 L = 1e-3 m³），**不是 1e-12**。
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Optional, Tuple

#: SI → 工程单位的合并系数：√(kg/m³ · mN/m · µm) / (mPa·s) → 无量纲
#: = √(1e-3 · 1e-6) / 1e-3 = 1e-4.5 / 1e-3 = 1e-1.5 = 0.0316228
PRACTICAL_COEF = 10.0 ** -1.5

#: 论文 Table 1（.ods 原始数据）：ink_no -> (ρ kg/m³, η mPa·s, γ mN/m, Z 论文值)
#: 组成 = (去离子水 %, 乙二醇 %, 甘油 %)
TABLE1: Dict[int, Tuple[float, float, float, float, Tuple[int, int, int]]] = {
    1:  (998.4,  1.000, 71.6, 53.6,  (100,  0,  0)),
    2:  (1037.0, 1.613, 69.3, 33.2,  ( 90,  0, 10)),
    3:  (1066.5, 1.843, 68.4, 29.3,  ( 80,  0, 20)),
    4:  (1093.2, 2.773, 67.3, 19.5,  ( 70,  0, 30)),
    5:  (1126.3, 4.140, 66.5, 13.2,  ( 60,  0, 40)),
    6:  (1036.7, 1.223, 67.0, 43.1,  ( 90, 10,  0)),
    7:  (1043.3, 1.990, 61.3, 25.4,  ( 70, 30,  0)),
    8:  (1046.7, 4.520, 52.3, 10.5,  ( 40, 60,  0)),
    9:  (1094.4, 5.940, 50.1,  7.88, ( 30, 70,  0)),
    10: (1106.8, 8.883, 49.3,  5.26, ( 20, 80,  0)),
    11: (1113.5, 16.38, 45.3,  2.74, (  0, 100, 0)),
    12: (1071.1, 3.450, 56.4, 14.3,  ( 50,  0, 50)),
}

#: 论文所用的喷嘴孔径（喷孔直径，不是滴直径）
PAPER_ORIFICE_UM = 40.0

#: 论文 Eq.1 的「可正常成滴」区间
PRINTABILITY_Z_MIN = 1.0
PRINTABILITY_Z_MAX = 14.0


# --------------------------------------------------------------------------- #
# 核心公式
# --------------------------------------------------------------------------- #
def z_si(rho_kgm3: float, eta_pas: float, gamma_npm: float, d_m: float) -> float:
    """SI 单位下的 Z = √(ργD)/η。

    参数:
        rho_kgm3: 密度 ρ [kg/m³]
        eta_pas:  剪切粘度 η [Pa·s]
        gamma_npm: 表面张力 γ [N/m]
        d_m:      喷孔直径 D [m]
    """
    if min(rho_kgm3, eta_pas, gamma_npm, d_m) <= 0:
        raise ValueError("ρ / η / γ / D 必须全为正数")
    return math.sqrt(rho_kgm3 * gamma_npm * d_m) / eta_pas


def z_practical(rho_kgm3: float, eta_mpas: float, gamma_mnm: float, d_um: float) -> float:
    """工程单位下的 Z（最常用的一组）。

    参数:
        rho_kgm3: 密度 ρ [kg/m³]
        eta_mpas: 剪切粘度 η [mPa·s]
        gamma_mnm:表面张力 γ [mN/m]
        d_um:     喷孔直径 D [µm]
    """
    return PRACTICAL_COEF * math.sqrt(rho_kgm3 * gamma_mnm * d_um) / eta_mpas


def reynolds(rho_kgm3: float, v_ms: float, d_m: float, eta_pas: float) -> float:
    """Re = ρvD/η（SI）。"""
    return rho_kgm3 * v_ms * d_m / eta_pas


def weber(rho_kgm3: float, v_ms: float, d_m: float, gamma_npm: float) -> float:
    """We = ρv²D/γ（SI）。"""
    return rho_kgm3 * v_ms ** 2 * d_m / gamma_npm


def ohnesorge(z: float) -> float:
    """Oh = 1/Z。"""
    return 1.0 / z


def printability(z: float) -> str:
    """按论文的 1 < Z < 14 判定成滴区间。

    论文正文："It is widely assumed that Z is required to be in the 1<Z<14 range
    for proper drop formation"（引用 Derby 2010 / Jang 2009）。
    """
    if z < PRINTABILITY_Z_MIN:
        return "过低（Z<1）：黏性相对过强，难以成滴"
    if z > PRINTABILITY_Z_MAX:
        return "过高（Z>14）：惯性相对过强，易出卫星滴/拖尾"
    return "在 1<Z<14 成滴区间内"


# --------------------------------------------------------------------------- #
# 反解：给定 Z 与墨水物性，求缺失的那个量
# --------------------------------------------------------------------------- #
def solve_diameter_um(z: float, rho_kgm3: float, eta_mpas: float, gamma_mnm: float) -> float:
    """反解喷孔直径 D [µm]：由 Z = √(ργD)/η 整理得 D = (Zη)²/(ργ)。"""
    eta_pas = eta_mpas * 1e-3
    gamma = gamma_mnm * 1e-3
    d_m = (z * eta_pas) ** 2 / (rho_kgm3 * gamma)
    return d_m * 1e6


def solve_viscosity_mpas(z: float, rho_kgm3: float, gamma_mnm: float, d_um: float) -> float:
    """反解粘度 η [mPa·s]：η = √(ργD)/Z。"""
    return z_practical(rho_kgm3, 1.0, gamma_mnm, d_um) / z


def solve_surface_tension_mnm(z: float, rho_kgm3: float, eta_mpas: float, d_um: float) -> float:
    """反解表面张力 γ [mN/m]：γ = (Zη)²/(ρD)。"""
    eta_pas = eta_mpas * 1e-3
    d_m = d_um * 1e-6
    gamma = (z * eta_pas) ** 2 / (rho_kgm3 * d_m)
    return gamma * 1e3


# --------------------------------------------------------------------------- #
# 自检：用论文 Table 1 验证实现，并反解 D
# --------------------------------------------------------------------------- #
def verify_table1(d_um: float = PAPER_ORIFICE_UM) -> Tuple[bool, List[str]]:
    """用论文 Table 1 的 ρ/η/γ/Z 自检本实现。

    返回 (是否全部通过, 报告行列表)。
    """
    lines: List[str] = []
    ok = True
    errs: List[float] = []
    for no, (rho, eta, gamma, z_paper, comp) in sorted(TABLE1.items()):
        z = z_practical(rho, eta, gamma, d_um)
        err = (z - z_paper) / z_paper * 100.0
        errs.append(abs(err))
        d_eq = solve_diameter_um(z_paper, rho, eta, gamma)
        if abs(err) > 2.0:
            ok = False
        lines.append(
            "  Ink%-2d 水/EG/甘油 %3d/%3d/%3d  Z论文 %6.2f  Z算 %8.4f  误差 %+6.2f%%  "
            "反解D %6.3f µm  %s"
            % (no, comp[0], comp[1], comp[2], z_paper, z, err, d_eq,
               printability(z_paper).split("：")[0])
        )
    mean = sum(errs) / len(errs)
    lines.append("")
    lines.append("  D 取 %.0f µm 时：max|误差| = %.2f%%, 平均 = %.2f%%"
                 % (d_um, max(errs), mean))
    d_all = [solve_diameter_um(z, rho, eta, gamma)
             for _, (rho, eta, gamma, z, _) in sorted(TABLE1.items())]
    lines.append("  反解等效 D：%.2f–%.2f µm，均值 %.2f µm（论文喷嘴标称 %g µm）"
                 % (min(d_all), max(d_all), sum(d_all) / len(d_all), PAPER_ORIFICE_UM))
    lines.append("  ⇒ 公式中的 D 是**喷孔直径**，不是滴直径（论文正文措辞不严谨）")
    return ok, lines


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _fmt_kv(pairs: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for chunk in pairs.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise argparse.ArgumentTypeError("需要 key=value 形式，收到 %r" % chunk)
        k, v = chunk.split("=", 1)
        out[k.strip()] = float(v)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Z 数（Ohnesorge 数倒数）计算器 — Z = √(ργD)/η")
    ap.add_argument("--table1", action="store_true",
                    help="用论文 Table 1 原始数据自检实现（并反解等效喷孔直径）")
    ap.add_argument("--d", type=float, default=None,
                    help="喷孔直径 D [µm]（默认 %g，与 --rho/--eta/--gamma 一起用时可省）"
                         % PAPER_ORIFICE_UM)
    ap.add_argument("--rho", type=float, help="密度 ρ [kg/m³]")
    ap.add_argument("--eta", type=float, help="剪切粘度 η [mPa·s]")
    ap.add_argument("--gamma", type=float, help="表面张力 γ [mN/m]")
    ap.add_argument("--z", type=float, help="已知 Z；配合 --rho/--eta/--gamma/--d 时反解缺的量")
    ap.add_argument("--solve", choices=["d", "eta", "gamma"],
                    help="反解哪一个量（需给 --z 与其余三个）")
    args = ap.parse_args(argv)

    did = False
    if args.table1:
        print("## 自检：论文 Table 1（Kim2022 Table 1, .ods 原始数据）")
        ok, lines = verify_table1(args.d if args.d is not None else PAPER_ORIFICE_UM)
        print("\n".join(lines))
        print("\n%s" % ("✓ 实现与论文数据一致" if ok else "✗ 存在超出 2% 的偏差"))
        did = True

    if args.rho is not None or args.eta is not None or args.gamma is not None:
        if args.rho is None or args.eta is None or args.gamma is None:
            ap.error("--rho / --eta / --gamma 必须同时给出")
        d_um = args.d if args.d is not None else PAPER_ORIFICE_UM
        z = z_practical(args.rho, args.eta, args.gamma, d_um)
        print("\n## 计算结果")
        print("  ρ = %.4g kg/m³, η = %.4g mPa·s, γ = %.4g mN/m, D = %.4g µm"
              % (args.rho, args.eta, args.gamma, d_um))
        print("  Z  = %.4f   (Oh = %.4f)" % (z, ohnesorge(z)))
        print("  %s" % printability(z))
        did = True

    if args.solve:
        if args.z is None:
            ap.error("--solve 需要 --z")
        d_um = args.d if args.d is not None else PAPER_ORIFICE_UM
        if args.solve == "d":
            v = solve_diameter_um(args.z, args.rho, args.eta, args.gamma)
            unit = "µm"
        elif args.solve == "eta":
            v = solve_viscosity_mpas(args.z, args.rho, args.gamma, d_um)
            unit = "mPa·s"
        else:
            v = solve_surface_tension_mnm(args.z, args.rho, args.eta, d_um)
            unit = "mN/m"
        print("  反解 %s = %.4g %s（给定 Z = %.4g）" % (args.solve, v, unit, args.z))
        did = True

    if not did:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
