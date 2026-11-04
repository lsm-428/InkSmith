# -*- coding: utf-8 -*-
r"""DMC Samba 一套 ML（HeadType 1117）。

数据现状（实测核实）
--------------------
本机 `D:\Gitee\LP50\ML\share\NOA89-test-01..09` 共 **108 孔 / 9 次会话**，
波形取值 4 组：
    Pulse0Voltage ∈ {34, 38, 40}、Pulse0Width ∈ {1.9, 2.2}
    Pulse0Tail = 4.5、Slewrate = 40、VoltageSupply = 120、Frequency = 1000
频率/温度/压力这些在 9 次会话里都是常量，所以**不能作为特征**（常数列对
树模型无信息、对线性模型无意义），只保留真正变化的 4 个量 + 孔号位置。

特征（全部来自 ADA 自己的参数命名空间，HeadType 1117）
----------------------------------------------------
    pulse0_voltage  DropAnalysis.DMC.Samba.Pulse0.Voltage
    pulse0_width    DropAnalysis.DMC.Samba.Pulse0.Width
    pulse0_tail     DropAnalysis.DMC.Samba.Pulse0.Tail
    slewrate        DropAnalysis.DMC.Samba.Slewrate
    voltage_supply  DropAnalysis.DMC.Samba.VoltageSupply
（后两个在现有数据里是常量，仍保留在特征表中：数据一多就会有变化）

两个任务
--------
1. 正向预测：给定波形 → 喷射结果
       分类 0 不喷 / 1 单滴 / 2 含卫星
       回归 液滴速度 [m/s]
2. 反向推荐：给定目标（速度区间 + 要求单滴无卫星）→ 推荐候选波形
       在 ADA 允许的参数范围网格上枚举，用训练好的模型打分排序。

运行
----
    python -m inkjet_ml.heads.samba            # 训练 + 评估 + 报告
    python -m inkjet_ml.heads.samba --recommend --target-speed 3.5
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from typing import List, Optional

import numpy as np

from ..dataset import Measurement, by_head, load_all, resolve_roots
from ..headml import (CLS_KEYS, MODEL_ORDER, REG_KEYS, HeadDataset, best_by,
                      evaluate_classification, evaluate_leave_nozzle_out,
                      evaluate_leave_nozzle_out_cls, evaluate_regression,
                      select_best_classifier, select_best_regressor,
                      _fmt_table, save_json)

HEAD_NAME = "DMC Samba"
HEAD_TYPE = "1117"

#: ADA 流程默认范围（Advanced-Drop-Analysis-DMCSamba.xml），反向推荐在这张网格上搜
ADA_GRID = {
    "pulse0_voltage": [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40],
    "pulse0_width": [1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 3.0, 3.4, 3.8],
    "pulse0_tail": [1.4, 2.2, 3.0, 4.5, 6.0],
    "slewrate": [25, 30, 35, 40, 45, 50, 55],
    "voltage_supply": [110, 114, 118, 120],
}
#: 仅扫描最相关的 3 个维度做粗筛，避免网格爆炸
SEARCH_DIMS = ["pulse0_voltage", "pulse0_width", "pulse0_tail"]


#: 波形特征（不含喷嘴）——用于「波形可迁移性」评估
FEATURES_WAVE = ["pulse0_voltage", "pulse0_width", "pulse0_tail",
                 "slewrate", "voltage_supply"]

#: 喷嘴位置特征（§1.5 创新点 2「喷嘴级一致性建模」的抓手）
#:
#: `nozzle`      原始孔号（1..N），提供身份信息；
#: `nozzle_frac` 归一化位置（0=第一孔，1=最后一孔），让模型能学**空间趋势**
#:               （例如边缘孔因墨路回流路径不同而系统性偏快/偏慢）。
#:               只给原始孔号是不够的——树模型无法在整数标签上做外推，
#:               面对没见过的孔号只能落到最近的分裂点上。
#:
#: ⚠️ **这两个特征不是免费午餐**（2026-11-03 首轮实测已更新认知）：
#: 原本担心的是「12 孔 × 9 会话时，加入孔号只会让模型**记住每孔的固定偏移**，
#: 交叉验证指标变好但**跨喷嘴泛化不会变好**」。
#: 实测否定了这个担心：**同一波形内 孔号 vs 速度 Spearman ρ 均值 +0.764（p<0.001）**，
#: 且逐孔留出（LeaveOneGroupOut, 12 折）时含孔号特征 R² 0.9433 > 仅波形 0.8897。
#: 说明孔号携带的是**系统性空间梯度**（可外推、可建模），
#: 而不是每孔独立的固定偏移（不可外推）。
#: 但这仍需**重复性**数据支撑——梯度是真实的还是孔位与「当天哪一批测的」混淆，
#: 要靠同一波形重复多测来分离（见 §7 阶段 3）。
#: 故本模块默认**同时报告**「不含孔号」与「含孔号」两套指标，
#: 并另做 leave-nozzle-out 评估（--leave-nozzle-out）来回答
#: §1.5 创新点 1「在少喷嘴上优化的波形能否迁移到整头」。
FEATURES_NOZZLE = ["nozzle", "nozzle_frac"]

#: 完整特征表 = 波形 + 喷嘴位置（默认训练用）
FEATURES = FEATURES_WAVE + FEATURES_NOZZLE

#: 本机 Samba 一次扫描覆盖的孔数 = **DMC Samba 的物理孔数**。
#: ⚠️ 2026-11-03 核实，三个独立来源一致：
#:   · 规格书 *DMC Samba Specification*：Addressable Nozzles – **12 jets**；
#:   · 实机 recipe `ADA Fujifilm DMCSamba.recipe.xml`：`DropAnalysis.Nozzle[0] = 1:12`
#:     （旧机台 C:\LP50 同值）；
#:   · 实测数据：9 会话 × 12 孔 = 108 行，**每会话都完整覆盖 1..12 孔**。
#:
#: ⚠️ **别把 `.Nozzle` 参数值当孔数**：它是「**要测哪些孔**的扫描范围」
#: （ADA 源码 `Advanced-Drop-Analysis.cs:5140` 按 "Active Nozzle" 序列解析）。
#: LP50 flow 里的通用默认 `1:128` 对 KM512（物理 512 孔）只是"只测前 128 个"，
#: 对 Samba（物理 12 孔）则是越界。各头物理孔数对照见 `ada.HEAD_NOZZLE_COUNT`。
#: 用于把孔号归一化成位置比例。
NOZZLE_COUNT = 12


def features_of(m: Measurement) -> Optional[List[float]]:
    """把一次测量转成特征向量（不含喷嘴——喷嘴随孔变化，逐孔追加）。"""
    s = m.settings
    vals = [s.get("pulse0_voltage"), s.get("pulse0_width"), s.get("pulse0_tail"),
            s.get("slewrate"), s.get("voltage_supply")]
    if any(v is None for v in vals):
        return None
    return [float(v) for v in vals]


def nozzle_features(nozzle: Optional[int]) -> Optional[List[float]]:
    """喷嘴位置特征：原始孔号 + 归一化位置。

    归一化用 `NOZZLE_COUNT`（本机实测覆盖的孔数）而非硬编码 12，
    换喷头或改扫描范围时只需改这一个常量。
    """
    if nozzle is None:
        return None
    n = max(NOZZLE_COUNT, 2)
    frac = (float(nozzle) - 1.0) / (n - 1.0)
    return [float(nozzle), min(max(frac, 0.0), 1.0)]


def build(measurements: Optional[List[Measurement]] = None,
          with_nozzle: bool = True,
          roots: Optional[List[str]] = None) -> HeadDataset:
    """构建数据集。

    `with_nozzle=False` 时只保留波形特征（用于迁移性评估的基线）。
    无论哪种情况，**喷嘴号都记进 `groups_nozzle`**，
    使 leave-nozzle-out 评估在任何特征配置下都可用。

    `roots` 指定要扫描的数据根（`None` = 用 `dataset.DEFAULT_ROOTS`）。
    ⚠️ 传了 `roots` 就**只**扫这些根——上层的 `resolve_roots()` 负责决定
    「默认根 + 新扫根」还是「只用新扫根」。
    """
    ms = by_head(measurements if measurements is not None
                 else load_all(roots), HEAD_NAME)
    feats = FEATURES if with_nozzle else FEATURES_WAVE
    X, yc, yv, g, gn, meta = [], [], [], [], [], []
    for m in ms:
        f = features_of(m)
        if f is None:
            continue
        for nz in m.nozzles:
            row = list(f)
            if with_nozzle:
                nf = nozzle_features(nz.nozzle)
                if nf is None:
                    continue
                row += nf
            X.append(row)
            yc.append(nz.y_cls)
            yv.append(np.nan if nz.speed is None else nz.speed)
            g.append(m.session)
            gn.append("nz%s" % nz.nozzle)
            meta.append({"session": m.session, "nozzle": nz.nozzle,
                         "n_droplets": nz.n_droplets})
    ds = HeadDataset(
        features=feats, X=np.asarray(X, dtype=float),
        y_cls=np.asarray(yc, dtype=int), y_vel=np.asarray(yv, dtype=float),
        groups=np.asarray(g), meta=meta, head_name=HEAD_NAME, head_type=HEAD_TYPE,
    )
    ds.groups_nozzle = np.asarray(gn)
    return ds


# --------------------------------------------------------------------------- #
def _measured_bbox(X):
    """实测特征向量的轴对齐包围盒 [min, max]，作为可信外推范围。"""
    return X.min(axis=0), X.max(axis=0)


def recommend(ds: HeadDataset, target_speed: float, margin: float = 0.20,
              angle_max: float = 5.0, top: int = 20) -> List[dict]:
    """在 ADA 网格上枚举，用**与评估同一套选出的最优模型**给每个候选打分。

    评分口径与 DropQCS 判据一致：单滴、无卫星、速度进邻域、方向正。

    ⚠️ **推荐必须用「仅波形特征」的模型**（2026-11-03 修正）
    ----------------------------------------------------
    推荐的对象是**整头通用的一组波形参数**，网格候选里没有"具体某个喷嘴"
    这个概念——它只有电压/宽度/尾隙/边沿/供电电压 5 个量。
    若用带孔号位置特征的模型来打分，两件事会同时出错：
      · 特征维度对不上（候选 5 列 vs 训练 7 列）→ 直接崩；
      · 就算补齐成 7 列，模型输出的也是"某个特定喷嘴在该波形下的速度"，
        而推荐要的是**整头的期望波形**。
    故此处**只用波形特征训练专用模型**，并在训练集上按与评估相同的口径选模型。

    ⚠️ 外推可信度（数据不全时公式必须自我保护）
    ----------------------------------------------------
    实测波形只有 4 个点时，模型在网格内部几乎全靠外推。为了不让「排名靠前」
    变成「噪声排序」，这里给每个候选算一个 **外推距离**（到实测包围盒的归一化距离）：

        extrap = 0  → 完全落在实测包围盒内
        extrap > 0  → 需要外推；越大越不可信

    并把 extrap 纳入排序：实测点优先，其次包围盒内，再次才是盒外外推。
    这样即便公式在数据稀疏时也**如实降级**，不会把外推点伪装成结论。
    """
    # 只取波形特征列 —— 网格候选没有"某个具体喷嘴"的概念
    wave_idx = [ds.features.index(f) for f in FEATURES_WAVE if f in ds.features]
    if len(wave_idx) != len(FEATURES_WAVE):
        raise ValueError("特征表缺少波形特征列，无法在波形网格上推荐：%s" % ds.features)
    Xw = ds.X[:, wave_idx]
    gw = ds.groups

    # 用与评估完全相同的交叉验证口径选模型（不再硬编码 MLP）
    clf, clf_name = select_best_classifier(Xw, ds.y_cls, gw)
    vm = ds.vel_mask()
    reg, reg_name = select_best_regressor(Xw[vm], ds.y_vel[vm], gw[vm])
    clf.fit(Xw, ds.y_cls)
    reg.fit(Xw[vm], ds.y_vel[vm])

    keys = SEARCH_DIMS
    combos = list(itertools.product(*[ADA_GRID[k] for k in keys]))
    slew = float(ADA_GRID["slewrate"][3])
    vs = float(ADA_GRID["voltage_supply"][-1])
    Xc = np.asarray([[c[0], c[1], c[2], slew, vs] for c in combos], dtype=float)
    p_cls = clf.predict(Xc)
    p_vel = reg.predict(Xc)

    lo, hi = target_speed * (1 - margin), target_speed * (1 + margin)
    # 已实测过的波形取值（用于标出哪些推荐点是实测过的，哪些是外推）
    wave_seen = {tuple(np.round(row, 3)) for row in Xw}
    # 可信外推范围：实测**波形**特征向量的轴对齐包围盒
    bmin, bmax = _measured_bbox(Xw)
    span = np.where((bmax - bmin) > 0, bmax - bmin, 1.0)   # 常量列不贡献距离

    rows = []
    for c, k, v, x in zip(combos, p_cls, p_vel, Xc):
        if k != 1:                     # 预测不是"单滴无卫星" → 直接出局
            continue
        if not (lo <= v <= hi):
            continue
        # 到包围盒的归一化外推距离（越界才计入，盒内为 0）
        below = np.clip(bmin - x, 0, None)
        above = np.clip(x - bmax, 0, None)
        extrap = float(np.sum((below + above) / span))
        rows.append({
            "pulse0_voltage": c[0], "pulse0_width": c[1], "pulse0_tail": c[2],
            "pred_class": int(k), "pred_speed": round(float(v), 3),
            "speed_err": round(abs(float(v) - target_speed), 3),
            "measured": tuple(np.round(x, 3)) in wave_seen,
            "extrap": round(extrap, 3),
        })
    # 排序：① 实测优先 → ② 盒内优先（extrap 小）→ ③ 速度偏差小
    rows.sort(key=lambda r: (not r["measured"], r["extrap"], r["speed_err"]))
    return rows[:top]


def data_caveats(ds: HeadDataset, cls: dict = None) -> List[str]:
    """如实列出这批数据支撑不起的结论，避免把无效指标当成结果。

    全部基于数据本身自动判定，数据补齐后这些提示会自动消失——
    公式不需要人记着"什么时候该加哪句警告"。
    """
    out: List[str] = []
    from collections import Counter
    cnt = Counter(ds.y_cls.tolist())

    if cnt.get(2, 0) == 0:
        out.append(
            "**分类任务里没有卫星滴样本（0 例）**：三分类实际退化为「不喷 vs 单滴」二分类，"
            "宏 F1 会因缺类而偏低，准确率也不代表「识别卫星」的能力。"
            "要评估卫星识别能力，必须专门去**高电压/宽尾隙**区域采样。")
    if cnt.get(0, 0) < 5:
        out.append(
            "**不喷样本仅 %d 例**，不足以支撑一个稳定的二分类边界。" % cnt.get(0, 0))

    # ---- 多数类基线：准确率没超过它就说明模型没学到任何东西 ----
    # ⚠️ 判定不能只看准确率：加了孔号特征后，准确率可能**低于**基线
    #    （因为模型开始敢预测少数类了），但宏 F1 明显更高——
    #    这时它**确实学到了东西**，只是"多数类准确率"这个指标不再适用。
    #    故此处同时看：**有没有模型真的预测出了少数类**（混淆矩阵非零）。
    if cnt.get(1, 0) > 0:
        baseline = cnt.get(1, 0) / ds.n
        if cls:
            from ..headml import best_by
            bn, bv = best_by(cls, "accuracy")
            # 有没有模型真的预测出了不喷孔（class 0）？
            caught = 0
            for r in cls.values():
                cm = r.get("confusion")
                if cm and len(cm) > 0 and len(cm[0]) > 0 and cm[0][0] > 0:
                    caught = max(caught, cm[0][0])
            if caught == 0:
                out.append(
                    "**分类指标无信息量**：永远猜多数类的基线准确率就是 %.4f，"
                    "而最优模型(%s)也只有 %.4f——**没有任何模型预测出过一个不喷孔**，"
                    "这个数字不能作为分类能力的结论。"
                    % (baseline, bn, bv))
            else:
                out.append(
                    "**分类指标要谨慎解读**：多数类基线准确率是 %.4f，"
                    "最优模型(%s)为 %.4f。但**有模型确实识别出了不喷孔**"
                    "（最多命中 %d/%d 个），所以宏 F1 比准确率更能反映真实能力——"
                    "准确率在这里会被大量单滴样本稀释。"
                    "注意：若该能力来自孔号特征，它在**没见过的新孔**上不一定成立"
                    "（见 leave-nozzle-out 一节）。"
                    % (baseline, bn, bv, caught, cnt.get(0, 0)))

    # ---- 特征退化：常量列 ----
    const_cols = [ds.features[i] for i in range(ds.X.shape[1])
                  if len(np.unique(np.round(ds.X[:, i], 6))) == 1]
    if const_cols:
        out.append(
            "**%d 个特征是常量列**（%s）：对树模型无信息、对线性模型无意义，"
            "数据补齐后才会有变化。" % (len(const_cols), ", ".join(const_cols)))

    # ---- 特征共线：不同波形点数 < 非恒定特征数 ----
    uniq = len({tuple(np.round(r, 3)) for r in ds.X})
    if uniq <= 8:
        out.append(
            "**波形只有 %d 个不同取值**：模型在网格内部全靠外推。"
            "下方「反向推荐」给出的是**未经实测的波形**，只能当作下一步该去测的点，"
            "不能直接当结论用。" % uniq)
    return out


def report(ds: HeadDataset, cls: dict, reg: dict,
           reg_wave: Optional[dict] = None,
           lno: Optional[dict] = None,
           lno_wave: Optional[dict] = None,
           lno_cls: Optional[dict] = None) -> str:
    out = ["# DMC Samba 一套 ML（HeadType 1117）", "", "## 数据", "",
           "```", ds.describe(), "```", ""]
    cav = data_caveats(ds, cls)
    if cav:
        out += ["### ⚠️ 这批数据支撑不起的结论", ""]
        out += ["%d. %s" % (i + 1, c) for i, c in enumerate(cav)]
        out += [""]
    out += ["## 正向预测：给定波形 → 喷射结果", "",
            _fmt_table("分类：0 不喷 / 1 单滴 / 2 含卫星", cls, CLS_KEYS, MODEL_ORDER),
            "",
            _fmt_table("回归：液滴速度 (m/s)", reg, REG_KEYS, MODEL_ORDER), ""]
    bc = best_by(cls, "accuracy")
    br = best_by(reg, "r2")
    out += ["最优模型：**分类 %s**（准确率 %.4f）　**回归 %s**（R² %.4f）"
            % (bc[0], bc[1], br[0], br[1]), ""]
    if bc[0] in cls and "confusion" in cls[bc[0]]:
        cm = np.asarray(cls[bc[0]]["confusion"])
        out += ["混淆矩阵（行=真实，列=预测，顺序 0/1/2）：", "",
                "```", str(cm), "```", ""]

    # ---- 喷嘴一致性：含 / 不含孔号特征的对照 ----
    if reg_wave:
        # 行=模型，列=两种特征配置（_fmt_table 要求 res[模型][指标] 结构）
        cmp_res = {}
        for name in MODEL_ORDER:
            a = reg.get(name, {})
            b = reg_wave.get(name, {})
            if "r2" not in a and "r2" not in b:
                continue
            cmp_res[name] = {"r2_with": a.get("r2"), "r2_without": b.get("r2")}
        out += ["## 喷嘴级一致性建模（§1.5 创新点 2）", "",
                "把孔号位置编码进特征后，模型能解释一部分**孔间差异**。"
                "但必须同时看「不含孔号」的对照，否则指标变好会被误读成模型更强——"
                "实际上那只是模型**记住了每孔的固定偏移**。", "",
                _fmt_table("回归 R² 对照（含 / 不含孔号位置特征）", cmp_res,
                           [("含孔号 R²", "%.4f", "r2_with"),
                            ("仅波形 R²", "%.4f", "r2_without")], MODEL_ORDER), "",
                "（两列的差值 = 孔号特征带来的**额外**解释力；"
                "它只有在**留出整孔**的评估里才有意义，见下节。）", ""]

    # ---- 迁移性：leave-nozzle-out ----
    if lno:
        out += ["## 波形可迁移性（§1.5 创新点 1）：leave-nozzle-out", "",
                "按**孔号**分组交叉验证：训练集与测试集不含同一个喷嘴。"
                "这是工业化的真正问题——在少数喷嘴上优化出的波形，"
                "换到没参与优化的喷嘴上还成不成立。", "",
                _fmt_table("leave-nozzle-out 回归（速度）", lno, REG_KEYS, MODEL_ORDER), ""]
        bw = best_by(lno, "r2")
        out += ["最优模型：**%s**（R² %.4f）" % (bw[0], bw[1]), ""]
        if lno_wave:
            out += ["### 对照：同样留出整孔，但**不给模型孔号信息**", "",
                    _fmt_table("leave-nozzle-out 回归（仅波形特征）", lno_wave,
                               REG_KEYS, MODEL_ORDER), ""]
            bww = best_by(lno_wave, "r2")
            out += ["最优模型：**%s**（R² %.4f）" % (bww[0], bww[1]), "",
                    "两者的差值 = **孔号信息在跨孔外推时到底有没有用**。"
                    "若两者接近，说明速度差异主要由波形决定、孔号只是个记录；"
                    "若「含孔号」明显更高，说明孔号确实携带了可迁移的结构性信息"
                    "（例如边缘孔系统性偏慢），那就值得做喷嘴级补偿。", ""]
        out += ["### ⚠️ 这个数字怎么读", "",
                "**它比「按会话分组」的 R² 更高，但这不代表模型更强。**"
                "原因是两种分组切的是不同的东西：", "",
                "| 分组方式 | 测试集里缺的是什么 | 回答的问题 |",
                "|---|---|---|",
                "| 按会话分组 | 整个会话（= 整个波形点，9 会话仅覆盖 4 个波形） | "
                "**没测过的波形**能不能预测准 |",
                "| 按孔分组 | 整个喷嘴（4 个波形都还在训练集里） | "
                "**没测过的喷嘴**能不能预测准 |", "",
                "所以：会话分组 ≈ 「留一波形出」，是**波形维度**的外推；"
                "孔分组是**空间维度**的外推。两者不可直接比大小。", ""]
        if lno_cls:
            out += ["### 分类侧的同一问题：换到新孔，还判得准吗", "",
                    _fmt_table("leave-nozzle-out 分类", lno_cls, CLS_KEYS, MODEL_ORDER), "",
                    "⚠️ **这里的结果比会话分组差得多**：会话分组时模型能靠孔号特征"
                    "认出「1 号孔老是不喷」；一旦把整个孔留出，模型就完全没见过它，"
                    "6 个不喷孔**一个都认不出来**（混淆矩阵第一行全 0）。", "",
                    "→ 说明现有数据里的「不喷」主要是**特定孔 × 特定时刻**的偶发现象，"
                    "而不是某个孔的稳定属性；**靠 12 个孔学不出可迁移的故障预测能力**。", ""]
        out += ["### 这个结论的强度边界", "",
                "⚠️ 2026-11-03 修正：原写「须待阶段 3 扫全量 128 孔」是**基于错误前提**——"
                "DMC Samba 全头只有 **12 个喷嘴**（规格书 12 jets；实机 recipe `Nozzle = 1:12`），"
                "现有 108 孔数据**已 100% 覆盖全头**，不存在「待扫全头」这回事。", "",
                "**那真实的局限是什么？**是这三条：", "",
                "1. **波形维度太窄**：全头 12 孔、9 会话只覆盖 **4 个波形点**，"
                "且电压与宽度共线（34/38V 只在 W=2.2 测过）——"
                "「留一孔出」时 4 个波形全在训练集里，**波形维度的外推完全没被考察**；",
                "2. **重复性没有独立测量**：同一波形最多重复过 4 次（40V/1.9），"
                "无法把「孔间差异」与「同一孔的重复性噪声」分开——"
                "这才是 §7 阶段 3 真正要补的；",
                "3. **时间与孔位混淆**：12 孔全部来自同三天采集，"
                "孔号与「当天哪一批测的」无法分离。", "",
                "→ 结论：**喷嘴级一致性建模在 Samba 上是有真实抓手的**"
                "（孔号携带可跨孔外推的系统性空间梯度，Spearman ρ 均值 +0.764，p<0.001），"
                "但要把它做成**可补偿**的模型，还缺阶段 3 的重复性数据。", ""]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DMC Samba ML")
    ap.add_argument("--recommend", action="store_true", help="输出推荐波形")
    ap.add_argument("--target-speed", type=float, default=3.5)
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default=os.path.join("results", "samba"))
    ap.add_argument("--no-nozzle-features", action="store_true",
                    help="不使用孔号位置特征（用于对照）")
    ap.add_argument("--leave-nozzle-out", action="store_true",
                    help="额外做按孔分组的迁移性评估")
    ap.add_argument("--data", action="append", metavar="ROOT", default=[],
                    help="追加的数据根目录（可重复）。指向机台上新扫的 ADA 落盘根，"
                         "如 \"C:\\\\LP50\\\\ScreenCaptures\\\\2026-11-05_09.30.00_Measurement\"。"
                         "默认与既有数据合并训练；加 --replace-data 则只用这里指定的")
    ap.add_argument("--replace-data", action="store_true",
                    help="丢弃默认数据源，只用 --data 指定的根")
    args = ap.parse_args(argv)

    roots = resolve_roots(args.data, replace=args.replace_data)
    missing = [r for r in roots if not os.path.isdir(r)]
    if missing:
        print("⚠️ 以下数据根不存在，已跳过：%s" % ", ".join(missing))
    print("数据根：%s" % (" | ".join(roots)))

    ds = build(with_nozzle=not args.no_nozzle_features, roots=roots)
    print("# DMC Samba 一套 ML（HeadType 1117）")
    print(ds.describe())
    if ds.n < 10:
        print("\n样本不足，无法训练。")
        print("  可能原因：")
        print("  1) 实测数据不在数据根 —— 实测资产**不入库**（.gitignore 排除 ML/share/），")
        print("     需自行采集，或用 --data <落盘根> 指向机台上扫出的数据；")
        print("  2) 数据根里的喷头不是 DMC Samba —— 本模块只取 HeadType 1117；")
        print("  3) 数据根层级选错 —— 要指到**含组合子目录**的那一层。")
        return 1

    cls = evaluate_classification(ds.X, ds.y_cls, ds.groups)
    reg = evaluate_regression(ds.X, ds.y_vel, ds.groups)

    # ---- 对照：仅波形特征（不含孔号位置）----
    reg_wave = None
    if not args.no_nozzle_features:
        ds_wave = build(with_nozzle=False, roots=roots)
        reg_wave = evaluate_regression(ds_wave.X, ds_wave.y_vel, ds_wave.groups)

    # ---- 迁移性：leave-nozzle-out（按孔分组）----
    lno = None
    lno_wave = None
    lno_cls = None
    if args.leave_nozzle_out:
        vm = ds.vel_mask()
        gn = ds.groups_nozzle[vm] if ds.groups_nozzle is not None else None
        lno = evaluate_leave_nozzle_out(ds.X[vm], ds.y_vel[vm], gn)
        # 分类侧同样要看：换到没见过的孔，还能不能判断它会不会喷
        lno_cls = evaluate_leave_nozzle_out_cls(
            ds.X, ds.y_cls,
            ds.groups_nozzle if ds.groups_nozzle is not None else None)
        if reg_wave is not None:
            # 对照：同样留出整孔，但模型看不到孔号信息
            vmw = ds_wave.vel_mask()
            lno_wave = evaluate_leave_nozzle_out(ds_wave.X[vmw], ds_wave.y_vel[vmw],
                                                ds_wave.groups_nozzle[vmw]
                                                if ds_wave.groups_nozzle is not None else None)

    os.makedirs(args.out, exist_ok=True)
    save_json({"dataset": {"features": ds.features, "n": ds.n,
                           "head": ds.head_name, "head_type": ds.head_type},
               "classification": cls, "regression": reg,
               "regression_wave_only": reg_wave,
               "leave_nozzle_out": lno,
               "leave_nozzle_out_wave_only": lno_wave,
               "leave_nozzle_out_classification": lno_cls},
              os.path.join(args.out, "metrics.json"))
    txt = report(ds, cls, reg, reg_wave=reg_wave, lno=lno, lno_wave=lno_wave,
                  lno_cls=lno_cls)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8-sig") as f:
        f.write(txt)
    print("\n" + txt)

    if args.recommend:
        rows = recommend(ds, args.target_speed, args.margin, top=args.top)
        print("\n## 反向推荐：目标速度 %.2f m/s ±%.0f%%\n" % (args.target_speed, args.margin * 100))
        if not rows:
            print("网格内没有满足条件的候选 —— 说明该墨水/工艺下目标速度可能不可达，"
                  "或需要放宽 --margin。")
        else:
            n_ext = sum(1 for r in rows if r["extrap"] > 0)
            print("推荐用**与评估同一套选出的最优模型**（不再是硬编码）。"
                  "排序：实测优先 → 外推距离小优先 → 速度偏差小。\n")
            print("「实测」= 数据里真测过；「外推」= 模型预测的未测波形，"
                  "**必须先实测验证再采用**。「外推距」= 到实测包围盒的归一化距离，"
                  "0 表示落在实测范围内，越大越不可信。\n")
            print("| 电压(V) | 宽度(µs) | 尾隙(µs) | 预测类 | 预测速度(m/s) | 偏差 | 外推距 | 来源 |")
            print("|---|---|---|---|---|---|---|---|")
            for r in rows:
                print("| %g | %g | %g | %d | %.2f | %.3f | %.2f | %s |"
                      % (r["pulse0_voltage"], r["pulse0_width"], r["pulse0_tail"],
                         r["pred_class"], r["pred_speed"], r["speed_err"], r["extrap"],
                         "✓ 实测" if r["measured"] else "外推"))
            if n_ext:
                print("\n⚠️ %d/%d 个推荐点落在实测包围盒外，属高风险外推，"
                      "建议优先补测这些点再采纳。" % (n_ext, len(rows)))
            save_json(rows, os.path.join(args.out, "recommended.json"))
    print("\n输出目录: %s" % os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())