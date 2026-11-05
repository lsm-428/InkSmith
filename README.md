# 墨匠 InkSmith

**喷墨驱动波形的机器学习设计** · *ML-driven inkjet drive waveform design*

基于 OLPEN LP50 落滴分析（ADA / DropQCS）实测数据，用机器学习反推压电喷头的
驱动波形，把「人工试配方」变成「数据驱动的波形推荐 + 闭环验证」。

## 目录

```
├── PROJECT_RECORD.md      # 单一事实来源：全部结论、勘误、实测数据
├── QUICKSTART.md          # 操作手册：四步循环（设计→扫描→排序→推荐）+ 命令速查
├── requirements.txt       # 运行依赖
├── inkjet_ml/             # 主包
│   ├── dataset.py         # 读取 ADA 落盘的 Results.csv / Measurement.xml
│   ├── headml.py          # 5 个模型 + 分组交叉验证（三套喷头共用，可横向比较）
│   ├── ada.py             # 生成 ADA 扫描配方（扫哪些参数、填什么范围）
│   ├── ada_rank.py        # 读回扫描结果，按 DropQCS 判据给波形组合排序
│   ├── ncalc.py           # NCalc 表达式求值器（纯 Python，AST 白名单）
│   ├── dropqcs.py         # DropQCS 规则集执行器（孔级 + 整头 + [Result] 链式）
│   ├── dropqcs_vars.py    # 从 Results.csv 还原 20+ 官方 DropQCS 变量
│   ├── inkprops.py        # 墨水物性：Z 数（Ohnesorge 倒数）计算与反解
│   └── heads/             # samba.py / km512.py / km1024i.py 喷头专属模型
├── docs/evidence/         # 论文 PDF、Table 原始数据（.ods）、论文分享 PPT
└── explore/               # 数据探索脚本与标签
```

## 快速开始

```bash
uv venv && uv pip install -r requirements.txt

# ===== 设计扫描方案 =====
uv run python -m inkjet_ml.ada --verify            # 配置自检（19/19）
uv run python -m inkjet_ml.ada --list-heads        # 列出支持的喷头
uv run python -m inkjet_ml.ada --plan screen       # DOE 方案预览
uv run python -m inkjet_ml.ada --config screen     # 可照填的 recipe 片段

# ===== 读回结果 =====
uv run python -m inkjet_ml.ada_rank "<落盘根>" --rules "<RulesConfig.xml>"   # 官方规则判合格

# ===== 训练与推荐 =====
uv run python -m inkjet_ml.heads.samba --recommend --target-speed 3.5
uv run python -m inkjet_ml.heads.samba --recommend --target-speed 3.5 --leave-nozzle-out
uv run python -m inkjet_ml.heads.samba --recommend --data "<新落盘根>"       # 接入新扫数据

# ===== 墨水 =====
uv run python -m inkjet_ml.inkprops --table1       # Z 数实现自检

# ===== 自检 =====
uv run python -m unittest inkjet_ml.test_ncalc        # 规则求值器（16 项）
uv run python -m unittest inkjet_ml.test_data_roots   # 数据接入链路（12 项）
```

四步循环（生成扫描方案 → 机台执行 → 读回排序 → 训练推荐）与故障对照见
[`QUICKSTART.md`](QUICKSTART.md)；结论与依据见 [`PROJECT_RECORD.md`](PROJECT_RECORD.md)。

## 覆盖的喷头

| 喷头 | 物理孔数 | 备注 |
|---|---|---|
| DMC Samba | 12 | 75 DPI，工作温度上限 60 °C |
| KM512 | 512 | |
| KM1024i | 1024 | |
| Spectra S-class（SE·SM·SL） | 128 | |
| SX3 | 128 | 已停产 |
| Canon C29 | 256 | |

> `DropAnalysis.Nozzle` 是**要测哪些孔的扫描范围**，不等于喷头总孔数。

## 关键设计取舍

- **recipe 参数名随喷头而变**：通用参数（`DropAnalysis.X`）在
  DMCSamba / CanonC29 / SpectraS-class 里带 `[0]` 下标，在 KM512 / KM1024i 里不带；
  喷头专属参数（`DropAnalysis.DMC.Samba.*`）一律不带。写错 → ADA **静默不扫描**。
  `ada.py` 从真实 recipe 反查，不再手写。
- **目标速度默认 3.5 m/s**（不是 5.0）：后者会让合格率普遍为 0。
- **Z 数 = 1/Oh = √(ργD)/η**，其中 D 取**喷孔直径**（非滴直径）。
  ρ / η / γ 三个物性量 LP50 **测不到**，必须外部送检后当特征导入。
- **每个延时点都是不同次发射的液滴**，不是同一滴的轨迹；延时越接近 0 越难被
  ADA 判为"液滴"，样本量小。
- **DropQCS 判据用纯 Python 求值**（`ncalc.py` / `dropqcs.py`，零新依赖），
  不用 .NET 的 `NCalc.dll`——后者只在 V4730 新基线机台上有，且要求装 .NET Runtime。
  官方 8 套预置规则 + Default 全部可离线执行。
- **「整头」= 一个测量场景**（一份 `Results.csv`）：官方相对判据里的
  `[PrintheadSpeedAverage]` 只在本场景内汇总，**绝不跨组合池化**——
  池化会让每个组合自比自，判据退化成"单组合内离散度"。

## 当前结论摘要

| 结论 | 数值 |
|---|---|
| 实测数据规模 | 38 会话 / 437 孔 / 6 类墨水（DMC Samba，2026-11-05 扩容后） |
| 形态分布 | 单滴 343 / 不喷 76 / 含卫星 18（2 滴 16 + 3 滴 2） |
| 方差分解 | 会话间 88.3% / 会话内 11.7%（108 孔口径，见 PROJECT_RECORD §8.6.6） |
| 波形组合 | 全库 12 个（每墨水 1–4 个），仍非正交 DOE |
| 分组交叉验证（437 孔实跑） | 分类 kNN acc 0.80 / 宏F1 0.41；回归 XGBoost R² 0.25 / RMSE 0.70 m/s |

**当前瓶颈是数据不是代码**：跨墨水后回归 R² 从单墨水 0.81 掉到 0.25——
速度由墨水（Z/黏度）、温度、压力共同决定而特征里还没有墨水维度；
六支墨水的 Z 已可算（`docs/evidence/ink_properties.md`），加 `Z` 特征是最直接的下一步。
卫星 18 例在分组 CV 下仍全部漏判，需按墨水补采高 Z/多脉冲区样本。

> **实测数据不入库**（机台本地资产，见 `.gitignore`）。复现需要自行采集，
> 或用 `explore/labels.csv` 里的同源标签表。

## 许可

[AGPL-3.0](LICENSE) — Copyright (c) 2026 lsm-428

> 修改后经网络提供服务，须按第 13 条提供对应源码。

