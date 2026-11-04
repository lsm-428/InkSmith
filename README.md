# 墨匠 InkSmith

**喷墨驱动波形的机器学习设计** · *ML-driven inkjet drive waveform design*

基于 OLPEN LP50 落滴分析（ADA / DropQCS）实测数据，用机器学习反推压电喷头的
驱动波形，把「人工试配方」变成「数据驱动的波形推荐 + 闭环验证」。

> 本仓库的 `ML/` 目录是全部内容来源。原 LP50 机台仓库（`lsm-428/LP50`）另存有
> AutoADA 集成、Recipes 机台配方与官方文档，不在此重复。

## 目录

```
ML/
├── PROJECT_RECORD.md      # 单一事实来源：全部结论、勘误、实测数据
├── inkjet_ml/             # 主包
│   ├── dataset.py         # 读取 ADA 落盘的 Results.csv / Measurement.xml
│   ├── headml.py          # 5 个模型 + 分组交叉验证（三套喷头共用，可横向比较）
│   ├── ada.py             # 生成 ADA 扫描配方（扫哪些参数、填什么范围）
│   ├── ada_rank.py        # 读回扫描结果，按 DropQCS 判据给波形组合排序
│   ├── inkprops.py        # 墨水物性：Z 数（Ohnesorge 倒数）计算与反解
│   └── heads/             # samba.py / km512.py / km1024i.py 喷头专属模型
├── docs/evidence/         # 论文 PDF、Table 原始数据（.ods）、论文分享 PPT
└── explore/               # 数据探索脚本与标签
```

## 快速开始

```bash
cd ML
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt

.venv/Scripts/python.exe -m inkjet_ml.ada --verify            # 配置自检（19 项）
.venv/Scripts/python.exe -m inkjet_ml.ada --list-heads        # 列出支持的喷头
.venv/Scripts/python.exe -m inkjet_ml.ada --plan screen       # DOE 方案预览
.venv/Scripts/python.exe -m inkjet_ml.ada --config screen     # 生成可照填的 recipe 片段

.venv/Scripts/python.exe -m inkjet_ml.heads.samba --recommend --target-speed 3.5
.venv/Scripts/python.exe -m inkjet_ml.heads.samba --recommend --target-speed 3.5 --leave-nozzle-out
.venv/Scripts/python.exe -m inkjet_ml.ada_rank "share/NOA89-test-05" --target-speed 3.5

.venv/Scripts/python.exe -m inkjet_ml.inkprops --table1        # Z 数实现自检
```

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

## 当前结论摘要

| 结论 | 数值 |
|---|---|
| 实测数据规模 | 9 会话 × 12 孔 = 108 孔 |
| 形态分布 | 单滴 102 / 不喷 6 / 卫星滴 0 |
| 方差分解 | 会话间 88.3% / 会话内 11.7% |
| 唯一波形点 | 4 组 |
| leave-nozzle-out 回归 R² | 含孔号 0.9433 / 仅波形 0.8897 |
| 分组交叉验证宏 F1 | 含孔号 0.7135 / 不含 0.4857 |

**当前瓶颈是数据不是代码**：现有 108 孔是「成滴窗口切片」，卫星滴 0 例、
波形只有 4 个点、电压与宽度共线。

## 许可

代码部分随本仓库沿用上游约定；第三方论文与表格仅供本地查阅，未入库。
