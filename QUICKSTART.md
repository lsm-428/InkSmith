# 快速开始（QUICKSTART）

> 本项目用机器学习建立「墨水 + 驱动波形 → 喷射结果」的预测模型，并闭环反解最优驱动波形。
> 平台：LP50 全自动喷墨打印系统（ADA 落滴自动分析 + DropQCS 落滴质检）。
>
> **本文档只讲"怎么操作"。** 结论与依据在 [`PROJECT_RECORD.md`](PROJECT_RECORD.md)；
> 论文原文与原始表格在 [`docs/evidence/`](docs/evidence/README.md)
> （平台官方手册仅本机查阅，不入库）。
>
> 文中命令均以 `uv run` 执行，输出为**实跑截取**（2026-11-04，Windows + UV 环境）。
> 环境准备：`uv venv` → `uv pip install -r requirements.txt`。

---

## 0. 这个仓库里有什么

本目录就是 git 仓库根，项目代码与文档全在这里。

| 位置 | 内容 | 入库 |
|---|---|---|
| `inkjet_ml/`、`*.md`、`explore/labels.csv` | 源码与文档 | ✅ |
| `share/` | DMC Samba 实测数据（9 会话 / 108 孔） | ❌ `.gitignore` |
| `results/`、`.venv/`、`.workbuddy/` | 训练产物 / 环境 / 本地状态 | ❌ `.gitignore` |

机台新扫的数据不在此目录，用 `--data <落盘根>` 接进训练（见 §4）。

---

## 0.1 一分钟看懂全流程

```
   ① 生成扫描方案          ② 机台执行 ADA          ③ 读回结果排序        ④ 训练并反向推荐
   ada.py --plan screen  →  HMI 里跑 ADA       →   ada_rank <落盘根>  →  heads/samba --recommend
   ada.py --config narrow    （人在机台边）        （按判据排波形组合）    （给目标速度 → 推最优波形）
        ↓                                                                            │
        └────────────────── ④ 推荐的波形成为下一轮扫描范围 ←────────────────────────────┘
```

四个模块的分工：

| 模块 | 职责 | 何时用 |
|---|---|---|
| `inkprops.py` | 墨水物性（ρ/η/γ）→ **Z 数** | 换墨水后，算 Z |
| `ada.py` | **扫什么参数、填什么范围** | 每次上机前 |
| `ada_rank.py` | **读回落盘结果、排序、判合格** | 机台扫完 |
| `heads/samba.py` | **训练模型 + 反向推荐波形** | 攒够新数据后 |

---

## 1. 第 ① 步：生成扫描方案

### 1.1 先看这版机器能扫什么

```powershell
uv run python -m inkjet_ml.ada --dims
```

输出（节选，真实解析自 V4730 的 `95 ADA` 流程）：

```
## DMCSamba 可扫描维度（真实解析自 95 ADA 流程，基线 V4730 / HMI 4.7.3）

| # | UpdateSpeed | 勾选 | 参数全名                       | 当前值        | 单位 | 水平数 | 来源     |
|---|-------------|------|--------------------------------|---------------|------|--------|----------|
| 2 | 8            | □    | DropAnalysis.HeadTemperature   | 30:5:55       | ºC   | 6      | 流程默认 |
| 4 | 7            | □    | DropAnalysis.InkPressure       | -25:2.5:-10   | mbar | 7      | 流程默认 |
| 9 | 3            | □    | DropAnalysis.DMC.Samba.Pulse0.Voltage | 10:2.5:40 | V  | 13     | 流程默认 |
| 10| 3            | □    | DropAnalysis.DMC.Samba.Pulse0.Width   | 1:0.1:4   | µs   | 31     | 流程默认 |
```

> ⚠️ **别照抄默认区间**：`.Nozzle` 的默认值 `1:128` 对 Samba（物理 **12** 孔）是越界，
> 对 KM512（512 孔）只是"只测前 128 个"。用 `--plan` 给出的方案，别用默认值。

### 1.2 看扫描方案（5 档，按顺序推进）

```powershell
uv run python -m inkjet_ml.ada --plan screen      # 阶段1 粗筛：找可行波形域
uv run python -m inkjet_ml.ada --plan narrow      # 阶段2 围绕最优点局部加密
uv run python -m inkjet_ml.ada --plan window      # 阶段2b 压力/温度独立成维
uv run python -m inkjet_ml.ada --plan delay       # 阶段2c 成滴形貌延时序列（LEDDelay 0–220 µs）
uv run python -m inkjet_ml.ada --plan uniform     # 阶段3 全头重复性扫描
```

`--plan screen` 实际输出（节选）：

```
## 阶段1 · 粗筛可行波形域（电压×宽度正交 · 4 维稀疏网格）

| 勾选 | 参数全名                          | 要填的范围/序列      | 单位 | 取值数 |
|------|-----------------------------------|----------------------|------|--------|
| ✓    | DropAnalysis.DMC.Samba.Pulse0.Voltage | 20;26;32;40        | V    | 4      |
| ✓    | DropAnalysis.DMC.Samba.Pulse0.Width   | 1.4;2.0;2.6;3.4    | µs   | 4      |
| ✓    | DropAnalysis.DMC.Samba.Pulse0.Tail    | 1.4;2.6;4.0;6.0    | µs   | 4      |
| ✓    | DropAnalysis.DMC.Samba.Slewrate       | 30;40;50           | V/µs | 3      |
| □    | DropAnalysis.InkPressure             | -1                 | mbar | 1（固定在实测工作点）|

**波形参数组合数** = 192 张/孔 × **4 孔** = **768 张 noz*.png**
```

### 1.3 导出可直接照填 HMI 的 recipe

```powershell
uv run python -m inkjet_ml.ada --config narrow
```

输出是**完整 XML**，直接抄进 HMI 的 ADA recipe：

```xml
<Recipe xmlns:xsd="..." xmlns:xsi="...">
  <Name>阶段2 · 围绕最优点局部加密（含第二脉冲必要性判定）</Name>
  <FlowFile>Advanced-Drop-Analysis-DMCSamba.xml</FlowFile>
  <ParameterValues>
    <Parameter Name="DropAnalysis.BaseFolder[0]" Value="C:\LP50\ScreenCaptures" />
    <Parameter Name="DropAnalysis.Nozzle.Checkbox[0]" Value="1" />
    <Parameter Name="DropAnalysis.Nozzle[0]" Value="1;2;3;4;5;6" />
    <Parameter Name="DropAnalysis.DMC.Samba.Pulse0.Voltage" Value="26;28;30;32" />
    <Parameter Name="DropAnalysis.DMC.Samba.Pulse0.Voltage.Checkbox" Value="1" />
    ...
  </ParameterValues>
</Recipe>
```

**照填要点**：
- `Checkbox = 1` 的参数才进扫描，**没勾的维持原值**；
- 参数名后缀 `[0]` 与否，**不同喷头不一样**（`ada.py` 已按真实 recipe 反查，不要手改）；
- `InkPressure = -1` 是**故意**超出 ADA 默认区间（-25…-10）——真实工作点必须被包含，
  否则新数据与既有 108 孔工作点不同、无法合并。

### 1.4 校验填得对不对

```powershell
uv run python -m inkjet_ml.ada --verify --head DMCSamba
```

应输出 `19/19 全部一致`（与实机 flow 文件交叉校验）。

---

## 2. 第 ② 步：机台执行 ADA

**这一步没有代码**——在 HMI 4.7.3 里跑 ADA，人在机台边。

产出落在 `DropAnalysis.BaseFolder`（默认 `C:\LP50\ScreenCaptures\`）下，
**每个波形组合一个独立子目录**，目录名用 `FolderPrefix` 编码参数
（`P0V24` = Pulse0 电压 24V、`P0W2.2` = 宽度 2.2µs、`noz1` = 1 号孔）：

```
<落盘根>\
    P0V40\P0W2.2\P0T4.5\
        Measurement.xml      ← 该组合的真实设定值
        Results.csv          ← 逐孔读数
        noz1.png / noz1.xml  ← 图像 + 单孔测量
        ...
```

---

## 3. 第 ③ 步：读回结果并排序

```powershell
uv run python -m inkjet_ml.ada_rank "<落盘根>"
```

输出（真实跑，share 数据）：

```
共解析 9 个扫描组合（每个组合 = 一个目录下的一个 Results.csv）
判据：NrDroplets==1 且 2.80<=Speed<=4.20 m/s 且 |Angle|<=2.0°

波形组合                                       孔数   不喷   单滴   卫星   合格率   速度均  速度σ  体积均(pl)
----------------------------------------------------------------------------------------------
(单波形)                                        12     0     12     0  100.0%    3.69  0.221     3.53
(单波形)                                        12     0     12     0  100.0%    3.81  0.209     3.35
(单波形)                                        12     0     12     0  100.0%    3.83  0.174     3.70
```

### 用官方 DropQCS 规则判（推荐）

机台自带的 **8 套示例规则 + `Default`**（共 9 个 `*.RulesConfig.xml`）可直接用：

```powershell
uv run python -m inkjet_ml.ada_rank "<落盘根>" --rules "D:\Oleton\Inkjet\LP50 V4730 4-2026 & Canon FAT\Rules\Examples\SingleDroplet.RulesConfig.xml"
```

| 规则 | 判什么 |
|---|---|
| `SingleDroplet` | 恰好 1 滴、无卫星 |
| `AtLeastOneDroplet` | 至少 1 滴（允许卫星） |
| `AverageSpeedWithinMargin` | 速度在**本场景**均值 ±20% 内 |
| `PerformMaintenance` | 0 滴或 >2 滴 → 触发维护动作 |
| `NestedNozzleRule` / `DualHead` / … | `[Result]` 链式 / 双头判据 |

> ⚠️ **整头语义 = 一个测量场景**（一份 `Results.csv`）。
> `[PrintheadSpeedAverage]` 只在本场景内汇总，**绝不跨组合池化**——
> 池化会让每个组合自比自，判据退化成"单组合内离散度"，失去"该孔是否偏离本头平均水平"的意义。

### 导出排序表

```powershell
uv run python -m inkjet_ml.ada_rank "<落盘根>" --top 20 --csv 排序表.csv
```

---

## 4. 第 ④ 步：训练 + 反向推荐

```powershell
uv run python -m inkjet_ml.heads.samba --recommend --target-speed 3.5
```

输出（真实跑，截选）：

```
数据根：D:\Gitee\LP50\ML\share
样本: 108  特征: 7  会话: 9
类别分布: 不喷 6 / 单滴 102 / 卫星 0

## 反向推荐：目标速度 3.50 m/s ±20%

| 电压(V) | 宽度(µs) | 尾隙(µs) | 预测类 | 预测速度(m/s) | 偏差 | 外推距 | 来源 |
|---|---|---|---|---|---|---|---|
| 34 | 2.2 | 4.5 | 1 | 3.11 | 0.392 | 0.00 | ✓ 实测 |
| 38 | 2.2 | 4.5 | 1 | 2.82 | 0.684 | 0.00 | ✓ 实测 |
| 40 | 2   | 4.5 | 1 | 3.82 | 0.319 | 0.00 | 外推   |

⚠️ 13/20 个推荐点落在实测包围盒外，属高风险外推，建议优先补测这些点再采纳。
```

**怎么读这张表**：
- `来源 = ✓ 实测` → 数据里真测过，可信；
- `来源 = 外推` → 模型猜的，**必须先实测验证再采用**；
- `外推距` → 到实测包围盒的归一化距离，0 = 落在实测范围内，越大越不可信。

### 把新扫的数据接进来（2026-11-04 新增）

```powershell
# 默认：既有数据 + 新扫数据 合并训练
uv run python -m inkjet_ml.heads.samba --recommend --target-speed 3.5 `
    --data "<落盘根>"

# 只用新扫的数据（丢弃默认源）
uv run python -m inkjet_ml.heads.samba --recommend --target-speed 3.5 `
    --data "<落盘根>" --replace-data

# 多个根可重复给 --data
uv run python -m inkjet_ml.heads.samba --data "<根A>" --data "<根B>"
```

`--data` 三个喷头通用（`samba` / `km512` / `km1024i`），语义见 `dataset.resolve_roots()`。

### 其他有用的开关

```powershell
# 迁移性评估：换到没见过的孔还能不能判断（§1.5 创新点 1）
uv run python -m inkjet_ml.heads.samba --leave-nozzle-out

# 对照：不给模型孔号信息
uv run python -m inkjet_ml.heads.samba --no-nozzle-features
```

---

## 5. 换墨水：算 Z 数

```powershell
uv run python -m inkjet_ml.inkprops --table1                    # 论文 Table 1 自检（12 种墨水）
uv run python -m inkjet_ml.inkprops --rho 1071 --eta 3.45 --gamma 56.4 --d 40   # 正算 Z
# 反解喷孔直径（--solve 必须配 --z）
uv run python -m inkjet_ml.inkprops --rho 1071 --eta 3.45 --gamma 56.4 --z 14.25 --solve d
```

`--table1` 会自检实现并给出结论：

```
  D 取 40 µm 时：max|误差| = 1.40%, 平均 = 0.24%
  反解等效 D：39.74–41.15 µm，均值 40.08 µm（论文喷嘴标称 40 µm）
  ⇒ 公式中的 D 是**喷孔直径**，不是滴直径（论文正文措辞不严谨）
```

> ⚠️ **论文那 12 种墨水不是你的生产墨水。**
> LP50 的生产墨水不在 Table 1 内，ρ/η/γ 需**另行实测**
> （锥板黏度 / 气泡压力法表面张力 / 密度）——这是跨墨水泛化的硬前提。

---

## 6. 常用命令速查

```powershell
# ===== 设计扫描方案 =====
uv run python -m inkjet_ml.ada --dims                    # 这版机器能扫什么
uv run python -m inkjet_ml.ada --plan screen             # 粗筛方案
uv run python -m inkjet_ml.ada --config narrow           # 可照填的 recipe XML
uv run python -m inkjet_ml.ada --verify --head DMCSamba  # 交叉校验（应 19/19）
uv run python -m inkjet_ml.ada --rules                   # DropQCS 判据文件内容

# ===== 读回结果 =====
uv run python -m inkjet_ml.ada_rank "<落盘根>"                          # 硬编码判据排序
uv run python -m inkjet_ml.ada_rank "<落盘根>" --rules "<RulesConfig>"   # 官方规则判
uv run python -m inkjet_ml.ada_rank "<落盘根>" --csv 排序表.csv          # 导出

# ===== 训练与推荐 =====
uv run python -m inkjet_ml.heads.samba --recommend --target-speed 3.5
uv run python -m inkjet_ml.heads.samba --recommend --data "<新落盘根>"   # 接入新数据
uv run python -m inkjet_ml.heads.samba --leave-nozzle-out                # 迁移性评估
uv run python -m inkjet_ml.heads.km512                                   # 管线自检
uv run python -m inkjet_ml.heads.km1024i                                 # 管线自检

# ===== 墨水 =====
uv run python -m inkjet_ml.inkprops --table1

# ===== 自检 =====
uv run python -m unittest inkjet_ml.test_ncalc -v        # 规则求值器（16 项）
uv run python -m unittest inkjet_ml.test_data_roots -v   # 数据接入链路（12 项）
```

---

## 7. 当前的现实边界（2026-11-04）

**已跑通** ✅
- 四个模块都能独立运行并产出可用输出；
- KM1024i 用 V4730 真实落盘数据（1024 孔）完整走通训练管线；
- 8 套示例规则 + `Default` 全部可离线执行；
- 规则求值器 16 项 + 数据接入链路 12 项 = **28 项回归测试**锁住。

**还没跑通** ⬜
- **实机扫描一次都还没跑**——DOE 方案已就位，但新数据一个点都没采
  （`PROJECT_RECORD.md` §9 标着「⬜ 仍待实机执行」）；
- **Z 数进不了特征**——生产墨水的 ρ/η/γ 未实测，跨墨水泛化暂时做不了；
- **现有数据太薄**——9 个会话只覆盖 4 个波形点（40V/2.2、40V/1.9、34V/2.2、38V/2.2），
  电压与宽度**共线**，尾隙/斜率/电源电压是常量列。
  所以推荐表里 13/20 是高风险外推——**先补数据，再信推荐**。

**下一步建议顺序**

| 优先级 | 动作 | 为什么 |
|---|---|---|
| 1 | 实机跑一轮 `--plan screen` | 768 张图，解掉电压×宽度共线 |
| 2 | 实测生产墨水 ρ/η/γ | 解锁 Z 数，才能做跨墨水泛化（论文立论之一） |
| 3 | 跑 `--plan window` | 压力/温度独立成维，找成滴窗口 |

---

## 8. 出问题看哪儿

| 症状 | 多半是 | 去哪查 |
|---|---|---|
| `样本不足，无法训练` | 数据根选错 / 喷头不匹配 | 打印的「数据根：」那行核对路径 |
| 推荐全是「外推」 | 实测波形点太少 | 先补扫描（`--plan screen`） |
| `ada --verify` 不是 19/19 | 装的是旧机台版本 | §8.7 参数基线：V4730 / HMI 4.7.3 |
| 速度判据合格率普遍为 0 | 目标速度没按实测标定 | `--target-speed` 默认 3.5，**换墨必须重标定** |
| 机台扫完读不到数据 | 落盘根选错层级 | 要指到**含组合子目录**的那一层 |
| 训练跑很久不结束 | 组数太少退化成留一法 | 已修（见 `test_data_roots.py::TestCVFolds`） |

**文档分工**：[`README.md`](README.md) 项目概览 · [`QUICKSTART.md`](QUICKSTART.md) 操作手册 ·
[`PROJECT_RECORD.md`](PROJECT_RECORD.md) 结论与依据（单一事实来源） ·
[`docs/evidence/`](docs/evidence/README.md) 论文与原始表格。

---

## 9. 环境提示

- **改完代码要清 `inkjet_ml\__pycache__`** —— 曾因缓存导致新加的错误处理不生效
  （症状：CLI 该报错却静默通过）。
- 训练产物在 `results\<head>\`（`report.md` / `metrics.json` / `recommended.json`），
  该目录被 `.gitignore` 排除，重跑不产生版本库改动。
- `explore\*.json` 是 ADA 资产解析产物，同样被 `.gitignore` 排除，可随时重新解析。
- `.workbuddy/`（AI 助手记忆与工作日志）也是本地状态，不入库。
- 实测数据 `share/`、平台官方手册 `docs/DropQCS/` 与 `docs/Advanced_Drop_Analysis/`
  同样只在本机保留，不入库。
