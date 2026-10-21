# AGENTS.md

**墨匠 InkSmith** —— *学会打印，而不是渲染*。膜层半色调：**ED-FS 误差扩散引擎**
把连续调厚度图转成 one-pass 可打印位图。目标 = 膜层打印（非吸收基材上
复刻厚度图）。主线 film 灰度：**1-bit 现役基线，多档（`--drops N`）已接线并
追平 FS（#55），待真机档位标定落地**。

> 2026-10-21 同日四次手术，详录 PITFALLS：#50 CMYK 全彩路线移除（含权重/数据）、
> #51 手术复核修复 8 处静默失真、#52 **NN 整体拆除**、#53 gen_ai 依赖声明缺失、
> #54 本文档体系精简。**NN 判决证据**：模型先验在行墨量 std 上 12/12 劣于零先验
> （wedge 0.0244→0.0027），PSNR 仅 −0.03dB ⇒ 网络是负资产，不是冗余。

Python + PyTorch（**仅作张量/卷积后端，无训练、无权重**），无构建系统、无 CI、
无单元测试。回归 = `eval.py batch` + `eval.py wedge`（毫秒级，无权重依赖）。

**现役基线**（2026-10-21 实测，photos_val 12 张，film σ=25µm@1440dpi）：

| 配置 | batch ED | vs FS 25.82 | 100µm 膜厚 RMS |
|---|---|---|---|
| N=1（1-bit，默认回归口径） | 24.64 | −1.18 | 0.39µm |
| **N=4（多档，已接线 #55）** | **25.81** | **−0.01（追平）** | 0.32µm |

**ED 的 1-bit 缺口大头是量化步长不是 1D 机制**（#55）——档位加密后单调收敛到 ~0，
蛇形 2D 核升级的预期收益需在 N=4 基线上**重新定价**。GS 基线 24.67；N≥4 后加档
饱和（#26），实用点 N=4。回归 = `batch`（N=1）须逐位 24.64/25.82/24.67。

## 文档优先级与分工（会冲突，按此顺序信）

**PITFALLS.md**（最新实证，动引擎前必读）> **WORKFLOW.md**（分阶段操作手册）>
**README.md**（对外介绍）。`CLAUDE.md` 只是指针。
四份文档**不重复维护同一事实**：物理推导与决策依据只写在 PITFALLS（one-pass
物理账见 #52，σ 下限见 #11，机制定型见 #32/#35），本文只留**操作结论**。

## 命令（一律 `uv run`，工作目录 = 仓库根）

根目录 6 个平铺脚本：`eval.py`（推理+评测+诊断）/ `webapp.py`（网页）/
`ed_fs.py`（主引擎）/ `printer.py`（物理模拟）/ `calc_sigma.py`（σ 出处）/
`gen_ai.py`（SD 内容图，备用）。

```bash
uv sync                       # 主线依赖（torch 2.7.1+cu118，pyproject+uv.lock 钉死）
uv sync --extra gen-ai        # 需要 gen_ai.py 时才加（diffusers 等 ~115M，#53）

# 推理（主引擎 ED-FS；无权重/ckpt，物理参数即 CLI，默认已验证配方）
uv run python eval.py infer 图.jpg --out out/photo   # ED-FS 位图 + 模拟预览
uv run python eval.py one   图.jpg                   # 单图三方 + inkcov 审计
uv run python eval.py batch                          # 批量三方（默认 photos_val，N=1 回归口径）
uv run python eval.py batch --drops 4                # 多档 film（25.81 追平 FS，#55）
uv run python eval.py batch --dpi 2880               # 换打印分辨率（σ=2.83px）
uv run python eval.py wedge                          # 灰度尺偏差+条纹+100µm RMS（改引擎前必跑）

uv run python webapp.py       # 网页对照（ED/FS/GS，端口 8123）
uv run python calc_sigma.py   # σ 出处计算（读仓库外 printheads.json）
uv run python gen_ai.py --n 100                     # SD 内容图（备用，需 GPU）
```

## 架构

```
eval.py       推理+评测+诊断四合一（ED/FS/GS 三路线，无 ckpt）
              infer(位图+预览) | one(单图三方+inkcov) | batch(批量三方) | wedge(灰度尺)
              底层（psnr / ed_halftone / floyd_steinberg / gs_halftone / inkcov /
              film_target_and_base）全在顶层，webapp.py 直接 import 复用——不要再拆模块。
webapp.py     网页版三路线对照；默认 ED 口径（#51 前曾是硬阈值，差 ~10dB）
ed_fs.py      主引擎：FS 式逐列串行误差扩散，L 维通用（多档白拿）。
              已知短板 = 1D 误差流；蛇形/2D 核升级在此文件（#52 主战场）。
printer.py    FilmSimulator(膜基线性体积，主线) / PrinterSimulator(纸基 exp)。
              现为评测渲染器；可微实现留给离线优化（DBS/CVO）。
calc_sigma.py 喷头参数 + 实测铺展 -> σ（出处可溯）
gen_ai.py     SD 内容图（备用；依赖是可选组，见 #53）
```

（2026-10-21 扁平化：`halftone/` 包随 NN 拆除后只剩 2 文件，已移回根目录。
PITFALLS 历史条目里的 `halftone/*.py` 路径是当时事实，不改。）

## 物理与口径（结论，推导见 PITFALLS）

- **σ 下限**：`σ[px] = σ[µm]×dpi/25400`，**σ<1.4px 相邻墨滴不融合、one-pass 膜呈
  麻点**（#11）。已验证配方 σ=25µm@1440dpi=1.42px。`--sigma-um`/`--dpi` 是**实测
  物理参数，不是画质旋钮**（出处走 `calc_sigma.py`）。
- **one-pass**：`ink∈{0,1}` 每像素至多一滴（6pL）⇒ `T_max=V/p²` 由 dpi 决定
  （1440≈19.3µm、2880≈77µm）。**dpi 管量程、不管颗粒**；薄端细度靠档位（多档
  ED）。完整物理账见 PITFALLS #52。
- **film 语义**：灰度 = 相对膜厚（1=满覆盖）。FS/GS 基线必须**反相**输入对齐
  （`film_target_and_base`，pad 而非 resize，#43）；自写基线别漏这步。
- **位图落盘口径全库统一：墨=黑/0**（infer/one/webapp 一致，可直接喂 RIP/inkcov；
  #51 前的 infer 是负片，别引旧产物）。
- 引擎唯一行为参数 `--bias-gain`（默认 4.0）。`wedge` 现状：ED 行墨量 std
  **0.0027**（FS 参照 0.0071）；旧 0.0244 是模型先验注入的聚簇（#52）。

## 引擎铁律（违反必然出错，实证见 PITFALLS）

1. **ED 权重和必须严格为 1**（右7/16+右下5/16+右上3/16+右二1/16）：否则覆盖率
   两端塌陷（#42，实测 0.067→0.000）。
2. **对比数字必须注明引擎**：ED/FS/GS 别混引 NN 时代数字（14.80/24.67，#45/#52）。
3. **基线先数值自检**：PIL `convert("1")` 是 bool 数组，除 255 会伪造假胜利（#6）；
   基线与被测共享同一几何路径（pad 而非 resize，#43）。位图字节语义随
   PIL/gs 版本翻转，换环境必重跑端点块自检（#48）。
4. **每次只改一个变量**；先数值诊断（psnr/覆盖率/行墨量 std），不靠肉眼读图。
5. **验收 = `batch` 三方 PSNR + `wedge` 行墨量 std**（ED std<0.05，FS 参照<0.01）；
   改 `ed_fs` 前后必跑 `wedge` 防条纹回归（#12）；ED 实现改动先跑**常数目标全灰阶
   扫描**验覆盖率（#42）。
6. **别给引擎加学习**：密度 = ED 权重和=1 精确满足、排布 = 蓝噪声/DBS 经典地盘，
   NN 已证负资产（#45/#52）。要增量先走"实测核 + 离线优化"。**dithering 是决策
   机制的属性，损失鼓励不出来**（#35）。

## 工程环境

- **GPU 已非必需**：推理纯 CPU 毫秒级。历史卡 Quadro P2000 5G。
- **沙箱会藏 GPU**：判据 `/proc/driver/nvidia/version` + `lsmod|grep nvidia`（#20）。
- **同一时间只跑一个 uv 进程**（缓存锁，先 `pgrep -af uv`，#10）；绕法 `.venv/bin/python`。
- **每个包必须有声明出处**（#53）：手工 `uv pip install` 的包不进锁，重建 `.venv`
  会静默丢。删 `.venv` 前跑 `pip list` 与 `uv.lock` 差集体检（命令见 #53）。
- 目录：评测集 `data/photos_val/`（12 张，唯一数据集）；`runs/` 空；产物 `out/`。
  `data/`、`runs/`、`out/`、`.venv/` 在 .gitignore。
- ghostscript 不吃 PNG（#7）；`calc_sigma.py` 读仓库外
  `../InteractivePrintingSimulator/printheads.json`，换机器会缺文件。

## 多档 film 路线（#55 已接线 N 档，剩余工作）

1. **档位标定（当前最高优先）**：`--drops N` 的名义档 [0,1/N..1]×顶滴体积要先过
   实测——③ 顶档/分级波形落墨线性度（平铺块打→测厚），实测值直接传 `ed_fs` 的
   lv 表（printer 吃 lv 加权和，天然支持非线性档）。N≥4 后加档饱和（#26/#55），
   实用点 N=4。
2. **蛇形 2D 核升级（收益待重新定价）**：1-bit 的 −1.18dB 大头已被证明是量化
   步长（#55），此项要在 N=4 基线上先测残余缺口再决定投入。
3. **工艺约束**（最大连续膜厚/台阶覆盖/孤立点）：离线优化 DBS/CVO 直接优化位图，
   目标函数用 `printer.py` 物理口径 + `thickness_rms_um`，**不训练网络**。
4. **验收 = 物理尺度均匀性**：`batch/wedge` 已输出 100µm 膜厚 RMS（µm，#55），
   不用视觉 ΔE（#44 反向适用）；注意 inkcov 对档位灰度图饱和（#48/#55），
   多档覆盖率以张量自检为准。
