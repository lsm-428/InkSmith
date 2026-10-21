# AGENTS.md

**墨匠 InkSmith** —— *学会打印，而不是渲染*（Learn to print, not to render）。
膜层半色调：**ED-FS 误差扩散引擎**把连续调厚度图转成 one-pass 可打印 1-bit 位图。
**项目目标 = 膜层打印**（非吸收基材上复刻连续调厚度图），主线 **1-bit film**，
下一步 **多档 film**（多档 ED + 离线优化；原"2-bit NN"路线随 NN 一并作废，#52）。

> 2026-10-21 同日三次手术：
> **#50** CMYK 2-bit 全彩路线移除（代码+权重+数据全删，ΔE 数字仅存档）。
> **#51** 删除手术后的复核修复（8 处静默失真）。
> **#52** **NN 路线整体拆除**：模型先验实测在行墨量 std 上 12/12 劣于零先验
> （wedge 0.0244→0.0027，~9× 均匀性提升；PSNR 仅 −0.03dB）——网络由"冗余"
> （#45）改判"负资产"。train/model/losses/data/gen_data/probes 与全部权重、
> 训练数据已删。**引用任何 NN 时代的数字前必读 #52。**

Python + PyTorch（仅作张量/卷积后端，**无训练、无权重**），无构建系统、无 CI、
无单元测试。回归 = `eval.py batch`（毫秒级、无权重依赖）+ `eval.py wedge`。

**当前状态**（2026-10-21 实测，1-bit film σ=25µm@1440dpi，photos_val 12 张）：
- **ED-FS 24.64 | FS 25.82 | GS 24.67**（ED vs FS −1.18，vs GS −0.03）。
- **现役最优仍是 PIL FS**；ED-FS 差在机制（逐列并行把 2D 误差流压成 1D，#52），
  不在缺网络。**主战场 = ED 引擎升级（蛇形逐行/2D 核），验收线 = 纯 ED ≥ FS。**
- 数据：评测集 `data/photos_val`（12 张，唯一幸存数据集）。NN 训练数据/权重已删；
  内容图可用 `gen_ai.py` 重造（SD 缓存 `~/.cache/huggingface` 仍在）。

## 文档优先级（会冲突，按此顺序信）

**PITFALLS.md**（最新实证，动引擎前必读）> **WORKFLOW.md**（分阶段操作手册）>
**README.md**（项目介绍）。`CLAUDE.md` 只是指向本文档的指针。

## 命令（一律 `uv run`，工作目录 = 仓库根）

根目录 4 个脚本：**eval.py**（推理+评测+诊断）/ **webapp.py**（网页对照）/
**gen_ai.py**（SD 内容图，备用）/ **calc_sigma.py**（σ 计算，出处可溯）。

```bash
uv sync                       # 主线依赖（torch 2.7.1+cu118，pyproject+uv.lock 钉死）
uv sync --extra gen-ai        # 需要 gen_ai.py 时才加（diffusers/transformers/accelerate，#53）

# === 推理（主引擎 ED-FS；无需任何权重/ckpt，物理参数即 CLI，默认=已验证配方）===
uv run python eval.py infer 图.jpg --out out/photo     # ED-FS 位图+预览
uv run python eval.py one   图.jpg                     # 单图三方 + inkcov 审计
uv run python eval.py batch                            # 批量三方（默认 photos_val）
uv run python eval.py batch --dpi 2880                 # 换打印分辨率（σ=2.83px）

# === 诊断（改引擎前必跑）===
uv run python eval.py wedge                            # 灰度尺分段偏差+条纹检测

# === 其他 ===
uv run python webapp.py                 # 网页对照（ED/FS/GS，端口 8123，无 ckpt）
uv run python calc_sigma.py             # σ 出处计算（读仓库外 printheads.json）
uv run python gen_ai.py --n 100         # SD 内容图（备用；首次联网下 4GB 权重）
```

- 物理参数默认 `--printer film --sigma-um 25 --dpi 1440`（σ=1.42px）。**它们不是
  调参旋钮**：必须实测可溯（`calc_sigma.py`）。σ[px]=σ[µm]×dpi/25400；
  **σ<1.4px 时相邻墨滴不融合，one-pass 膜呈麻点**（#11 原训练下限，#52 改读）。
- one-pass 语义：`ink∈{0,1}` 每像素至多一滴；满覆盖膜厚 `T_max=V/p²`
  （6pL：1440dpi≈19.3µm、2880dpi≈77µm）——**dpi 是量程旋钮**。100µm 窗内
  膜厚波动下限由滴数 `k=T̄·s²/V` 决定、与 dpi 无关：**薄端细度靠档位（多档
  ED），不靠 dpi、更不靠 NN**。
- 唯一行为参数 `--bias-gain`（ED 偏置增益，默认 4.0）。
- `wedge` 现状：ED 行墨量 std **0.0027**（FS 参照 0.0071）——均匀性已是引擎
  应有水平；旧 NN+ED 的 0.0244 是模型先验注入的聚簇（#52）。

## 架构

```
eval.py         推理+评测+诊断四合一（ED/FS/GS 三路线，无 ckpt）：
                infer(ED-FS 位图+预览) | one(单图三方+inkcov) | batch(批量三方) |
                wedge(灰度尺分段偏差+条纹检测)
                底层（psnr / ed_halftone / FS / GS 基线 / inkcov / film 目标反相）
                都在顶层，webapp.py 直接 import 复用——不要再拆新模块。
webapp.py       网页版对照（ED/FS/GS 三路线；import eval 复用核心；无 ckpt）
ed_fs.py        主引擎：FS 式逐列串行误差扩散（L 维通用，多档白拿）。
                权重和严格为 1（右7/16+右下5/16+右上3/16+右二1/16，#42）。
                已知短板 = 1D 误差流；蛇形/2D 核升级在此文件（#52 主战场）。
printer.py      FilmSimulator(膜基线性体积，主线) / PrinterSimulator(纸基 exp)。
                现为评测渲染器；可微实现留给离线优化（DBS/CVO）复用。
calc_sigma.py   喷头参数 + 实测铺展 -> σ 计算器（出处可溯）
gen_ai.py       SD 内容图生成（备用，主线不依赖）
```

（仓库结构 2026-10-21 扁平化：`halftone/` 包随 NN 拆除后只剩 2 个文件，
已移到根目录；PITFALLS 历史条目里的 `halftone/*.py` 路径是当时事实，不改。）

- film 语义：灰度 = 相对膜厚（1=满覆盖），FS/GS 基线必须**反相**输入对齐
  （`film_target_and_base`，pad 而非 resize，#43）；自写基线别漏这步。
- **位图落盘口径全库统一：墨=黑/0**（infer/one/webapp 一致，可直接喂 RIP/inkcov；
  #51 修复前的 infer 是负片，别再引旧产物）。
- `build_printer(args字典)`：eval 传 `vars(args)`，字段名即 CLI 参数名。
- 位图与 dpi 无关（同网格下）；`--dpi` 影响铺展渲染/评估换算。按物理尺寸
  换 dpi 重采样目标图时位图内容随之变——那是量程/颗粒的换算，不是口径漂移。

## 引擎铁律（违反必然出错，均有实证，详见 PITFALLS）

1. **ED 权重和必须严格为 1**：否则覆盖率两端塌陷（#42，实测 0.067→0.000）。
2. **对比数字必须注明引擎**：NN+硬阈值 14.80 的惨案属历史；现役三路线
   ED/FS/GS 别混引 NN 时代数字（#45/#52）。
3. **基线先数值自检**：PIL `convert("1")` 是 bool 数组，除 255 会伪造假胜利（#6）；
   基线与被测必须共享同一几何路径（pad 而非 resize，#43）。
4. **每次只改一个变量**；先数值诊断（psnr/覆盖率/行墨量 std），不靠肉眼读图调参。
5. **验收 = `batch` 三方 PSNR + `wedge` 行墨量 std**（ED std < 0.05，FS 参照
   <0.01）；改 `ed_fs` 必跑 `wedge` 防条纹回归（#12）。
6. **别给引擎加学习**：密度=ED 权重和=1 精确满足、排布=蓝噪声/DBS 经典地盘，
   NN 已证负资产（#45/#52）。要增量先走"实测核 + 离线优化"。

## 工程环境

- **GPU 已非必需**：推理纯 CPU 毫秒级（torch 自动选卡）。历史：Quadro P2000。
- **沙箱会藏 GPU**：判据 `/proc/driver/nvidia/version` + `lsmod|grep nvidia`（#20）。
- **同一时间只跑一个 uv 进程**（缓存锁，先 `pgrep -af uv`，#10）；锁冲突用
  `.venv/bin/python`。
- **每个包必须有声明出处**（#53）：手工 `uv pip install` 的包不进锁，重建
  `.venv` 会静默丢。删 `.venv` 前先做体检：`pip list` 与 `uv.lock` 差集非空
  即有声明缺失。`gen_ai` 依赖在可选组：`uv sync --extra gen-ai`。
- 目录：评测集 `data/photos_val/`（12 张，唯一数据集）；`runs/` 已清空；产物
  `out/`。`data/`、`runs/`、`out/`、`.venv/` 都在 .gitignore。
- ghostscript 不吃 PNG，inkcov 前需 PIL 包 PDF（`eval.py one` 已处理）。
- `calc_sigma.py` 读仓库外 `../InteractivePrintingSimulator/printheads.json`，换机器会缺文件。

## 多档 film 路线（下一步；原"2-bit NN"作废）

`ed_fs` 的 L 维已通用（`levels` 任意），缺的是：
1. **引擎升级**：蛇形逐行扫描恢复 2D 误差流——验收 = 1-bit 纯 ED ≥ PIL FS（25.82）；
2. **多档接线**：档位表 [0,⅓,⅔,1] 名义值先行，实测线性度（标定 ③）后替换；
3. **工艺约束**（最大连续膜厚/台阶覆盖/孤立点）：走离线优化（DBS/CVO 直接优化
   位图，目标函数用 `printer.py` 物理口径），不训练网络；
4. **验收指标 = 物理尺度均匀性**（100µm 膜厚 RMS + 档位混合统计），不用视觉 ΔE
   ——膜层裁判是工艺，高频抖动被物理平均掉本就是设计意图。
