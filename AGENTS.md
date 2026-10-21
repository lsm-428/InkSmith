# AGENTS.md

**墨匠 InkSmith** —— *学会打印，而不是渲染*（Learn to print, not to render）。
神经半色调：NN 输入连续调图，直接输出可打印位图。**项目目标 = 膜层打印**
（非吸收基材上复刻连续调厚度图），主线 **1-bit film**，下一步 **2-bit film**。

> 2026-10-21：**CMYK 2-bit 全彩路线已整体移除**（代码删除）。研究结论归档在
> PITFALLS.md #16–#30/#35–#48。移除理由与决策链见 PITFALLS #50。历史权重仍在
> 磁盘（`runs/cmyk_sd*/`、`runs/keep/`），要复现那些 ΔE 数字须 checkout `5714501`。

Python + PyTorch，无构建系统、无 CI、**无单元测试**。回归 = `train.py --smoke` + `eval.py batch --ed`。

**当前状态**（2026-10-21 实测，1-bit film，photos_val 12 张）：
- 硬阈值 PSNR **14.80**；`--ed` ED-FS 后处理 **24.67**；FS 公允基线 **25.82**
  （**--ed 仍差 1.15，12/12 全输**）、GS **24.67**（打平）。
- `--ed` 下模型权重贡献 **+0.02dB**（零 logits 也 24.67，PITFALLS #45/#11）。
  **膜基线性物理下逐像素决策赢不了 FS 是已证明的结构性结论，不是没调好参。**
- 数据：`data/images_gray`（500 张脚本合成灰度）为 1-bit 训练集；`data/images`
  为 `train.py` 默认目录（编排模式空则自动重建）。SD 图 `data/images_ai` 随
  CMYK 路线停用（膜层是灰度厚度图，不需要彩色数据）。

## 文档优先级（会冲突，按此顺序信）

**PITFALLS.md**（最新实证，改训练动力学前必读）> **WORKFLOW.md**（分阶段操作手册）>
**README.md**（项目介绍）。`CLAUDE.md` 只是指向本文档的指针。

## 命令（一律 `uv run`，工作目录 = 仓库根）

根目录 5 个脚本：**train.py**（训练）/ **eval.py**（评测+诊断）/ **webapp.py**（网页）；
**gen_data.py**（合成灰度数据）/ **gen_ai.py**（SD 数据，CMYK 遗留，暂搁）/
**calc_sigma.py**（σ 计算）。

```bash
uv sync                       # torch 2.7.1+cu118，pyproject 钉死

# === 训练（默认自动编排：退火多轮 -> 硬 STE 微调）===
uv run python train.py --printer film --sigma-um 25 --dpi 1440 --out runs/onebit
uv run python train.py --smoke     # ~10 秒冒烟；改代码后先跑这个
uv run python train.py --rounds 1 --epochs 20 --hard-ste ...  # 手工单段（一般不用）

# === eval.py：评测 2 + 诊断 4 子命令 ===
uv run python eval.py one   图.jpg --ckpt runs/onebit/best.pt --ed   # 单图三方 + inkcov
uv run python eval.py batch          --ckpt runs/onebit/best.pt --ed   # 批量三方（默认 photos_val）
uv run python eval.py infer 图.jpg   --ckpt runs/onebit/best.pt --ed   # 出打印位图+预览
uv run python eval.py wedge --ckpt runs/onebit/best.pt --ed   # 灰度尺分段偏差+条纹检测（改配方前必跑）
uv run python eval.py data        # 训练集平坦区/灰阶分布
uv run python eval.py norm        # 网络能否区分绝对灰阶
uv run python eval.py compare --a A.pt --b B.pt        # 两 ckpt 逐一对比

# === 其他 ===
uv run python webapp.py                 # 网页对照（默认 runs/onebit/best.pt，8123）
uv run python gen_data.py --n 500       # 生成合成灰度图
uv run python calc_sigma.py             # σ 出处计算（读仓库外 printheads.json）
```

- `--out` 决定实验目录，勿与 paper 实验混用（默认 `resume=True`，同名会续到旧权重，PITFALLS #38）。
- `--rounds 1` 是单段模式，`--epochs` 才生效；默认 `--rounds 4` 走自动编排
  （温度几何退火 1.0→0.3→0.09→0.03、w_bin 逐轮加强 0.05·轮、按孤立点率/psnr 自动调参）。
  ⚠ w_bin 逐轮加强在 2026-10-21 之前是**死代码**（PITFALLS #47，已修）。
- **唯一可信的指标是硬阈值 psnr**（`train.py::evaluate` / `eval.py`）。`train.log`
  里退火轮的 psnr 是软前向值，好看但会骗人（PITFALLS #4/#14）。
- `train.py --help` / `eval.py` / `webapp.py` 现均正常（曾 `--help` 因 `100%` 抛
  ValueError，已修，PITFALLS #47 附带）。

## 架构

```
train.py        训练：train_once(单段) + main(自动编排多轮/调参/硬STE微调) + evaluate(验证)
eval.py         评测+诊断七合一（共用 triplet 跑三条路线）：
                infer(位图) | one(单图三方) | batch(批量三方) | wedge(灰度尺/条纹) |
                data(数据集) | norm(架构) | compare(双ckpt)
webapp.py       网页版对照（1-bit 三路线；import eval.triplet 复用评测核心）
gen_data.py     合成灰度训练图（渐变/斑块/分形/几何/条纹/混合）
calc_sigma.py   喷头参数 + 实测铺展 -> σ 计算器（出处可溯）
halftone/model.py     U-Net + soft/hard_binarize(1-bit) + soft/hard_levels(多档，2-bit film 复用)
                      多档时 forward 返回 (B,out_ch,H,W,levels)
halftone/printer.py   FilmSimulator(膜基，线性体积，主线) / PrinterSimulator(纸基 exp 吸收)
                      由 build_printer(ckpt["args"]) 按 "printer" 字段选
halftone/ed_fs.py     FS 式逐列串行误差扩散。**推理后处理专用**（训练前向发散，#33）。
                      权重和严格为 1（右7/16+右下5/16+右上3/16+右二1/16，否则覆盖率两端塌陷 #42）
halftone/losses.py    HVS 低频匹配损失 + 孤立点惩罚（1-bit 单通道）
halftone/data.py      数据加载（RgbCrops，灰度存 3 通道，训练取 [:, :1]）
```

- 评测底层（psnr / FS / GS 基线 / inkcov / ckpt 重建 / film 目标反相）都在
  `eval.py` 顶层，`webapp.py` 直接 import 复用——不要再拆新模块。
- 权重是 `{"model","args"}`，infer/eval/webapp 从 ckpt 重建模型和打印机 →
  **改 argparse 默认值不影响已有 ckpt**；`levels>1` 的 ckpt 当前只有历史 CMYK
  权重，1-bit 主线 ckpt 恒 `levels=1`。
- film 语义：灰度 = 相对膜厚（1=满覆盖），eval/webapp 自动**反相** FS/GS 输入
  对齐（`film_target_and_base`）；自写基线别漏这步。
- σ 换算：film 下 `σ[px]=σ[µm]×dpi/25400`；位图与 dpi 无关，`--dpi` 只影响预览/评估渲染。

## 训练铁律（违反必然失败，均有实证，详见 PITFALLS）

1. **禁止 `--hard-ste` 从零训练**：必死（flatness degradation #1），只能退火后微调。
2. **打印模糊 σ 必须 ≳ 1.4px**：`--sigma-um 25 --dpi 1440`（=1.42px）是结晶下限；
   σ<1.4px 退火全程躺平（#11）。
3. **推理必须加 `--ed`**：硬阈值逐像素决策膜基下赢不了 FS（#11）；`--ed` 后
   24.67dB 但仍差 FS 1.15（#43），且增益几乎全来自扩散机制（模型贡献 +0.02dB，#45）。
4. **基线先数值自检**：PIL `convert("1")` 是 bool 数组，除 255 会伪造假胜利（#6）；
   基线与被测必须共享同一几何路径，一边 resize 一边 pad 会白送对方分数（#43）。
5. **每次只改一个变量**；先数值诊断（psnr/覆盖率/行墨量 std），不靠肉眼读图调参。
6. **模型可能用水平条纹代替像素级抖动**：判据是位图**行墨量 std**（`eval.py wedge`）
   >0.05 即条纹退化解（FS 参照 <0.01）。已由 `--ed` 修复（std 0.157→0.024），
   别靠加正则治（#12 已排除清单）。`striping_penalty` 默认关（`--w-stripe 0`）。
7. **验收用硬阈值 psnr + wedge 行墨量 std**；`train.log` 软 psnr 不作验收。

## 工程环境

- GPU 是 Quadro P2000（5G，Pascal），batch 16×256² 已满。动卡前先
  `nvidia-smi --query-compute-apps` 确认无残留（pkill 常杀不净，#9）。
- **沙箱会藏 GPU**：沙箱内 `nvidia-smi` 报错/`cuda.is_available()=False` 全是假象。
  判据：`/proc/driver/nvidia/version` 有内容 + `lsmod|grep nvidia` 有模块 ⇒ 驱动正常（#20）。
- **同一时间只跑一个 uv 进程**（缓存锁冲突，先 `pgrep -af uv`，#10）。锁冲突可用 `.venv/bin/python`。
- 训练必须落盘：`train.log`（append+行缓冲）+ `pipeline_log.jsonl`（逐轮），后台超时被杀这是唯一幸存记录（#8）。
- 目录：训练图 `data/images/`（或 `--data` 指定），验证集 `data/val/`（种子 123 固定 30 张，勿混训练集，
  丢失重建 `gen_data.py --n 30 --seed 123 --out data/val`），权重 `runs/<实验>/`，产物 `out/`。
  `data/`、`runs/`、`out/`、`.venv/` 都在 .gitignore。
- ghostscript 不吃 PNG，inkcov 前需 PIL 包 PDF（`eval.py one` 已处理）。
- `calc_sigma.py` 读仓库外 `../InteractivePrintingSimulator/printheads.json`，换机器会缺文件。
- `--data` 目录为空时 train.py 会悄悄用合成图冒烟——正式训练前确认目录非空。

## 2-bit film 路线（下一步，尚未接线）

现有 `UNet(levels>1)` + `soft_levels`/`hard_levels` + `ed_fs`（L 维通用）已在，缺的是
`train_once` 单通道多档前向（原 CMYK 分支删掉后需重建为 film 版）与物理损失。**验收
指标换成物理尺度均匀性**（100µm 膜厚 RMS + 档位混合 std），不用视觉 ΔE——膜层裁判是
工艺，高频抖动被物理平均掉本就是设计意图。开工前先做消融证明网络非冗余（见 README 路线图 2）。
