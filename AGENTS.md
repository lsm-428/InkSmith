# AGENTS.md

**墨匠 InkSmith** —— *学会打印，而不是渲染*（Learn to print, not to render）。
神经半色调：NN 输入连续调图，直接输出喷墨可打印的位图，跳过传统
Floyd–Steinberg 误差扩散。两条路线并存：

| 路线 | 入口 | 输出 | 物理建模 |
|---|---|---|---|
| **1-bit 灰度** | `train.py --printer film/paper` | 1-bit 二值位图 | 可微打印模拟器焊进损失函数（反向补偿网点扩大） |
| **CMYK 2-bit 全彩** | `train.py --cmyk` | 4 通道 × 4 档墨量 | ICC 标定 LUT256 + Neugebauer + Lab ΔE2000（无可微模拟器） |

Python + PyTorch，无构建系统、无 CI、**无单元测试**。

**当前状态**（2026-10-20）：CMYK 路线已训练收敛（未达可用线）——`runs/cmyk_sd/best.pt`（400 张
Stable Diffusion 生成图训练）在 photos_val 硬域 ΔE00 **8.92**，对 gs 工业
RIP **-21.54**（12/12 全胜）；距可用线 3.0 仍有 2.97 倍差距。1-bit 路线
`runs/onebit/best.pt`：硬阈值 14.80dB（输 FS），**加 `--ed` 后 24.67dB，
对 FS +5.19 / 对 GS +5.55，12/12 全胜**——逐像素硬阈值不是 1-bit 的正确
用法，ED-FS 后处理才是（PITFALLS #11 修正 + #42）。
训练数据已从脚本合成升级为 AI 生成（PITFALLS #39/#40）。

## 文档优先级（会冲突，按此顺序信）

**PITFALLS.md**（最新实证，改训练动力学前必读） > **WORKFLOW.md**（分阶段操作手册） >
**README.md**（项目介绍）。`CLAUDE.md` 只是一个指向本文档的指针。
文档冲突以 PITFALLS 为准：
WORKFLOW 阶段 4 推荐 `--dpi-min/--dpi-max` 随机 dpi 训练，已被
PITFALLS #11 推翻（线性物理下无收益，单 dpi 训练即可）——代码仍支持该选项，
别因为"文档写了"就启用。

## 命令（一律 `uv run`，工作目录 = 仓库根）

根目录 7 个脚本，3 个主入口 + 4 个工具：
**train.py**（训练）/ **eval.py**（评测+诊断）/ **webapp.py**（网页）；
**gen_data.py**（脚本合成数据）/ **gen_ai.py**（SD 生成数据）/
**calib_target.py**（网点扩大标定靶）/ **calc_sigma.py**（σ 计算）。

```bash
uv sync                       # 装环境（torch 2.7.1+cu118，pyproject 钉死版本）

# === 训练（train.py 默认就是自动编排：退火多轮 -> 硬 STE 微调）===
uv run python train.py --printer film --sigma-um 25 --dpi 1440 --out runs/onebit
uv run python train.py --smoke     # ~10 秒冒烟；改代码后先跑这个
uv run python train.py --rounds 1 --epochs 20 --hard-ste ...  # 手工单段（默认自动，一般不用）

# === CMYK 2-bit 全彩路线（阶段 5；无可微打印模拟器，损失=Lab ΔE2000）===
uv run python gen_data.py --n 800 --rgb --icc /usr/share/color/icc/colord/FOGRA39L_coated.icc \
    --cmyk-frac 0.15 --sat 0.6 --gamma 1.4 --size 512   # 照片化分布
uv run python train.py --cmyk --batch 4 --data data/images_ai --out runs/cmyk_sd
                                                  # 默认 w_cov 0.3/w_tac 1.0/tac_max 3.2
uv run python train.py --smoke --cmyk             # 冒烟（约 10 秒，1 epoch）
# 注意 --batch 4：CMYK 256 组合张量链显存峰值远高于 1-bit，batch 16 会爆显存

# === eval.py：评测 3 个 + 诊断 4 个子命令 ===
uv run python eval.py one   图.jpg --ckpt runs/onebit/best.pt   # 单图三方对照 + inkcov 审计
uv run python eval.py batch          --ckpt runs/onebit/best.pt   # 批量，默认 data/photos_val
uv run python eval.py infer 图.jpg   --ckpt runs/onebit/last.pt   # 出打印位图+预览
uv run python eval.py wedge --ckpt runs/onebit/best.pt   # 灰度尺分段偏差 + 条纹检测（改配方前必跑）
uv run python eval.py data        # 训练集平坦区/灰阶分布
uv run python eval.py norm        # 网络能否区分绝对灰阶
uv run python eval.py compare --a A.pt --b B.pt        # 两 ckpt 逐一对比
uv run python eval.py cmyk data/photos_val --ckpt runs/cmyk_sd/best.pt   # CMYK：ΔE00+四通道 inkcov
uv run python eval.py cmyk data/photos_val --ckpt runs/cmyk_sd/best.pt --rip   # 再加 gs RIP 基线
uv run python eval.py cmyk data/photos_val --ckpt runs/cmyk_sd/best.pt --ed --rip  # + ED-FS 后处理

# === 其他 ===
uv run python webapp.py                 # 网页版对照，单图/多图上传，默认 8123
uv run python gen_data.py --n 500       # 生成合成训练图
uv run python calc_sigma.py             # σ 出处计算
uv run python calib_target.py --out out/dotgain   # 网点扩大标定靶（打印后用 --analyze）
```

- `--out` 决定实验目录，勿与 paper 实验混用（会 resume 到旧权重上）。
- `--rounds 1` 是单段模式，`--epochs` 才生效；默认 `--rounds 4` 走自动编排
  （温度几何退火 1.0→0.03、w_bin 逐轮加强、按孤立点率/psnr 自动调参）。
- **别用 `train.py --help`**：help 文本含 `100%固含`，argparse `%` 格式化会抛
  `ValueError`（正常运行不受影响）；查参数直接读源码里的 `add_argument` 段。
  `eval.py` / `webapp.py` 的 `--help` 正常。
- **回归手段 = `train.py --smoke` + `eval.py one`**，没有 pytest / lint / typecheck。
  CMYK 路线的回归 = `train.py --smoke --cmyk` + `eval.py cmyk --limit 2`。
- **唯一可信的指标是硬阈值 psnr**（`train.py::evaluate` 和 `eval.py`）。
  `train.log` 里的 psnr 是软前向值，好看但会骗人（PITFALLS #4）。
  CMYK 路线同理：可信指标是 `evaluate_cmyk` / `eval.py cmyk` 的**硬 argmax
  ΔE2000**，train.log 里的 `de` 是软前向值。
- 默认 ckpt 不一致：`eval.py infer` 默认 `last.pt`，`one`/`batch` 默认 `best.pt`。
- **1-bit 验收必须加 `--ed`**：硬阈值的逐像素决策在 film 下赢不了 FS
  （14.80 vs 19.48，PITFALLS #11）；ED-FS 后处理后 24.67dB、对 FS +5.19、
  12/12 全胜。不带 `--ed` 的 1-bit 数字不代表这条路线的最好水平。

## 架构

```
train.py        训练：train_once(单段) + main(自动编排多轮/调参/硬STE微调) + evaluate(验证)
                （--cmyk 走 CMYK 2-bit 路线：evaluate_cmyk 硬域 ΔE2000）
eval.py         评测+诊断八合一（共用 triplet 跑三条路线）：
                infer(位图) | one(单图三方) | batch(批量三方) |
                wedge(灰度尺/条纹) | data(数据集) | norm(架构) | compare(双ckpt) |
                cmyk(CMYK ΔE00 + 四通道 inkcov；--rip 加 gs 工业 RIP 基线；
                     --ed 加 ED-FS 误差扩散后处理)
webapp.py       网页版对照服务（import eval.triplet 复用评测核心）
gen_data.py     合成训练图（--rgb 彩色，--icc 走 ICC 分离，--sat/--gamma 照片化分布）
                calib_target.py 网点扩大标定靶（256 块 4 档全组合 PDF + 扫描分析）
halftone/model.py     U-Net + soft_binarize/hard_binarize(1-bit) + soft_levels/hard_levels(多档)
                      多档时 forward 返回 (B,out_ch,H,W,levels)
halftone/printer.py   PrinterSimulator(paper，exp 吸收=饱和区) / FilmSimulator(film，线性体积)
                      由 build_printer(ckpt["args"]) 按 "printer" 字段选（CMYK 路线不用）
halftone/cmyk.py      色彩层：LUT16/LUT256（ICC 标定）+ Neugebauer 收缩 + sRGB↔Lab↔CMYK
halftone/ed_fs.py     FS 式逐列串行误差扩散（**推理后处理专用**，训练前向会发散 PITFALLS #33）
halftone/losses.py    HVS 低频损失 + 孤立点惩罚（CMYK 联合掩码）；halftone/data.py 数据加载
```

评测底层（psnr / FS / GS 基线 / inkcov / ckpt 重建 / film 目标反相）都在
`eval.py` 顶层，`webapp.py` 直接 import 复用——不要再拆出新模块。

- 权重是 `{"model", "args"}`，infer/eval/diag/webapp 从 ckpt 重建模型和打印机 →
  **改 argparse 默认值不影响已有 ckpt**，也别指望 ckpt 里的超参和代码默认值同步。
- film 模式语义：灰度 = 相对膜厚（1=满覆盖），所以 eval/diag/webapp 会自动**反相**
  FS/GS 的输入来对齐（`film_target_and_base` 负责）；自写基线时别漏掉这一步。
- σ 换算：film 下 `σ[px] = σ[µm] × dpi / 25400`；位图本身与 dpi 无关，
  `--dpi` 只影响预览/评估渲染。

## 训练铁律（违反必然失败，均有实证，详见 PITFALLS）

1. **禁止 `--hard-ste` 从零训练**：必死（flatness degradation），只能退火后微调。
2. **打印模糊 σ 必须 ≳ 1.4px**：12.5µm@1440dpi=0.71px 会让退火全程躺平；
   已验证配方是 `--sigma-um 25 --dpi 1440`（≈1.42px）。**推理必须加 `--ed`**
   （ED-FS 后处理，见铁律 #6 与 PITFALLS #42）：硬阈值 14.80dB → ed_fs 24.67dB。
3. **基线先数值自检**：PIL `convert("1")` 是 bool 数组，除 255 会伪造假胜利（PITFALLS #6）。
4. 每次只改一个变量；先做数值诊断（psnr/覆盖率/张量均值），不要靠肉眼读图调参。
5. **跟 gs RIP 比彩色，必须先让 gs 用同一个 ICC 分离**：直接 `tiffsep1` 渲
   sRGB PDF 会用 gs 自带默认 profile（gray50 的 K=0.074 vs FOGRA39L 0.184），
   且 gs 10.05.1 不吃 `-dOverrideICC`/`-sDefaultCMYKProfile`。正确做法是
   PIL `ImageCms` 先分离成 CMYK PDF 再让 gs 只做半色调。见 PITFALLS #18。
6. **模型可能用水平条纹代替像素级抖动**：灰度尺上 PSNR 会低 13dB 且呈 S 形。
   判据是位图**行墨量 std**（`eval.py wedge` 输出）：>0.05 即条纹退化解，
   FS 参照 <0.01。详见 PITFALLS #12。条纹退化解**已由 `--ed` 修复**
   （行墨量 std 0.157→0.024），不要靠加正则治——试过的指标全部误伤真实低频，
   见 PITFALLS #12 的"已排除"清单。
7. **CMYK 多档必须有 `w_cov`**：只优化 Lab ΔE 时网络会找跨通道替身
   （K 1/3 落成 C 档 2，ΔE 10.4），软期望与硬 argmax 是两个目标。PITFALLS #17。
8. **自定义 loss 先用全零输入跑 backward**：CIEDE2000 在 a=b=0 处有 4 个
   0/0 型反传炸弹（`sqrt(0)`/`atan2(0,0)`/`**7`），随机输入永远碰不到。PITFALLS #16。
9. **dithering 不能靠损失鼓励出来**：w_cov、降 T、gap penalty、硬 STE 直通、
   4 独立 sigmoid 头全部证伪（PITFALLS #22/#24/#25/#26）。dithering 是**决策
   机制**的属性，正确来源是推理后处理 `halftone/ed_fs.py`（eval.py cmyk --ed）。
   训练前向走串行扩散会发散（PITFALLS #33）。

## 工程环境

- GPU 是 Quadro P2000（5G，Pascal），batch 16×256² 已满。动卡前先
  `nvidia-smi --query-compute-apps` 确认无残留进程（pkill 常杀不净，残留会 OOM）。
- **沙箱会把 GPU 藏掉**：沙箱内 `nvidia-smi` 可能报 "couldn't communicate with
  the NVIDIA driver"、`torch.cuda.is_available()=False`、`/dev/nvidia*` 不存在，
  **这些都是假象**。判据：`cat /proc/driver/nvidia/version` 有内容 +
  `lsmod | grep nvidia` 有模块 ⇒ 驱动是好的，问题在访问权限。别据假象改训练
  计划（PITFALLS #20，白跑过两天 CPU 冒烟）。
- **CMYK 全彩路线显存峰值远高于 1-bit**（256 组合张量链），batch 16×256²
  会 `CUBLAS_STATUS_ALLOC_FAILED`（在 backward 的 `cublasCreate(handle)`
  才炸，不是 forward）。实际安全值 `--batch 4`。
- **同一时间只跑一个 uv 进程**（缓存锁冲突；报锁错先 `pgrep -af uv`）。
- 训练必须落盘：`train.log`（append+行缓冲）、`pipeline_log.jsonl`（逐轮）——
  后台任务超时会被杀，这是唯一幸存记录。
- 数据/产物目录：训练图 `data/images/`，验证集 `data/val/`（种子 123 固定 30 张，
  **勿混入训练集**；丢失可重建 `uv run python gen_data.py --n 30 --seed 123 --out data/val`），
  权重 `runs/<实验名>/`，推理产物 `out/`；`data/`、`runs/`、`out/`、`.venv/` 都在 .gitignore 里。
- **CMYK 路线的彩色训练数据**：`data/images/`（800 张脚本合成 RGB）与
  `data/images_ai/`（1000 张 SD 生成）都是 RGB，`--cmyk` 直接用。
  SD 图是主数据源（ΔE00 低 18%，PITFALLS #39）。
- `uv` 缓存锁冲突时直接用 `.venv/bin/python`（不用 `uv run`）。
- **CMYK 默认跳过硬 STE 微调**：多档下它会毁掉退火成果（ΔE 6.38→42.04，
  覆盖率 0.336→0.562，PITFALLS #19）。best.pt 来自退火末态而非 finetune 后。
  要强行启用加 `--cmyk-finetune`。1-bit 老路不受影响（`--hard-ste` 仍照旧）。
- `train.py --data` 目录为空时会悄悄用合成图冒烟——正式训练前确认 `data/images/` 非空。
- `calc_sigma.py` 读仓库外的 `../InteractivePrintingSimulator/printheads.json`
  （本机存在）；换机器会缺文件。
- ghostscript 不吃 PNG 输入，墨量审计前需 PIL 包成 PDF（`eval.py one` 已处理）。
- **RIP 彩色基线的正确形态**：`eval.py::rip_cmyk()` = PIL `ImageCms` 按同一 ICC
  分离成 CMYK PDF → `gs -sDEVICE=tiffsep1 -r72` 出四张 1-bit 位图。中间产物落在
  `out/rip_gs/`。tiffsep1 的 tif 是 mode "1"（bool，**True=白**），墨覆盖率 =
  `1 - arr.astype(np.float32)`，别除 255（PITFALLS #18）。
- RIP ΔE 是**保守下界**：RIP 位图要过 LUT256 得量化到 4 档（`one_hot_from_cov`），
  丢掉了网点大小承载的亚像素覆盖率信息。真实 RIP 比这个数更好。

## 数据生成：AI 生成训练图（Stable Diffusion，替代脚本合成，见 PITFALLS #22）
uv run python gen_ai.py --n 2 --steps 15 --dry-run     # 小批试跑（不落盘）
uv run python gen_ai.py --n 400 --steps 20 --size 512 --out data/images_ai
uv run python gen_ai.py --n 100 --init data/images --strength 0.55  # 图生图

- 首次需联网下 SD1.5 fp16 权重（~5.4G，走 hf-mirror；huggingface.co 直连不通）。
  之后纯本地（`local_files_only=True` + `variant="fp16"`，缺 variant 会报
  IncompleteSnapshotError）。
- Quadro P2000 实测 512² 20 步 **21-35 s/张**（FP16 + attention slicing）。
  400 张 ≈ 3.5 h，必须后台跑。
- `--init` 图生图模式：以脚本合成图作 init，strength 0.55 保留结构多样性。
- 自动过滤 SD 失败模式（近黑/近白/std<0.03/单色>85%），失败 seed 直接跳过。
