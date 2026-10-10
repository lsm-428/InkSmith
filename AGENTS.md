# AGENTS.md

神经半色调（neural-halftone）：U-Net 输入灰度图，输出喷墨可打印的 1-bit 二值位图；
可微打印模拟器焊在损失函数里（网络反向补偿网点扩大）。Python + PyTorch，
无构建系统、无 CI、**无单元测试**。

## 文档优先级（会冲突，按此顺序信）

**PITFALLS.md**（最新实证，改训练动力学前必读） > **WORKFLOW.md**（分阶段操作手册） >
**CLAUDE.md / README.md**（入门，偏旧，未覆盖 film 模式）。文档冲突以 PITFALLS 为准：
WORKFLOW 阶段 4 推荐 `--dpi-min/--dpi-max` 随机 dpi 训练，已被
PITFALLS #11 推翻（线性物理下无收益，单 dpi 训练即可）——代码仍支持该选项，
别因为"文档写了"就启用。

## 命令（一律 `uv run`，工作目录 = 仓库根）

根目录只有 4 个入口：**训练 train.py / 评测+诊断 eval.py / 网页 webapp.py**（+2 个工具）。

```bash
uv sync                       # 装环境（torch 2.7.1+cu118，pyproject 钉死版本）

# === 训练（train.py 默认就是自动编排：退火多轮 -> 硬 STE 微调）===
uv run python train.py --printer film --sigma-um 25 --dpi 1440 --out runs/film
uv run python train.py --smoke     # ~10 秒冒烟；改代码后先跑这个
uv run python train.py --rounds 1 --epochs 20 --hard-ste ...  # 手工单段（默认自动，一般不用）

# === eval.py：评测 3 个 + 诊断 4 个子命令 ===
uv run python eval.py one   图.jpg --ckpt runs/film/best.pt   # 单图三方对照 + inkcov 审计
uv run python eval.py batch          --ckpt runs/film/best.pt   # 批量，默认 data/photos_val
uv run python eval.py infer 图.jpg   --ckpt runs/film/last.pt   # 出打印位图+预览
uv run python eval.py wedge --ckpt runs/film/best.pt   # 灰度尺分段偏差 + 条纹检测（改配方前必跑）
uv run python eval.py data        # 训练集平坦区/灰阶分布
uv run python eval.py norm        # 网络能否区分绝对灰阶
uv run python eval.py compare --a A.pt --b B.pt        # 两 ckpt 逐一对比

# === 其他 ===
uv run python webapp.py                 # 网页版对照，单图/多图上传，默认 8123
uv run python gen_data.py --n 500       # 生成合成训练图
uv run python calc_sigma.py             # σ 出处计算
```

- `--out` 决定实验目录，勿与 paper 实验混用（会 resume 到旧权重上）。
- `--rounds 1` 是单段模式，`--epochs` 才生效；默认 `--rounds 4` 走自动编排
  （温度几何退火 1.0→0.03、w_bin 逐轮加强、按孤立点率/psnr 自动调参）。
- **别用 `train.py --help`**：help 文本含 `100%固含`，argparse `%` 格式化会抛
  `ValueError`（正常运行不受影响）；查参数直接读源码里的 `add_argument` 段。
  `eval.py` / `webapp.py` 的 `--help` 正常。
- **回归手段 = `train.py --smoke` + `eval.py one`**，没有 pytest / lint / typecheck。
- **唯一可信的指标是硬阈值 psnr**（`train.py::evaluate` 和 `eval.py`）。
  `train.log` 里的 psnr 是软前向值，好看但会骗人（PITFALLS #4）。
- 默认 ckpt 不一致：`eval.py infer` 默认 `last.pt`，`one`/`batch` 默认 `best.pt`。
- 验收别默认"赢 FS"：线性物理（film）下 FS 接近最优，NN 保真度赢不了是预期结论，
  不是 bug（PITFALLS #11）。FS/GS 基线用于对照，不是必须击败的目标。

## 架构

```
train.py        训练：train_once(单段) + main(自动编排多轮/调参/硬STE微调) + evaluate(验证)
eval.py         评测+诊断七合一（共用 triplet 跑三条路线）：
                infer(位图) | one(单图三方) | batch(批量三方) |
                wedge(灰度尺/条纹) | data(数据集) | norm(架构) | compare(双ckpt)
webapp.py       网页版对照服务（import eval.triplet 复用评测核心）
gen_data.py     合成训练图；calc_sigma.py σ 参数计算器
halftone/model.py     U-Net + soft_binarize(退火) / hard_binarize(截断 STE)
halftone/printer.py   PrinterSimulator(paper，exp 吸收=饱和区) / FilmSimulator(film，线性体积)
                      由 build_printer(ckpt["args"]) 按 "printer" 字段选
halftone/losses.py    HVS 低频损失 + 孤立点惩罚；halftone/data.py 数据加载
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
   已验证配方是 `--sigma-um 25 --dpi 1440`（≈1.42px，硬 STE 后 20.98dB）。
3. **基线先数值自检**：PIL `convert("1")` 是 bool 数组，除 255 会伪造假胜利（PITFALLS #6）。
4. 每次只改一个变量；先做数值诊断（psnr/覆盖率/张量均值），不要靠肉眼读图调参。
5. **模型可能用水平条纹代替像素级抖动**：灰度尺上 PSNR 会低 13dB 且呈 S 形。
   判据是位图**行墨量 std**（`eval.py wedge` 输出）：>0.05 即条纹退化解，
   FS 参照 <0.01。详见 PITFALLS #12。**注意：目前所有尝试过的条纹惩罚指标都会
   误伤真实低频内容，别轻易加正则，见 PITFALLS #12 的"已排除"清单。**

## 工程环境

- GPU 是 Quadro P2000（5G，Pascal），batch 16×256² 已满。动卡前先
  `nvidia-smi --query-compute-apps` 确认无残留进程（pkill 常杀不净，残留会 OOM）。
- **同一时间只跑一个 uv 进程**（缓存锁冲突；报锁错先 `pgrep -af uv`）。
- 训练必须落盘：`train.log`（append+行缓冲）、`pipeline_log.jsonl`（逐轮）——
  后台任务超时会被杀，这是唯一幸存记录。
- 数据/产物目录：训练图 `data/images/`，验证集 `data/val/`（种子 123 固定 30 张，
  **勿混入训练集**；丢失可重建 `uv run python gen_data.py --n 30 --seed 123 --out data/val`），
  权重 `runs/<实验名>/`，推理产物 `out/`；`data/`、`runs/`、`out/`、`.venv/` 都在 .gitignore 里。
- `train.py --data` 目录为空时会悄悄用合成图冒烟——正式训练前确认 `data/images/` 非空。
- 当前工作区含多处未提交修改/未跟踪文件，动之前先 `git status --short` 确认再说（提交后可删此行）。
- `calc_sigma.py` 读仓库外的 `../InteractivePrintingSimulator/printheads.json`
  （本机存在）；换机器会缺文件。
- ghostscript 不吃 PNG 输入，墨量审计前需 PIL 包成 PDF（`eval.py one` 已处理）。
