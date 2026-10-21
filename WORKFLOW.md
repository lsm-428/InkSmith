# 工作流程：神经半色调（图像 → 喷墨膜厚位图）

> 目标：NN 输入连续灰度（厚度）图，直接输出喷墨可打的位图，跳过传统误差扩散。
> 本文是流程的可复现操作手册，按阶段组织。
>
> 2026-10-21：**CMYK 全彩路线（旧阶段 5）已整体移除**——代码删除，结论归档
> PITFALLS #16–#30/#35–#48，决策链 #50。本手册只覆盖 1-bit film 主线；
> 下一步 2-bit film 见 README 路线图。

## 阶段 0：环境（一次性）

```bash
uv sync        # 按 pyproject.toml 装 torch 2.7.1+cu118
```

验证 GPU：`uv run python -c "import torch; print(torch.cuda.is_available())"` 应输出 `True`。
⚠ 沙箱会遮蔽 GPU（PITFALLS #20），报错先查 `/proc/driver/nvidia/version`。

## 阶段 1：训练数据

主线是**灰度厚度图**，一条来源：

| 来源 | 命令 | 说明 |
|---|---|---|
| 脚本合成 | `uv run python gen_data.py --n 500` | 6 类频率成分：渐变/斑块/分形噪声/几何/条纹/混合（默认输出 `data/images_gray/`） |
| 真实厚度图 | 手动拷贝进 `data/images_gray/` | 目标图案越接近实际打印越好 |

- 数据量建议 500~2000；重新生成会覆盖同名文件（种子固定可复现）。
- 验证集 `data/val/`：种子 123 固定 30 张，**勿混入训练集**；
  丢失重建 `uv run python gen_data.py --n 30 --seed 123 --out data/val`。
- `data/images_ai`（SD 彩色图）与 `data/images`（脚本彩色图）已随 CMYK 路线
  **删除**；`gen_ai.py` 保留在仓库备用（主线不再依赖，膜层是灰度厚度图）。

## 阶段 2：训练

**自动模式（推荐，默认行为）**：
`uv run python train.py --printer film --sigma-um 25 --dpi 1440 --out runs/onebit`

两段式训练（踩坑后的正确姿势，详见 PITFALLS #1/#4）：

1. **退火阶段**（默认每轮 10 epoch × 4 轮）：`sigmoid(logits/T)` 软前向，
   T 逐轮 1.0→0.3→0.09→0.03，`w_bin` 逐轮加强（0.05·轮；2026-10-21 修复后
   才真实生效，PITFALLS #47）。软域学覆盖率。
2. **硬 STE 微调**（默认 1 轮 20 epoch）：硬前向 + 截断 STE 反传，
   消灭软值作弊，训练指标从此 = 推理指标。**严禁从零直接硬 STE**（#1）。

策略：孤立点过多则 `w_iso` 加倍（封顶 0.2）；psnr 上涨则续训；停滞则 `lr` 减半。
每轮结束在验证集测硬阈值 psnr/孤立点率/明暗偏差。
产物：`best.pt`、`train.log`（逐 epoch）、`pipeline_log.jsonl`（逐轮，中途被杀不丢）。

手动/微调单段（`--rounds 1`，一般不用；**默认从 `<--out>/last.pt` 续训**，
从零须显式 `--no-resume`，#51）：

```bash
uv run python train.py --rounds 1 --data data/images_gray --epochs 20 --length 2000
# 冒烟（~10 秒）：uv run python train.py --smoke
```

- 每个 epoch 产出：`<--out>/epoch_NNN.png`（三行：目标 / 二值墨点 / 模拟打印）。
- 参考速度：P2000 上约 70s/epoch（batch 16，256×256）。

**验收标准**（数值优先，肉眼只作辅助）：
1. 硬阈值验证 psnr（**唯一可信指标**，见"别用 train.log 的 psnr"）；
2. `uv run python eval.py wedge --ckpt runs/onebit/best.pt --ed`：分段覆盖偏差
   |x|<0.05，且**行墨量 std < 0.05**（>0.05 说明用水平条纹代替像素抖动，#12）；
3. `epoch_*.png` 第二行（墨点）是细腻抖动纹理，不是大块团聚/条纹。

> **别用 `train.log` 里的 psnr 做验收**：退火轮是软前向值，可靠"半滴墨"作弊到
> 30dB+，硬阈值后只剩 13dB（PITFALLS #4/#14）。每轮日志里的 `[验证] psnr=` 才是硬阈值。

不达标时的调节方向：

| 症状 | 调法 |
|---|---|
| 墨点团聚成块 / 孤立点多 | 调大 `--w-iso`（pipeline 自动加倍，封顶 0.2） |
| 退火全程躺平不涨 | σ 必须 ≳1.4px：`--sigma-um 25 --dpi 1440`（#11） |
| 整体偏黑/偏白 | film 看 `eval.py one` 的 inkcov 与目标覆盖率 |
| 灰度尺呈 S 形/阶跃畸变 | `eval.py wedge` 行墨量 std 判条纹退化解（#12/#13）；`--ed` 已治 |

## 阶段 3：推理出打印位图（**必加 `--ed`**）

```bash
uv run python eval.py infer 照片.jpg --ckpt runs/onebit/best.pt --ed --out out/照片
```

产出两个文件：
- `*_bitmap.png` — 0/1 二值图，**这就是发给打印机的位图**（**墨=黑/0**，与
  `one`/`webapp` 落盘及 inkcov 审计口径一致）；
- `*_preview.png` — 经扩散模型模拟的膜厚效果，用于打印前自检。

⚠ 不带 `--ed` 的硬阈值输出是逐像素独立决策，膜基下 14.80dB 远逊 FS 25.82，
且含条纹退化解（#12/#13/#42）。`--ed` 把量化误差扩散给后续像素，行墨量 std
0.157→0.024、PSNR→24.67；**但仍差 FS 1.15（#43/#45），模型权重贡献 +0.02dB**。

## 阶段 4：真实打印机标定（主线闭环，部分完成）

**已完成**：物理参数已落实到代码，出处可溯。

- 硬件：KM1024iSHE（6 pL，360 dpi/通道 ×4 行），非吸收膜基材，100% 固含墨水
- 实测墨滴铺展直径 50 µm → 高斯等效 σ = 12.5 µm → `σ[px] = σ[µm]×dpi/25400`
- 膜基物理 = 体积守恒线性叠加（无吸收饱和）= `printer.py::FilmSimulator`
- 训练（已验证配方）：`train.py --printer film --sigma-um 25 --dpi 1440 --out runs/onebit`
  （σ=25µm@1440dpi=1.42px 是两段式配方能结晶的**下限**，#11）
- 推理：`eval.py infer 图.jpg --ckpt runs/onebit/best.pt --ed --dpi 1440`（--dpi 只影响预览）
- 参数出处：`uv run python calc_sigma.py`

**待做（按对膜层的意义排序）**：
1. **档位墨量线性度（③）**：单通道 4 档平铺块各打一张 → 扫描/测厚 → 验证
   实际落墨量 ≈ 0/⅓/⅔/1；6pL 喷头档位间常非线性。2-bit film 训练前完成；
2. **铺展核实测（②）**：孤点序列打 → 高分辨扫描 → 拟合真实核替换高斯假设；
3. **dpi 对齐（⑤）**：刻度尺靶 → 实测 dpi，供扫描分析与图案对齐。

（网点扩大/坏嘴标定：坏嘴归打印机维护不做；网点扩大的单通道等效物是②，
旧 CMYK 标定靶脚本 `calib_target.py` 已随路线删除。）

## 阶段 5：2-bit film（规划中，未实现）

单通道 × 4 档（0/⅓/⅔/1 名义值）× FilmSimulator。要点：

- **可复用**：`UNet(levels=4)`、`soft_levels/hard_levels`、`ed_fs`（L 维通用）、
  `FilmSimulator`（4 档墨量卷积=线性体积，物理上直接成立）。
- **需新建**：`train_once` 的单通道多档前向 + 膜厚损失（ΔE/ICC 链已删，
  换 `printer(ink)` 的膜厚 MSE）；推理位图从"0/1 点阵"变"每像素档位号"。
- **验收指标 = 物理尺度均匀性**：100µm 窗口膜厚 RMS 误差 + 档位混合统计
  （`texture_stats` 的 loc_std 同源）。不用视觉 ΔE——膜层的裁判是工艺，
  高频混合被物理平均掉正是设计意图。
- **先决问题（研究本体）**：膜基线性物理下 ED 已近信息天花板（#11/#45），
  2-bit 必须回答"网络贡献了什么"。候选增量：档位使用约束、最大连续膜厚、
  台阶/平坦度损失——ED 结构上不知道的。开工第一步是消融对照
  （ED(零 logits) vs ED(模型) vs 硬阈值 vs 最近档），证明非冗余再谈调参。
- 档位标定（阶段 4.③）拿到实测值前，`LEVELS` 先用名义值（2026-10-21 决策）。

## 文件索引

```
train.py             训练入口（默认自动编排；--rounds 1 为单段）
eval.py              评测+诊断：infer / one / batch / wedge / data / norm / compare
                     （compare 也吃 --ed，#51 起真正透传；改配方前对比两 ckpt 必加）
webapp.py            网页版对照（1-bit 三路线；默认 ED 口径，#51 起与 eval --ed 一致）
gen_data.py          合成灰度数据
calc_sigma.py        喷头参数 + 实测铺展 -> σ 计算器
halftone/model.py    U-Net + STE 二值化 + 多档 levels 参数（2-bit film 复用）
halftone/printer.py  可微打印模拟器（FilmSimulator 主线 / PrinterSimulator 纸基保留）
halftone/ed_fs.py    FS 式逐列串行误差扩散（推理后处理，#32/#35/#42）
halftone/losses.py   HVS 低频损失 + 孤立点惩罚（1-bit 单通道）
halftone/data.py     数据加载（RgbCrops）
data/images_gray/    训练图（灰度，默认目录）
runs/                训练产物（CMYK 历史归档已删除，现仅存 onebit/smoke）
out/                 推理产物
```
