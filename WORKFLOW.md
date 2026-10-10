# 工作流程：神经半色调（图像 → 喷墨二值打印图）

> 目标：NN 输入连续灰度图，直接输出喷墨可打的 1-bit 位图，跳过传统误差扩散。
> 本文档是整个流程的可复现操作手册，按阶段组织。

## 阶段 0：环境（一次性）

```bash
uv sync        # 按 pyproject.toml 装 torch 2.7.1+cu118（EFD 项目同款配置）
```

验证 GPU：`uv run python -c "import torch; print(torch.cuda.is_available())"` 应输出 `True`。

## 阶段 1：训练数据

两条路线，可叠加（都放 `data/images/`）：

| 来源 | 命令 | 说明 |
|---|---|---|
| 合成数据 | `uv run python gen_data.py --n 500` | 6 类频率成分：渐变/斑块/分形噪声/几何/条纹/混合 |
| 真实照片 | 手动拷贝进 `data/images/` | 内容越接近实际打印目标越好 |

- 合成数据量：`--n` 建议 500~2000。
- 重新生成会覆盖同名文件（种子固定，可复现）。

## 阶段 2：训练

**自动模式（推荐，默认行为）**：
`uv run python train.py --printer film --sigma-um 25 --dpi 1440 --out runs/film`

两段式训练（这是踩坑后的正确姿势，详见 PITFALLS.md #1/#4）：

1. **退火阶段**（默认每轮 10 epoch × 4 轮）：`sigmoid(logits/T)` 软前向，
   T 逐轮 1.0→0.3→0.09→0.03，`w_bin` 逐轮加强。软域学覆盖率。
2. **硬 STE 微调**（默认 1 轮 20 epoch）：硬前向 + 截断 STE 反传，
   消灭软值作弊，训练指标从此 = 推理指标。**严禁从零直接硬 STE**。

策略：孤立点过多则 `w_iso` 加倍（封顶 0.2）；psnr 上涨则续训；停滞则 `lr` 减半。
每轮结束在验证集（`data/val`，种子 123 固定生成 30 张）测三项指标：
psnr（**硬阈值**，唯一可信）、孤立点率、明暗偏差。
产物：`best.pt`（最佳权重）、`train.log`（逐 epoch）、
`pipeline_log.jsonl`（逐轮，中途被杀不丢）。

手动/微调单段（`--rounds 1`，一般不用）：

```bash
uv run python train.py --rounds 1 --data data/images --epochs 20 --length 2000
# 冒烟（~10 秒）：uv run python train.py --smoke
```

- 每个 epoch 产出：`runs/halftone/epoch_NNN.png`（三行：目标 / 二值墨点 / 模拟打印）
- 权重：`runs/halftone/last.pt`（含训练超参，eval/infer 自动读取）
- 参考速度：P2000 上约 70s/epoch（batch 16，256×256）

**验收标准**（数值优先，肉眼只作辅助）：
1. 硬阈值验证 psnr 达标（**唯一可信指标**，见下方"别用 train.log 的 psnr"）；
2. 跑 `uv run python eval.py wedge --ckpt runs/film/best.pt`：分段覆盖偏差 |x|<0.05，
   且**行墨量 std < 0.05**（>0.05 说明模型用水平条纹代替像素抖动，见 PITFALLS #12）；
3. `epoch_*.png` 第二行（墨点）是细腻抖动纹理，不是大块团聚/条纹——这项只看图，
   因为它对应上面那个数字判据。

> **别用 `train.log` 里的 psnr 做验收**：那是软前向值，可以靠"半滴墨"作弊到
> 30dB，硬阈值后只剩 13dB（PITFALLS #4）。每轮日志里的 `[验证] psnr=` 才是硬阈值。

不达标时的调节方向：

| 症状 | 调法 |
|---|---|
| 墨点团聚成块 / 孤立点多 | 调大 `--w-iso`（pipeline 会自动加倍，封顶 0.2） |
| 灰度尺呈 S 形畸变 | 查 `eval.py wedge` 的行墨量 std，是条纹退化解（见 PITFALLS #12） |
| 中间调发脏/层次糊 | 检查 `--sigma-um` 是否匹配真实墨点（film）或调大 `--w-hvs` |
| 整体偏黑/偏白 | paper 调 `--gain`；film 看 `eval.py one` 的 inkcov 与目标覆盖率 |
| 退火全程躺平不涨 | 打印模糊 σ 必须 ≳1.4px（film: `--sigma-um 25 --dpi 1440`），见 PITFALLS #11 |

## 阶段 3：推理出打印位图

```bash
uv run python eval.py infer 照片.jpg --ckpt runs/halftone/last.pt --out out/照片
```

产出两个文件：
- `*_bitmap.png` — 0/1 二值图，**这就是发给打印机的位图**；
- `*_preview.png` — 经扩散模型模拟的纸上效果，用于打印前自检。

## 阶段 4：真实打印机标定（部分完成）

**已完成**：物理参数已落实到代码。

- 硬件：KM1024iSHE（6 pL，360 dpi/通道 ×4 行），非吸收膜基材，100% 固含墨水
- 实测墨滴铺展直径 50 µm → 高斯等效 σ = 12.5 µm → `σ[px] = 12.5×dpi/25400`
- 膜基物理 = 体积守恒线性叠加（无吸收饱和），对应 `printer.py` 的
  `FilmSimulator`；纸基吸收型保留为 `PrinterSimulator`
- 训练（单 dpi，已验证配方）：`train.py --printer film --sigma-um 25 \
    --dpi 1440 --out runs/film`
  σ=25µm@1440dpi = 1.42px，是两段式配方能结晶的**下限**（σ<1.4px 会让退火全程
  躺平，见 PITFALLS #11）。
- ~~训练（连续 dpi）~~：`--dpi-min/--dpi-max` 随机 dpi 模式代码仍支持，但
  **线性物理下已验证无收益**（只稀释纹理结晶信号，且网络无法感知当前 batch 的 σ）。
  单 dpi 训练即可，靠图案 dpi 无关性泛化。详见 PITFALLS #11。
- 推理：`eval.py infer 图.jpg --ckpt runs/film/best.pt --dpi 1440`
  （--dpi 只影响预览渲染，位图与 dpi 无关）
- 参数出处可溯：`uv run python calc_sigma.py`（读 printheads.json + 实测铺展）

**语义注意**：film 模式下灰度值 ↔ 相对膜厚（1=满覆盖标称膜厚），
评价指标是膜厚复刻精度；`eval.py` 会自动用反相图给 FS/GS 加网以对齐语义。

**待做**：打印测试图 → 实测膜厚分布（轮廓仪）→ 拟合真实铺展核替换高斯。

## 阶段 5（未做）：全彩 CMYK

单通道练稳后扩展全彩。架构上只有输出通道数、打印模型、损失三处变化，
U-Net 与训练循环复用。

### 核心：可微 Neugebauer 叠印模型（替换 printer.py）

```
四通道二值图 → 各通道高斯扩散 → 覆盖率 C,M,Y,K ∈ [0,1]
→ Demichel 方程：16 种叠印组合面积比 = 各通道覆盖率/互补的乘积
   例：P(蓝) = C·M·(1-Y)·(1-K)
→ 纸上颜色 = Σ₁₆ 面积比 × 原色反射率（Yule-Nielsen n 值修正）
→ 模拟彩色图 vs 目标：Lab 空间 ΔE loss
```

### 彩色特有的四个问题

| 问题 | 对策 |
|---|---|
| 叠印非线性（dot-on/off-dot） | Neugebauer 面积比近似；更高精度换 spectral Neugebauer |
| 通道间摩尔纹 | loss 加通道去相关项；人眼对彩色低频纹极敏感 |
| 总墨量上限 TAC (~240-320%) | 惩罚四通道覆盖率之和超限的像素 |
| 灰成分替代 GCR | 暗部给 K 墨更低的"成本"，让模型偏好用 K 替 CMY |

### 标定

打印 16 原色色靶 → 扫描/拍照 → 拟合每种原色反射率与 n 值。
建议先完成阶段 4 的单色标定流程，彩色标定复用同一条管线。

### 顺序建议

1. 单色流程全链路验证（阶段 1-4）；
2. 先上 **2-bit 四档输出**（无/小/中/大墨滴）— 单通道内验证多档墨滴框架；
3. 再全彩：4 通道 × 1-bit，Neugebauer 模型；
4. 最终形态：4 通道 × 2-bit，即每个像素 4 种墨各 4 档。

## 文件索引

```
train.py             训练入口（默认自动编排；--rounds 1 为单段）
eval.py              评测+诊断：infer 推理 / one / batch / wedge / data / norm / compare
webapp.py            网页版对照（单图/多图上传）
gen_data.py          合成数据生成
calc_sigma.py        喷头参数 + 实测铺展 -> σ 计算器（出处可溯）
halftone/model.py    U-Net + STE 二值化
halftone/printer.py  可微打印模拟器（阶段 4 要改的就是这里）
halftone/losses.py   HVS 低频损失 + 孤立点惩罚
halftone/data.py     数据加载
data/images/         训练图
runs/                训练产物
out/                 推理产物
```
