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
| **AI 生成（推荐，CMYK 路线主数据源）** | `uv run python gen_ai.py --n 400` | Stable Diffusion 1.5 文生图，照片域纹理。ΔE00 比脚本合成低 18%（PITFALLS #39） |
| 脚本合成 | `uv run python gen_data.py --n 500 --rgb --icc <icc>` | 6 类频率成分：渐变/斑块/分形噪声/几何/条纹/混合 |
| 真实照片 | 手动拷贝进 `data/images/` | 内容越接近实际打印目标越好 |

- 数据量：脚本合成建议 500~2000；SD 生成建议 ≥400（1000 张是 2026-10-20 的规模）。
- 重新生成会覆盖同名文件（种子固定，可复现）；`gen_ai.py` 已支持 resume
  （从目录已有张数继续编号，不会覆盖）。
- **`data/images` 与 `data/images_ai` 当前都是 RGB**（脚本用 `--rgb` 生成，
  文件名 `gen_*_color.png`），`--cmyk` 训练直接可用，不需要再转彩色。
- 注意 PITFALLS #40：**不要**试图用"脚本 + 调色"复现 SD 的效果，
  两者的颜色分布统计量几乎一致，差的是内容纹理真实性。

## 阶段 2：训练

**自动模式（推荐，默认行为）**：
`uv run python train.py --printer film --sigma-um 25 --dpi 1440 --out runs/onebit`

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

- 每个 epoch 产出：`<--out>/epoch_NNN.png`（三行：目标 / 二值墨点 / 模拟打印）。
  `--out` 默认 `runs/halftone`，**实际实验一律显式指定**（如实测用的
  `runs/cmyk_sd`），别依赖默认值。
- 权重：`<--out>/best.pt`、`<--out>/last.pt`（含训练超参，eval/infer 自动读取）
- 参考速度：P2000 上约 70s/epoch（batch 16，256×256）

**验收标准**（数值优先，肉眼只作辅助）：
1. 硬阈值验证 psnr 达标（**唯一可信指标**，见下方"别用 train.log 的 psnr"）；
2. 跑 `uv run python eval.py wedge --ckpt runs/onebit/best.pt`：分段覆盖偏差 |x|<0.05，
   且**行墨量 std < 0.05**（>0.05 说明模型用水平条纹代替像素抖动，见 PITFALLS #12）；
3. `epoch_*.png` 第二行（墨点）是细腻抖动纹理，不是大块团聚/条纹——这项只看图，
   因为它对应上面那个数字判据。

> **别用 `train.log` 里的 psnr 做验收**：那是软前向值，可以靠"半滴墨"作弊到
> 30dB，硬阈值后只剩 13dB（PITFALLS #4）。每轮日志里的 `[验证] psnr=` 才是硬阈值。

不达标时的调节方向：

| 症状 | 调法 |
|---|---|
| 墨点团聚成块 / 孤立点多 | 调大 `--w-iso`（pipeline 会自动加倍，封顶 0.2） |
| 灰度尺呈 S 形畸变 | 查 `eval.py wedge` 的行墨量 std，是条纹退化解（见 PITFALLS #12）。**已解决**：推理加 `--ed`（14.80→24.67dB，详见 PITFALLS #42） |
| 中间调发脏/层次糊 | 检查 `--sigma-um` 是否匹配真实墨点（film）或调大 `--w-hvs` |
| 整体偏黑/偏白 | paper 调 `--gain`；film 看 `eval.py one` 的 inkcov 与目标覆盖率 |
| 退火全程躺平不涨 | 打印模糊 σ 必须 ≳1.4px（film: `--sigma-um 25 --dpi 1440`），见 PITFALLS #11 |

## 阶段 3：推理出打印位图

```bash
uv run python eval.py infer 照片.jpg --ckpt runs/cmyk_sd/best.pt --out out/照片
# （1-bit 路线用 1 通道权重；CMYK 权重走 eval.py cmyk 子命令，见阶段 5）
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
    --dpi 1440 --out runs/onebit`
  σ=25µm@1440dpi = 1.42px，是两段式配方能结晶的**下限**（σ<1.4px 会让退火全程
  躺平，见 PITFALLS #11）。
- ~~训练（连续 dpi）~~：`--dpi-min/--dpi-max` 随机 dpi 模式代码仍支持，但
  **线性物理下已验证无收益**（只稀释纹理结晶信号，且网络无法感知当前 batch 的 σ）。
  单 dpi 训练即可，靠图案 dpi 无关性泛化。详见 PITFALLS #11。
- 推理：`eval.py infer 图.jpg --ckpt runs/onebit/best.pt --dpi 1440`
  （--dpi 只影响预览渲染，位图与 dpi 无关）
- 参数出处可溯：`uv run python calc_sigma.py`（读 printheads.json + 实测铺展）

**语义注意**：film 模式下灰度值 ↔ 相对膜厚（1=满覆盖标称膜厚），
评价指标是膜厚复刻精度；`eval.py` 会自动用反相图给 FS/GS 加网以对齐语义。

**待做**：打印测试图 → 实测膜厚分布（轮廓仪）→ 拟合真实铺展核替换高斯。

## 阶段 5（已实现、训练已收敛，仍未达可用线）：全彩 CMYK 2-bit

> 2026-10-17 落地：`--cmyk` 全链路（train.py / eval.py cmyk / halftone/cmyk.py）。
> 2026-10-19 v4 全量训练收敛（1300 张脚本合成+灰度）：ΔE00 10.85。
> 2026-10-20 SD 照片域数据重训（400 张）：**当前最优 `runs/cmyk_sd/best.pt`，
> photos_val 硬域 ΔE00 8.92**（关 ED），对 gs 工业 RIP **-21.54**（12/12 全胜）。
> ED-FS 后处理在此权重上**有害**（8.92→10.85），见下方权重演进与 PITFALLS #39。

### 已实现的模型（与阶段 5 原计划的差异）

```
sRGB 目标图（in_ch=3）
 → U-Net 输出 out_ch=4 × levels=4 logits → (B,4,H,W,4)
 → 互斥 softmax（退火 T）/ 硬 argmax one-hot（STE 微调）
 → 每通道覆盖率 cov = Σ p·[0,1/3,2/3,1]
 → LUT256 查表（256 种 4 档组合的 Lab，FOGRA39L 标定）
 → Lab 空间 ΔE2000 损失
```

- **没有可微打印模拟器**：CMYK 网点扩大未标定（阶段 5 的"标定"一节未做），
  所以 `--cmyk` 时 printer=None，损失直接比 Neugebauer 纸上 Lab。
- **LUT256 取代 16 项 Demichel 面积加权**：4 档下组合数是 4⁴=256，
  用 ICC 直接标定 256 个色块的 Lab（`build_lut256`），比"先算 16 项面积
  再乘 16 个原色反射率"少一层建模误差，也不需要 Yule-Nielsen n 值。
  代价是硬编码单一 ICC（默认 FOGRA39L，`--icc` 可换）。
- **ICC 同时给出软目标**：`srgb_to_cov()` 把 sRGB 分离成四通道连续覆盖率，
  作为 `l_cov` 的软监督目标（见 PITFALLS #17，w_cov=0.3）。

### 命令

```bash
# 1. 生成彩色训练图（照片化分布：--sat/--gamma 拉高 OOG 与动态范围）
uv run python gen_data.py --n 800 --rgb --icc /usr/share/color/icc/colord/FOGRA39L_coated.icc \
    --cmyk-frac 0.15 --sat 0.6 --gamma 1.4 --size 512
# 2. 训练（默认 FOGRA39L、w_cov=0.3、w_tac=1.0、tac_max=3.2、target_de=3.0）
uv run python train.py --cmyk --batch 4 --data data/images_ai --out runs/cmyk_sd
# 3. 评测：硬域 ΔE00 + 四通道 gs inkcov 审计 + OOG% + TAC（--rip 加工业基线）
uv run python eval.py cmyk data/photos_val --ckpt runs/cmyk_sd/best.pt --rip
# 4. 推理后处理：ED-FS 误差扩散（拿 dithering，多样性 1.77 vs 硬 argmax 1.1）
uv run python eval.py cmyk data/photos_val --ckpt runs/cmyk_sd/best.pt --ed --rip
```

**注意 `--batch 4`**：CMYK 的 256 组合张量链显存峰值远高于 1-bit，
batch 16×256² 会 `CUBLAS_STATUS_ALLOC_FAILED`（PITFALLS #20）。

### 调参结论（GPU 全量实测，PITFALLS #17–#36）

| 项 | 结论 | 依据 |
|---|---|---|
| `--w-cov 0.3` | 必须 >0；0 时跨通道作弊（K 1/3 落成 C 档 2） | PITFALLS #17 |
| `--w-tac 1.0` | 保留；w_tac=0 时 TAC 失控到 1.85 | 真实数据 3 epoch 扫参 |
| `--w-lab 1.0` | ΔE 主损失 | — |
| `--tac-max 3.2` | TAC 上限（四通道墨量和） | 12 张照片单像素 TAC 99 分位 2.84 |
| 退火终点 T | **0.003**（原 0.03，多档下 gap 不收敛） | PITFALLS #19/#23 |
| 硬 STE 微调 | **CMYK 默认跳过**（毁退火成果，ΔE 6.38→42.04） | PITFALLS #19 |
| `--gap-min` | 无效（被 l_lab 淹 2-3 个数量级） | PITFALLS #24 |
| `--cmyk-ste-train` | 无效（STE 梯度与 argmax 不同步） | PITFALLS #25 |
| `--seed-noise` | 无效（多样性没起来，ΔE 恶化 16.10） | PITFALLS #28 |
| 邻域窗口损失 | 无效（软域作弊，ΔE 13.56） | PITFALLS #30 |
| `--ed-fs` 训练前向 | **发散**（ΔE 48-60），勿用 | PITFALLS #33 |
| `eval.py cmyk --ed` | ★**有效**（推理后处理，多样性 1.77，ΔE 持平） | PITFALLS #35/#36 |

**残余 ΔE 的真因**（PITFALLS #26/#27）：不是 4 档量化误差（4 档可表达
ICC 目标到 0.32/0.28/0.27/0.11），而是**单像素 argmax 天花板 3-5 ΔE00**——
模型被逼用非 ICC 覆盖率换取色准。突破靠 ED-FS 推理后处理（亚档位覆盖率
来自串行机制而非损失鼓励）。**但 ED 是否有利取决于权重**：在 v4 权重上
ΔE 10.69→10.52（略好），在 SD 训练的权重上 8.92→10.85（有害，PITFALLS #39）。

**剩余差距与下一步**（当前 8.92 vs 目标 3.0，2.97 倍）：
1. ~~全量训练~~ 已完成（4 轮退火，photos_val 硬域 ΔE00 见下表）；
2. ~~OOG~~ 已缓解（照片化数据；photos_val 实测 OOG 0.3-32.4%，
   最高 0006 的 32.4%，仍是残余误差来源之一）；
3. **训练数据升级为 AI 生成**（2026-10-20，最大单项增益）：`gen_ai.py`
   用 Stable Diffusion 1.5 文生图替代 `gen_data.py` 脚本合成。400 张 SD 图
   训出的模型 ΔE00 **10.85→8.92**（-18%，PITFALLS #39）。分布统计量对比
   证明优势不在颜色分布而在**内容真实性**（PITFALLS #40：SD 图与脚本图的
   饱和度 0.436 vs 0.415、亮度 0.222 vs 0.206 几乎一致，只有 OOG 差
   23.8% vs 16.9%，但模型差 18%）⇒ 不要试图用"脚本 + 调色"复现，要照片域
   就用 SD；
4. **网点扩大标定**（最大未开采优势）：`calib_target.py` 已生成 256 块标定靶
   （`out/dotgain/dotgain_target.pdf`），且 `--analyze` 已能输出
   `measured_lut256.npy`（PITFALLS #41，链路已通、只欠硬件）。等真机打印 →
   扫描 → `--analyze` → 拷成 `runs/<exp>/lut256.npy` 改 meta `"measured": true`
   → 接进训练。NN 可把墨水/纸张偏差焊进损失反向补偿，RIP 结构上做不到；
5. GCR（暗部偏好 K 墨）与通道去相关项尚未实现；
6. 训练数据量：当前 1000 张 SD 图（2026-10-20 下午完成），
   真实照片 12 张仍未入训练集。

### 权重演进（photos_val 12 张硬域，`eval.py cmyk --rip`）

| 权重 | 训练数据 | 关 ED ΔE00 | 开 ED ΔE00 | 对 gs RIP |
|---|---|---|---|---|
| `runs/keep/cmyk_v4_best.pt` | 800 脚本 + 500 灰度 | 10.85 | 10.85 | -19.62 |
| `runs/cmyk_sd/best.pt` | 400 张 SD 生成图 | **8.92** | 10.85 | -21.54 |

★**ED-FS 后处理在 SD 数据上反而有害**（PITFALLS #39）：v4 上无影响，
sd 上恶化 +1.93（8.92→10.85）。ED-FS 原本是补偿"NN 无 dithering"
（PITFALLS #27 多样性 1.03-1.39），sd 训练后 NN 已有抖动能力，ED 开始与
NN 自己的决策打架。⇒ **换数据/配方后必须重测 `--ed` 开/关两遍**，
不能沿用"ED 总是更好"（PITFALLS #35 的结论只在 v4 上成立）。

## 文件索引

```
train.py             训练入口（默认自动编排；--rounds 1 为单段；--cmyk 全彩 2-bit）
eval.py              评测+诊断：infer 推理 / one / batch / wedge / data / norm / compare
                     / cmyk（ΔE00 + 四通道 inkcov；--rip 工业基线；--ed ED-FS 后处理）
webapp.py            网页版对照（单图/多图上传）
gen_data.py          合成数据生成（--rgb 彩色；--icc 走 ICC 分离；--sat/--gamma 照片化）
calib_target.py      网点扩大标定靶生成 + 扫描分析（256 块 4 档全组合）
calc_sigma.py        喷头参数 + 实测铺展 -> σ 计算器（出处可溯）
halftone/model.py    U-Net + STE 二值化 + 多档（in_ch/out_ch/levels）
halftone/printer.py  可微打印模拟器（1-bit；CMYK 路线不用）
halftone/cmyk.py     色彩层：LUT16/LUT256 + Neugebauer + sRGB↔Lab↔CMYK（ICC）
halftone/ed_fs.py    FS 式逐列串行误差扩散（推理后处理，PITFALLS #32/#35/#42）
halftone/losses.py   HVS 低频损失 + 孤立点惩罚（CMYK 联合掩码）
halftone/data.py     数据加载（RgbCrops，灰度自动转 3 通道）
data/images/         训练图
runs/                训练产物
out/                 推理产物
```
