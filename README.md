<div align="center">

# 墨匠 InkSmith

**学会打印，而不是渲染** · *Learn to print, not to render*

</div>

神经半色调：NN 输入连续调图，**直接输出可打印的位图**，跳过传统
Floyd–Steinberg 误差扩散流程。**项目目标 = 膜层打印**（非吸收基材上复刻
连续调厚度图），主线 **1-bit 灰度 film**，下一步 **2-bit film**
（单通道 4 档墨量）。

> 2026-10-21：**CMYK 2-bit 全彩路线已整体移除**（代码删除，研究结论归档于
> [PITFALLS.md](PITFALLS.md) #16–#30、#35–#48）。理由：项目裁判是膜层工艺
> 而非传统数码打印；#45/#46 已证明在"复刻 ICC 连续调"目标下 NN 上界 ≈
> 30 行经典多档 ED（ΔE00 2.21）、模型贡献为 0——该路线没有增量价值。
> 历史权重仍在磁盘（`runs/cmyk_sd*/`、`runs/keep/`），复现评测需回到
> git 提交 `5714501`。

## 原理（1-bit film 主线）

```
训练：灰度图 → U-Net → logits → STE 软/硬二值化 → 可微打印模拟器(高斯扩散, 膜基线性体积)
            → 模拟膜厚效果 vs 目标图：L1 + HVS低频MSE + 孤立点惩罚 → 反传
推理：logits → 硬阈值 或 ED-FS 串行误差扩散（--ed，必须加）→ 可打印位图
```

膜基物理（`--printer film`）：非吸收基材 + 100% 固含，墨滴 = 高斯斑
线性体积叠加，无吸收饱和。物理参数实测可溯：KM1024iSHE 6 pL 墨滴铺展
直径 50 µm → σ = 12.5 µm，`calc_sigma.py` 出处可查。

## 当前状态（2026-10-21 实测，全部可复现）

| 指标 | 数值 |
|---|---|
| 硬阈值位图 PSNR（photos_val 12 张，film σ=25µm@1440dpi） | 14.80 |
| `--ed` ED-FS 后处理 | **24.67** |
| FS 基线（公允值，几何对齐修复后） | **25.82**（NN --ed 仍差 1.15，12/12 全输） |
| GS 筛网基线 | 24.67（打平） |
| `--ed` 消融（零 logits 对照） | 24.67 —— 模型权重贡献 +0.02dB（PITFALLS #45） |

⚠ **膜基线性物理下逐像素硬阈值赢不了 FS**（PITFALLS #11）；NN 的假设增量
在 2-bit 档位（亚档位覆盖率 + 工艺约束损失），1-bit 上已证伪。

## 用法

```bash
uv sync

# 冒烟（合成数据，验证流程，~10 秒；改代码后先跑这个）
uv run python train.py --smoke

# 正式训练（默认自动编排：退火多轮 → 硬 STE 微调）
uv run python train.py --printer film --sigma-um 25 --dpi 1440 --out runs/onebit

# 推理：出打印位图 + 模拟预览（--ed 必须加）
uv run python eval.py infer photo.jpg --ckpt runs/onebit/best.pt --ed --out out/photo
# 产出 out/photo_bitmap.png（0/1 打印位图）与 out/photo_preview.png（模拟膜厚效果）

# 三方对照（模型 vs Floyd–Steinberg vs Ghostscript 筛网）
uv run python eval.py one photo.jpg --ckpt runs/onebit/best.pt --ed
uv run python eval.py batch --ckpt runs/onebit/best.pt --ed   # 12 张：24.67 vs FS 25.82

# 诊断（改配方前必跑）
uv run python eval.py wedge --ckpt runs/onebit/best.pt --ed   # 灰度尺分段偏差 + 条纹检测
uv run python webapp.py                                     # 网页对照（默认 runs/onebit/best.pt）
```

完整命令与架构见 [AGENTS.md](AGENTS.md)；分阶段操作手册见 [WORKFLOW.md](WORKFLOW.md)，
历史教训见 [PITFALLS.md](PITFALLS.md)（改训练动力学前必读）。

## 关键超参

- `--sigma-um` / `--dpi`：film 墨滴扩散 σ（µm）与打印分辨率；σ[px]=σ[µm]×dpi/25400，
  已验证配方 σ≥1.4px（25µm@1440dpi=1.42px 是下限，PITFALLS #11）。
- `--gain`：paper 模式网点扩大系数（纸基吸收模型用，主线 film 不涉及）。
- `--w-iso`：孤立点惩罚权重（默认 0.05，编排自动加倍封顶 0.2）。

## 路线图（2026-10-21 重排：膜层单线）

1. **2-bit film**（当前主线延伸，最大交付形态）：单通道 × 4 档墨量
   （0/⅓/⅔/1 名义值）× FilmSimulator 线性体积；模型/损失/ED 基础设施已在
   （`UNet(levels=4)`/`soft_levels`/`hard_levels`/`ed_fs` L 维通用），缺训练
   接线。**评价指标 = 物理尺度均匀性**（100µm 尺度膜厚 RMS + 档位混合统计），
   不是视觉口径——膜层裁判是工艺，高频混合被物理平均掉本来就是设计意图。
2. **NN 的增量验证**（研究问题本体）：膜基线性物理下经典 ED 已近信息天花板
   （#11/#45）；2-bit 的候选增量 = 工艺约束（最大连续膜厚/台阶覆盖/界面
   平坦度）损失项——这些是 ED 结构上不知道、可以焊进损失的。先做
   ED(零先验) vs ED(模型 logits) vs 硬阈值消融，证明网络非冗余再谈调参。
3. **档位标定（③）**：4 档实际落墨量是否线性（6pL 喷头档位间常非线性），
   打平铺块+扫描/测厚实测；先按名义值跑通，标定作为后续单变量实验。
4. **铺展核实测（②）**：孤点序列打→扫描→拟合真实核替换高斯假设。
5. ~~网点扩大标定（CMYK LUT256）~~ → 随 CMYK 路线移除；若 2-bit film
   需要实测物理，走 3/4 的单通道版本。
