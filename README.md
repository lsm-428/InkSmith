<div align="center">

# 墨匠 InkSmith

**学会打印，而不是渲染** · *Learn to print, not to render*

</div>

> **Smith 作动词是"锻造"。** 这个项目的标志性动作就是把可微打印机物理**焊进
> 损失函数**，然后拿真实墨水/纸张标定去**锻造**它——锻造的是墨，不是渲染像素。
> 这不是自我贴金：CMYK 路线实测 ΔE00 **8.92 vs 工业 RIP 30.46**，12/12 全胜
> gs tiffsep1，比工业工具更会放墨。

神经半色调：NN 输入连续调图，**直接输出喷墨可打印的位图**，跳过传统
Floyd–Steinberg 误差扩散流程。两条路线：

| 路线 | 输入 | 输出 | 物理建模 | 状态 |
|---|---|---|---|---|
| **1-bit 灰度** `--printer film/paper` | 灰度图 | 1-bit 二值位图 | 可微打印模拟器（网点扩大预补偿） | 原型；film 线性物理下保真度赢不了 FS（PITFALLS #11），优势在 paper/exp 吸收模式 |
| **CMYK 2-bit 全彩** `--cmyk` | sRGB 图 | 4 通道 × 4 档墨量 | ICC 标定 LUT256 + Neugebauer + Lab ΔE2000（无可微模拟器） | **已收敛**：photos_val 硬域 ΔE00 **8.92**，对 gs 工业 RIP **-21.54**（12/12 全胜） |

## 原理

```
# 1-bit 路线
灰度图 → U-Net → logits → STE 硬二值化 → 可微打印模拟器(高斯扩散+网点扩大)
        → 模拟纸上效果 vs 目标图：L1 + HVS低频MSE + 孤立点惩罚 → 反传

# CMYK 2-bit 路线
sRGB → U-Net → 4通道×4档 logits → softmax(退火)/argmax(硬) → LUT256 查表
      → Lab 空间 ΔE2000 损失；推理时可套 ED-FS 串行误差扩散拿 dithering
```

- `halftone/model.py` — U-Net + 直通估计器（STE）二值化 + 多档（`soft_levels`/`hard_levels`）
- `halftone/printer.py` — 可微打印机模型（墨滴=高斯斑）。两套物理：`PrinterSimulator`
  （纸基吸收，exp 饱和）与 `FilmSimulator`（非吸收膜基 + 100% 固含，线性体积）
- `halftone/cmyk.py` — 色彩层：ICC(FOGRA39L) 标定 LUT256 + Neugebauer 收缩 + sRGB↔Lab↔CMYK
- `halftone/ed_fs.py` — FS 式逐行串行误差扩散。**推理后处理专用**（训练前向会发散，PITFALLS #33）
- `halftone/losses.py` — HVS 低频匹配损失 + 孤立墨点/白洞惩罚（CMYK 四通道联合掩码）

## 用法

```bash
uv sync

# 冒烟（合成数据，验证流程，~10 秒）
uv run python train.py --smoke
uv run python train.py --smoke --cmyk          # CMYK 路线冒烟

# 1-bit 正式训练（默认自动编排：退火多轮 → 硬 STE 微调）
uv run python train.py --printer film --sigma-um 25 --dpi 1440 --out runs/film

# CMYK 正式训练（注意 --batch 4：256 组合张量链显存峰值高，batch 16 会爆）
uv run python train.py --cmyk --batch 4 --out runs/cmyk

# 推理：出打印位图 + 模拟预览
uv run python eval.py infer photo.jpg --ckpt runs/film/last.pt --out out/photo
# 产出 out/photo_bitmap.png（0/1 打印位图）与 out/photo_preview.png（模拟纸上效果）

# 三方对照（模型 vs Floyd–Steinberg vs Ghostscript 筛网）
uv run python eval.py one photo.jpg --ckpt runs/film/best.pt

# CMYK 评测（ΔE00 + 四通道 inkcov；--rip 加工业 RIP 基线；--ed 加误差扩散后处理）
uv run python eval.py cmyk data/photos_val --ckpt runs/cmyk/best.pt --rip
```

完整命令与架构见 [AGENTS.md](AGENTS.md)；分阶段操作手册见 [WORKFLOW.md](WORKFLOW.md)，
历史教训见 [PITFALLS.md](PITFALLS.md)（改训练动力学前必读）。

## 关键超参

- `--sigma`：墨滴扩散半径（像素）。打印位图分辨率下，一个墨点在纸上的实际半径。
- `--gain`：网点扩大系数（dot gain），典型 1.2~1.8，可用打印测试图扫描标定。
- `--w-iso`：孤立点惩罚权重。喷墨打不出可靠孤点时调大。
- CMYK 路线：`--w-cov 0.3`（必须 >0，否则跨通道作弊，PITFALLS #17）、
  `--w-tac 1.0`、`--tac-max 3.2`、退火终点 T 0.003。

## 路线图（2026-10-20 更新）

已完成：

1. ~~2-bit 四档墨滴~~ → 已并入 CMYK 路线（`--levels 4`）；
2. ~~CMYK 四通道 + 套印模型~~ → **已实现并收敛**：`--cmyk` 走 ICC 标定
   （FOGRA39L）+ LUT256 Neugebauer + Lab ΔE2000，`runs/cmyk_sd/best.pt`
   在 photos_val 上硬域 ΔE00 **8.92**，对 gs 工业 RIP **-21.54**（12/12 全胜，
   同 ICC 分离只比半色调算法）。详见 `WORKFLOW.md` 阶段 5。

下一步（按优先级）：

1. **网点扩大标定**（最大未开采优势）：`calib_target.py` 生成 256 块标定靶
   → 真机打印 → `--analyze` 实测 Lab 重建 LUT256 → 接进训练。NN 可把
   墨水/纸张偏差焊进损失反向补偿，RIP 结构上做不到；
2. 用 `halftone/ed_fs.py` 的串行误差扩散修 1-bit 老路的条纹退化解
   （PITFALLS #12/#13）；
3. GCR（暗部偏好 K 墨）与通道去相关项；
4. 把 ΔE00 从 8.92 压到 3.0 可用线（需标定 LUT + 更多训练数据）。
