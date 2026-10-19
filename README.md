# neural-halftone

神经半色调原型：NN 输入连续灰度图，**直接输出喷墨可打印的 1-bit 二值点阵图**，
跳过传统误差扩散（Floyd–Steinberg）流程。

## 原理

```
灰度图 → U-Net → logits → STE 硬二值化 → 可微打印模拟器(高斯扩散+网点扩大)
        → 模拟纸上效果 vs 目标图：L1 + HVS低频MSE + 孤立点惩罚 → 反传
推理时：logits > 0 → 真 0/1 位图 → 打印
```

- `halftone/model.py` — U-Net + 直通估计器（STE）二值化
- `halftone/printer.py` — 可微打印机模型（墨滴=高斯斑）。两套物理：`PrinterSimulator`
  （纸基吸收，exp 饱和）与 `FilmSimulator`（非吸收膜基 + 100% 固含，线性体积）
- `halftone/losses.py` — HVS 低频匹配损失 + 孤立墨点/白洞惩罚

## 用法

```bash
uv sync

# 冒烟（合成数据，验证流程，~10 秒）
uv run python train.py --smoke

# 正式训练（默认自动编排：退火多轮 → 硬 STE 微调）
uv run python train.py --printer film --sigma-um 25 --dpi 1440 --out runs/film

# 推理：出打印位图 + 模拟预览
uv run python eval.py infer photo.jpg --ckpt runs/film/last.pt --out out/photo
# 产出 out/photo_bitmap.png（0/1 打印位图）与 out/photo_preview.png（模拟纸上效果）

# 三方对照（模型 vs Floyd–Steinberg vs Ghostscript 筛网）
uv run python eval.py one photo.jpg --ckpt runs/film/best.pt
```

完整命令与架构见 [AGENTS.md](AGENTS.md)；分阶段操作手册见 [WORKFLOW.md](WORKFLOW.md)，
历史教训见 [PITFALLS.md](PITFALLS.md)（改训练动力学前必读）。

## 关键超参

- `--sigma`：墨滴扩散半径（像素）。打印位图分辨率下，一个墨点在纸上的实际半径。
- `--gain`：网点扩大系数（dot gain），典型 1.2~1.8，可用打印测试图扫描标定。
- `--w-iso`：孤立点惩罚权重。喷墨打不出可靠孤点时调大。

## 路线图（2026-10-19 更新）

已完成：

1. ~~2-bit 四档墨滴~~ → 已并入下面的 CMYK 路线（`--levels 4`）；
2. ~~CMYK 四通道 + 套印模型~~ → **已实现**：`--cmyk` 走 ICC 标定
   （FOGRA39L）+ LUT256 Neugebauer + Lab ΔE2000，`runs/cmyk_v4/best.pt`
   在 photos_val 上硬域 ΔE00 **10.77**，对 gs 工业 RIP **-19.17**（6/6 全胜，
   同 ICC 分离只比半色调算法）。详见 `WORKFLOW.md` 阶段 5。

下一步（按优先级）：

1. **网点扩大标定**（最大未开采优势）：`calib_target.py` 生成 256 块标定靶
   → 真机打印 → `--analyze` 实测 Lab 重建 LUT256 → 接进训练。NN 可把
   墨水/纸张偏差焊进损失反向补偿，RIP 结构上做不到；
2. 用 `halftone/ed_fs.py` 的串行误差扩散修 1-bit 老路的条纹退化解
   （PITFALLS #12/#13）；
3. GCR（暗部偏好 K 墨）与通道去相关项；
4. 把 ΔE00 从 10.77 压到 3.0 可用线（需标定 LUT + 更多训练数据）。
