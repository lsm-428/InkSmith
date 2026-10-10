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

## 下一步

1. 用真实打印-扫描数据标定 `sigma`/`gain`，或把打印机模型换成标定好的点扩散函数；
2. 输出从 1-bit 扩展为 2-bit 四档（无/小/中/大墨滴）；
3. 扩展 CMYK 四通道，通道间用套印模型耦合。
