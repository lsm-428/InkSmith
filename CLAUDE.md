# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目目标

神经半色调（learned halftoning）：NN 输入连续灰度图，直接输出喷墨可打印的 1-bit
二值位图，跳过传统误差扩散。长期路线：单通道 1-bit → 单通道多档墨滴 → CMYK 全彩
（Neugebauer 叠印模型）。完整流程见 `WORKFLOW.md`，历史教训见 `PITFALLS.md`——
**改训练动力学之前必读 PITFALLS.md**。

## 常用命令

```bash
uv sync                                        # 装环境（torch 2.7.1+cu118，P2000 可用）
uv run python gen_data.py --n 500              # 生成合成训练图 -> data/images/
uv run python pipeline.py --sigma 1.5 --gain 2.0   # 自动两段式训练（主入口）
uv run python train.py --smoke                 # 1 分钟流程冒烟（改动后先跑这个）
uv run python train.py --resume --epochs 10    # 手动续训
uv run python verify.py 图.jpg                 # 验证：模型 vs Floyd-Steinberg，PSNR + gs 墨量审计
uv run python infer.py 图.jpg --out out/x      # 出打印位图 *_bitmap.png + 模拟预览 *_preview.png
```

无单元测试；`--smoke` + `verify.py` 就是回归手段。FS 基线是目前要打败的标杆
（狠物理 σ=1.5/gain=2.0 下验证集均值 18.88 dB）。

## 架构（三文件就是全部）

```
train.py / pipeline.py          入口：单次训练 / 自动多轮+策略调参+硬STE微调
halftone/model.py               U-Net + 两种二值化（soft 退火 / hard 截断STE）
halftone/printer.py             可微打印模拟器：高斯扩散 + exp 吸收（dot gain）
halftone/losses.py              HVS 低频损失 + 孤立点惩罚
```

核心设计：**打印机物理焊进损失函数**。训练时前向经过 `PrinterSimulator`
（可微），loss 在"模拟纸上效果 vs 目标图"上算，网络因此学会反向补偿网点扩大。
推理时只做硬阈值，不走模拟器。`sigma`/`gain` 是物理参数，对应 WORKFLOW 阶段 4
的真机标定，不是画质调参旋钮。

## 训练动力学的铁律（违反必然失败，均有实证）

1. **禁止从零硬 STE 训练**：CNN 长不出 Nyquist 抖动（flatness degradation），
   几千步 loss 躺平。必须软退火（T 1.0→0.03）→ 硬 STE 微调两段式，pipeline 已实现。
2. **只信硬阈值验证 psnr**：软前向可达 30dB 但硬阈值后可能只有 13dB
   （软值"半滴墨"作弊）。pipeline 的 evaluate 是硬阈值，train.log 里的 psnr 是软值。
3. **基线先数值自检再对比**：`Image.convert("1")` 返回 bool，除 255 会毁掉
   FS 基线造成假胜利（PITFALLS #6）。
4. 打印模拟器的饱和区 = 优化死区（exp 吸收 + sigmoid 双饱和曾锁死实心填充）。

## 工程注意

- GPU 是 Quadro P2000（5G，Pascal）：batch 16×256² 已满；动手前
  `nvidia-smi --query-compute-apps` 确认无残留进程（pkill 常杀不净）。
- 日志必须落盘：`train.log`（append+行缓冲）、`pipeline_log.jsonl`（逐轮）。
  后台任务超时会杀进程，落盘日志是唯一幸存记录。
- 同一时间只跑一个 `uv` 进程（缓存锁冲突）。
- ghostscript 不吃 PNG 输入，inkcov 审计前用 PIL 包 PDF（verify.py 已处理）。
- 训练用图放 `data/images/`，验证集 `data/val/`（种子 123 固定，勿混入训练）。
