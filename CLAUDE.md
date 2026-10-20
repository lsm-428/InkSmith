# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**项目指引已迁移到 [AGENTS.md](AGENTS.md)** —— 命令、架构、训练铁律、工程环境都在那里，
本文件不再重复维护，避免两份文档漂移。请直接读 `AGENTS.md`。

补充说明（AGENTS.md 未展开的部分）：

- 项目目标：神经半色调（learned halftoning）——NN 输入连续调图，直接输出喷墨可打印的
  位图，跳过传统误差扩散。项目名 **墨匠 InkSmith**（*学会打印，而不是渲染*）。
  两条路线并存：
  - **1-bit 灰度**（`--printer film/paper`）：可微打印模拟器焊进损失函数，
    反向补偿网点扩大。**推理必须加 `--ed`**：硬阈值 14.80dB（输 FS 19.48），
    ED-FS 后处理后 **24.67dB，对 FS +5.19 / 对 GS +5.55，12/12 全胜**；
  - **CMYK 2-bit 全彩**（`--cmyk`，2026-10-20 已收敛）：ICC 标定 LUT256 +
    Neugebauer + Lab ΔE2000，无可微打印模拟器（网点扩大未标定）。
    当前 `runs/cmyk_sd/best.pt` 在 photos_val 硬域 ΔE00 **8.92**，
    对 gs 工业 RIP **-21.54**（12/12 全胜）。
- 核心设计：**打印机物理焊进损失函数**。训练时前向经过可微打印模拟器，loss 在
  "模拟纸上效果 vs 目标图"上算，网络因此学会反向补偿网点扩大；推理时只做硬阈值，
  不走模拟器。`sigma`/`gain`/`sigma-um` 是物理参数（对应真机标定），不是画质调参旋钮。
- **dithering 的正确来源**：不是损失能鼓励出来的，是决策机制的属性。
  `halftone/ed_fs.py`（FS 式串行误差扩散）作**推理后处理**使用；当训练前向会发散
  （PITFALLS #33）。1-bit 与 CMYK 都靠它；但 ED 是否有利取决于权重——
  CMYK 的 SD 权重上 ED 反而有害（PITFALLS #39），每次换配方必须开/关各测一遍。
- 长期路线细节（Neugebauer 叠印、通道去相关、TAC 上限、GCR）见 `WORKFLOW.md`
  阶段 5（**已实现、训练已收敛，但仍未达可用线**，不再是"待做"）。