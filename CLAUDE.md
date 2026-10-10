# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**项目指引已迁移到 [AGENTS.md](AGENTS.md)** —— 命令、架构、训练铁律、工程环境都在那里，
本文件不再重复维护，避免两份文档漂移。请直接读 `AGENTS.md`。

补充说明（AGENTS.md 未展开的部分）：

- 项目目标：神经半色调（learned halftoning）——NN 输入连续灰度图，直接输出喷墨可打印的
  1-bit 二值位图，跳过传统误差扩散。长期路线：单通道 1-bit → 单通道多档墨滴 → CMYK 全彩
  （Neugebauer 叠印模型）。
- 核心设计：**打印机物理焊进损失函数**。训练时前向经过可微打印模拟器，loss 在
  "模拟纸上效果 vs 目标图"上算，网络因此学会反向补偿网点扩大；推理时只做硬阈值，
  不走模拟器。`sigma`/`gain`/`sigma-um` 是物理参数（对应真机标定），不是画质调参旋钮。
- 长期路线细节（Neugebauer 叠印、通道去相关、TAC 上限、GCR）见 `WORKFLOW.md` 阶段 5。