# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

**项目指引已迁移到 [AGENTS.md](AGENTS.md)** —— 命令、架构、训练铁律、工程环境都在那里，
本文件不再重复维护。请直接读 `AGENTS.md`。

补充说明（AGENTS.md 未展开的部分）：

- 项目目标：神经半色调（learned halftoning），**落地场景是膜层打印**（非吸收基材上
  复刻连续调厚度图），不是传统数码打印。主线 **1-bit film**（`--printer film`），
  下一步 **2-bit film**（单通道 4 档墨量）。
- **CMYK 2-bit 全彩路线已于 2026-10-21 整体移除**（代码删除）。它的研究结论完整
  归档在 PITFALLS.md #16–#30/#35–#48，移除决策链见 #50：在"复刻 ICC 连续调分离"
  目标下，网络理论天花板 = 经典多档误差扩散（ΔE00 2.21），模型贡献被 ED 洗成 0，
  且项目裁判是膜层工艺而非彩色数码打印——该路线没有增量价值。历史权重在磁盘
  （`runs/cmyk_sd*/`、`runs/keep/`，gitignore 不入库），复现其 ΔE 数字须 checkout
  git 提交 `5714501`。
- 核心设计：**打印物理焊进损失函数**。训练时前向经过可微打印模拟器（膜基=线性体积
  叠加，`FilmSimulator`），loss 在"模拟膜厚 vs 目标图"上算；推理只做硬阈值/ED-FS。
  `sigma-um`/`sigma`/`dpi` 是物理参数（实测可溯，`calc_sigma.py`），不是画质调参旋钮。
- **1-bit film 的已证结论**（引用数字前必读）：膜基线性物理下逐像素决策赢不了 FS
  （PITFALLS #11）；`--ed` 后 24.67 vs FS 25.82（差 1.15，#43），模型权重贡献
  +0.02dB（#45）。"NN 在 1-bit film 上冗余"是结构性负结果，不是没调好参。
- **dithering 的正确来源**：不是损失能鼓励出来的，是决策机制的属性。
  `halftone/ed_fs.py`（FS 式串行误差扩散）作**推理后处理**；当训练前向会发散
  （#33）；权重和必须严格为 1（#42，否则覆盖率两端塌陷）。
- 评测指标：1-bit 用硬阈值 PSNR（唯一可信，#4/#14）+ `wedge` 行墨量 std（#12）。
  **2-bit film 计划改用物理尺度均匀性**（100µm 膜厚 RMS + 档位混合统计）——
  膜层被工艺平均，视觉 ΔE 口径（#44 的教训）不适用。
- 长期路线细节见 `WORKFLOW.md` 阶段 5（2-bit film 规划）与 README 路线图。
