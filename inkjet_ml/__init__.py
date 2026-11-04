"""喷墨驱动波形机器学习（LP50 原生路线）。

**每个喷头一套独立 ML** —— 它们在 ADA 里的参数命名空间完全不同，特征不能共用：

    python -m inkjet_ml.heads.samba      DMC Samba (1117)  有 108 孔实测数据
    python -m inkjet_ml.heads.km512      KM512     (3)     管线就绪，待实测数据
    python -m inkjet_ml.heads.km1024i    KM1024i   (80)    管线就绪，待实测数据

配套：
    dataset   —— 统一读取 ADA 落盘的 Results.csv / Measurement.xml
    headml    —— 5 个模型 + 分组交叉验证（三套共用，保证可横向比较）
    ada       —— 生成 ADA 扫描配方（扫哪些参数、填什么范围）
    ada_rank  —— 读回扫描结果，按 DropQCS 判据给波形组合排序
    inkprops  —— 墨水物性 → Z 数计算器（Z = √(ργD)/η，含论文 Table 1 自检）
    ncalc     —— DropQCS 规则表达式求值器（纯 Python，不依赖 .NET / NCalc.dll）
    dropqcs   —— DropQCS 规则集执行器（NozzlesRule / HeadsRule 两级 + [Result] 链式）
    dropqcs_vars —— 从 Results.csv 还原官方 DropQCS 变量

`ada` / `ada_rank` 全程只用 ADA 自己的读数（Nr. of droplets / Speed / Volume / Angle），
不引入 Z / 黏度 / 表面张力等测不到的物理量，也不计算耗时。

`ada_rank` 的判据有两条路径：
  · 默认：内置三条硬编码判据（单滴 + 速度带 + 方向容差）；
  · `--rules <RulesConfig.xml>`：直接执行**官方 DropQCS 规则集**
    （NCalc 表达式，支持相对判据 / [Result] 链式 / 双头判据 / 位置判据）。
"""

__all__ = ["ada", "ada_rank", "dataset", "headml", "heads", "inkprops",
           "ncalc", "dropqcs", "dropqcs_vars", "test_ncalc"]