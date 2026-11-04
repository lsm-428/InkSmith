# -*- coding: utf-8 -*-
"""每个喷头一套独立的 ML。

**喷头之间绝不共用特征**——它们在 ADA 里的参数命名空间完全不同：

    DMC Samba (1117)   Pulse{0..3}.{Voltage,Width,Tail} + Slewrate + VoltageSupply
    KM512     (3)      LowVoltage/HighVoltage/DropPeriod/OnPulseWidth/
                       OffPulseWidth/OnToOffDelay/PhaseLength/Grayscale
    KM1024i   (80)     KM1024iWaveformPulse{n}Section{m}Width (105 个分段) +
                       KM1024i.Width / VoltageAdjustLeft|Right /
                       KM1024iDriverPowerVoltageAdjustLow|High

每个模块都是 `python -m inkjet_ml.heads.<name>` 可直接运行。
"""

__all__ = ["samba", "km512", "km1024i"]