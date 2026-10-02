"""可微打印机模拟器：墨滴 = 高斯扩散斑，重叠处按吸收模型变暗。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel1d(sigma):
    radius = max(2, math.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


class PrinterSimulator(nn.Module):
    """输入墨点图 ink（1=该像素落墨），输出纸上灰度（1=白纸，0=全黑）。

    sigma: 墨滴扩散半径（像素），dot gain 的几何来源
    gain:  吸收系数，>1 时模拟墨点重叠导致的网点扩大
    """

    def __init__(self, sigma=1.0, gain=1.5):
        super().__init__()
        k1 = gaussian_kernel1d(sigma)
        k2 = k1[:, None] @ k1[None, :]
        self.register_buffer("kernel", k2[None, None])
        self.gain = gain

    def forward(self, ink):
        coverage = F.conv2d(ink, self.kernel, padding=self.kernel.shape[-1] // 2)
        printed_ink = 1.0 - torch.exp(-self.gain * coverage)
        return 1.0 - printed_ink
