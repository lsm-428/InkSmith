"""打印机模拟器（现用作评测渲染器）：墨滴 = 高斯扩散斑，重叠处按吸收模型变暗。

NN 训练已于 #52 移除；本模块现供 eval/webapp 把位图渲染成膜厚预览并算 PSNR。
保留可微实现（无 detach，纯张量运算）是为将来离线优化（DBS/CVO）可直接复用。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel1d(sigma):
    radius = max(2, math.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(x, sigma_px):
    """按给定 σ(像素) 现算核做高斯模糊（物理尺度评价/损失用）。"""
    k1 = gaussian_kernel1d(sigma_px).to(x.device)
    k = (k1[:, None] @ k1[None, :])[None, None]
    return F.conv2d(x, k, padding=k.shape[-1] // 2)


class PrinterSimulator(nn.Module):
    """纸基吸收型：输入墨点图 ink（1=落墨），输出纸上灰度（1=白纸，0=全黑）。

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


class FilmSimulator(nn.Module):
    """非吸收膜基 + 100% 固含：体积守恒线性铺展。

    与纸基的本质区别：墨不渗不蒸发，膜厚与落墨体积线性相加，
    无吸收饱和（exp 死区物理上不存在）。
    输入墨点图（1=落墨），输出相对膜厚（1=满覆盖标称膜厚）。

    物理出处（KM1024iSHE）：6 pL 墨滴实测铺展直径 50 µm，
    高斯等效 σ ≈ 直径/4 = 12.5 µm；sigma_um/dpi 可随实测替换。
    """

    def __init__(self, sigma_um=12.5, dpi=720):
        super().__init__()
        self.sigma_um, self.dpi = sigma_um, dpi
        self.sigma_px = sigma_um * dpi / 25400.0
        self.register_buffer("kernel", self._kernel2d(self.sigma_px))

    @staticmethod
    def _kernel2d(sigma_px):
        k1 = gaussian_kernel1d(sigma_px)
        return (k1[:, None] @ k1[None, :])[None, None]

    def forward(self, ink, sigma_px=None):
        """sigma_px 为空用构造时的固定 dpi；传入则按该 σ 现算核（连续 dpi 训练用）。"""
        k = self._kernel2d(sigma_px).to(ink.device) if sigma_px else self.kernel
        # 线性体积叠加：满覆盖 -> 1.0（标称膜厚），无饱和
        return F.conv2d(ink, k, padding=k.shape[-1] // 2)


def build_printer(a):
    """从 checkpoint 保存的 args 字典重建打印机；旧 checkpoint 无 printer 字段时按纸基。"""
    if a.get("printer", "paper") == "film":
        return FilmSimulator(sigma_um=a.get("sigma_um", 12.5), dpi=a.get("dpi", 720))
    return PrinterSimulator(sigma=a["sigma"], gain=a["gain"])
