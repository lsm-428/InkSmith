"""损失函数：HVS 低频匹配 + 孤立点惩罚。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .printer import gaussian_kernel1d


class HVSLoss(nn.Module):
    """人眼近似低通滤波器：只惩罚低频误差。

    半色调的本质就是把误差能量推到人眼不敏感的高频（蓝噪声），
    所以直接对打印结果和目标做低通后再比 MSE。
    """

    def __init__(self, sigma=2.0):
        super().__init__()
        k1 = gaussian_kernel1d(sigma)
        k2 = k1[:, None] @ k1[None, :]
        self.register_buffer("kernel", k2[None, None])

    def _blur(self, x):
        return F.conv2d(x, self.kernel, padding=self.kernel.shape[-1] // 2)

    def forward(self, printed, target):
        return F.mse_loss(self._blur(printed), self._blur(target))


def isolated_dot_penalty(ink):
    """惩罚孤立墨点和孤立白洞：单像素孤点喷墨不可靠，也易堵头。"""
    k = torch.ones(1, 1, 3, 3, device=ink.device)
    neighbors = F.conv2d(ink, k, padding=1) - ink
    iso_ink = ink * torch.clamp(1.0 - neighbors, min=0.0)
    paper = 1.0 - ink
    paper_neighbors = F.conv2d(paper, k, padding=1) - paper
    iso_paper = paper * torch.clamp(1.0 - paper_neighbors, min=0.0)
    return iso_ink.mean() + iso_paper.mean()
