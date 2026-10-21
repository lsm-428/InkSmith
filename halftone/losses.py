"""损失函数：HVS 低频匹配 + 孤立点惩罚。（条纹惩罚已实现但默认关闭，见下）"""
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
    """惩罚孤立墨点和孤立白洞：单像素孤点喷墨不可靠，也易堵头（1-bit 单通道）。"""
    k = torch.ones(1, 1, 3, 3, device=ink.device)
    neighbors = F.conv2d(ink, k, padding=1) - ink
    iso_ink = ink * torch.clamp(1.0 - neighbors, min=0.0)
    paper = 1.0 - ink
    paper_neighbors = F.conv2d(paper, k, padding=1) - paper
    iso_paper = paper * torch.clamp(1.0 - paper_neighbors, min=0.0)
    return iso_ink.mean() + iso_paper.mean()


def striping_penalty(ink, win=16):
    """⚠ 未启用，保留仅供实验。默认 w_stripe=0，不参与训练。

    意图：抑制"水平条纹代替像素级抖动"的退化解（PITFALLS #12）。
    实现为 win×win 局部窗口内行/列墨量的方差。

    **为什么先别用**：试过 6 个变体，没有一个能把"条纹"与"正常抖动/真实低频
    结构"分开——条纹的本质是"相邻像素同值"，而任何平滑的真实内容二值化后
    也是相邻同值。实测（横条纹 / 像素抖动 / 真实渐变三者的惩罚值）：

        行列墨量方差(全局)   0.21 / 0.011 / 0.086   ← 误伤真实低频 8 倍
        局部窗口方差         0.125 / 0.021 / 0.050   ← 只差 2.5 倍，不够
        方向平滑残差         0.019 / 0.789 / 0.0004  ← 方向反了
        邻域自相关          +1.96 / -0.007 / +1.98   ← 梯度二值化后与条纹同值
        行内占比极端度       0.054 / 0.235 / 0.009   ← 顺序完全反了
        模糊前后保真差       0.011 / 0.077 / 0.004   ← 条纹反而更优

    要解决条纹应从别处入手（高频噪声注入 / 上采样方式），不要靠这个正则。
    启用方式：train.py --w-stripe 0.05
    """
    row = F.avg_pool2d(ink, (1, win), stride=1)
    col = F.avg_pool2d(ink, (win, 1), stride=1)
    return (row.var(dim=3, unbiased=False).mean()
            + col.var(dim=2, unbiased=False).mean())
