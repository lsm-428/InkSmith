"""U-Net：灰度图 -> 每像素墨点 logits。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """输入 1 通道灰度（1=白），输出 1 通道 logits（>0 表示该像素落墨）。

    要求输入边长能被 8 整除（3 次 2 倍下采样）。

    up: 解码上采样方式。`nearest` 产生块状结构，是"水平条纹退化解"的温床
    （见 PITFALLS #12）；`bilinear` 更平滑，倾向像素级纹理。
    该值随 ckpt 存进 args，推理端自动重建——改它必须重训，旧权重不兼容。
    """

    def __init__(self, base=32, up="nearest"):
        super().__init__()
        self.up = up
        self.d1 = DoubleConv(1, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.b = DoubleConv(base * 4, base * 4)
        self.u3 = DoubleConv(base * 8, base * 2)
        self.u2 = DoubleConv(base * 4, base)
        self.u1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def _up(self, x):
        return F.interpolate(x, scale_factor=2, mode=self.up)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(F.avg_pool2d(c1, 2))
        c3 = self.d3(F.avg_pool2d(c2, 2))
        b = self.b(F.avg_pool2d(c3, 2))
        x = self.u3(torch.cat([self._up(b), c3], 1))
        x = self.u2(torch.cat([self._up(x), c2], 1))
        x = self.u1(torch.cat([self._up(x), c1], 1))
        return self.head(x)


def soft_binarize(logits, temperature):
    """软二值化：sigmoid(logits/T)。T 大时近乎线性（好学覆盖率），
    T 小时逼近硬阈值。训练用退火 T，推理直接 logits>0 硬阈值。"""
    return torch.sigmoid(logits / temperature)


class _ClipSTE(torch.autograd.Function):
    """截断直通：前向硬阈值，反传在 |x|<4 内透传梯度。

    只在退火完成后用于微调——此时覆盖率已学好，只需消除软值作弊，
    让训练指标与硬阈值推理一致。从零训练直接用它会塌缩（见 PITFALLS #1）。
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad * (x.abs() < 4).float()


def hard_binarize(logits):
    return _ClipSTE.apply(logits)
