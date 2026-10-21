"""数据集：文件夹随机灰度/彩色裁块；文件夹为空时退化为合成图冒烟。"""
import glob
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


class RgbCrops(Dataset):
    """每次取一张图随机裁 size×size sRGB 块，返回 3×H×W（[0,1]）。

    灰度图会被转成 3 通道（值复制），1-bit 训练路径取第 0 通道。
    """

    def __init__(self, root="data/images_gray", size=256, length=2000):
        self.paths = sorted(
            p for p in glob.glob(f"{root}/**/*", recursive=True)
            if p.lower().endswith(_EXTS)
        )
        self.size = size
        self.length = length
        self.synthetic = len(self.paths) == 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.synthetic:
            img = self._synth(idx)
        else:
            img = self._load(idx)
        if random.random() < 0.5:
            img = torch.flip(img, dims=[-1])
        if random.random() < 0.5:
            img = torch.flip(img, dims=[-2])
        return img

    def _load(self, idx):
        path = self.paths[idx % len(self.paths)]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        s = self.size
        if min(w, h) < s:
            scale = s / min(w, h)
            img = img.resize((max(s, round(w * scale)), max(s, round(h * scale))))
            w, h = img.size
        i = random.randint(0, h - s)
        j = random.randint(0, w - s)
        arr = np.asarray(img.crop((j, i, j + s, i + s)), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _synth(self, idx):
        """合成平滑彩色图（随机高斯斑块叠加），仅用于流程冒烟。"""
        g = torch.Generator().manual_seed(idx)
        s = self.size
        x = torch.linspace(0, 1, s)
        xx, yy = torch.meshgrid(x, x, indexing="ij")
        img = torch.rand((3, s, s), generator=g)
        for _ in range(int(torch.randint(3, 8, (), generator=g))):
            cx, cy = torch.rand(2, generator=g)
            sigma = 0.05 + 0.2 * torch.rand((), generator=g)
            amp = torch.rand((), generator=g)
            hue = torch.rand(3, 1, 1, generator=g)
            blob = amp * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
            img = img + hue * blob
        img = (img - img.amin(dim=(1, 2), keepdim=True)) / (
            img.amax(dim=(1, 2), keepdim=True) - img.amin(dim=(1, 2), keepdim=True) + 1e-6)
        return img


# 旧名兼容
GrayCrops = RgbCrops
