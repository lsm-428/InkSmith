"""数据集：文件夹随机灰度裁块；文件夹为空时退化为合成图冒烟。"""
import glob
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


class GrayCrops(Dataset):
    """每次取一张图随机裁 size×size 灰度块，返回 1×H×W，1=白。

    length 是逻辑长度（每 epoch 样本数），与图片张数无关。
    """

    def __init__(self, root="data/images", size=256, length=2000):
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
        img = Image.open(path).convert("L")
        w, h = img.size
        s = self.size
        if min(w, h) < s:
            scale = s / min(w, h)
            img = img.resize((max(s, round(w * scale)), max(s, round(h * scale))))
            w, h = img.size
        i = random.randint(0, h - s)
        j = random.randint(0, w - s)
        arr = np.asarray(img.crop((j, i, j + s, i + s)), dtype=np.float32) / 255.0
        return torch.from_numpy(arr)[None]

    def _synth(self, idx):
        """合成平滑灰度图（随机高斯斑块叠加），仅用于流程冒烟。"""
        g = torch.Generator().manual_seed(idx)
        s = self.size
        x = torch.linspace(0, 1, s)
        xx, yy = torch.meshgrid(x, x, indexing="ij")
        img = torch.rand((), generator=g) * torch.ones(s, s)
        for _ in range(int(torch.randint(3, 8, (), generator=g))):
            cx, cy = torch.rand(2, generator=g)
            sigma = 0.05 + 0.2 * torch.rand((), generator=g)
            amp = torch.rand((), generator=g)
            img = img + amp * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        return img[None]
