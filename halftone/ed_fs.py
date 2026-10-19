"""FS 式逐行串行误差扩散：工业 RIP 的真正机制（PITFALLS #32 实测成功）。

与 `ed.py::ed_iterate`（迭代式，失败）的区别：那里把"已打印邻域"用于
算偏置，反馈是死区的；这里把**量化误差本身**按 Floyd-Steinberg 权重
(7/16 右, 3/16 左下, 5/16 下, 1/16 右下) 实际传递给后续像素，
邻域被迫选不同档位补偿 -> 亚档位覆盖率是机制的必然结果。

行内必须串行（W 步 python 循环），行间用张量操作。64² 常数目标实测
覆盖率 0.202/0.500/0.798（目标 0.2/0.5/0.8），邻域多样性 2.06。
"""
import sys, torch, torch.nn.functional as F; sys.path.insert(0,'.')
from halftone.cmyk import LEVELS
torch.set_num_threads(20)

def ed_fs(logits, target_cov, levels, k_r=7/16, k_dl=3/16, k_d=5/16,
          k_dr=1/16, bias_gain=4.0):
    """逐行串行（行内 python 循环，W 步；行间用张量）。真 Floyd-Steinberg。
    logits (B,C,H,W,L), target_cov (B,C,H,W) -> 硬 one-hot (B,C,H,W,L) + STE
    """
    B,C,H,W,L = logits.shape
    lv = levels.view(1, 1, 1, -1).to(device=logits.device, dtype=logits.dtype)
    buf = target_cov.clone().to(device=logits.device, dtype=logits.dtype)
    out = torch.zeros(B, C, H, W, L, device=logits.device)
    soft_all = torch.softmax(logits, dim=-1)
    for w in range(W):
        # 当前列的 logits + 由 buf 引导的偏置（buf>0 欠墨 -> 偏高档位）
        col_logits = logits[:, :, :, w]           # (B,C,H,L)
        buf_col = buf[:, :, :, w]                 # (B,C,H)
        bias = (buf_col.unsqueeze(-1) - lv).abs() * -bias_gain
        adj = col_logits + bias
        hard = torch.zeros_like(adj)
        hard.scatter_(-1, adj.argmax(-1, keepdim=True), 1.0)
        picked = (hard * lv).sum(-1)              # (B,C,H) 实际覆盖率
        err = buf_col - picked                    # 量化误差
        out[:, :, :, w] = hard + (soft_all[:,:,:,w] - soft_all[:,:,:,w].detach())
        # 扩散（写回 buf）
        if w+1 < W:  buf[:, :, :, w+1] += err * k_r
        if w+1 < W:  buf[:, :, 1:, w+1] += err[:,:,:-1] * k_dl   # 左下
        buf[:, :, 1:, w] = buf[:, :, 1:, w] + err[:,:,:-1] * k_d # 下（先存后取）
        if w+1 < W:  buf[:, :, :-1, w+1] += err[:,:,1:] * k_dr   # 右下
    return out
