"""FS 式逐列串行误差扩散：工业 RIP 的真正机制（PITFALLS #32 实测成功）。

与 `ed.py::ed_iterate`（迭代式，失败）的区别：那里把"已打印邻域"用于
算偏置，反馈是死区的；这里把**量化误差本身**按权重实际传递给后续像素，
邻域被迫选不同档位补偿 -> 亚档位覆盖率是机制的必然结果。

按列串行（W 步 python 循环），列内 H 行一次性决策。因为整列同时决策，
误差只能扩散到**后续列**——这是本实现与教科书 FS（逐行、左→右）的
关键差异，也是权重必须重新分配的原因：

    右   (w+1, r)    7/16
    右下 (w+1, r+1)  5/16
    右上 (w+1, r-1)  3/16
    右二 (w+2, r)    1/16      四项和 = 16/16 = 1.0

**为什么权重和必须严格为 1**（PITFALLS #42）：早先版本把 5/16 那项写到
当前列（`buf[:,:,1:,w]`），而当前列所有行已在同一轮按同一份 buf 快照
决策完毕——那是一次**死写入**。有效权重和只剩 11/16 = 0.6875，31% 的
量化误差被丢弃。后果是覆盖率两端塌陷：稳态 buf = P/(1-0.6875) = 3.2P，
需 buf>0.5 才落墨 ⇒ P<0.156 输出恒 0、P>0.844 输出恒 1。实测
tgt 0.067→0.000、0.867→1.000（正确应为 0.067/0.867）。

覆盖率精度是 ED 存在的前提，误差扩散权重和必须等于 1。
"""
import sys, torch, torch.nn.functional as F; sys.path.insert(0, '.')
torch.set_num_threads(20)


def ed_fs(logits, target_cov, levels, k_r=7/16, k_dr=5/16, k_ur=3/16,
          k_rr=1/16, bias_gain=4.0):
    """逐列串行（列内整列并行，W 步；列间张量）。FS 权重重排到后续列。

    logits (B,C,H,W,L), target_cov (B,C,H,W) -> 硬 one-hot (B,C,H,W,L) + STE
    """
    B, C, H, W, L = logits.shape
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
        out[:, :, :, w] = hard + (soft_all[:, :, :, w] - soft_all[:, :, :, w].detach())
        # 扩散：全部指向**后续列**（写当前列是死写入，见模块 docstring）
        if w + 1 < W:
            buf[:, :, :, w + 1] += err * k_r                        # 右
            buf[:, :, 1:, w + 1] += err[:, :, :-1] * k_dr           # 右下
            buf[:, :, :-1, w + 1] += err[:, :, 1:] * k_ur           # 右上
        if w + 2 < W:
            buf[:, :, :, w + 2] += err * k_rr                       # 右二
    return out
