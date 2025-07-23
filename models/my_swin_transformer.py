# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")

class FSM(nn.Module):
    """
    特征选择模块
    """
    def __init__(self, num_heads,window_size,H,W):
        super().__init__()
        self.H = H
        self.W = W
        self.num_heads = num_heads
        self.window_size = window_size

    def forward(self, x, attn_list):
        """
        Args:
            x: (B, L , C)
            attn_list : [(B*num_window, num_heads, L, L),...]  每个WindowAttention的attn
        Returns:
            (B,C)
        """
        B_x, L, C = x.shape  # 从输入特征x获取正确的批量大小B_x
        idx_list = []
        for attn in attn_list:
            # attn的原始形状：(B*num_windows, num_heads, L_window, L_window)
            # 其中L_window = window_size * window_size = 7*7=49
            B_num_windows, num_heads, L_window, _ = attn.shape

            # 计算窗口数量（总窗口数 = B*num_windows // B_x）
            num_windows = B_num_windows // B_x

            # 计算特征图的窗口划分数量（H方向和W方向）
            num_windows_h = self.H // self.window_size
            num_windows_w = self.W // self.window_size
            assert num_windows == num_windows_h * num_windows_w, "窗口数量不匹配"

            # 重塑attn为：(B, 窗口行数, 窗口列数, 头数, 窗口内序列长, 窗口内序列长)
            attn_reshaped = attn.view(
                B_x, num_windows_h, num_windows_w,
                num_heads, L_window, L_window
            )

            # 将窗口维度与序列长度维度合并，恢复为特征图空间尺寸 (H, W)
            # H = 窗口行数 * window_size，W = 窗口列数 * window_size
            attn_spatial = attn_reshaped.permute(
                0, 3, 1, 4, 2, 5  # 维度顺序：(B, 头数, 窗口行数, 窗口内高, 窗口列数, 窗口内宽)
            ).contiguous()

            # 合并窗口维度和窗口内维度，得到 (B, num_heads, H, W)
            attn_spatial = attn_spatial.view(
                B_x, num_heads,
                num_windows_h * L,  # H = 窗口行数 * L
                num_windows_w * L  # W = 窗口列数 * L
            )

            # 后续注意力聚合逻辑（以attn_spatial为基础）
            attn_avg = attn_spatial.mean(dim=2)  # [B, num_heads, W] （或根据需求调整聚合维度）
            # 注意：需定义self.K_head（例如在__init__中添加self.K_head = K_head参数）
            _, idx = torch.topk(attn_avg, k=1, dim=2)  # [B, num_heads, K_head]
            idx_list.append(idx.squeeze(-1))  # [B, num_heads]

        # 特征选择逻辑
        idx_all = torch.cat(idx_list, dim=1)  # [B, depths*num_heads]
        batch_idx = torch.arange(B_x, device=x.device).unsqueeze(1).expand_as(idx_all)
        selected = x[batch_idx, idx_all]  # [B, depths*num_heads, C]
        selected = selected.mean(dim=1)  # [B, C]
        return selected# ------------------------------------------------------------------
# EACA: External-Dependency & Cross-Spatial Attention Module
# Reference: 《结合Swin及多尺度特征融合的细粒度图像分类》
# ------------------------------------------------------------------
class ExternalDependencyAttention(nn.Module):
    """
    外部依赖注意力（EA）
    输入: [B, L, C]   输出: [B, L, C]
    """
    def __init__(self, dim, r=8):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim // r, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(dim // r, dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, L, C]  ->  [B, C, L]
        x_t = x.transpose(1, 2)
        attn = torch.sigmoid(self.conv1(x_t))
        out = self.conv2(attn)          # [B, C, L]
        out = out.transpose(1, 2)       # [B, L, C]
        return self.norm(x + out)       # 残差+归一化


class CrossSpatialAttention(nn.Module):
    """
    跨空间注意力（CA）
    输入: [B, L, C]  输出: [B, L, C]
    """
    def __init__(self, dim, input_resolution):
        super().__init__()
        H, W = input_resolution
        self.H, self.W = H, W
        self.conv_h = nn.Conv1d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.conv_w = nn.Conv1d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, L, C = x.shape
        assert L == self.H * self.W, "Shape mismatch in CrossSpatialAttention"
        x_hw = x.transpose(1, 2).view(B, C, self.H, self.W)      # [B,C,H,W]

        # 水平 & 垂直方向池化
        x_h = x_hw.mean(dim=3)        # [B,C,H]
        x_w = x_hw.mean(dim=2)        # [B,C,W]

        # 卷积 + 激活
        attn_h = torch.sigmoid(self.conv_h(x_h))      # [B,C,H]
        attn_w = torch.sigmoid(self.conv_w(x_w))      # [B,C,W]

        # 外积得到二维注意力
        attn = attn_h.unsqueeze(3) * attn_w.unsqueeze(2)  # [B,C,H,W]
        out = x_hw * attn                                 # 逐元素加权
        out = out.view(B, C, L).transpose(1, 2)           # [B,L,C]
        return self.norm(x + out)                         # 残差+归一化


class EACA(nn.Module):
    """
    融合外部依赖与跨空间注意力模块
    输入: [B, L, C]  输出: [B, L, C]
    """
    def __init__(self, dim, input_resolution, r=4):
        super().__init__()
        self.ea = ExternalDependencyAttention(dim, r)
        self.ca = CrossSpatialAttention(dim, input_resolution)

    def forward(self, x):

        x = self.ea(x) + self.ca(x)   # 并行融合
        return x


# -------------------------------------------------------------
# FFM: 多尺度特征融合模块
# 参考《结合Swin及多尺度特征融合的细粒度图像分类》
# -------------------------------------------------------------
class FFM(nn.Module):
    """
    处理 EACA 输出序列 [B, L, C] 的多阶段互补融合
    输入: list[tensor]  [B, L, C]，长度 P
    输出: list[tensor]  [B, L, C]，长度 P，同输入形状
    """

    def __init__(self):
        super().__init__()

    def pair_fuse(self, x1, x2):
        """
        x1: [B, L1, C]
        x2: [B, L2, C]
        返回: x1_to_2, x2_to_1  同输入形状
        """
        B, L1, C = x1.shape
        _, L2, _ = x2.shape

        # [B, C, L] 方便矩阵乘法
        f1 = x1.transpose(1, 2)   # [B, C, L1]
        f2 = x2.transpose(1, 2)   # [B, C, L2]

        # 相似度矩阵 M = f1^T @ f2  → [B, L1, L2]
        M = torch.bmm(f1.transpose(1, 2), f2) / (C ** 0.5)

        # 互补权重
        A12 = torch.softmax(-M, dim=-1)      # [B, L1, L2]
        A21 = torch.softmax(-M.transpose(1, 2), dim=-1)  # [B, L2, L1]

        # 加权求和
        x1_to_2 = torch.bmm(f2, A21)  # [B, C, L1]
        x2_to_1 = torch.bmm(f1, A12)  # [B, C, L2]

        # 转回 [B, L, C]
        x1_to_2 = x1_to_2.transpose(1, 2)  # [B, L1, C]
        x2_to_1 = x2_to_1.transpose(1, 2)  # [B, L2, C]

        return x1_to_2, x2_to_1

    def forward(self, multi_feats):
        """
        multi_feats: list[tensor] [B, L, C]，长度 P
        返回: 融合后的 list[[tensor]....] [P ,B, L, C]
        """
        P = len(multi_feats)
        if P < 2:
            return multi_feats

        out = [torch.zeros_like(f) for f in multi_feats]   # [B, L, C]

        for i in range(P):
            for j in range(P):
                if i == j:
                    continue
                x_i_to_j, x_j_to_i = self.pair_fuse(multi_feats[i], multi_feats[j])
                out[i] = out[i] + x_i_to_j + multi_feats[i]     # 累加互补信息 残差加回原始特征
                out[j] = out[j] + x_j_to_i + multi_feats[j]

        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,FSM = None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        self.FSM = FSM
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if self.FSM is not None:
            return x,attn
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False,FSM = None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.FSM = FSM

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,FSM=self.FSM)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        if self.FSM is not None:
            attn_windows,attn = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.FSM is not None:
            return x,attn
        else:
            return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False,EACA = None,FSM = None,eaca_dim = None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.FSM = FSM
        self.eaca_dim = eaca_dim if eaca_dim is not None else dim
        self.proj_eaca = nn.Linear(self.dim, self.eaca_dim, bias=False)
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process,FSM=FSM)
            for i in range(depth)])

        if EACA is not None:
            self.EACA = EACA(dim = dim,input_resolution=input_resolution)
        else:
            self.EACA = None
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        if self.FSM is not None:
            attn_list = []
            for blk in self.blocks:
                if self.use_checkpoint:
                    x,attn = checkpoint.checkpoint(blk, x)
                else:
                    x,attn = blk(x)
                attn_list.append(attn)
            x_fsm = self.FSM(x,attn_list)
        else:
            for blk in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)


        if self.EACA is not None:
            x_eaca = self.proj_eaca(x)
            x_eaca = self.EACA(x_eaca)
        if self.downsample is not None:
            x = self.downsample(x)

        if self.FSM is not None:
            return x,x_eaca,x_fsm
        elif self.EACA is not None:
            return x,x_eaca
        else:
            return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class SwinFC(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.ffm = FFM()

        # 新增：预定义通道对齐卷积层（根据网络结构确定需要处理的阶段数）
        self.channel_align_convs = nn.ModuleList()

        # 计算各阶段EACA输出的通道数（根据Swin的通道增长规律：embed_dim * 2^i）
        eaca_dims = [int(embed_dim * (2 ** i)) for i in range(1,len(depths))]
        # 目标通道数：选择最大通道数或指定值（如最后一个阶段的通道数）
        self.target_eaca_dim = max(eaca_dims) if eaca_dims else embed_dim

        # 为每个阶段创建通道对齐层
        for dim in eaca_dims:
            if dim != self.target_eaca_dim:
                self.channel_align_convs.append(
                    nn.Conv1d(
                        in_channels=dim,
                        out_channels=self.target_eaca_dim,
                        kernel_size=1,
                        bias=False
                    )
                )
            else:
                self.channel_align_convs.append(nn.Identity())  # 通道数一致时直接通过
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 计算当前阶段的输入分辨率
            current_resolution = (
                patches_resolution[0] // (2 ** i_layer),
                patches_resolution[1] // (2 ** i_layer)
            )
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=current_resolution,
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process,
                               EACA=EACA if (i_layer !=0) else None,
                               FSM=FSM(H=current_resolution[0], W=current_resolution[1],num_heads=num_heads[i_layer],window_size=window_size) if (i_layer == self.num_layers-1) else None,)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        eaca_list = []
        for layer in self.layers[:-1]:
            if layer.EACA is not None:
                x,x_eaca = layer(x)
                eaca_list.append(x_eaca)
            else:
                x = layer(x)
        #最后stage单独接收
        x, x_eaca,x_fsm = self.layers[-1](x)
        eaca_list.append(x_eaca)
        aligned_eaca = []

        for i, feat in enumerate(eaca_list):
            # [B, L, C] -> [B, C, L] 进行1D卷积
            feat_aligned = self.channel_align_convs[i](feat.transpose(1, 2)).transpose(1, 2)
            aligned_eaca.append(feat_aligned)

        xx_list = self.ffm(aligned_eaca)

        x_list = []
        for xx in xx_list:
            xx = self.norm(xx)  # B L C
            xx = self.avgpool(xx.transpose(1, 2))  # B C 1
            xx = torch.flatten(xx, 1)
            x_list.append(xx)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x_list.append(x)
        return x_list,x_fsm

    def forward(self, x):
        x_list, x_fsm = self.forward_features(x)
        logits_list = [self.head(xx) for xx in x_list]
        return logits_list, x_fsm

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

# class SwinTransformer(nn.Module):
#     r""" Swin Transformer
#         A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
#           https://arxiv.org/pdf/2103.14030
#
#     Args:
#         img_size (int | tuple(int)): Input image size. Default 224
#         patch_size (int | tuple(int)): Patch size. Default: 4
#         in_chans (int): Number of input image channels. Default: 3
#         num_classes (int): Number of classes for classification head. Default: 1000
#         embed_dim (int): Patch embedding dimension. Default: 96
#         depths (tuple(int)): Depth of each Swin Transformer layer.
#         num_heads (tuple(int)): Number of attention heads in different layers.
#         window_size (int): Window size. Default: 7
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
#         qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
#         drop_rate (float): Dropout rate. Default: 0
#         attn_drop_rate (float): Attention dropout rate. Default: 0
#         drop_path_rate (float): Stochastic depth rate. Default: 0.1
#         norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
#         ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
#         patch_norm (bool): If True, add normalization after patch embedding. Default: True
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
#         fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
#     """
#
#     def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
#                  embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
#                  window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                  norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
#                  use_checkpoint=False, fused_window_process=False, **kwargs):
#         super().__init__()
#
#         self.num_classes = num_classes
#         self.num_layers = len(depths)
#         self.embed_dim = embed_dim
#         self.ape = ape
#         self.patch_norm = patch_norm
#         self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
#         self.mlp_ratio = mlp_ratio
#
#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)
#         num_patches = self.patch_embed.num_patches
#         patches_resolution = self.patch_embed.patches_resolution
#         self.patches_resolution = patches_resolution
#
#         # absolute position embedding
#         if self.ape:
#             self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#             trunc_normal_(self.absolute_pos_embed, std=.02)
#
#         self.pos_drop = nn.Dropout(p=drop_rate)
#
#         # stochastic depth
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
#
#         # build layers
#         self.layers = nn.ModuleList()
#         for i_layer in range(self.num_layers):
#             layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
#                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
#                                                  patches_resolution[1] // (2 ** i_layer)),
#                                depth=depths[i_layer],
#                                num_heads=num_heads[i_layer],
#                                window_size=window_size,
#                                mlp_ratio=self.mlp_ratio,
#                                qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                drop=drop_rate, attn_drop=attn_drop_rate,
#                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
#                                norm_layer=norm_layer,
#                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
#                                use_checkpoint=use_checkpoint,
#                                fused_window_process=fused_window_process,
#                                eaca_dim=self.num_features)
#             self.layers.append(layer)
#
#         self.norm = norm_layer(self.num_features)
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'absolute_pos_embed'}
#
#     @torch.jit.ignore
#     def no_weight_decay_keywords(self):
#         return {'relative_position_bias_table'}
#
#     def forward_features(self, x):
#         x = self.patch_embed(x)
#         if self.ape:
#             x = x + self.absolute_pos_embed
#         x = self.pos_drop(x)
#
#         for layer in self.layers:
#             x = layer(x)
#
#         x = self.norm(x)  # B L C
#         x = self.avgpool(x.transpose(1, 2))  # B C 1
#         x = torch.flatten(x, 1)
#         return x
#
#     def forward(self, x):
#         x = self.forward_features(x)
#         x = self.head(x)
#         return x
#
#     def flops(self):
#         flops = 0
#         flops += self.patch_embed.flops()
#         for i, layer in enumerate(self.layers):
#             flops += layer.flops()
#         flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
#         flops += self.num_features * self.num_classes
#         return flops
