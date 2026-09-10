# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr
import math
from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from sam2.modeling.sam2_utils import DropPath, MLP

def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x

#**************************

class MoEAdapter(nn.Module):
    """down -> act -> up"""
    def __init__(self,dim,bottleneck_dim,dropout=0.0):
        super().__init__()
        self.down = nn.Linear(dim,bottleneck_dim,bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim,dim,bias=False)
        self.drop = nn.Dropout(dropout)

        nn.init.zeros_(self.up.weight)
    def forward(self,x):
        return self.up(self.drop(self.act(self.down(x))))

class MoEAdaptMLPBlock(nn.Module):
    """
        包装原始 mlp： y = mlp(x) + scale * MoEAdapter(x, prior)
        """
    def __init__(
            self,
            mlp: MLP,
            dim: int,
            expert_num: int,
            embedding_dim: int,
            bottleneck_dim: int,
            adapter_dropout: float = 0.0,
            adapter_scalar: float | None = None,
            use_x_pool_in_gate: bool = False,
    ) -> None:
        super().__init__()
        self.orig_mlp = mlp
        self.dim = dim
        self.expert_num = expert_num
        self.embedding_dim = embedding_dim
        self.use_x_pool_in_gate = use_x_pool_in_gate

        # experts
        self.experts = nn.ModuleList([
            MoEAdapter(dim, bottleneck_dim, dropout=adapter_dropout)
            for _ in range(expert_num)
        ])

        # gate：5 路 prior (modal, organ1, organ2, organ3, task)
        gate_in = 5 * embedding_dim + (dim if use_x_pool_in_gate else 0)
        self.gate = nn.Linear(gate_in, expert_num, bias=True)

        if adapter_scalar is None:
            self.adapter_scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("adapter_scale", torch.tensor(float(adapter_scalar)), persistent=False)

        self.reset_parameters()
    def _pool_x(self, x):
        return x.mean(dim=(1, 2))  # [B,C]

    def reset_parameters(self) -> None:
        for e in self.experts:
            nn.init.kaiming_uniform_(e.down.weight, a=math.sqrt(5))
            nn.init.zeros_(e.up.weight)

    def forward(self, x: torch.Tensor, prior_embeds: dict) -> tuple:
        """
                x: [B, H, W, C] 或 [B, N, C]
                prior_embeds: {
                    "modal": [B, embedding_dim],
                    "organ": list of 4 or 5 tensors, 每个 [B, embedding_dim]
                }
                返回: (out, gate_weights)
        """

        y = self.orig_mlp(x)
        modal = prior_embeds["modal"]
        organ = prior_embeds["organ"]
        if len(organ) == 5:
            organ0, organ1, organ2, organ3, task = organ
        elif len(organ) == 4:
            organ0, organ1, organ2, task = organ
            organ3 = torch.zeros_like(task)
        else:
            raise ValueError(
                f"Expected 4 or 5 organ priors, but got {len(organ)}"
            )

        gate_in = torch.cat([modal, organ1, organ2, organ3, task], dim=-1)

        if self.use_x_pool_in_gate:
            x_pool = self._pool_x(x)
            gate_in = torch.cat([gate_in, x_pool], dim=-1)

        w = F.softmax(self.gate(gate_in), dim=-1)  # [B*, expert_num]

        expert_outs = []
        for e in self.experts:
            expert_outs.append(e(x))
        stack = torch.stack(expert_outs, dim=1)  # [B*, E, H, W, C]
        moe = torch.einsum("behwc,be->bhwc", stack, w)
        out = y + self.adapter_scale * moe
        return out, w
#**************************

class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
        use_MoE = False,
        expert_num: int = 4,
        embedding_dim: int = 16,
        adapter_bn: int = 16,
        adapter_scalar=None,
        use_x_pool_in_gate: bool = False,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.use_MoE = use_MoE

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)

        base_mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if use_MoE:
            self.mlp = MoEAdaptMLPBlock(
                mlp=base_mlp,
                dim=dim_out,  # token dim C
                expert_num=expert_num,
                embedding_dim=embedding_dim,  
                bottleneck_dim=adapter_bn,
                adapter_dropout=0.1,
                adapter_scalar=adapter_scalar,
                use_x_pool_in_gate=use_x_pool_in_gate,  
            )
        else:
            self.mlp = base_mlp

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor,
                prior_embeds: torch.Tensor=None) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size) #划分了窗口，可以通过x[0]查看划分后的第一个窗口

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)

        if self.use_MoE:
            out, gate = self.mlp(self.norm2(x), prior_embeds)  # 注意：传 norm2(x)
            x = x + self.drop_path(out)
            return x, gate
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        weights_path=None,
        return_interm_layers=True,  # return feats from every stage
        use_MoE = False,
        pos : list = None,
        expert_num: int = 4,
        embedding_dim: int = 16,
        adapter_bn: int = 16,
        adapter_scalar: float = 0.1,
        use_x_pool_in_gate: bool = False,
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = nn.ModuleList()

        if pos is None and use_MoE:
            pos = list(range(depth))
        elif pos is None:
            pos = []
        
        self.pos = pos
        self.use_MoE = use_MoE

        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
            
            use_moe_in_this_block = use_MoE and (i in pos)
            
            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
                use_MoE=use_moe_in_this_block,
                expert_num=expert_num,
                embedding_dim=embedding_dim,
                adapter_bn=adapter_bn,
                adapter_scalar=adapter_scalar,
                use_x_pool_in_gate=use_x_pool_in_gate,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

        if weights_path is not None:
            with g_pathmgr.open(weights_path, "rb") as f:
                chkpt = torch.load(f, map_location="cpu")
            logging.info("loading Hiera", self.load_state_dict(chkpt, strict=False))

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor,
                modal: torch.Tensor=None, organ: torch.Tensor=None) -> List[torch.Tensor]:
        prior_embeds = {"modal": modal, "organ": organ}

        x = self.patch_embed(x)
        # x: (B, H, W, C)
        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        expert_activation = []

        for i, blk in enumerate(self.blocks):
            if blk.use_MoE:
                x, gate = blk(x, prior_embeds)
                if gate is not None:
                    expert_activation.append(gate)
            else:
                x = blk(x, prior_embeds)  # 非 MoE 层传 prior_embeds会被忽略
            
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs, expert_activation

    def get_layer_id(self, layer_name):
        # https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        num_layers = self.get_num_layers()

        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("pos_embed") != -1:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1

    def get_num_layers(self) -> int:
        return len(self.blocks)
