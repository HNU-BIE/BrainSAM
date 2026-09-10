# -*- coding: utf-8 -*-
"""
Lightweight Prompt Embedding Generator (LPEG) for BrianSAM
从 FPN neck 输出的图像特征自动生成 prompt 点坐标，模拟用户点击。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LPEG(nn.Module):
    """
    Lightweight Prompt Embedding Generator

    从 FPN neck 输出的图像特征生成前景热力图，
    取 top-k 位置作为 SAM2 的 point prompt 坐标。

    Args:
        feat_dim:    FPN neck 输出的通道数 (SAM2 默认 256)
        hidden_dim:  中间层维度
        num_points:  生成的 prompt 点数量
        image_size:  输入图像的尺寸 (用于坐标缩放)
    """

    def __init__(
        self,
        feat_dim: int = 256,
        hidden_dim: int = 64,
        num_points: int = 1,
        image_size: int = 512,
    ):
        super().__init__()
        self.num_points = num_points
        self.image_size = image_size

        # 热力图生成：从图像特征预测前景概率
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.heatmap_head:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image_embedding: torch.Tensor):
        """
        Args:
            image_embedding: [B, C, H, W] FPN neck 输出的特征图

        Returns:
            point_inputs: dict，SAM2 格式的 point prompt
                - "point_coords": [B, num_points, 2] 像素坐标 (x, y)
                - "point_labels": [B, num_points]     标签 (1=前景)
        """
        B, C, H, W = image_embedding.shape

        heatmap = self.heatmap_head(image_embedding)  # [B, 1, H, W]
        heatmap = torch.sigmoid(heatmap)

        flat = heatmap.view(B, -1)  # [B, H*W]
        topk_val, topk_idx = flat.topk(self.num_points, dim=-1)  # [B, K]

        # 3) 索引转坐标 (特征图尺度)
        feat_y = (topk_idx // W).float()  # [B, K]
        feat_x = (topk_idx % W).float()   # [B, K]

        # 4) 缩放到原图像素坐标
        scale_x = self.image_size / W
        scale_y = self.image_size / H
        pixel_x = (feat_x + 0.5) * scale_x  # +0.5 取像素中心
        pixel_y = (feat_y + 0.5) * scale_y

        # 5) 组装 SAM2 格式
        point_coords = torch.stack([pixel_x, pixel_y], dim=-1)  # [B, K, 2]
        point_labels = torch.ones(
            B, self.num_points, dtype=torch.int32, device=image_embedding.device
        )

        point_inputs = {
            "point_coords": point_coords,
            "point_labels": point_labels,
        }

        return point_inputs

    def forward_with_heatmap(self, image_embedding: torch.Tensor):
        """
        同 forward，但额外返回热力图用于可视化或监督训练。

        Returns:
            point_inputs: SAM2 格式的 point prompt
            heatmap:      [B, 1, H, W] 前景概率图
        """
        B, C, H, W = image_embedding.shape

        heatmap = self.heatmap_head(image_embedding)
        # heatmap = torch.sigmoid(heatmap)

        flat = torch.sigmoid(heatmap).view(B, -1)
        topk_val, topk_idx = flat.topk(self.num_points, dim=-1)

        feat_y = (topk_idx // W).float()
        feat_x = (topk_idx % W).float()

        scale_x = self.image_size / W
        scale_y = self.image_size / H
        pixel_x = (feat_x + 0.5) * scale_x
        pixel_y = (feat_y + 0.5) * scale_y

        point_coords = torch.stack([pixel_x, pixel_y], dim=-1)
        point_labels = torch.ones(
            B, self.num_points, dtype=torch.int32, device=image_embedding.device
        )

        point_inputs = {
            "point_coords": point_coords,
            "point_labels": point_labels,
        }

        return point_inputs, heatmap


