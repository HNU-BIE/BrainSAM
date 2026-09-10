import logging
import numpy as np

import os
from sam2.modeling.sam2_utils import MLP
from sam2.modeling.sam2_base import SAM2Base
from training.utils.data_utils import BatchedVideoDatapoint

join = os.path.join

import math
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from data.data_info import *

def get_index(map_idx):
    idx = map_idx
    for i in range(list(map_idx.items())[-1][0]):
        if i not in idx:
            idx[i] = 0
    idx = dict(sorted(idx.items(),key=lambda item: item[0]))
    return torch.LongTensor(list(idx.values()))



import logging
import numpy as np
import torch
import torch.nn as nn
from sam2.modeling.sam2_base import SAM2Base
from data.data_info import *


def get_index(map_idx):
    idx = map_idx
    for i in range(list(map_idx.items())[-1][0]):
        if i not in idx:
            idx[i] = 0
    idx = dict(sorted(idx.items(), key=lambda item: item[0]))
    return torch.LongTensor(list(idx.values()))


class BrainSAMBase(SAM2Base):
    def __init__(
            self,
            image_encoder,
            embedding_dim: int = 64,
            pos: list = None,
            memory_attention=None,
            memory_encoder=None,
            **kwargs,
    ):
        super().__init__(image_encoder, memory_attention, memory_encoder, **kwargs)

        if pos:
            self.pos = pos
        else:
            self.pos = list(range(len(self.image_encoder.trunk.blocks)))

        modal_index = get_index(modal_map_idx)
        brain_index_1 = get_index(brain_level_1_map_idx)
        brain_index_2 = get_index(brain_level_2_map_idx)

        self.image_encoder.register_buffer('modal_index', modal_index, persistent=True)
        self.image_encoder.register_buffer('brain_index_1', brain_index_1, persistent=True)
        self.image_encoder.register_buffer('brain_index_2', brain_index_2, persistent=True)

        modal_embed = nn.Embedding(len(modal_index), embedding_dim)
        organ_embed_0 = nn.Embedding(1, embedding_dim)
        organ_embed_1 = nn.Embedding(len(brain_level_1_map_idx), embedding_dim)
        organ_embed_2 = nn.Embedding(len(brain_level_2_map_idx), embedding_dim)
        organ_embed_3 = nn.Embedding(len(task_list) + 1, embedding_dim)

        nn.init.normal_(modal_embed.weight, mean=0.0, std=0.02)
        nn.init.zeros_(organ_embed_0.weight)
        nn.init.normal_(organ_embed_1.weight, mean=0.0, std=0.02)
        nn.init.normal_(organ_embed_2.weight, mean=0.0, std=0.02)
        nn.init.normal_(organ_embed_3.weight, mean=0.0, std=0.02)

        self.image_encoder.modal_embed = modal_embed
        self.image_encoder.organ_embed = nn.ModuleList([
            organ_embed_0, organ_embed_1, organ_embed_2, organ_embed_3
        ])

    def forward_image(self, img_batch, modal=None, organ=None):
        backbone_out = self.image_encoder(img_batch, modal, organ)
        if self.use_high_res_features_in_sam:
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1])
        return backbone_out