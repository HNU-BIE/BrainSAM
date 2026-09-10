# -*- coding: utf-8 -*-
"""
BrainSAM pre-training script
"""
import logging
import numpy as np

import os
from sam2.modeling.sam2_utils import MLP
from sam2.modeling.sam2_base import SAM2Base
from training.utils.data_utils import BatchedVideoDatapoint
from sam2.modeling.sam2_utils import (
    get_next_point,
    sample_box_points
)
from sam2.utils.misc import concat_points
join = os.path.join

import math
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from data.data_info import *
from sam2.modeling.LPEG import LPEG

def get_index(map_idx):
    idx = map_idx
    for i in range(list(map_idx.items())[-1][0]):
        if i not in idx:
            idx[i] = 0
    idx = dict(sorted(idx.items(),key=lambda item: item[0]))
    return torch.LongTensor(list(idx.values()))

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
            use_x_pool_in_gate: bool = True,
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


        # gate：5 路 prior (modal, organ0, organ1, organ2, task)
        gate_in = 5 * embedding_dim + (dim if use_x_pool_in_gate else 0)
        self.gate = nn.Linear(gate_in, expert_num, bias=True)

        if adapter_scalar is None:
            self.adapter_scale = 0.1
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
                    "organ": list of 5 tensors, 每个 [B, embedding_dim]
                }
                返回: (out, gate_weights)
        """

        y = self.orig_mlp(x)
        modal = prior_embeds["modal"]
        organ = prior_embeds["organ"]
        organ0, organ1, organ2, task = organ

        gate_in = torch.cat([modal, organ0, organ1, organ2, task], dim=-1)

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

class BrainSAM(SAM2Base):
    def __init__(
            self,
            image_encoder,
            memory_attention=None,
            memory_encoder=None,
            prob_to_use_pt_input_for_train=0.0,
            prob_to_use_pt_input_for_eval=0.0,
            prob_to_use_box_input_for_train=0.0,
            prob_to_use_box_input_for_eval=0.0,
            # if it is greater than 1, we interactive point sampling in the 1st frame and other randomly selected frames
            num_frames_to_correct_for_train=1,  # default: only iteratively sample on first frame
            num_frames_to_correct_for_eval=1,  # default: only iteratively sample on first frame
            rand_frames_to_correct_for_train=False,
            rand_frames_to_correct_for_eval=False,
            # how many frames to use as initial conditioning frames (for both point input and mask input; the first frame is always used as an initial conditioning frame)
            # - if `rand_init_cond_frames` below is True, we randomly sample 1~num_init_cond_frames initial conditioning frames
            # - otherwise we sample a fixed number of num_init_cond_frames initial conditioning frames
            # note: for point input, we sample correction points on all such initial conditioning frames, and we require that `num_frames_to_correct` >= `num_init_cond_frames`;
            # these are initial conditioning frames because as we track the video, more conditioning frames might be added
            # when a frame receives correction clicks under point input if `add_all_frames_to_correct_as_cond=True`
            num_init_cond_frames_for_train=1,  # default: only use the first frame as initial conditioning frame
            num_init_cond_frames_for_eval=1,  # default: only use the first frame as initial conditioning frame
            rand_init_cond_frames_for_train=True,
            # default: random 1~num_init_cond_frames_for_train cond frames (to be constent w/ previous TA data loader)
            rand_init_cond_frames_for_eval=False,
            # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
            # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
            add_all_frames_to_correct_as_cond=False,
            # how many additional correction points to sample (on each frame selected to be corrected)
            # note that the first frame receives an initial input click (in addition to any correction clicks)
            num_correction_pt_per_frame=7,
            # method for point sampling during evaluation
            # "uniform" (sample uniformly from error region) or "center" (use the point with the largest distance to error region boundary)
            # default to "center" to be consistent with evaluation in the SAM paper
            pt_sampling_for_eval="center",
            # During training, we optionally allow sampling the correction points from GT regions
            # instead of the prediction error regions with a small probability. This might allow the
            # model to overfit less to the error regions in training datasets
            prob_to_sample_from_gt_for_train=0.0,
            use_act_ckpt_iterative_pt_sampling=False,
            # whether to forward image features per frame (as it's being tracked) during evaluation, instead of forwarding image features
            # of all frames at once. This avoids backbone OOM errors on very long videos in evaluation, but could be slightly slower.
            forward_backbone_per_frame_for_eval=False,
            freeze_image_encoder=True,
            #*******
            lpeg = None,
            register_buffters_for_prior_idx=True,
            use_boundary_prior=True,
            embedding_dim=64,
            #*******
            **kwargs,

    ):
        super().__init__(image_encoder, memory_attention, memory_encoder, **kwargs)
        self.use_act_ckpt_iterative_pt_sampling = use_act_ckpt_iterative_pt_sampling
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval

        # Point sampler and conditioning frames
        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_box_input_for_train = prob_to_use_box_input_for_train
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_input_for_eval = prob_to_use_box_input_for_eval
        if prob_to_use_pt_input_for_train > 0 or prob_to_use_pt_input_for_eval > 0:
            logging.info(
                f"Training with points (sampled from masks) as inputs with p={prob_to_use_pt_input_for_train}"
            )
            assert num_frames_to_correct_for_train >= num_init_cond_frames_for_train
            assert num_frames_to_correct_for_eval >= num_init_cond_frames_for_eval

        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
        # Initial multi-conditioning frames
        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
        self.register_buffters_for_prior_idx = register_buffters_for_prior_idx
        self.use_boundary_prior = use_boundary_prior
        # A random number generator with a fixed initial seed across GPUs
        self.rng = np.random.default_rng(seed=42)

        if freeze_image_encoder:
            trainable_keywords = [
                'modal_embed', 'organ_embed',
                'mlp.experts', 'mlp.gate', 'mlp.adapter_scale',
            ]
            for name, p in self.image_encoder.named_parameters():
                if any(keyword in name for keyword in trainable_keywords):
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        
        #***************************************************************************************
        if register_buffters_for_prior_idx:
            modal_index = get_index(modal_map_idx)
            brain_index_1 = get_index(brain_level_1_map_idx)
            brain_index_2 = get_index(brain_level_2_map_idx)
            brain_index_3 = get_index(brain_level_3_map_idx)

            self.image_encoder.register_buffer('modal_index',modal_index, persistent=True)
            self.image_encoder.register_buffer('brain_index_1',brain_index_1, persistent=True)
            self.image_encoder.register_buffer('brain_index_2',brain_index_2, persistent=True)
            self.image_encoder.register_buffer('brain_index_3',brain_index_3, persistent=True)

            modal_embed = nn.Embedding(len(modal_index),embedding_dim)
            organ_embed_0 = nn.Embedding(1,embedding_dim)
            organ_embed_1 = nn.Embedding(len(brain_level_1_map_idx),embedding_dim)
            organ_embed_2 = nn.Embedding(len(brain_level_2_map_idx),embedding_dim)
            organ_embed_3 = nn.Embedding(len(brain_level_3_map_idx), embedding_dim)  
            organ_embed_4 = nn.Embedding(len(task_list) + 1, embedding_dim)  # tasks
            

            nn.init.normal_(modal_embed.weight,   mean=0.0, std=0.02)
            nn.init.zeros_(organ_embed_0.weight)   # organ_0 是 size=1 的 global brain embed，保持零没问题
            nn.init.normal_(organ_embed_1.weight,  mean=0.0, std=0.02)
            nn.init.normal_(organ_embed_2.weight,  mean=0.0, std=0.02)
            nn.init.normal_(organ_embed_3.weight,  mean=0.0, std=0.02)
            nn.init.normal_(organ_embed_4.weight,  mean=0.0, std=0.02)

            self.image_encoder.modal_embed = modal_embed
            self.image_encoder.organ_embed = nn.ModuleList([organ_embed_0, organ_embed_1,
                                                            organ_embed_2, organ_embed_3,
                                                            organ_embed_4
                                                            ])

        self.lpeg = lpeg #一个轻量化的Prompt生成模块，用于模拟用户点击
        if self.lpeg is not None:
            lpeg_params = sum(p.numel() for p in self.lpeg.parameters())
            lpeg_trainable = sum(p.numel() for p in self.lpeg.parameters() if p.requires_grad)
            print(f"LPEG params: {lpeg_params}, trainable: {lpeg_trainable}")


        
    # ***************************************************************************************
    #******
    def forward(self, x: BatchedVideoDatapoint,):
        #模态和组织编码
        if self.register_buffters_for_prior_idx:
            flat_prior = x.flat_prior_batch
            modal_ids = flat_prior[:, 0].long()
            modal_index = self.image_encoder.modal_index[modal_ids]
            modal_embed = self.image_encoder.modal_embed(modal_index)

            organ_1_ids = flat_prior[:, 1].long()
            organ_2_ids = flat_prior[:, 2].long()
            organ_3_ids = flat_prior[:, 3].long()
            task_ids = flat_prior[:, 4].long()
            
            organ_index_0 = torch.zeros(flat_prior.shape[0], dtype=torch.long, device=flat_prior.device)
            organ_embed_0 = self.image_encoder.organ_embed[0](organ_index_0)

            organ_index_1 = self.image_encoder.brain_index_1[organ_1_ids]
            organ_embed_1 = self.image_encoder.organ_embed[1](organ_index_1)

            organ_index_2 = self.image_encoder.brain_index_2[organ_2_ids]
            organ_embed_2 = self.image_encoder.organ_embed[2](organ_index_2)

            organ_index_3 = self.image_encoder.brain_index_3[organ_3_ids]
            organ_embed_3 = self.image_encoder.organ_embed[3](organ_index_3)

            organ_embed_4 = self.image_encoder.organ_embed[4](task_ids)

            organ_embed = [organ_embed_0, organ_embed_1, organ_embed_2, organ_embed_3,organ_embed_4]

            
            if self.use_boundary_prior:
                # 构造边界先验表
                pv = x.prior_batch[0]    # [B_video, 4]  每video一行(各帧相同取t=0)
                pv_modal = self.image_encoder.modal_embed(
                    self.image_encoder.modal_index[pv[:, 0].long()])
                pv_o1 = self.image_encoder.organ_embed[1](
                    self.image_encoder.brain_index_1[pv[:, 1].long()])
                pv_o2 = self.image_encoder.organ_embed[2](
                    self.image_encoder.brain_index_2[pv[:, 2].long()])
                pv_o3 = self.image_encoder.organ_embed[3](pv[:, 3].long())
                pv_o4 = self.image_encoder.organ_embed[4](pv[:, 4].long())
                self._boundary_prior_table = torch.cat(
                    [pv_modal, pv_o1, pv_o2, pv_o3, pv_o4], dim=-1)   # [B_video, 256]
                # print(f"Constructed boundary prior table with shape: {self._boundary_prior_table.shape}")  # debug
            else:
                self._boundary_prior_table = None
                print("Boundary prior will not be used. _boundary_prior_table is set to None.")
        else:
            modal_embed = None
            organ_embed = None
            boundary_prior = None

        if self.training or not self.forward_backbone_per_frame_for_eval:
            # precompute image features on all frames before tracking
            backbone_out = self.forward_image(x.flat_img_batch, modal_embed, organ_embed)
            expert_acts = backbone_out.get("expert_activation", [])
            self._monitor = {}
            if expert_acts:
                # gate 分布：所有层的平均 gate 权重
                all_gates = torch.stack(expert_acts, dim=0)  # [num_layers, B, E]
                mean_gate = all_gates.mean(dim=(0, 1))  # [E]
                self._monitor = {
                    f"gate/expert_{i}_weight": mean_gate[i].item()
                    for i in range(mean_gate.shape[0])
                }
                # gate 熵：衡量专家分化程度，熵越低说明路由越集中
                entropy = -(all_gates * (all_gates + 1e-8).log()).sum(dim=-1).mean()
                self._monitor["gate/entropy"] = entropy.item()

                # adapter_scale
                for name, module in self.image_encoder.trunk.named_modules():
                    if isinstance(module, MoEAdaptMLPBlock):
                        self._monitor["adapter/scale"] = module.adapter_scale.item()
                        break  # 只取第一个作为代表

            if self.use_boundary_prior and self.register_buffters_for_prior_idx:
                self._monitor.update({
                    "prior/organ_embed_1_norm": self.image_encoder.organ_embed[1].weight.norm().item(),
                    "prior/organ_embed_2_norm": self.image_encoder.organ_embed[2].weight.norm().item(),
                    "prior/organ_embed_3_norm": self.image_encoder.organ_embed[3].weight.norm().item(),
                    "prior/organ_embed_4_norm": self.image_encoder.organ_embed[4].weight.norm().item(),
                    "prior/modal_embed_norm":   self.image_encoder.modal_embed.weight.norm().item(),
                    "prior/boundary_film_norm": self.sam_mask_decoder.boundary_film.weight.norm().item(),
                    "prior/sim_mouse_human": F.cosine_similarity(
                        self.image_encoder.organ_embed[2].weight[0:1],
                        self.image_encoder.organ_embed[2].weight[3:4],
                    ).item(),
                    "prior/sim_smooth_complicate": F.cosine_similarity(
                        self.image_encoder.organ_embed[1].weight[0:1],
                        self.image_encoder.organ_embed[1].weight[2:3],
                    ).item(),
                    "prior/film_gamma_mean": getattr(self.sam_mask_decoder, '_film_gamma_mean', 0.0),
                    "prior/film_beta_mean":  getattr(self.sam_mask_decoder, '_film_beta_mean', 0.0),
                    "prior/film_delta_bt":   getattr(self.sam_mask_decoder, '_film_delta_bt', 0.0),
                })

        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}

        #提示编码
        backbone_out = self.prepare_prompt_inputs(backbone_out, x)

        previous_stages_out = self.forward_tracking(backbone_out, x)
        expert_acts = backbone_out.get("expert_activation", [])

        lpeg_info = None
        if self.lpeg is not None and self.training:
            lpeg_heatmaps = backbone_out.get("lpeg_heatmaps", {})
            gt_masks = backbone_out.get("gt_masks_per_frame", {})
            if lpeg_heatmaps:
                lpeg_info = (lpeg_heatmaps, gt_masks)
        
        return previous_stages_out,expert_acts,lpeg_info

    def _prepare_backbone_features_per_frame(self, img_batch, img_ids):
        """Compute the image backbone features on the fly for the given img_ids."""
        # Only forward backbone on unique image ids to avoid repetitive computation
        # (if `img_ids` has only one element, it's already unique so we skip this step).
        if img_ids.numel() > 1:
            unique_img_ids, inv_ids = torch.unique(img_ids, return_inverse=True)
        else:
            unique_img_ids, inv_ids = img_ids, None

        # Compute the image features on those unique image ids
        image = img_batch[unique_img_ids]
        backbone_out = self.forward_image(image)
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)
        # Inverse-map image features for `unique_img_ids` to the final image features
        # for the original input `img_ids`.
        if inv_ids is not None:
            image = image[inv_ids]
            vision_feats = [x[:, inv_ids] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids] for x in vision_pos_embeds]

        return image, vision_feats, vision_pos_embeds, feat_sizes

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        """
        Prepare input mask, point or box prompts. Optionally, we allow tracking from
        a custom `start_frame_idx` to the end of the video (for evaluation purposes).
        """
        # Load the ground-truth masks on all frames (so that we can later
        # sample correction points from them)

        gt_masks_per_frame = {
            stage_id: masks.unsqueeze(1)  # [B, 1, H_im, W_im]
            for stage_id, masks in enumerate(input.masks)
        }
        # gt_masks_per_frame = input.masks.unsqueeze(2) # [T,B,1,H_im,W_im] keep everything in tensor form
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        num_frames = input.num_frames
        backbone_out["num_frames"] = num_frames

        # Randomly decide whether to use point inputs or mask inputs
        if self.training:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_train
            prob_to_use_box_input = self.prob_to_use_box_input_for_train
            num_frames_to_correct = self.num_frames_to_correct_for_train
            rand_frames_to_correct = self.rand_frames_to_correct_for_train
            num_init_cond_frames = self.num_init_cond_frames_for_train
            rand_init_cond_frames = self.rand_init_cond_frames_for_train
        else:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_eval
            prob_to_use_box_input = self.prob_to_use_box_input_for_eval
            num_frames_to_correct = self.num_frames_to_correct_for_eval
            rand_frames_to_correct = self.rand_frames_to_correct_for_eval
            num_init_cond_frames = self.num_init_cond_frames_for_eval
            rand_init_cond_frames = self.rand_init_cond_frames_for_eval
        if num_frames == 1:
            # here we handle a special case for mixing video + SAM on image training,
            # where we force using point input for the SAM task on static images
            prob_to_use_pt_input = 1.0
            num_frames_to_correct = 1
            num_init_cond_frames = 1
        assert num_init_cond_frames >= 1
        # (here `self.rng.random()` returns value in range 0.0 <= X < 1.0)
        use_pt_input = self.rng.random() < prob_to_use_pt_input
        if rand_init_cond_frames and num_init_cond_frames > 1:
            # randomly select 1 to `num_init_cond_frames` frames as initial conditioning frames
            num_init_cond_frames = self.rng.integers(
                1, num_init_cond_frames, endpoint=True
            )
        if (
                use_pt_input
                and rand_frames_to_correct
                and num_frames_to_correct > num_init_cond_frames
        ):
            # randomly select `num_init_cond_frames` to `num_frames_to_correct` frames to sample
            # correction clicks (only for the case of point input)
            num_frames_to_correct = self.rng.integers(
                num_init_cond_frames, num_frames_to_correct, endpoint=True
            )
        backbone_out["use_pt_input"] = use_pt_input

        # Sample initial conditioning frames
        if num_init_cond_frames == 1:
            init_cond_frames = [start_frame_idx]  # starting frame
        else:
            # starting frame + randomly selected remaining frames (without replacement)
            init_cond_frames = [start_frame_idx] + self.rng.choice(
                range(start_frame_idx + 1, num_frames),
                num_init_cond_frames - 1,
                replace=False,
            ).tolist()
        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]
        # Prepare mask or point inputs on initial conditioning frames
        backbone_out["mask_inputs_per_frame"] = {}  # {frame_idx: <input_masks>}
        backbone_out["point_inputs_per_frame"] = {}  # {frame_idx: <input_points>}
        backbone_out["lpeg_heatmaps"] = {}
        
        for t in init_cond_frames:
            #*********
            if self.lpeg is not None:
                img_ids = input.flat_obj_to_img_idx[t]
                current_feat = backbone_out["backbone_fpn"][-1][img_ids]
                _, heatmap = self.lpeg.forward_with_heatmap(current_feat)
                backbone_out["lpeg_heatmaps"][t] = heatmap
                #*********

            if not use_pt_input:
                backbone_out["mask_inputs_per_frame"][t] = gt_masks_per_frame[t]
            else:
                # During training # P(box) = prob_to_use_pt_input * prob_to_use_box_input
                use_box_input = self.rng.random() < prob_to_use_box_input
                if use_box_input:
                    points, labels = sample_box_points(
                        gt_masks_per_frame[t],
                    )
                else:
                    # (here we only sample **one initial point** on initial conditioning frames from the
                    # ground-truth mask; we may sample more correction points on the fly)
                    points, labels = get_next_point(
                        gt_masks=gt_masks_per_frame[t],
                        pred_masks=None,
                        method=(
                            "uniform" if self.training else self.pt_sampling_for_eval
                        ),
                    )

                point_inputs = {"point_coords": points, "point_labels": labels}
                backbone_out["point_inputs_per_frame"][t] = point_inputs

        # Sample frames where we will add correction clicks on the fly
        # based on the error between prediction and ground-truth masks
        if not use_pt_input:
            # no correction points will be sampled when using mask inputs
            frames_to_add_correction_pt = []
        elif num_frames_to_correct == num_init_cond_frames:
            frames_to_add_correction_pt = init_cond_frames
        else:
            assert num_frames_to_correct > num_init_cond_frames
            # initial cond frame + randomly selected remaining frames (without replacement)
            extra_num = num_frames_to_correct - num_init_cond_frames
            frames_to_add_correction_pt = (
                    init_cond_frames
                    + self.rng.choice(
                backbone_out["frames_not_in_init_cond"], extra_num, replace=False
            ).tolist()
            )
        backbone_out["frames_to_add_correction_pt"] = frames_to_add_correction_pt

        return backbone_out

    #***
    def forward_image(self, img_batch, modal=None, organ=None):
        backbone_out = self.image_encoder(img_batch, modal, organ)
        if self.use_high_res_features_in_sam:
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
        return backbone_out


    def forward_tracking(
            self, backbone_out, input: BatchedVideoDatapoint, return_dict=False
    ):
        """Forward video tracking on each frame (and sample correction clicks)."""
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            # Prepare the backbone features
            # - vision_feats and vision_pos_embeds are in (HW)BC format
            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            ) = self._prepare_backbone_features(backbone_out)

        # Starting the stage loop
        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]
        # first process all the initial conditioning frames to encode them as memory,
        # and then conditioning on them to track the remaining frames
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]
        output_dict = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        for stage_id in processing_order:
            # Get the image features for the current frames
            # img_ids = input.find_inputs[stage_id].img_ids
            img_ids = input.flat_obj_to_img_idx[stage_id]
            if img_feats_already_computed:
                # Retrieve image features according to img_ids (if they are already computed).
                current_vision_feats = [x[:, img_ids] for x in vision_feats]
                current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
            else:
                # Otherwise, compute the image features on the fly for the given img_ids
                # (this might be used for evaluation on long videos to avoid backbone OOM).
                (
                    _,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                ) = self._prepare_backbone_features_per_frame(
                    input.flat_img_batch, img_ids
                )

            #****
            frame_prior = None
            if getattr(self, "_boundary_prior_table", None) is not None:
                num_frames_total = input.num_frames
                video_idx = img_ids // num_frames_total          # [O] 当前帧每个object的video
                frame_prior = self._boundary_prior_table[video_idx]  # [O, 256] 精确对齐!
                # print(f"Frame {stage_id} boundary prior shape: {frame_prior.shape}")  # debug
            #********

            # Get output masks based on this frame's prompts and previous memory
            current_out = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id, None),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id, None),
                gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id, None),
                frames_to_add_correction_pt=frames_to_add_correction_pt,
                output_dict=output_dict,
                num_frames=num_frames,
                boundary_prior=frame_prior,    
            )
            # Append the output, depending on whether it's a conditioning frame
            add_output_as_cond_frame = stage_id in init_cond_frames or (
                    self.add_all_frames_to_correct_as_cond
                    and stage_id in frames_to_add_correction_pt
            )
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out

        if return_dict:
            return output_dict
        # turn `output_dict` into a list for loss function
        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        # Make DDP happy with activation checkpointing by removing unused keys
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]

        return all_frame_outputs

    def track_step(
            self,
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse=False,  # tracking in reverse time order (for demo usage)
            run_mem_encoder=True,  # Whether to run the memory encoder on the predicted masks.
            prev_sam_mask_logits=None,  # The previously predicted SAM mask logits.
            frames_to_add_correction_pt=None,
            gt_masks=None,
            boundary_prior=None,
    ):
        if frames_to_add_correction_pt is None:
            frames_to_add_correction_pt = []
        current_out, sam_outputs, high_res_features, pix_feat = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
            boundary_prior=boundary_prior,
        )

        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            boundary_logits,
            boundary_alpha,
        ) = sam_outputs

        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]
        current_out["multistep_pred_boundary_logits"] = [boundary_logits]
        current_out["multistep_pred_boundary_alpha"] = [boundary_alpha]

        # Optionally, sample correction points iteratively to correct the mask
        if frame_idx in frames_to_add_correction_pt:
            point_inputs, final_sam_outputs = self._iter_correct_pt_sampling(
                is_init_cond_frame,
                point_inputs,
                gt_masks,
                high_res_features,
                pix_feat,
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                object_score_logits,
                current_out,
                boundary_logits=boundary_logits,
                boundary_alpha=boundary_alpha,
                boundary_prior=boundary_prior,
            )
            (
                _,
                _,
                _,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
                boundary_logits,
                boundary_alpha,
            ) = final_sam_outputs

        # Use the final prediction (after all correction steps for output and eval)
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )
        return current_out

    def _iter_correct_pt_sampling(
            self,
            is_init_cond_frame,
            point_inputs,
            gt_masks,
            high_res_features,
            pix_feat_with_mem,
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            object_score_logits,
            current_out,
            boundary_logits=None,
            boundary_alpha=None,
            boundary_prior=None,
    ):

        assert gt_masks is not None
        all_pred_masks = [low_res_masks]
        all_pred_high_res_masks = [high_res_masks]
        all_pred_multimasks = [low_res_multimasks]
        all_pred_high_res_multimasks = [high_res_multimasks]
        all_pred_ious = [ious]
        all_point_inputs = [point_inputs]
        all_object_score_logits = [object_score_logits]

        all_boundary_logits = [boundary_logits]
        all_boundary_alpha = [boundary_alpha]

        for _ in range(self.num_correction_pt_per_frame):
            # sample a new point from the error between prediction and ground-truth
            # (with a small probability, directly sample from GT masks instead of errors)
            if self.training and self.prob_to_sample_from_gt_for_train > 0:
                sample_from_gt = (
                        self.rng.random() < self.prob_to_sample_from_gt_for_train
                )
            else:
                sample_from_gt = False
            # if `pred_for_new_pt` is None, only GT masks will be used for point sampling
            pred_for_new_pt = None if sample_from_gt else (high_res_masks > 0)
            new_points, new_labels = get_next_point(
                gt_masks=gt_masks,
                pred_masks=pred_for_new_pt,
                method="uniform" if self.training else self.pt_sampling_for_eval,
            )
            point_inputs = concat_points(point_inputs, new_points, new_labels)
            # Feed the mask logits of the previous SAM outputs in the next SAM decoder step.
            # For tracking, this means that when the user adds a correction click, we also feed
            # the tracking output mask logits along with the click as input to the SAM decoder.
            mask_inputs = low_res_masks
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            if self.use_act_ckpt_iterative_pt_sampling and not multimask_output:
                sam_outputs = torch.utils.checkpoint.checkpoint(
                    self._forward_sam_heads,
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                    boundary_prior=boundary_prior,
                    use_reentrant=False,
                )
            else:
                sam_outputs = self._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                    boundary_prior=boundary_prior,
                )
            (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                _,
                object_score_logits,
                boundary_logits,
                boundary_alpha,
            ) = sam_outputs

            all_pred_masks.append(low_res_masks)
            all_pred_high_res_masks.append(high_res_masks)
            all_pred_multimasks.append(low_res_multimasks)
            all_pred_high_res_multimasks.append(high_res_multimasks)
            all_pred_ious.append(ious)
            all_point_inputs.append(point_inputs)
            all_object_score_logits.append(object_score_logits)

            all_boundary_logits.append(boundary_logits)
            all_boundary_alpha.append(boundary_alpha)

        # Concatenate the masks along channel (to compute losses on all of them,
        # using `MultiStepIteractiveMasks`)
        current_out["multistep_pred_masks"] = torch.cat(all_pred_masks, dim=1)
        current_out["multistep_pred_masks_high_res"] = torch.cat(
            all_pred_high_res_masks, dim=1
        )
        current_out["multistep_pred_multimasks"] = all_pred_multimasks
        current_out["multistep_pred_multimasks_high_res"] = all_pred_high_res_multimasks
        current_out["multistep_pred_ious"] = all_pred_ious
        current_out["multistep_point_inputs"] = all_point_inputs
        current_out["multistep_object_score_logits"] = all_object_score_logits
        current_out["multistep_pred_boundary_logits"] = all_boundary_logits
        current_out["multistep_pred_boundary_alpha"] = all_boundary_alpha
        return point_inputs, sam_outputs