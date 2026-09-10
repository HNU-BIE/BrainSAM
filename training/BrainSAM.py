import os

from sam2.modeling.sam2_utils import MLP
from sam2.modeling.sam2_base import SAM2Base
from training.utils.data_utils import BatchedVideopoint
from sam2.modeling.sam2_utils import (
    get_next_point,
    sample_box_points
)
join = os.path.join

import math
from typing import Type

import torch
import torch.nn as nn
from data.data_info import *

def get_index(map_idx):
    idx = map_idx
    for i in range(list(map_idx.items())[-1][0]):
        if i not in idx:
            idx[i] = 0
    idx = dict(sorted(idx.items(),key=lambda item: item[0]))
    return torch.LongTensor(list(idx.values()))

class MoEAdaptMLPBlock(nn.Module):
    def __init__(
            self,
            mlp: MLP,
            expert_num: int,
            embedding_num: int,
            embedding_dim: int,
            adapter_bn: int,
            adapter_act: Type[nn.Module] = nn.GELU,
            adapter_dropout: float = 0.1,
            adapter_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.mlp = mlp
        self.expert_num = expert_num

        dim = mlp.embedding_dim
        self.adapter_input_embed = nn.Linear(dim, embedding_dim)
        self.adapter_gate = nn.Linear(embedding_dim * embedding_num, expert_num)
        self.adapter_organ_gate = nn.Linear(embedding_dim * 4, 4)

        self.adapter_down = nn.ModuleList([
            nn.Linear(dim, adapter_bn) for _ in range(expert_num)
        ])
        self.adapter_up = nn.ModuleList([
            nn.Linear(adapter_bn, dim) for _ in range(expert_num)
        ])
        self.adapter_act = adapter_act()
        self.adapter_dropout = adapter_dropout
        if adapter_dropout is None:
            self.adapter_scale = nn.Parameter(torch.ones(1))
        else:
            self.adapter_scale = adapter_scalar

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.expert_num):
            nn.init.kaiming_uniform_(self.adapter_down[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.adapter_down[i].bias)
            nn.init.zeros_(self.adapter_up[i].weight)
            nn.init.zeros_(self.adapter_up[i].bias)

    def forward(self, x: torch.Tensor,
                modal: torch.Tensor, organ: torch.Tensor) -> torch.Tensor:
        adpt = []
        for i in range(self.expert_num):
            adpt.append(self.adapter_up[i](nn.functional.dropout(
                self.adapter_act(self.adapter_down[i](x)),
                p=self.adapter_dropout,
                training=self.training
            )))
        adpt = torch.stack(adpt, dim=1)

        organ_0, organ_1, task = organ
        organ = torch.cat([organ_0, organ_1], dim=1)#首尾拼接送进线性层
        organ_gate = self.adapter_organ_gate(organ)
        organ_gate = torch.softmax(organ_gate, dim=-1)
        organ = torch.stack([organ_0, organ_1], dim=1)
        organ = torch.einsum('bec,be->bc', organ, organ_gate)#消失的字母代表在这个维度求和

        input = x.mean(dim=1).mean(dim=1)
        input = self.adapter_input_embed(input)

        gate = torch.cat([organ, task, modal, input], dim=-1)#首尾拼接送进线性层
        gate = self.adapter_gate(gate)
        gate = torch.softmax(gate, dim=1)

        adpt = torch.einsum('bec,be->bc', adpt, gate)
        return self.mlp(x) + adpt * self.adapter_scale, gate

class BrainSAM(SAM2Base):
    def __init__(
          self,embedding_dim: int = 64,
            pos: list = None,
            expert_num: int = 8,
            bottleneck_dim: int = 256,
            **kwargs,

    ):
        super(BrainSAM,self).__init__()

        #在哪些层插入适配器
        if pos:
            self.pos = pos
        else:
            self.pos = list(range(self.image_encoder.trunk.blocks))

        modal_index = get_index(modal_map_idx)
        brain_index_1 = get_index(brain_level_1_map_idx)
        brain_index_2 = get_index(brain_level_2_map_idx)

        self.image_encoder.register_buffer('modal_index',modal_index,False)
        self.image_encoder.register_buffer('brain_index_1',brain_index_1,False)
        self.image_encoder.register_buffer('brain_index_2',brain_index_2,False)

        modal_embed = nn.Embedding(len(modal_index),embedding_dim)
        organ_embed_0 = nn.Embedding(1,embedding_dim)
        organ_embed_1 = nn.Embedding(len(brain_level_1_map_idx),embedding_dim)
        organ_embed_2 = nn.Embedding(len(brain_level_2_map_idx),embedding_dim)
        nn.init_zeros_(modal_embed.weight)
        nn.init_zeros_(organ_embed_0.weight)
        nn.init_zeros_(organ_embed_1.weight)
        nn.init_zeros_(organ_embed_2.weight)

        self.image_encoder.modal_embed = modal_embed
        self.image_encoder.organ_embed = nn.ModuleList([organ_embed_0, organ_embed_1])

        for idx, blk in enumerate(self.image_encoder.trunk.blocks):
            if idx not in self.pos:
                continue

            blk.mlp = MoEAdaptMLPBlock(
                mlp=blk.mlp,
                expert_num=len(modal_index),
                embedding_num=expert_num,
                embedding_dim=4,
                adapter_bn=bottleneck_dim,
            )

    def forward(self, x: BatchedVideopoint,
                modal: torch.Tensor, organ: torch.Tensor):
        #模态和组织编码
        modal_index = self.image_encoder.trunk.modal_indx[modal]
        modal_embed = self.image_encoder.modal_embed(modal_index)

        organ_1, organ_2, task = organ
        organ_index_0 = torch.zeros(B, dtype=torch.long, device=x.device)
        organ_embed_0 = self.image_encoder.organ_embed[0](organ_index_0)
        organ_index_1 = self.image_encoder.organ_index_1(organ_1)
        organ_embed_1 = self.image_encoder.organ_embed[1](organ_index_1)
        # organ_index_2 = self.image_encoder.organ_index_2(organ_2)
        # organ_embed_2 = self.image_encoder.organ_embed[2](organ_index_2)
        organ_embed = [organ_embed_0, organ_embed_1, task]
        if self.training or not self.forward_backbone_per_frame_for_eval:
            # precompute image features on all frames before tracking
            backbone_out = self.forward_image(x.flat_img_batch, modal_embed, organ_embed)
        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        
        #提示编码
        backbone_out = self.prepare_prompt_inputs(backbone_out, x)

        previous_stages_out = self.forward_tracking(backbone_out, x)

        return previous_stages_out

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
        # gt_masks_per_frame = {
        #     stage_id: targets.segments.unsqueeze(1)  # [B, 1, H_im, W_im]
        #     for stage_id, targets in enumerate(input.find_targets)
        # }
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
        for t in init_cond_frames:
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
        )

        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]

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
            )
            (
                _,
                _,
                _,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
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
    ):

        assert gt_masks is not None
        all_pred_masks = [low_res_masks]
        all_pred_high_res_masks = [high_res_masks]
        all_pred_multimasks = [low_res_multimasks]
        all_pred_high_res_multimasks = [high_res_multimasks]
        all_pred_ious = [ious]
        all_point_inputs = [point_inputs]
        all_object_score_logits = [object_score_logits]
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
                    use_reentrant=False,
                )
            else:
                sam_outputs = self._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                )
            (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                _,
                object_score_logits,
            ) = sam_outputs
            all_pred_masks.append(low_res_masks)
            all_pred_high_res_masks.append(high_res_masks)
            all_pred_multimasks.append(low_res_multimasks)
            all_pred_high_res_multimasks.append(high_res_multimasks)
            all_pred_ious.append(ious)
            all_point_inputs.append(point_inputs)
            all_object_score_logits.append(object_score_logits)

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

        return point_inputs, sam_outputs